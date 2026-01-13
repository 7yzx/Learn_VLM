import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
    """
    Docstring for MHA, 多头注意力
    """
    def __init__(self, embed_dim, head_num):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_num = head_num
        
        self.head_dim = embed_dim // head_num
        
        assert(self.head_dim * head_num == embed_dim), "Embedding dimension must be divisible by number of heads"
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, feature, mask):
        B, seq_len, _ = feature.size()
        
        Q = self.q_linear(feature).view(B, seq_len, self.head_num, self.head_dim)    # (B, seq_len, embed_dim) -> (B, seq_len, embed_dim) -> (B, seq_len, head_num, head_dim)
        K = self.k_linear(feature).view(B, seq_len, self.head_num, self.head_dim)
        V = self.v_linear(feature).view(B, seq_len, self.head_num, self.head_dim)
        
        Q = Q.transpose(1, 2)  # (B, head_num, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim)) # Scaled dot-product attention (B, head_num, seq_len, seq_len)
        if mask is not None:
            mask = mask.view(B, 1, 1, seq_len)  # (B, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        out = torch.matmul(attention, V)  # (B, head_num, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, seq_len, self.embed_dim)  # (B, seq_len, embed_dim)
        out = self.out_linear(out) # (B, seq_len, embed_dim)
        return out


class GroupQueryAttention(nn.Module):
    """
    Group query Attention
    把q 分成g组 每组query 共享 K V
    """
    def __init__(self, embed_dim=512, num_heads=8, group_num=4, droup_out=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.group_num = group_num
        self.droup_out = droup_out
        # head_dim: 每个注意力头的维度。例如 512 / 8 = 64
        self.head_dim = embed_dim // num_heads
        # group_heads: 核心概念。每个 KV 组对应多少个 Q 头。
        # 例如：8个头，分4组。那么每组有 2 个 Q 头共享 1 个 KV 头。
        self.group_heads = num_heads // group_num
        
        # --- 定义投影层 ---
        # Q 的投影：保持完整的维度 (embed_dim -> embed_dim) ，同时有 num_heads 个 Q. 关系 embed_dim = num_heads * head_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        
        # K 和 V 的投影：维度被压缩了！这是 GQA 的精髓。 
        # 输出维度不是 embed_dim，而是 group_num * head_dim。也就是只生成 group_num 个 K 和 V，而不是 num_heads 个。
        self.k_proj = nn.Linear(embed_dim, self.group_num * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.group_num * self.head_dim)
        
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(droup_out)
    
    def forward(self, x, key_padding_mask=None):
        # 输入 x 的维度: (Batch_Size, Seq_Len, Embed_Dim)
        bs, seq_len, _ = x.size()
        # 1. 线性投影
        q = self.q_proj(x) # (B, S, 512) 
        k = self.k_proj(x) # (B, S, 4*64=256)
        v = self.v_proj(x) # (B, S, 4*64=256)
        
        # 2. 拆分多头与维度变换 (Reshape)
        # (B, S, 512) -> (B, S, num_heads, head_dim) -> （B，num_heads, S, head_dim)
        # (B, 100, 512)->(B, 100, 8, 64)->(B, 8, 100, 64)
        q = q.view(bs, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # K (B, S, group_num * head_dim) -> (B, S, group_num, head_dim) -> (B, group_num, head_dim, S)
        # (B, 100, 256) -> (B, 100, 4, 64) -> (B, 4, 64, 100)
        k = k.view(bs, seq_len, self.group_num, self.head_dim).permute(0, 2, 3, 1)
        
        # V (B, S, group_num * head_dim) -> (B, S, group_num, head_dim) -> (B, group_num, S, head_dim)
        # (B, 100, 256) -> (B, 100, 4, 64) -> (B, 4, 100, 64)
        v = v.view(bs, seq_len, self.group_num, self.head_dim).permute(0, 2, 1, 3)
        
        # 3. 广播/复制 KV (GQA 核心步骤)
        # 现在的 K 是 (B, 4, 64, S)，但 Q 有 8 个头。我们需要把 K 里的 4 个组复制，扩展成 8 个头。
        # unsqueeze(2): 在第2维插入一个维度 -> (B, 4, 1, 64, S)
        # expand: 把插入的维度复制 group_heads(2) 次 -> (B, 4, 2, 64, S)
        # contiguous: 重新整理内存，因为 expand 只是视图，不物理复制，后面 view 需要连续内存
        k = k.unsqueeze(2).expand(-1, -1, self.group_heads, -1, -1).contiguous()
        # view: 将 (groups, group_heads) 合并为 num_heads
        # (B, 4, 2, 64, S) -> (B, 8, 64, S)
        # 这里的 8 = 2 * 4
        k = k.view(bs, self.num_heads, self.head_dim, seq_len)
        
        # 对 V 做同样的操作
        # (B, 2, S, 64) -> (B, 2, 1, S, 64) -> (B, 2, 4, S, 64)
        v = v.unsqueeze(2).expand(-1, -1, self.group_heads, -1, -1).contiguous()
        # (B, 2, 4, S, 64) -> (B, 8, S, 64)
        v = v.view(bs, self.num_heads, seq_len, self.head_dim)
        
        # matmul: (B, 8, S, 64) @ (B, 8, 64, S) -> (B, 8, S, S)
        attn_scores = torch.matmul(q, k) / (self.head_dim ** 0.5)
        
        if key_padding_mask is not None:
            # [B, S] -> [B, 1, 1, S]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_padding_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1) # (B, 8, S, S)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v) # (B, 8, S, 64)
        
        # 多头拼接
        output = output.permute(0, 2, 1, 3).contiguous().view(bs, seq_len, self.embed_dim)  # (B, S, 512)
        output = self.o_proj(output)  # (B, S, 512)
        return output
    
if __name__ == "__main__":
    x = torch.randn(2, 100, 512)  # (B, S, E)
    mask = torch.ones(2, 100).bool()  # (B, S)
    print("---- MHA Test ----")
    mha = MHA(embed_dim=512, head_num=8)
    out = mha(x, mask)
    print(out.shape)  # Expected output: (2, 100, 512)
    
    
    print("---- GQA Test ----")
    gqa = GroupQueryAttention(embed_dim=512, num_heads=8, group_num=4)
    out = gqa(x, key_padding_mask=mask)
    print(out.shape)  # Expected output: (2, 100, 512)
    
    
    
        
        
        
        
        

        