import torch
import torch.nn as nn


class MHA(nn.Module):
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
        
        scores = torch.matmual(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim)) # Scaled dot-product attention (B, head_num, seq_len, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        out = torch.matmul(attention, V)  # (B, head_num, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, seq_len, self.embed_dim)  # (B, seq_len, embed_dim)
        out = self.out_linear(out) # (B, seq_len, embed_dim)
        return out


        