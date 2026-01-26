import torch
import torch.nn as nn
from .transformer import TransformerEncoder, PositionalEmbedding

def tokenizer(text, encode=True, mask=None, max_seq_length=32):
  if encode:
    # CLIP 在输入文本的开头和结尾分别加上 [BOS] (Begin Of Sequence) 和 [EOS] (End Of Sequence) 标记。
    # 这里用 chr(2) 代表 BOS，chr(3) 代表 EOS
    out = chr(2) + text + chr(3)

    # 代码确保所有输出长度一致为 max_seq_length。太长就切掉，太短就补 chr(0)。
    if len(out) > max_seq_length:
      out = out[:max_seq_length]

    out = out + "".join(
        [chr(0) for _ in range(max_seq_length - len(out))]
    )

    out = torch.IntTensor(list(out.encode("utf-8")))

    mask = torch.ones(len(out.nonzero()))

    if len(mask) < max_seq_length:
      mask = torch.cat((mask, torch.zeros(max_seq_length - len(mask)))).type(torch.IntTensor)
    else:
      mask = mask.type(torch.IntTensor)
  else:
    out = [chr(x) for x in text[1: len(mask.nonzero()) - 1]]
    out = "".join(out)
    mask = None
  
  return out, mask

class TextEncoder(nn.Module):
  def __init__(self, vocab_size, d_model, max_seq_length, n_layers,
               n_heads, emb_dim):
    super().__init__()
    self.max_seq_length = max_seq_length
    self.embed = nn.Embedding(vocab_size, d_model)

    self.positional_embedding = PositionalEmbedding(d_model, max_seq_length)

    self.transformer_encoder = nn.ModuleList(
        [TransformerEncoder(d_model, n_heads) for _ in range(n_layers)]
    )

    self.projection = nn.Parameter(torch.randn(d_model, emb_dim))

  def forward(self, text, mask=None):
    x = self.embed(text)
    x = self.positional_embedding(x)

    for encoder_layer in self.transformer_encoder:
      x = encoder_layer(x, mask=mask) # [B, max_seq_length, d_model]
    

    # Transformer 输出的 x 是一个序列，包含每个单词的特征（比如 [SOT, "一只", "狗", EOT, Pad, Pad...]）。
    # 取出每句话最后一个有效字符（即 EOS 标记）对应的向量。python 高级索引 给的是两个列表进行索引

    x = x[
        torch.arange(text.shape[0]), 
        torch.sub(torch.sum(mask[:,0], dim=1),1)
          ]
    
    # 投影到多模态公共空间。
    if self.projection is not None:
      x = x @ self.projection
    
    # L2 归一化
    x = x / torch.norm(x, dim=-1, keepdim=True)

    return x
  
  
class TextEncoder_Retrieval(nn.Module):
  def __init__(self, vocab_size, d_model, max_seq_length, n_layers,n_heads, emb_dim):
      super().__init__()

      self.max_seq_length = max_seq_length

      self.embed = nn.Embedding(vocab_size, d_model)

      self.positional_embedding = PositionalEmbedding(d_model, max_seq_length)

      self.transformer_encoder = nn.ModuleList([TransformerEncoder(d_model, n_heads) for _ in range(n_layers)])

      self.projection = nn.Parameter(torch.randn(d_model, emb_dim))


 # # For image retrieval
  def forward(self, text, mask=None):
        x = self.embed(text)
        x = self.positional_embedding(x)
    
        for encoder_layer in self.transformer_encoder:
            x = encoder_layer(x, mask=mask)
    
        if mask is not None:
            # Get the lengths of each sequence (i.e., find the last non-padded token)
            seq_lengths = mask.sum(dim=1) - 1  # Subtract 1 to get the index
            x = x[torch.arange(text.shape[0]), seq_lengths]
        else:
            x = x[:, -1]  # If no mask is provided, take the last token in the sequence.
    
        if self.projection is not None:
            x = x @ self.projection
    
        x = x / torch.norm(x, dim=-1, keepdim=True)
    
        return x
