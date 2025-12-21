import torch
import torch.nn as nn
import numpy as np

class TransformerEncoder(nn.Module):
  def __init__(self, d_model, n_heads, mlp_ratio =4):
      super().__init__()

      self.d_model = d_model
      self.n_heads = n_heads

      self.ln1 = nn.LayerNorm(d_model)

      self.mha = MultiheadAttention(d_model, n_heads)

      self.ln2 = nn.LayerNorm(d_model)

      self.mlp = nn.Sequential(
          nn.Linear(d_model, d_model*mlp_ratio),
          nn.GELU(),
          nn.Linear(d_model * mlp_ratio, d_model)
      )

  #For clip even though its a encoder model it requires mask ->to account for padded for max seq_length
  def forward(self, x, mask = None):

      x_n = self.mha(self.ln1(x), mask = mask)
      x = x + self.mlp(self.ln2(x_n))

      return x  # x.shape -->  [B,max_seq_len,d_model]


class AttentionHead(nn.Module):
  def __init__(self, d_model, qkv_dim):
    super().__init__()

    self.qkv_dim = qkv_dim

    self.query = nn.Linear(d_model, qkv_dim)
    self.key = nn.Linear(d_model, qkv_dim)
    self.value = nn.Linear(d_model, qkv_dim)

  def forward(self, x, mask=None):
    # x.shape --> [B, max_seq_len, d_model]
    Q = self.query(x)
    K = self.key(x)
    V = self.value(x)

    attention = Q @ K.transpose(-2, -1) #[B, max_seq_len, max_seq_len]
    attention = attention / (self.qkv_dim ** 0.5)
    # apply attention mask for padded sequence
    if mask is not None:
      mask = attention.masked_fill(
          mask == 0, float("-inf")
      )# torch.tensor.masked_fill

    attention = torch.softmax(attention , dim=-1) # (softmax(Q_K^T)/sqrt(d_k)).V

    attention = attention @ V

    return attention # Y_i

class MultiheadAttention(nn.Module):
  def __init__(self, d_model, n_heads):
    super().__init__()
    # d_model --> embed dimension
    # n_heads --> nums of heads
    self.qkv_dim = d_model // n_heads

    self.W_o = nn.Linear(d_model, d_model)
    self.multi_head = nn.ModuleList(
        [AttentionHead(d_model, self.qkv_dim) for _ in range(n_heads)]
    )

  def forward(self, x, mask=None):
    # x.shape --> [B, max_seq, d_model]
    # concatenates the outputs

    out = torch.cat(
        [head(x , mask=mask) for head in self.multi_head], dim=-1
    ) # [ B, max_seq_len, d_model]

    out = self.W_o(out) # [B, max_seq_len, d_model]

    return out

class PositionalEmbedding(nn.Module):
  def __init__(self, d_model, max_seq_len):
    super().__init__()
    self.d_model = d_model
    self.max_seq_len = max_seq_len
    pe = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    self.register_buffer("pe", pe.unsqueeze(0))

  def forward(self, x):
    # x.shape --> [B, max_seq_len, d_model]
    seq_len = x.size(1)
    return x + self.pe[:, :seq_len]
    # [B, max_seq_len, d_model] + [1, max_seq_len, d_model]
