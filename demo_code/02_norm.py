import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features)) # features shape Dim
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # B seq_len 1 
        std = x.std(dim=-1, keepdim=True) # B seqlen 1

        return self.gamma * (x - mean) / (std + self.eps) + self.beta




dim = 1024
ln = LayerNorm(dim)
x = torch.randn(2, 76, 1024)
normlize_x = ln(x)
print(f"x shape {x.shape}, normlize_x shape {normlize_x.shape}")
