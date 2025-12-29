import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, droupout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(droupout),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x):
        # x shape : (batch_size, seq_len, in_dim)
        return self.net(x)


in_dim = 1024
input_tensor = torch.randn(2, 10, in_dim)  # Example input tensor
mlp = MLP(in_dim=in_dim, hidden_dim=4*in_dim, out_dim=in_dim)        
output = mlp(input_tensor)
print(output.shape)  # Expected output shape: (2, 10, 1024)


