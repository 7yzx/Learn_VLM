import torch
from torch import nn
from .transformer import TransformerEncoder, PositionalEmbedding 

class VisionEncoder(nn.Module):
  def __init__(self, d_model, img_size, patch_size,
               n_channels, n_heads, n_layers, emb_dim):
    super().__init__()
    assert (
        img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0
    ), "image dimensions should be divisible by patch dim"
    assert d_model % n_heads == 0, "d_model should be divisible by n_heads"

    self.num_patches = (img_size[0] * img_size[1]) // (patch_size[0] * patch_size[1])
    # max_seq length

    self.max_seq_length = self.num_patches + 1

    self.linear_proj = nn.Conv2d(
        in_channels=n_channels,
        out_channels=d_model,
        kernel_size=patch_size,
        stride=patch_size[0],
    )

    self.cls_token = nn.Parameter(torch.randn(1,1,d_model),requires_grad=True)

    self.positional_embedding = PositionalEmbedding(d_model, self.max_seq_length)

    self.transformer_encoder = nn.ModuleList(
        [TransformerEncoder(d_model, n_heads) for _ in range(n_layers)]
    )

    self.projection = nn.Parameter(torch.randn(d_model, emb_dim))

  def forward(self, x, mask=None):
    x = self.linear_proj(x)
    # [B,C,H,W] -> (B, d_model, patch_col_d_model, patch_row_d_model)
    x = x.flatten(2).transpose(-2, -1)
    # (B, d_model, Patch_col_d_model, Patch_row_height) --> Flatten (B, d_model, Patch) --> .transpose(-2,-1) (B, Patch, d_model)

    x = torch.cat(
        (self.cls_token.expand(x.shape[0], -1, -1), x), dim=1
    )

    x = self.positional_embedding(x)

    for encoder_layer in self.transformer_encoder:
      x = encoder_layer(x, mask)

    x = x[:, 0, :]

    if self.projection is not None:
      x = x @ self.projection

    x = x / torch.norm(x, dim=-1,keepdim=True)

    return x



