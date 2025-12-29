import torch
import torch.nn as nn

class SimplifiedQFormerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        
        # 1. self-attention layer
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        self.ln1 = nn.LayerNorm(dim)
        
        # 2. cross-attention layer 
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        
        # 3. mlp
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        self.ln3 = nn.LayerNorm(dim)
    
    def forward(self, queries, image_features):
        # queries [B, num_queries, dim]
        # feature [B, num_patches, dim]
        
        # 1. self-attention
        residual = queries
        queries2, _ = self.self_attn(queries, queries, queries)
        queries = self.ln1(residual + queries2)
        
        # 2. cross-attention
        residual = queries
        queries2, _ = self.cross_attn(query=queries, key=image_features, value=image_features)
        queries = self.ln2(residual + queries2)
        
        # 3. mlp
        residual = queries
        queries2 = self.mlp(queries)
        queries = self.ln3(residual + queries2)
        
        return queries

class QFormer(nn.Module):
    def __init__(self, num_queries, embed_dim, num_heads, num_layers):
        super().__init__()        
        # learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        
        # stack of transformer blocks
        self.layers = nn.ModuleList([
            SimplifiedQFormerBlock(dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
    
    def forward(self, image_features):
        batch_size = image_features.size(0)
        queries = self.query_tokens.expand(batch_size, -1, -1)  # [B, num_queries, dim]
        for layer in self.layers:
            queries = layer(queries, image_features)
        return queries

img_feats = torch.randn(2, 197, 1024)  # Example image features
qformer = QFormer(num_queries=32, embed_dim=1024, num_heads=8, num_layers=6)
output = qformer(img_feats)
print(f"Image features shape: {img_feats.shape}")
print(f"Q-Former output shape: {output.shape}")  # Expected output shape: (2, 32, 1024)