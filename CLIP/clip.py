import torch
import torch.nn as nn
from models.vit import VisionEncoder
from models.tokenization import TextEncoder, TextEncoder_Retrieval
import numpy as np
class CLIP(nn.Module):
    def __init__(self, 
                emb_dim,
                vit_layers,
                vit_d_model, 
                img_size,
                patch_size,
                n_channels,
                vit_heads,
                vocab_size,
                max_seq_length,
                text_heads,
                text_layers,
                text_d_model,
                retrieval=False,
    ):
        super().__init__()
        self.vision_encoder = VisionEncoder(
                vit_d_model,
                img_size,
                patch_size,
                n_channels,
                vit_heads,
                vit_layers,
                emb_dim,
            )
        if retrieval:
            self.text_encoder = TextEncoder_Retrieval(
                vocab_size,
                text_d_model,
                max_seq_length,
                text_layers,
                text_heads,
                emb_dim,
            )
        else:
            self.text_encoder = TextEncoder(
                vocab_size,
                text_d_model,
                max_seq_length,
                text_layers,
                text_heads,
                emb_dim,
                )
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    def CLIPLoss(self, logits, device="cuda"):
        """
        Contrastive loss 
        """
        labels = torch.arange(logits.shape[0]).to(device)    
        loss_v = nn.functional.cross_entropy(logits.transpose(-2, -1), labels)
        loss_t = nn.functional.cross_entropy(logits, labels)
        return (loss_v + loss_t) / 2
    
    def forward(self, image, text, text_mask=None):
        V_e = self.vision_encoder(image) # [B, emb_dim]
        T_e = self.text_encoder(text, mask=text_mask) # [B, emb_dim]
        print(f"V e {V_e.shape}, T e {T_e.shape}")
        
        logits = (V_e @ T_e.transpose(-2, -1)) * torch.exp(self.temperature)
        loss = self.CLIPLoss(logits, device=self.device)
        return loss 


