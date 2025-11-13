# vision_encoder.py
import torch
import torch.nn as nn
import timm
from typing import Optional

class VisionEncoder(nn.Module):
    """
    Vision encoder compatible with both CLIP-style embedding
    and MAE patch outputs.
    """
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        embed_dim: Optional[int] = None,
        freeze: bool = False,
        return_patches: bool = False,
    ):
        super().__init__()
        self.return_patches = return_patches

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        self.out_dim = self.backbone.num_features

        if embed_dim is not None and embed_dim != self.out_dim:
            self.proj = nn.Linear(self.out_dim, embed_dim)
            self.out_dim = embed_dim
        else:
            self.proj = None

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        If return_patches=False: (B, D)
        If return_patches=True: (B, N, D)
        """
        if hasattr(self.backbone, "forward_features") and self.return_patches:
            # Patch-level output (no global pooling)
            x = self.backbone.patch_embed(x)
            x = x + self.backbone.pos_embed[:, 1:, :]
            x = self.backbone.pos_drop(x)
            for blk in self.backbone.blocks:
                x = blk(x)
            feats = x  # (B, N, D)
        else:
            feats = self.backbone(x)  # (B, D)

        if self.proj is not None:
            feats = self.proj(feats)
        return feats
