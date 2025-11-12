# mm_model.py
import torch
import torch.nn as nn
from vision_encoder import VisionEncoder
from text_encoder import TextEncoder
from projector import MLPProjector
from models.mae_decoder import MAEDecoder
from utils.mae import patchify, random_masking

class MultiModalModel(nn.Module):
    """
    multimodal model supporting:
    - CLIP contrastive / OT alignment
    - MAE reconstruction
    - MLM (via text encoder)
    """
    def __init__(
        self,
        vision_name="vit_base_patch16_224",
        text_name="bert-base-uncased",
        shared_dim=256,
        freeze_vision=False,
        freeze_text=False,
        patch_size=16,
    ):
        super().__init__()
        # encoders
        self.vision_encoder = VisionEncoder(
            model_name=vision_name,
            pretrained=True,
            return_patches=True,  
            freeze=freeze_vision,
        )
        self.text_encoder = TextEncoder(model_name=text_name, freeze=freeze_text)

        #  projectors for joint embedding space
        self.vision_proj = MLPProjector(self.vision_encoder.out_dim, shared_dim)
        self.text_proj = MLPProjector(self.text_encoder.hidden_size, shared_dim)

        # MAE decoder 
        self.mae_decoder = MAEDecoder(embed_dim=self.vision_encoder.out_dim, patch_size=patch_size)
        self.patch_size = patch_size

    @torch.no_grad()
    def image_embed(self, images: torch.Tensor):
        """Global pooled embedding (for CLIP/OT eval)."""
        feats = self.vision_encoder(images)
        z = self.vision_proj(feats.mean(dim=1) if feats.ndim == 3 else feats)
        return z

    @torch.no_grad()
    def text_embed(self, input_ids, attention_mask=None):
        pooled, _ = self.text_encoder(input_ids, attention_mask)
        z = self.text_proj(pooled)
        return z

    def forward(
        self,
        images=None,
        input_ids=None,
        attention_mask=None,
        mask_ratio=0.75,
    ):
        out = {}

        #  MAE + CLIP/OT
        if images is not None:
            patches = patchify(images, self.patch_size)  # (B, N, patch_dim)
            visible, mask, ids_restore = random_masking(patches, mask_ratio)
            z_visible = self.vision_encoder(images)  # (B, N, D)
            z_img = self.vision_proj(z_visible.mean(dim=1))
            out["z_img"] = z_img
            out["mask"] = mask

            # MAE reconstruction
            pred = self.mae_decoder(z_visible, ids_restore)
            out["pred_patches"] = pred
            out["target_patches"] = patches

        # MLM / CLIP / OT 
        if input_ids is not None:
            pooled, _ = self.text_encoder(input_ids, attention_mask)
            z_txt = self.text_proj(pooled)
            out["z_txt"] = z_txt

        return out
