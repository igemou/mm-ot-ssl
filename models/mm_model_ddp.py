# mm_model.py
import torch
import torch.nn as nn
from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from models.projector import MLPProjector
from models.mae_decoder import MAEDecoder, PretrainedMAEDecoder
from utils.mae import patchify, random_masking
from collections import defaultdict

from utils.losses import (
    clip_contrastive_loss,
    sinkhorn_ot_loss,
    anchored_ot_loss,
    gromov_wasserstein_loss, 
    mlm_loss,
    MAE_loss,
)

class DDPMultiModalModel(nn.Module):
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
        # self.mae_decoder = MAEDecoder(embed_dim=self.vision_encoder.out_dim, patch_size=patch_size)
        self.mae_decoder = PretrainedMAEDecoder()
        self.patch_size = patch_size

        self.mae_loss_fn = MAE_loss()
        self.mlm_loss_fn = mlm_loss(model_name=text_name)

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

    def pred(self,
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
            if hasattr(pred, "logits"):
                pred = pred.logits
            out["pred_patches"] = pred
            out["target_patches"] = patches

        # MLM / CLIP / OT 
        if input_ids is not None:
            pooled, _ = self.text_encoder(input_ids, attention_mask)
            z_txt = self.text_proj(pooled)
            out["z_txt"] = z_txt

        return out


    def forward(self,
                paired_batch = None,
                unpaired_batch = None,
                text_batch = None,
                image_batch = None,
                ot_loss = "gw",
                λ = defaultdict(int),
                alpha_anchor = 0
    ):
        preds = defaultdict(lambda: {})
        loss_dict = {}
        
        total_loss = 0.0
        
        # Paired CLIP loss
        if paired_batch is not None:
            out = self.pred(images=paired_batch["image"].to(self.device),
                            input_ids=paired_batch["input_ids"].to(self.device),
                            attention_mask=paired_batch["attention_mask"].to(self.device))
            for o, v in out.items():
                preds["paired"][o] = v.detach().cpu().numpy()

            #losses
            loss_clip = clip_contrastive_loss(out["z_img"], out["z_txt"])
            total_loss += λ["clip"] * loss_clip
            loss_dict["clip"] = loss_clip.item()

        # Unpaired OT 
        if unpaired_batch is not None and λ["ot"] > 0:
            if ot_loss == "anchored":
                images = torch.cat([paired_batch["image"], unpaired_batch["image"]], dim=0).to(self.device)
                input_ids = torch.cat([paired_batch["input_ids"], unpaired_batch["input_ids"]], dim=0).to(self.device)
                attn_mask = torch.cat([paired_batch["attention_mask"], unpaired_batch["attention_mask"]], dim=0).to(self.device)

                out = self.pred(images=images, input_ids=input_ids, attention_mask=attn_mask)
                for o, v in out.items():
                    preds["unpaired"][o] = v.detach().cpu().numpy()

                z_img, z_txt = out["z_img"], out["z_txt"]

                n_paired = len(paired_batch["image"])
                anchors = [(i, i) for i in range(n_paired)]
                loss_ot = anchored_ot_loss(z_img, z_txt, paired_indices=anchors, alpha=alpha_anchor)
            else:
                out = self.pred(images=unpaired_batch["image"].to(self.device),
                                input_ids=unpaired_batch["input_ids"].to(self.device),
                                attention_mask=unpaired_batch["attention_mask"].to(self.device))
                for o, v in out.items():
                    preds["unpaired"][o] = v.detach().cpu().numpy()
                
                z_img, z_txt = out["z_img"], out["z_txt"]

                if ot_loss == "gw":
                    loss_ot = gromov_wasserstein_loss(z_img, z_txt)
                else:
                    loss_ot = sinkhorn_ot_loss(z_img, z_txt)

            total_loss += λ["ot"] * loss_ot
            loss_dict["ot"] = loss_ot.item()

        # Text-only MLM 
        if text_batch is not None and λ["mlm"] > 0:
            loss_mlm = self.mlm_loss_fn(input_ids=text_batch["input_ids"].to(self.device),
                                        attention_mask=text_batch["attention_mask"].to(self.device),
                                        labels=text_batch["input_ids"].to(self.device),)
            total_loss += λ["mlm"] * loss_mlm
            loss_dict["mlm"] = loss_mlm.item()

        # Image-only MAE 
        if image_batch is not None and λ["mae"] > 0:
            out = self.pred(images=image_batch["image"].to(self.device))
            for o, v in out.items():
                preds["image"][o] = v.detach().cpu().numpy()

            loss_mae = self.mae_loss_fn(out["pred_patches"], out["target_patches"], out["mask"])
            total_loss += λ["mae"] * loss_mae
            loss_dict["mae"] = loss_mae.item()

        return preds, total_loss, loss_dict
