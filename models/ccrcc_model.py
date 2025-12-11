import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple 2-layer MLP used for encoders/decoders."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CcRCCMultiModalModel(nn.Module):
    """
    Feature-level multimodal model for ccRCC.

    Modalities:
      - ct       : CT feature vector (treated as "image" modality)
      - clingen  : clinical+genomic feature vector (treated as "text" modality)
    """

    def __init__(
        self,
        ct_dim: int,
        clingen_dim: int,
        shared_dim: int = 256,
        hidden_dim: int = 512,
        ct_mask_ratio: float = 0.25,
        clingen_mask_ratio: float = 0.25,
    ):
        """
        Args:
            ct_dim:          dimensionality of CT feature vectors
            clingen_dim:     dimensionality of clingen feature vectors
            shared_dim:      joint embedding dimension (for CLIP/OT)
            hidden_dim:      hidden size in encoders/decoders
            ct_mask_ratio:   default masking ratio for ct MAE
            clingen_mask_ratio: default masking ratio for clingen MLM
        """
        super().__init__()
        self.ct_dim = ct_dim
        self.clingen_dim = clingen_dim
        self.shared_dim = shared_dim
        self.hidden_dim = hidden_dim
        self.ct_mask_ratio = ct_mask_ratio
        self.clingen_mask_ratio = clingen_mask_ratio

        # Encoders → shared space
        self.ct_encoder = MLP(ct_dim, hidden_dim, shared_dim)
        self.clingen_encoder = MLP(clingen_dim, hidden_dim, shared_dim)

        self.ct_decoder = MLP(shared_dim, hidden_dim, ct_dim)
        self.clingen_decoder = MLP(shared_dim, hidden_dim, clingen_dim)

    @torch.no_grad()
    def image_embed(self, ct: torch.Tensor) -> torch.Tensor:
        """
        CT → shared embedding; normalized.
        ct: [B, ct_dim]
        """
        z = self.ct_encoder(ct)
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def text_embed(self, clingen: torch.Tensor) -> torch.Tensor:
        """
        clingen: [B, clingen_dim]
        """
        z = self.clingen_encoder(clingen)
        return F.normalize(z, dim=-1)

    def _mask_features(self, x: torch.Tensor, mask_ratio: float):
        """
        Randomly mask a fraction of feature dimensions.
        """
        B, D = x.shape
        device = x.device
        # independent Bernoulli per feature dim
        mask = (torch.rand(B, D, device=device) < mask_ratio)
        corrupted = x.clone()
        corrupted[mask] = 0.0
        return corrupted, mask

    def ct_mae_step(self, ct: torch.Tensor, mask_ratio: float = None):
        """
        MAE-style masked feature modeling for CT.
        """
        if mask_ratio is None:
            mask_ratio = self.ct_mask_ratio
        corrupted, mask = self._mask_features(ct, mask_ratio)
        z = self.ct_encoder(corrupted)
        pred = self.ct_decoder(z)
        return pred, ct, mask

    def clingen_mlm_step(self, clingen: torch.Tensor, mask_ratio: float = None):
        """
        MLM-style masked feature modeling for clingen.
        """
        if mask_ratio is None:
            mask_ratio = self.clingen_mask_ratio
        corrupted, mask = self._mask_features(clingen, mask_ratio)
        z = self.clingen_encoder(corrupted)
        pred = self.clingen_decoder(z)
        return pred, clingen, mask

    def forward(
        self,
        ct: torch.Tensor = None,
        clingen: torch.Tensor = None,
        do_recon: bool = False,
        ct_mask_ratio: float = None,
        clingen_mask_ratio: float = None,
    ):
        """
        Args:
            ct:        [B, ct_dim] CT feature vectors (image modality)
            clingen:   [B, clingen_dim] clingen feature vectors (text modality)
            do_recon:  if True, also perform masked reconstruction for available modalities
            ct_mask_ratio, clingen_mask_ratio: optional overrides

        """
        out = {}

        if ct is not None:
            z_ct = self.ct_encoder(ct)
            out["z_img"] = F.normalize(z_ct, dim=-1)

        if clingen is not None:
            z_cl = self.clingen_encoder(clingen)
            out["z_txt"] = F.normalize(z_cl, dim=-1)

        if do_recon and ct is not None:
            pred_ct, target_ct, mask_ct = self.ct_mae_step(
                ct, mask_ratio=ct_mask_ratio
            )
            out["ct_pred"] = pred_ct
            out["ct_target"] = target_ct
            out["ct_mask"] = mask_ct

        if do_recon and clingen is not None:
            pred_cl, target_cl, mask_cl = self.clingen_mlm_step(
                clingen, mask_ratio=clingen_mask_ratio
            )
            out["clingen_pred"] = pred_cl
            out["clingen_target"] = target_cl
            out["clingen_mask"] = mask_cl

        return out
