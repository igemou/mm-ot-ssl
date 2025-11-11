import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

# CLIP-style
def clip_contrastive_loss(z_img, z_txt, temperature: float = 0.07):
    """
    CLIP-style contrastive loss between image and text embeddings.
    """
    z_img = F.normalize(z_img, dim=-1)
    z_txt = F.normalize(z_txt, dim=-1)

    logits = z_img @ z_txt.t() / temperature
    labels = torch.arange(len(z_img), device=z_img.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2

# MLM loss
class mlm_loss(nn.Module):
    """
    MLM loss using a pretrained transformer
    """
    def __init__(self, model_name: str = "bert-base-uncased", freeze: bool = False):
        super().__init__()
        self.mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)
        if freeze:
            for p in self.mlm_model.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.mlm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

# Masked Autoencoder (MAE)
class MAE_loss(nn.Module):
    """
    Mean Absolute Error loss for image reconstruction
    """
    def __init__(self):
        super().__init__()

    def forward(self, recon_imgs, target_imgs, mask=None):
        if mask is not None:
            diff = (recon_imgs - target_imgs) * mask
            denom = mask.sum()
        else:
            diff = recon_imgs - target_imgs
            denom = diff.numel()
        return diff.abs().sum() / (denom + 1e-8)

# Optimal Transport loss
def sinkhorn_ot_loss(x, y, eps=0.1, iters=50, return_plan=False):
    """
    OT loss between two feature sets.
    """
    C = torch.cdist(x, y, p=2) ** 2
    K = torch.exp(-C / eps)

    r = torch.ones(x.size(0), device=x.device) / x.size(0)
    c = torch.ones(y.size(0), device=y.device) / y.size(0)

    u = torch.ones_like(r)
    v = torch.ones_like(c)
    for _ in range(iters):
        u = r / (K @ v)
        v = c / (K.t() @ u)

    P = torch.diag(u) @ K @ torch.diag(v)
    ot_loss = torch.sum(P * C)

    if return_plan:
        return ot_loss, P
    return ot_loss

# Anchoredd OT
def anchored_ot_loss(x, y, paired_indices=None, eps=0.1, iters=50, alpha=0.1):
    """
    to reduce transport cost for known paired samples.
    """
    C = torch.cdist(x, y, p=2) ** 2
    if paired_indices is not None:
        for i, j in paired_indices:
            if i < C.size(0) and j < C.size(1):
                C[i, j] *= alpha

    K = torch.exp(-C / eps)
    r = torch.ones(x.size(0), device=x.device) / x.size(0)
    c = torch.ones(y.size(0), device=y.device) / y.size(0)
    u = torch.ones_like(r)
    v = torch.ones_like(c)
    for _ in range(iters):
        u = r / (K @ v)
        v = c / (K.t() @ u)

    P = torch.diag(u) @ K @ torch.diag(v)
    return torch.sum(P * C)
