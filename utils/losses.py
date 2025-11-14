import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM
import ot

# CLIP-Style Contrastive Loss
def clip_contrastive_loss(z_img, z_txt, temperature: float = 0.07):
    """
    CLIP-style contrastive loss between image and text embeddings.
    """
    device = z_img.device
    z_img = F.normalize(z_img, dim=-1)
    z_txt = F.normalize(z_txt, dim=-1)

    logits = (z_img @ z_txt.t()) / temperature
    labels = torch.arange(len(z_img), device=device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2


# MLM Loss 
class mlm_loss(nn.Module):
    """
    MLM loss using a pretrained transformer (BERT, RoBERTa, etc.)
    Ensures all inputs & model are on the same device.
    """
    def __init__(self, model_name: str = "bert-base-uncased", freeze: bool = False, device=None):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mlm_model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)

        if freeze:
            for p in self.mlm_model.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        outputs = self.mlm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss


# MAE Loss
class MAE_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recon_imgs, target_imgs, mask=None):
        if not isinstance(recon_imgs, torch.Tensor) and hasattr(recon_imgs, "logits"):
            recon_imgs = recon_imgs.logits

        target_imgs = target_imgs.to(recon_imgs.device)
        if mask is not None:
            mask = mask.to(recon_imgs.device)

        # Expect shapes: recon_imgs, target_imgs: (B, L, P), mask: (B, L)
        diff = recon_imgs - target_imgs  # (B, L, P)

        if mask is not None:
            mask = mask.unsqueeze(-1).float()  # (B, L, 1)
            diff = diff * mask  # zero out unmasked patches

            denom = (mask.sum() * diff.shape[-1]).clamp_min(1.0)
        else:
            denom = diff.numel()

        return diff.abs().sum() / denom


# Optimal Transport (OT)
def sinkhorn_ot_loss(x, y, eps=0.1, iters=50, return_plan=False):
    """
    OT loss between two feature sets using Sinkhorn iterations.
    """
    device = x.device
    y = y.to(device)

    C = torch.cdist(x, y, p=2) ** 2
    K = torch.exp(-C / eps)

    r = torch.ones(x.size(0), device=device) / x.size(0)
    c = torch.ones(y.size(0), device=device) / y.size(0)

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


# Anchored Optimal Transport
def anchored_ot_loss(x, y, paired_indices=None, eps=0.1, iters=50, alpha=0.1):
    """
    Anchored OT loss: reduces transport cost for known paired samples.
    """
    device = x.device
    y = y.to(device)

    C = torch.cdist(x, y, p=2) ** 2
    if paired_indices is not None:
        for i, j in paired_indices:
            if i < C.size(0) and j < C.size(1):
                C[i, j] *= alpha

    K = torch.exp(-C / eps)
    r = torch.ones(x.size(0), device=device) / x.size(0)
    c = torch.ones(y.size(0), device=device) / y.size(0)

    u = torch.ones_like(r)
    v = torch.ones_like(c)
    for _ in range(iters):
        u = r / (K @ v)
        v = c / (K.t() @ u)

    P = torch.diag(u) @ K @ torch.diag(v)
    return torch.sum(P * C)

# Gromov–Wasserstein
def gromov_wasserstein_loss(x, y, eps=1e-3, loss_fun="square_loss"):
    """
    Gromov–Wasserstein distance between two feature distributions.
    GW aligns based on relational geometry
    """
    # Convert to numpy (POT only supports numpy)
    X = x.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()

    # Pairwise distance matrices
    Cx = ot.utils.dist(X, X)
    Cy = ot.utils.dist(Y, Y)

    # Uniform distributions
    p = ot.unif(len(X))
    q = ot.unif(len(Y))

    # Compute GW distance
    gw, _ = ot.gromov.gromov_wasserstein2(
        Cx, Cy, p, q, loss_fun=loss_fun, epsilon=eps, verbose=False, log=True
    )
    return torch.tensor(gw, device=x.device, dtype=torch.float32)