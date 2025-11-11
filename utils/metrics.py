import torch
import torch.nn.functional as F


# Cosine Similarity
def cosine_similarity(a, b):
    """
    Compute cosine similarity matrix between all pairs (a_i, b_j).
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.t()


# Retrieval Accuracy (Image→Text and Text→Image)
def retrieval_accuracy(z_img, z_txt, topk=(1, 5)):
    """
    Computes retrieval accuracy for image→text and text→image directions.
    """
    sim = cosine_similarity(z_img, z_txt)
    labels = torch.arange(z_img.size(0), device=z_img.device)
    results = {}

    # Image → Text retrieval
    rank_i = sim.argsort(dim=-1, descending=True)
    for k in topk:
        correct_i = (rank_i[:, :k] == labels.unsqueeze(1)).any(dim=1).float().mean()
        results[f"i2t@{k}"] = correct_i.item()

    # Text → Image retrieval
    rank_t = sim.t().argsort(dim=-1, descending=True)
    for k in topk:
        correct_t = (rank_t[:, :k] == labels.unsqueeze(1)).any(dim=1).float().mean()
        results[f"t2i@{k}"] = correct_t.item()

    return results

# Reconstruction PSNR (for MAE pretraining)
def reconstruction_psnr(recon, target):
    """
    Compute Peak Signal-to-Noise Ratio between reconstructed and target images.
    """
    mse = F.mse_loss(recon, target, reduction="mean")
    psnr = -10 * torch.log10(mse + 1e-8)
    return psnr.item()
