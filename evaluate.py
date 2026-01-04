import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.metrics import retrieval_accuracy

from models.mm_model import MultiModalModel
from data.flickr_dataset import FlickrMultiModalDataset

from data.ccrcc_dataset import CCRCCMultiModalDataset
from models.ccrcc_model import CcRCCMultiModalModel


def _strip_prefix_if_present(sd, prefix: str):
    if not isinstance(sd, dict):
        return sd
    keys = list(sd.keys())
    if keys and all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in sd.items()}
    return sd


def safe_load_state_dict(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)

    # Lightning-style wrapper
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    # Common wrappers/prefixes (DDP, custom)
    state = _strip_prefix_if_present(state, "module.")
    state = _strip_prefix_if_present(state, "model.")
    state = _strip_prefix_if_present(state, "net.")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): e.g. {missing[:5]}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): e.g. {unexpected[:5]}")


@torch.no_grad()
def extract_embeddings_flickr(model, dataloader, device, normalize=True):
    zs_img, zs_txt = [], []
    for batch in tqdm(dataloader, desc="Extracting embeddings (Flickr)"):
        img = batch["image"].to(device, non_blocking=True)
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)

        z_img = model.image_embed(img)
        z_txt = model.text_embed(ids, mask)

        if normalize:
            z_img = F.normalize(z_img, dim=-1)
            z_txt = F.normalize(z_txt, dim=-1)

        zs_img.append(z_img.detach().cpu())
        zs_txt.append(z_txt.detach().cpu())

    return torch.cat(zs_img, dim=0), torch.cat(zs_txt, dim=0)


@torch.no_grad()
def extract_embeddings_ccrcc(model, dataloader, device, normalize=True):
    zs_ct, zs_cl = [], []
    for batch in tqdm(dataloader, desc="Extracting embeddings (ccRCC)"):
        ct = batch["ct"].to(device, non_blocking=True)        # [B, ct_dim]
        cl = batch["clingen"].to(device, non_blocking=True)   # [B, clingen_dim]

        z_ct = model.image_embed(ct)
        z_cl = model.text_embed(cl)

        if normalize:
            z_ct = F.normalize(z_ct, dim=-1)
            z_cl = F.normalize(z_cl, dim=-1)

        zs_ct.append(z_ct.detach().cpu())
        zs_cl.append(z_cl.detach().cpu())

    return torch.cat(zs_ct, dim=0), torch.cat(zs_cl, dim=0)


def build_model_and_loader(args, device):
    pin = torch.cuda.is_available()

    if args.dataset == "flickr":
        model = MultiModalModel(
            vision_name=args.vision_name,
            text_name=args.text_name,
            shared_dim=args.shared_dim,
        ).to(device)

        dataset = FlickrMultiModalDataset(
            split="paired",
            paired_fraction=args.paired_fraction,
            train=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin,
        )
        return model, loader, extract_embeddings_flickr

    if args.dataset == "ccrcc":
        model = CcRCCMultiModalModel(
            ct_dim=args.ct_dim,
            clingen_dim=args.clingen_dim,
            shared_dim=args.shared_dim,
            hidden_dim=args.hidden_dim,
            ct_mask_ratio=args.ct_mask_ratio,
            clingen_mask_ratio=args.clingen_mask_ratio,
        ).to(device)

        dataset = CCRCCMultiModalDataset(
            root=args.ccrcc_root,
            split="paired",
            subset=args.ccrcc_subset,
            return_labels=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin,
        )
        return model, loader, extract_embeddings_ccrcc

    raise ValueError(f"Unknown dataset: {args.dataset}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating {args.dataset} checkpoint: {args.checkpoint} on {device}")

    model, loader, extractor = build_model_and_loader(args, device)

    safe_load_state_dict(model, args.checkpoint, device)
    model.eval()

    z_a, z_b = extractor(model, loader, device, normalize=True)
    metrics = retrieval_accuracy(z_a, z_b)

    print("\n=== Retrieval Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate cross-modal retrieval (Flickr30k or ccRCC)")

    p.add_argument("--dataset", choices=["flickr", "ccrcc"], required=True)
    p.add_argument("--checkpoint", type=str, required=True)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--shared_dim", type=int, default=256)

    p.add_argument("--vision_name", type=str, default="vit_base_patch16_224")
    p.add_argument("--text_name", type=str, default="bert-base-uncased")

    # Flickr-specific
    p.add_argument("--paired_fraction", type=float, default=1.0)

    # ccRCC-specific
    p.add_argument("--ccrcc_root", type=str, default="data")
    p.add_argument("--ccrcc_subset", choices=["train", "test"], default="test")
    p.add_argument("--ct_dim", type=int, default=512)
    p.add_argument("--clingen_dim", type=int, default=18)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--ct_mask_ratio", type=float, default=0.25)
    p.add_argument("--clingen_mask_ratio", type=float, default=0.25)

    args = p.parse_args()
    main(args)
