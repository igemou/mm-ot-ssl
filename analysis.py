import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.metrics import retrieval_accuracy

from data.flickr_dataset import FlickrMultiModalDataset
from models.mm_model import MultiModalModel

from data.ccrcc_dataset import CCRCCMultiModalDataset
from models.ccrcc_model import CcRCCMultiModalModel


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _strip_prefix_if_present(state_dict, prefix: str):
    # Handles DDP "module." etc.
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if keys and all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def safe_load_state_dict(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    state = _strip_prefix_if_present(state, "module.")
    state = _strip_prefix_if_present(state, "model.")
    state = _strip_prefix_if_present(state, "net.")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint ({len(missing)}): e.g. {missing[:5]}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint ({len(unexpected)}): e.g. {unexpected[:5]}")


@torch.no_grad()
def extract_flickr(model, loader, device, normalize=True):
    z_img_all, z_txt_all = [], []
    for batch in tqdm(loader, desc="Extracting embeddings (Flickr)"):
        img = batch["image"].to(device, non_blocking=True)
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)

        z_img = model.image_embed(img)
        z_txt = model.text_embed(ids, mask)

        if normalize:
            z_img = torch.nn.functional.normalize(z_img, dim=-1)
            z_txt = torch.nn.functional.normalize(z_txt, dim=-1)

        z_img_all.append(z_img.detach().cpu())
        z_txt_all.append(z_txt.detach().cpu())

    return torch.cat(z_img_all, 0), torch.cat(z_txt_all, 0)


@torch.no_grad()
def extract_ccrcc(model, loader, device, normalize=True):
    z_ct_all, z_cl_all = [], []
    ids_all = []

    for batch in tqdm(loader, desc="Extracting embeddings (ccRCC)"):
        ct = batch["ct"].to(device, non_blocking=True)
        cl = batch["clingen"].to(device, non_blocking=True)

        # model API: image_embed/text_embed; keep as-is for your CcRCCMultiModalModel
        z_ct = model.image_embed(ct)
        z_cl = model.text_embed(cl)

        if normalize:
            z_ct = torch.nn.functional.normalize(z_ct, dim=-1)
            z_cl = torch.nn.functional.normalize(z_cl, dim=-1)

        z_ct_all.append(z_ct.detach().cpu())
        z_cl_all.append(z_cl.detach().cpu())

        # Your dataset uses "case_id"
        if "case_id" in batch:
            # collated as list[str] by default collate_fn
            ids_all.extend(list(batch["case_id"]))
        else:
            ids_all.extend([None] * ct.shape[0])

    return torch.cat(z_ct_all, 0), torch.cat(z_cl_all, 0), ids_all


def project_and_plot(z_a, z_b, out_dir, title, seed=42, max_points=4000):
    try:
        from sklearn.manifold import TSNE
    except Exception:
        TSNE = None
    try:
        from umap import UMAP
    except Exception:
        UMAP = None

    z_a = z_a.detach().cpu()
    z_b = z_b.detach().cpu()

    emb = torch.cat([z_a, z_b], dim=0).numpy()
    n = emb.shape[0]
    n_a = z_a.shape[0]

    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        emb_sub = emb[idx]
        a_mask = idx < n_a
        b_mask = ~a_mask
    else:
        emb_sub = emb
        a_mask = np.arange(n) < n_a
        b_mask = ~a_mask

    figs = []
    if TSNE is not None:
        n_samples = emb_sub.shape[0]
        perp = min(30, max(5, (n_samples - 1) // 3))
        if perp >= n_samples:
            perp = max(1, n_samples - 1)

        tsne = TSNE(n_components=2, perplexity=perp, random_state=seed, init="pca")        
        proj_tsne = tsne.fit_transform(emb_sub)
        figs.append(("tsne", proj_tsne, a_mask, b_mask))
    else:
        print("[WARN] sklearn not available; skipping t-SNE.")

    if UMAP is not None:
        umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed)
        proj_umap = umap.fit_transform(emb_sub)
        figs.append(("umap", proj_umap, a_mask, b_mask))
    else:
        print("[WARN] umap-learn not available; skipping UMAP.")

    if not figs:
        return

    plt.figure(figsize=(12, 5))
    for i, (name, proj, am, bm) in enumerate(figs[:2], start=1):
        plt.subplot(1, 2, i)
        plt.scatter(proj[am, 0], proj[am, 1], s=10, alpha=0.6, label="Modality A")
        plt.scatter(proj[bm, 0], proj[bm, 1], s=10, alpha=0.6, label="Modality B")
        plt.title(name.upper())
        plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tsne_umap.png"), dpi=300)
    plt.close()


def write_retrieval_examples_flickr(dataset: FlickrMultiModalDataset, z_img, z_txt, out_dir, seed=42, k=5, num_queries=3):
    # z_* expected already on CPU
    sims = (z_img @ z_txt.T)

    rng = np.random.default_rng(seed)
    q_idx = rng.choice(len(z_img), size=min(num_queries, len(z_img)), replace=False)

    tokenizer = dataset.tokenizer

    path = os.path.join(out_dir, "retrieval_examples.txt")
    with open(path, "w") as f:
        for qi in q_idx:
            topk = torch.topk(sims[qi], k=min(k, sims.shape[1])).indices.tolist()
            f.write(f"\nImage query index {qi} top-{len(topk)} retrieved captions:\n")
            for j in topk:
                item = dataset[j]
                ids = item["input_ids"].tolist()
                text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                f.write(f"  [{j}] {text}\n")


def write_retrieval_examples_ccrcc(ids, z_ct, z_cl, out_dir, seed=42, k=5, num_queries=3):
    sims = (z_ct @ z_cl.T)
    rng = np.random.default_rng(seed)
    q_idx = rng.choice(len(z_ct), size=min(num_queries, len(z_ct)), replace=False)

    path = os.path.join(out_dir, "retrieval_examples.txt")
    with open(path, "w") as f:
        for idx in q_idx:
            topk = torch.topk(sims[idx], k=min(k, sims.shape[1])).indices.tolist()
            pid = ids[idx] if ids[idx] is not None else f"idx={idx}"
            f.write(f"\nCT query ({pid}) top-{len(topk)} retrieved clingen entries:\n")
            for j in topk:
                pid_j = ids[j] if ids[j] is not None else f"idx={j}"
                f.write(f"  [{j}] {pid_j}\n")


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint
    ckpt_name = os.path.basename(ckpt_path).replace(".pt", "").replace(".ckpt", "")
    exp_name = os.path.basename(os.path.dirname(ckpt_path))

    out_dir = os.path.join("results", "analysis_outputs", f"{args.dataset}_{exp_name}_{ckpt_name}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Saving analysis results to: {out_dir}")

    if args.dataset == "flickr":
        model = MultiModalModel(
            vision_name=args.vision_name,
            text_name=args.text_name,
            shared_dim=args.shared_dim,
        ).to(device)

        safe_load_state_dict(model, ckpt_path, device)
        model.eval()

        dataset = FlickrMultiModalDataset(
            split="paired",
            paired_fraction=args.paired_fraction,
            train=False,  # uses your manual val split; keep as-is
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        z_a, z_b = extract_flickr(model, loader, device)

    elif args.dataset == "ccrcc":
        model = CcRCCMultiModalModel(
            ct_dim=args.ct_dim,
            clingen_dim=args.clingen_dim,
            shared_dim=args.shared_dim,
            hidden_dim=args.hidden_dim,
            ct_mask_ratio=args.ct_mask_ratio,
            clingen_mask_ratio=args.clingen_mask_ratio,
        ).to(device)

        safe_load_state_dict(model, ckpt_path, device)
        model.eval()

        # FIX: your dataset uses subset="train"/"test", not train=False
        dataset = CCRCCMultiModalDataset(
            root=args.ccrcc_root,
            split="paired",
            subset=args.ccrcc_subset,
            return_labels=False,  # labels not needed for embedding analysis
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        z_a, z_b, ids = extract_ccrcc(model, loader, device)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    metrics = retrieval_accuracy(z_a, z_b)
    print("\n=== Retrieval performance ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    with open(os.path.join(out_dir, "retrieval_metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    np.savez(
        os.path.join(out_dir, "embeddings.npz"),
        z_a=z_a.numpy(),
        z_b=z_b.numpy(),
    )

    title = f"{args.dataset} embedding space\n{exp_name}/{ckpt_name}"
    project_and_plot(z_a, z_b, out_dir, title, seed=args.seed, max_points=args.max_points)


    print("\n[INFO] Writing qualitative retrieval examples...")
    if args.dataset == "flickr":
        write_retrieval_examples_flickr(dataset, z_a, z_b, out_dir,
                                seed=args.seed, k=args.topk, num_queries=args.num_queries)
    else:
        write_retrieval_examples_ccrcc(ids, z_a, z_b, out_dir,
                               seed=args.seed, k=args.topk, num_queries=args.num_queries)
    print(f"\n[INFO] Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyze pretrained multimodal model (Flickr or ccRCC)")
    p.add_argument("--dataset", choices=["flickr", "ccrcc"], required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--shared_dim", type=int, default=256)

    p.add_argument("--vision_name", type=str, default="vit_base_patch16_224")
    p.add_argument("--text_name", type=str, default="bert-base-uncased")
    p.add_argument("--paired_fraction", type=float, default=0.1)

    p.add_argument("--ccrcc_root", type=str, default="data")
    p.add_argument("--ccrcc_subset", choices=["train", "test"], default="test")
    p.add_argument("--ct_dim", type=int, default=512)
    p.add_argument("--clingen_dim", type=int, default=18)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--ct_mask_ratio", type=float, default=0.25)
    p.add_argument("--clingen_mask_ratio", type=float, default=0.25)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_points", type=int, default=4000, help="Max points used for TSNE/UMAP plots")
    p.add_argument("--num_queries", type=int, default=3)
    p.add_argument("--topk", type=int, default=5)

    args = p.parse_args()
    main(args)
