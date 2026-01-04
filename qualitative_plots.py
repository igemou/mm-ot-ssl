import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from models.mm_model import MultiModalModel
from data.flickr_dataset import FlickrMultiModalDataset


@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    imgs, txts = [], []
    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        z_img = model.image_embed(batch["image"].to(device))
        z_txt = model.text_embed(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        imgs.append(z_img.cpu())
        txts.append(z_txt.cpu())
    return torch.cat(imgs), torch.cat(txts)


def retrieve_topk(cos_sim_row, dataset, k=5):
    """Extract top-k captions given cosine sim vector."""
    topk_idx = torch.topk(cos_sim_row, k).indices.tolist()
    captions = []

    for j in topk_idx:
        try:
            caps = dataset[j]["caption"]
        except KeyError:
            caps = dataset.dataset[dataset.paired_idx[j]]["caption"]

        text = caps[0] if isinstance(caps, list) else caps
        captions.append(text)

    return captions


def plot_image_with_captions(pil_image, captions, save_path, title):
    """Creates side-by-side figure showing image + text."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Left image
    ax[0].imshow(pil_image)
    ax[0].axis("off")
    ax[0].set_title("Image", fontsize=14)

    # Captions on right
    ax[1].axis("off")
    cap_text = "\n\n".join([f"• {c}" for c in captions])
    ax[1].text(0, 0.5, cap_text, fontsize=11, va="center")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected_indices = [10, 123, 456]

    dataset = FlickrMultiModalDataset(split="paired", paired_fraction=1.0, train=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)


    for ckpt in args.checkpoints:
        ckpt_name = os.path.basename(os.path.dirname(ckpt))
        print(f"\n---- Loading checkpoint: {ckpt_name} ----")
        out_dir = f"results/qualitative_plots/{ckpt_name}"
        os.makedirs(out_dir, exist_ok=True)

        model = MultiModalModel(
            vision_name="vit_base_patch16_224",
            text_name="bert-base-uncased",
            shared_dim=args.shared_dim,
        ).to(device)

        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()

        z_img, z_txt = extract_embeddings(model, dataloader, device)

        z_img_norm = z_img / (z_img.norm(dim=1, keepdim=True) + 1e-8)
        z_txt_norm = z_txt / (z_txt.norm(dim=1, keepdim=True) + 1e-8)
        cos_sim = z_img_norm @ z_txt_norm.t()

        for idx in selected_indices:

            raw_image = dataset.dataset[ dataset.paired_idx[idx] ]["image"]
            captions = retrieve_topk(cos_sim[idx], dataset, k=5)

            save_path = os.path.join(out_dir, f"img_{idx}.png")
            title = f"{ckpt_name} — Image {idx}"

            plot_image_with_captions(raw_image, captions, save_path, title)
            print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qualitative comparison with plots")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="List of checkpoint paths"
    )
    parser.add_argument("--shared_dim", type=int, default=256)
    args = parser.parse_args()
    main(args)
