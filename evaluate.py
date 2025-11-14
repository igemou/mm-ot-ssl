import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.mm_model import MultiModalModel
from data.flickr_dataset import FlickrMultiModalDataset
from utils.metrics import retrieval_accuracy


@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """Embed all images and captions in the dataset."""
    imgs, txts = [], []
    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        z_img = model.image_embed(batch["image"].to(device))
        z_txt = model.text_embed(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        imgs.append(z_img)
        txts.append(z_txt)
    return torch.cat(imgs), torch.cat(txts)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Evaluating checkpoint {args.checkpoint} on {device}")

    model = MultiModalModel(
        vision_name=args.vision_name,
        text_name=args.text_name,
        shared_dim=args.shared_dim,
    ).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Load validation split 
    dataset = FlickrMultiModalDataset(split="paired", paired_fraction=1.0, train=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Extract embeddings
    z_img, z_txt = extract_embeddings(model, dataloader, device)

    # retrieval metrics
    metrics = retrieval_accuracy(z_img, z_txt)
    print("\n=== Retrieval Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cross-modal retrieval on Flickr30k")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pt file")
    parser.add_argument("--vision_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--text_name", type=str, default="bert-base-uncased")
    parser.add_argument("--shared_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
