import os
import torch
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import numpy as np
from umap import UMAP
from data.flickr_dataset import FlickrMultiModalDataset
from models.mm_model import MultiModalModel
from utils.metrics import retrieval_accuracy

parser = argparse.ArgumentParser(description="Analyze pretrained multimodal model")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="checkpoints/flickr_pretrain/epoch_10.pt",
    help="Path to model checkpoint (.pt file)",
)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--shared_dim", type=int, default=256)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Loading model from {args.checkpoint}...")
model = MultiModalModel(
    vision_name="vit_base_patch16_224",
    text_name="bert-base-uncased",
    shared_dim=args.shared_dim,
).to(DEVICE)
model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
model.eval()

dataset = FlickrMultiModalDataset(split="paired", paired_fraction=0.1)
loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

print("[INFO] Extracting embeddings...")
imgs, txts = [], []
with torch.no_grad():
    for batch in tqdm(loader):
        z_img = model.image_embed(batch["image"].to(DEVICE))
        z_txt = model.text_embed(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
        )
        imgs.append(z_img)
        txts.append(z_txt)

z_img = torch.cat(imgs)
z_txt = torch.cat(txts)

metrics = retrieval_accuracy(z_img, z_txt)
print("\nRetrieval performance:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

os.makedirs("analysis_outputs", exist_ok=True)
np.savez("analysis_outputs/embeddings_flickr.npz", z_img=z_img.cpu(), z_txt=z_txt.cpu())

print("\n[INFO] Running t-SNE and UMAP")
emb = torch.cat([z_img, z_txt]).cpu().numpy()
labels = np.array(["Image"] * len(z_img) + ["Text"] * len(z_txt))

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
proj_tsne = tsne.fit_transform(emb)

# UMAP
umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
proj_umap = umap.fit_transform(emb)

# Combined figure
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(proj_tsne[:len(z_img), 0], proj_tsne[:len(z_img), 1],
            s=10, alpha=0.6, label="Images")
plt.scatter(proj_tsne[len(z_img):, 0], proj_tsne[len(z_img):, 1],
            s=10, alpha=0.6, label="Text")
plt.title("t-SNE projection")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(proj_umap[:len(z_img), 0], proj_umap[:len(z_img), 1],
            s=10, alpha=0.6, label="Images")
plt.scatter(proj_umap[len(z_img):, 0], proj_umap[len(z_img):, 1],
            s=10, alpha=0.6, label="Text")
plt.title("UMAP projection")
plt.legend()

plt.suptitle(f"Shared Embedding Space after Anchored OT Pretraining\n{os.path.basename(args.checkpoint)}")
plt.tight_layout()
plt.savefig("analysis_outputs/tsne_umap_comparison.png", dpi=300)
plt.show()

print("\n[INFO] Computing qualitative retrieval examples...")
cos_sim = torch.nn.functional.cosine_similarity(
    z_img.unsqueeze(1), z_txt.unsqueeze(0), dim=-1
)

num_samples = 3
sample_ids = np.random.choice(len(z_img), num_samples, replace=False)

for idx in sample_ids:
    topk = torch.topk(cos_sim[idx], k=5).indices.cpu().tolist()
    print(f"\nImage {idx} top-5 retrieved captions:")
    for j in topk:
        try:
            captions = dataset[j]["caption"]
        except KeyError:
            captions = dataset.dataset[dataset.paired_idx[j]]["caption"]
        text = captions[0] if isinstance(captions, list) else captions
        print(f"  {text}")
