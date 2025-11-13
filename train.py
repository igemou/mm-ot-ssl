import os
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.mm_model import MultiModalModel
from data.flickr_dataset import FlickrMultiModalDataset
from utils.losses import (
    clip_contrastive_loss,
    sinkhorn_ot_loss,
    anchored_ot_loss,
    mlm_loss,
    MAE_loss,
)
from utils.metrics import retrieval_accuracy


class Trainer:
    """Unified trainer for Anchored Multimodal SSL pretraining."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MultiModalModel(
            vision_name=args.vision_name,
            text_name=args.text_name,
            shared_dim=args.shared_dim,
        ).to(self.device)

        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=1e-4
        )

        self.mlm_loss_fn = mlm_loss(model_name=args.text_name)
        self.mae_loss_fn = MAE_loss()

    
        self._build_flickr_datasets()
        os.makedirs(args.save_dir, exist_ok=True)

    def _build_flickr_datasets(self):
        """Prepare Flickr30k loaders for paired/unpaired/image/text modes."""
        self.train_loaders = {
            mode: DataLoader(
                FlickrMultiModalDataset(
                    split=mode, paired_fraction=self.args.paired_fraction
                ),
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=4,
                drop_last=True,
            )
            for mode in ["paired", "unpaired", "text_only", "image_only"]
        }

        self.val_loader = DataLoader(
            FlickrMultiModalDataset(split="paired", paired_fraction=0.1),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
        )

        # Persistent iterators for cycling through data
        self.iters = {k: iter(v) for k, v in self.train_loaders.items()}

    def _next(self, name):
        """Safely fetch next batch (restarts iterator if exhausted)."""
        try:
            return next(self.iters[name])
        except StopIteration:
            self.iters[name] = iter(self.train_loaders[name])
            return next(self.iters[name])

    def train(self):
        print(f"Starting pretraining for {self.args.epochs} epochs on {self.device}")
        λ = {
            "clip": self.args.lambda_clip,
            "ot": self.args.lambda_ot,
            "mlm": self.args.lambda_mlm,
            "mae": self.args.lambda_mae,
        }

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            epoch_losses = {k: 0.0 for k in λ}
            steps = len(self.train_loaders["paired"])

            for _ in tqdm(range(steps), desc=f"Epoch {epoch}"):
                self.opt.zero_grad(set_to_none=True)
                total_loss = 0.0

                # 1. Paired contrastive (CLIP-style)
                b = self._next("paired")
                out = self.model(
                    images=b["image"].to(self.device),
                    input_ids=b["input_ids"].to(self.device),
                    attention_mask=b["attention_mask"].to(self.device),
                )
                loss_clip = clip_contrastive_loss(out["z_img"], out["z_txt"])
                total_loss += λ["clip"] * loss_clip
                epoch_losses["clip"] += loss_clip.item()

                # 2. Unpaired alignment (OT or Anchored OT)
                if self.args.use_anchored_ot:
                    b_paired = self._next("paired")
                    b_unpaired = self._next("unpaired")

                    images = torch.cat(
                        [b_paired["image"], b_unpaired["image"]], dim=0
                    ).to(self.device)
                    input_ids = torch.cat(
                        [b_paired["input_ids"], b_unpaired["input_ids"]], dim=0
                    ).to(self.device)
                    attn_mask = torch.cat(
                        [b_paired["attention_mask"], b_unpaired["attention_mask"]],
                        dim=0,
                    ).to(self.device)

                    out = self.model(
                        images=images, input_ids=input_ids, attention_mask=attn_mask
                    )
                    z_img, z_txt = out["z_img"], out["z_txt"]

                    n_paired = len(b_paired["image"])
                    anchors = [(i, i) for i in range(n_paired)]
                    loss_ot = anchored_ot_loss(
                        z_img, z_txt, paired_indices=anchors, alpha=self.args.alpha_anchor
                    )
                else:
                    b = self._next("unpaired")
                    out = self.model(
                        images=b["image"].to(self.device),
                        input_ids=b["input_ids"].to(self.device),
                        attention_mask=b["attention_mask"].to(self.device),
                    )
                    loss_ot = sinkhorn_ot_loss(out["z_img"], out["z_txt"])

                total_loss += λ["ot"] * loss_ot
                epoch_losses["ot"] += loss_ot.item()

                # 3. Text-only masked language modeling
                b = self._next("text_only")
                loss_mlm = self.mlm_loss_fn(
                    input_ids=b["input_ids"].to(self.device),
                    attention_mask=b["attention_mask"].to(self.device),
                    labels=b["input_ids"].to(self.device),
                )
                total_loss += λ["mlm"] * loss_mlm
                epoch_losses["mlm"] += loss_mlm.item()

                # 4. Image-only masked autoencoding
                b = self._next("image_only")
                out = self.model(images=b["image"].to(self.device))
                loss_mae = self.mae_loss_fn(
                    out["pred_patches"], out["target_patches"], out["mask"]
                )
                total_loss += λ["mae"] * loss_mae
                epoch_losses["mae"] += loss_mae.item()

                # Backpropagation
                total_loss.backward()
                self.opt.step()

            # Log averaged losses per epoch
            avg = {k: v / steps for k, v in epoch_losses.items()}
            print(f"Epoch {epoch} | Losses: {avg}")

            torch.save(
                self.model.state_dict(),
                os.path.join(self.args.save_dir, f"epoch_{epoch}.pt"),
            )

            if epoch % self.args.eval_every == 0:
                self.evaluate(epoch)

    def evaluate(self, epoch):
        self.model.eval()
        imgs, txts = [], []
        with torch.no_grad():
            for batch in self.val_loader:
                z_img = self.model.image_embed(batch["image"].to(self.device))
                z_txt = self.model.text_embed(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )
                imgs.append(z_img)
                txts.append(z_txt)

        z_img = torch.cat(imgs)
        z_txt = torch.cat(txts)
        metrics = retrieval_accuracy(z_img, z_txt)
        print(f"[Epoch {epoch}] Validation retrieval: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Anchored MultiModal SSL Pretraining (Flickr30k)"
    )

    # Model configs
    parser.add_argument("--vision_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--text_name", type=str, default="bert-base-uncased")
    parser.add_argument("--shared_dim", type=int, default=256)

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--paired_fraction", type=float, default=0.2)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="checkpoints_flickr")

    # Loss weights
    parser.add_argument("--lambda_clip", type=float, default=1.0)
    parser.add_argument("--lambda_ot", type=float, default=0.5)
    parser.add_argument("--lambda_mlm", type=float, default=1.0)
    parser.add_argument("--lambda_mae", type=float, default=1.0)

    # Anchored OT
    parser.add_argument(
        "--use_anchored_ot",
        action="store_true",
        help="Enable anchored OT instead of plain OT.",
    )
    parser.add_argument(
        "--alpha_anchor",
        type=float,
        default=0.1,
        help="Anchor strength multiplier for known pairs.",
    )

    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()
