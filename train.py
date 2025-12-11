# import os
# import torch
# import argparse
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# from models.mm_model import MultiModalModel
# from data import FlickrMultiModalDataset, COCOMultiModalDataset
# from utils.losses import (
#     clip_contrastive_loss,
#     sinkhorn_ot_loss,
#     anchored_ot_loss,
#     gromov_wasserstein_loss, 
#     mlm_loss,
#     MAE_loss,
# )
# from utils.metrics import retrieval_accuracy


# class Trainer:
#     """Unified trainer for Anchored / GW Multimodal SSL pretraining."""

#     def __init__(self, args):
#         self.args = args
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = MultiModalModel(
#             vision_name=args.vision_name,
#             text_name=args.text_name,
#             shared_dim=args.shared_dim,
#         ).to(self.device)

#         self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-4)
#         self.mlm_loss_fn = mlm_loss(model_name=args.text_name)
#         self.mae_loss_fn = MAE_loss()

#         print(f"Dataset: {args.dataset}")

#         data_fns = {"flickr": self._build_flickr_datasets,
#                     "coco": self._build_coco_datasets}
#         data_fns[args.dataset]()
#         os.makedirs(args.save_dir, exist_ok=True)

#         self.best_score = -1
#         self.early_stop_counter = 0

#     def _build_flickr_datasets(self):
#         print("Building Flickr30k datasets...")

#         requested = ["paired", "unpaired", "text_only", "image_only"]
#         self.train_loaders = {}
#         self.iters = {}

#         for mode in requested:
#             dataset = FlickrMultiModalDataset(
#                 split=mode,
#                 paired_fraction=self.args.paired_fraction,
#                 train=True,
#             )

#             if len(dataset) == 0:
#                 print(f"[WARN] '{mode}' split is empty — skipping.")
#                 continue

#             loader = DataLoader(
#                 dataset,
#                 batch_size=self.args.batch_size,
#                 shuffle=True,
#                 num_workers=4,
#                 drop_last=True,
#             )

#             self.train_loaders[mode] = loader
#             self.iters[mode] = iter(loader)

#         if "paired" not in self.train_loaders:
#             raise RuntimeError("ERROR: 'paired' dataset cannot be empty.")

#         self.val_loader = DataLoader(
#             FlickrMultiModalDataset(split="paired", paired_fraction=0.1, train=False),
#             batch_size=self.args.batch_size,
#             shuffle=False,
#             num_workers=4,
#         )

#         print("Flickr30k loaded splits:", list(self.train_loaders.keys()))


#     def _build_coco_datasets(self):
#         print("Building COCO datasets...")

#         requested = ["paired", "unpaired", "text_only", "image_only"]
#         self.train_loaders = {}
#         self.iters = {}

#         for mode in requested:
#             dataset = COCOMultiModalDataset(
#                 split=mode,
#                 paired_fraction=self.args.paired_fraction,
#                 train=True,
#             )

#             if len(dataset) == 0:
#                 print(f"[WARN] '{mode}' split is empty — skipping.")
#                 continue

#             loader = DataLoader(
#                 dataset,
#                 batch_size=self.args.batch_size,
#                 shuffle=True,
#                 num_workers=4,
#                 drop_last=True,
#             )

#             self.train_loaders[mode] = loader
#             self.iters[mode] = iter(loader)

#         if "paired" not in self.train_loaders:
#             raise RuntimeError("ERROR: 'paired' dataset cannot be empty.")

#         self.val_loader = DataLoader(
#             COCOMultiModalDataset(split="paired", paired_fraction=0.1, train=False),
#             batch_size=self.args.batch_size,
#             shuffle=False,
#             num_workers=4,
#         )

#         print("COCO loaded splits:", list(self.train_loaders.keys()))

#     def _next(self, name):
#         """Safely fetch next batch."""
#         try:
#             return next(self.iters[name])
#         except StopIteration:
#             self.iters[name] = iter(self.train_loaders[name])
#             return next(self.iters[name])

#     def train(self):
#         print(f"Starting pretraining for {self.args.epochs} epochs on {self.device}")
#         λ = {
#             "clip": self.args.lambda_clip,
#             "ot": self.args.lambda_ot,
#             "mlm": self.args.lambda_mlm,
#             "mae": self.args.lambda_mae,
#         }

#         for epoch in range(1, self.args.epochs + 1):
#             self.model.train()
#             epoch_losses = {k: 0.0 for k in λ}
#             steps = len(self.train_loaders["paired"])

#             for _ in (pbar := tqdm(range(steps), desc=f"Epoch {epoch}")):
#                 self.opt.zero_grad(set_to_none=True)
#                 total_loss = 0.0

#                 # Paired CLIP loss
#                 b = self._next("paired")
#                 out = self.model(
#                     images=b["image"].to(self.device),
#                     input_ids=b["input_ids"].to(self.device),
#                     attention_mask=b["attention_mask"].to(self.device),
#                 )
#                 loss_clip = clip_contrastive_loss(out["z_img"], out["z_txt"])
#                 total_loss += λ["clip"] * loss_clip
#                 epoch_losses["clip"] += loss_clip.item()

#                 # Unpaired OT
#                 if "unpaired" in self.train_loaders and λ["ot"] > 0:
#                     b = self._next("unpaired")
#                     out = self.model(
#                         images=b["image"].to(self.device),
#                         input_ids=b["input_ids"].to(self.device),
#                         attention_mask=b["attention_mask"].to(self.device),
#                     )
#                     z_img, z_txt = out["z_img"], out["z_txt"]

#                     if self.args.use_anchored_ot:
#                         b_p = self._next("paired")
#                         images = torch.cat([b_p["image"], b["image"]], dim=0).to(self.device)
#                         input_ids = torch.cat([b_p["input_ids"], b["input_ids"]], dim=0).to(self.device)
#                         attn_mask = torch.cat([b_p["attention_mask"], b["attention_mask"]], dim=0).to(self.device)

#                         out = self.model(images=images, input_ids=input_ids, attention_mask=attn_mask)
#                         z_img, z_txt = out["z_img"], out["z_txt"]

#                         n_paired = len(b_p["image"])
#                         anchors = [(i, i) for i in range(n_paired)]
#                         loss_ot = anchored_ot_loss(
#                             z_img, z_txt, paired_indices=anchors, alpha=self.args.alpha_anchor
#                         )
#                     elif self.args.use_gw_ot:
#                         loss_ot = gromov_wasserstein_loss(z_img, z_txt)
#                     else:
#                         loss_ot = sinkhorn_ot_loss(z_img, z_txt)

#                     total_loss += λ["ot"] * loss_ot
#                     epoch_losses["ot"] += loss_ot.item()

#                 # Text MLM
#                 if "text_only" in self.train_loaders and λ["mlm"] > 0:
#                     b = self._next("text_only")
#                     loss_mlm = self.mlm_loss_fn(
#                         input_ids=b["input_ids"].to(self.device),
#                         attention_mask=b["attention_mask"].to(self.device),
#                         labels=b["input_ids"].to(self.device),
#                     )
#                     total_loss += λ["mlm"] * loss_mlm
#                     epoch_losses["mlm"] += loss_mlm.item()

#                 # Image MAE
#                 if "image_only" in self.train_loaders and λ["mae"] > 0:
#                     b = self._next("image_only")
#                     out = self.model(images=b["image"].to(self.device))
#                     loss_mae = self.mae_loss_fn(out["pred_patches"], out["target_patches"], out["mask"])
#                     total_loss += λ["mae"] * loss_mae
#                     epoch_losses["mae"] += loss_mae.item()

#                 total_loss.backward()
#                 self.opt.step()
#                 postfix = {
#                     "clip": loss_clip.item() if "clip" in epoch_losses else 0.0,
#                     "ot": loss_ot.item() if λ["ot"] > 0 else 0.0,
#                     "mlm": loss_mlm.item() if λ["mlm"] > 0 else 0.0,
#                     "mae": loss_mae.item() if λ["mae"] > 0 else 0.0,
#                 }
#                 pbar.set_postfix(postfix)
#             avg = {k: v / steps for k, v in epoch_losses.items()}
#             print(f"Epoch {epoch} | Losses: {avg}")

#             if epoch % self.args.eval_every == 0:
#                 score, metrics = self.evaluate(epoch)

#                 # Save best model
#                 if score > self.best_score:
#                     self.best_score = score
#                     self.early_stop_counter = 0
#                     best_path = os.path.join(self.args.save_dir, "best.pt")
#                     torch.save(self.model.state_dict(), best_path)
#                     print(f"New best model saved at epoch {epoch} (score {score:.4f})")

#                 else:
#                     self.early_stop_counter += 1
#                     print(f"No improvement. Early stop counter = {self.early_stop_counter}")

#                     if self.early_stop_counter >= self.args.patience:
#                         print("Early stopping triggered!")
#                         return


#     def evaluate(self, epoch):
#         self.model.eval()
#         imgs, txts = [], []
#         with torch.no_grad():
#             for batch in self.val_loader:
#                 z_img = self.model.image_embed(batch["image"].to(self.device))
#                 z_txt = self.model.text_embed(
#                     batch["input_ids"].to(self.device),
#                     batch["attention_mask"].to(self.device),
#                 )
#                 imgs.append(z_img)
#                 txts.append(z_txt)

#         z_img = torch.cat(imgs)
#         z_txt = torch.cat(txts)
#         # metrics = retrieval_accuracy(z_img, z_txt, self.device)
#         metrics = retrieval_accuracy(z_img, z_txt, topk=(1, 5))
#         print(f"[Epoch {epoch}] Validation retrieval: {metrics}")

#         score = 0.5 * (metrics["i2t@1"] + metrics["t2i@1"])
#         return score, metrics

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Anchored/GW MM-SSL Pretraining")

#     # Model
#     parser.add_argument("--vision_name", type=str, default="vit_base_patch16_224")
#     parser.add_argument("--text_name", type=str, default="bert-base-uncased")
#     parser.add_argument("--shared_dim", type=int, default=256)

#     # Dataset
#     parser.add_argument("--dataset", type=str, choices=["flickr", "coco"], default="flickr")

#     # Training
#     parser.add_argument("--epochs", type=int, default=50)
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--lr", type=float, default=1e-4)
#     parser.add_argument("--paired_fraction", type=float, default=0.2)
#     parser.add_argument("--eval_every", type=int, default=1)
#     parser.add_argument("--save_dir", type=str, default="checkpoints_flickr")

#     # Loss weights
#     parser.add_argument("--lambda_clip", type=float, default=1.0)
#     parser.add_argument("--lambda_ot", type=float, default=0.5)
#     parser.add_argument("--lambda_mlm", type=float, default=1.0)
#     parser.add_argument("--lambda_mae", type=float, default=1.0)

#     # OT Variants
#     parser.add_argument("--use_anchored_ot", action="store_true")
#     parser.add_argument("--alpha_anchor", type=float, default=0.1)
#     parser.add_argument("--use_gw_ot", action="store_true")

#     # Early stopping
#     parser.add_argument("--patience", type=int, default=5)

#     args = parser.parse_args()
#     trainer = Trainer(args)
#     trainer.train()


import os
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.mm_model import MultiModalModel
from models.ccrcc_model import CcRCCMultiModalModel   
from data.flickr_dataset import FlickrMultiModalDataset
from data.coco_dataset import COCOMultiModalDataset
from data.ccRCC_dataset import CcRCCMultiModalDataset        

from utils.losses import (
    clip_contrastive_loss,
    sinkhorn_ot_loss,
    anchored_ot_loss,
    gromov_wasserstein_loss,
    mlm_loss,
    MAE_loss,
    masked_mse_loss,
)
from utils.metrics import retrieval_accuracy

class Trainer:
    """Unified trainer for Anchored / GW Multimodal SSL pretraining."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Dataset: {args.dataset}")

        if args.dataset == "ccrcc":
            if args.ccrcc_root is None:
                raise ValueError("For dataset='ccrcc', you must provide --ccrcc_root")

            tmp_ds = CcRCCMultiModalDataset(
                root=args.ccrcc_root,
                mode="paired",
                subset="train",
                return_labels=False,
            )
            if len(tmp_ds) == 0:
                raise RuntimeError("ccRCC paired train split is empty.")
            sample = tmp_ds[0]
            ct_dim = sample["ct"].numel()
            cl_dim = sample["clingen"].numel()
            print(f"Inferred ccRCC dims: ct_dim={ct_dim}, clingen_dim={cl_dim}")

            self.model = CcRCCMultiModalModel(
                ct_dim=ct_dim,
                clingen_dim=cl_dim,
                shared_dim=args.shared_dim,
            ).to(self.device)

            # MLM/MAE for ccRCC are handled via masked feature 
            self.mlm_loss_fn = None
            self.mae_loss_fn = None

            self._build_ccrcc_datasets()

        else:
            # Flickr/COCO standard image+text model
            self.model = MultiModalModel(
                vision_name=args.vision_name,
                text_name=args.text_name,
                shared_dim=args.shared_dim,
            ).to(self.device)

            self.mlm_loss_fn = mlm_loss(model_name=args.text_name)
            self.mae_loss_fn = MAE_loss()

            data_fns = {
                "flickr": self._build_flickr_datasets,
                "coco": self._build_coco_datasets,
            }
            data_fns[args.dataset]()

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-4)
        os.makedirs(args.save_dir, exist_ok=True)

        self.best_score = -1
        self.early_stop_counter = 0

    def _build_flickr_datasets(self):
        print("Building Flickr30k datasets...")

        requested = ["paired", "unpaired", "text_only", "image_only"]
        self.train_loaders = {}
        self.iters = {}

        for mode in requested:
            dataset = FlickrMultiModalDataset(
                split=mode,
                paired_fraction=self.args.paired_fraction,
                train=True,
            )

            if len(dataset) == 0:
                print(f"[WARN] '{mode}' split is empty — skipping.")
                continue

            loader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=4,
                drop_last=True,
            )

            self.train_loaders[mode] = loader
            self.iters[mode] = iter(loader)

        if "paired" not in self.train_loaders:
            raise RuntimeError("ERROR: 'paired' dataset cannot be empty.")

        self.val_loader = DataLoader(
            FlickrMultiModalDataset(split="paired", paired_fraction=0.1, train=False),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
        )

        print("Flickr30k loaded splits:", list(self.train_loaders.keys()))

    def _build_coco_datasets(self):
        print("Building COCO datasets...")

        requested = ["paired", "unpaired", "text_only", "image_only"]
        self.train_loaders = {}
        self.iters = {}

        for mode in requested:
            dataset = COCOMultiModalDataset(
                split=mode,
                paired_fraction=self.args.paired_fraction,
                train=True,
            )

            if len(dataset) == 0:
                print(f"[WARN] '{mode}' split is empty — skipping.")
                continue

            loader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=4,
                drop_last=True,
            )

            self.train_loaders[mode] = loader
            self.iters[mode] = iter(loader)

        if "paired" not in self.train_loaders:
            raise RuntimeError("ERROR: 'paired' dataset cannot be empty.")

        self.val_loader = DataLoader(
            COCOMultiModalDataset(split="paired", paired_fraction=0.1, train=False),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
        )

        print("COCO loaded splits:", list(self.train_loaders.keys()))

    def _build_ccrcc_datasets(self):
        print("Building ccRCC datasets...")

        requested = ["paired", "unpaired", "text_only", "image_only"]
        self.train_loaders = {}
        self.iters = {}

        for mode in requested:
            dataset = CcRCCMultiModalDataset(
                root=self.args.ccrcc_root,
                mode=mode,
                subset="train",
                return_labels=False,  # pure SSL
            )

            if len(dataset) == 0:
                print(f"'{mode}' split is empty.")
                continue

            loader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=4,
                drop_last=True,
            )

            self.train_loaders[mode] = loader
            self.iters[mode] = iter(loader)

        if "paired" not in self.train_loaders:
            raise RuntimeError("ERROR: 'paired' dataset cannot be empty.")

        # Validation on paired test split
        self.val_loader = DataLoader(
            CcRCCMultiModalDataset(
                root=self.args.ccrcc_root,
                mode="paired",
                subset="test",
                return_labels=False,
            ),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
        )

        print("ccRCC loaded splits:", list(self.train_loaders.keys()))

    def _next(self, name):
        """Safely fetch next batch."""
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

            for _ in (pbar := tqdm(range(steps), desc=f"Epoch {epoch}")):
                self.opt.zero_grad(set_to_none=True)
                total_loss = 0.0

                b = self._next("paired")

                if self.args.dataset == "ccrcc":
                    out = self.model(
                        ct=b["ct"].to(self.device),
                        clingen=b["clingen"].to(self.device),
                    )
                else:
                    out = self.model(
                        images=b["image"].to(self.device),
                        input_ids=b["input_ids"].to(self.device),
                        attention_mask=b["attention_mask"].to(self.device),
                    )

                loss_clip = clip_contrastive_loss(out["z_img"], out["z_txt"])
                total_loss += λ["clip"] * loss_clip
                epoch_losses["clip"] += loss_clip.item()

                # Unpaired OT
                loss_ot = torch.tensor(0.0, device=self.device)
                if "unpaired" in self.train_loaders and λ["ot"] > 0:
                    b = self._next("unpaired")

                    if self.args.dataset == "ccrcc":
                        out = self.model(
                            ct=b["ct"].to(self.device),
                            clingen=b["clingen"].to(self.device),
                        )
                        z_img, z_txt = out["z_img"], out["z_txt"]

                        if self.args.use_gw_ot:
                            loss_ot = gromov_wasserstein_loss(z_img, z_txt)
                        elif self.args.use_anchored_ot:
                            # Anchored OT not wired for ccRCC right now
                            raise NotImplementedError(
                                "Anchored OT for ccRCC not implemented yet; "
                                "use Sinkhorn or GW."
                            )
                        else:
                            loss_ot = sinkhorn_ot_loss(z_img, z_txt)

                    else:
                        out = self.model(
                            images=b["image"].to(self.device),
                            input_ids=b["input_ids"].to(self.device),
                            attention_mask=b["attention_mask"].to(self.device),
                        )
                        z_img, z_txt = out["z_img"], out["z_txt"]

                        if self.args.use_anchored_ot:
                            b_p = self._next("paired")
                            images = torch.cat([b_p["image"], b["image"]], dim=0).to(self.device)
                            input_ids = torch.cat([b_p["input_ids"], b["input_ids"]], dim=0).to(self.device)
                            attn_mask = torch.cat([b_p["attention_mask"], b["attention_mask"]], dim=0).to(self.device)

                            out = self.model(images=images, input_ids=input_ids, attention_mask=attn_mask)
                            z_img, z_txt = out["z_img"], out["z_txt"]

                            n_paired = len(b_p["image"])
                            anchors = [(i, i) for i in range(n_paired)]
                            loss_ot = anchored_ot_loss(
                                z_img, z_txt, paired_indices=anchors, alpha=self.args.alpha_anchor
                            )
                        elif self.args.use_gw_ot:
                            loss_ot = gromov_wasserstein_loss(z_img, z_txt)
                        else:
                            loss_ot = sinkhorn_ot_loss(z_img, z_txt)

                    total_loss += λ["ot"] * loss_ot
                    epoch_losses["ot"] += loss_ot.item()

                # Unimodal MLM / MAE
                #   - Flickr/COCO: text MLM + image MAE (original)
                #   - ccRCC: clingen masked-feature + ct masked-feature

                # Text / clingen side
                if λ["mlm"] > 0:
                    if self.args.dataset == "ccrcc":
                        # use clingen masked feature reconstruction
                        # can use text_only split if present; otherwise reuse paired
                        loader_name = "text_only" if "text_only" in self.train_loaders else "paired"
                        b = self._next(loader_name)
                        cl = b["clingen"].to(self.device)
                        pred, target, mask = self.model.clingen_mlm_step(
                            cl, mask_ratio=self.args.mlm_mask_ratio
                        )
                        loss_mlm = masked_mse_loss(pred, target, mask)
                    else:
                        if "text_only" in self.train_loaders:
                            b = self._next("text_only")
                            loss_mlm = self.mlm_loss_fn(
                                input_ids=b["input_ids"].to(self.device),
                                attention_mask=b["attention_mask"].to(self.device),
                                labels=b["input_ids"].to(self.device),
                            )
                        else:
                            loss_mlm = torch.tensor(0.0, device=self.device)

                    total_loss += λ["mlm"] * loss_mlm
                    epoch_losses["mlm"] += loss_mlm.item()
                else:
                    loss_mlm = torch.tensor(0.0, device=self.device)

                # Image / CT side
                if λ["mae"] > 0:
                    if self.args.dataset == "ccrcc":
                        loader_name = "image_only" if "image_only" in self.train_loaders else "paired"
                        b = self._next(loader_name)
                        ct = b["ct"].to(self.device)
                        pred, target, mask = self.model.ct_mae_step(
                            ct, mask_ratio=self.args.mae_mask_ratio
                        )
                        loss_mae = masked_mse_loss(pred, target, mask)
                    else:
                        if "image_only" in self.train_loaders:
                            b = self._next("image_only")
                            out_mae = self.model(images=b["image"].to(self.device))
                            loss_mae = self.mae_loss_fn(
                                out_mae["pred_patches"],
                                out_mae["target_patches"],
                                out_mae["mask"],
                            )
                        else:
                            loss_mae = torch.tensor(0.0, device=self.device)

                    total_loss += λ["mae"] * loss_mae
                    epoch_losses["mae"] += loss_mae.item()
                else:
                    loss_mae = torch.tensor(0.0, device=self.device)

                total_loss.backward()
                self.opt.step()

                postfix = {
                    "clip": loss_clip.item(),
                    "ot": loss_ot.item() if λ["ot"] > 0 else 0.0,
                    "mlm": loss_mlm.item() if λ["mlm"] > 0 else 0.0,
                    "mae": loss_mae.item() if λ["mae"] > 0 else 0.0,
                }
                pbar.set_postfix(postfix)

            avg = {k: v / steps for k, v in epoch_losses.items()}
            print(f"Epoch {epoch} | Losses: {avg}")

            if epoch % self.args.eval_every == 0:
                score, metrics = self.evaluate(epoch)

                # Save best model
                if score > self.best_score:
                    self.best_score = score
                    self.early_stop_counter = 0
                    best_path = os.path.join(self.args.save_dir, "best.pt")
                    torch.save(self.model.state_dict(), best_path)
                    print(f"New best model saved at epoch {epoch} (score {score:.4f})")
                else:
                    self.early_stop_counter += 1
                    print(f"No improvement. Early stop counter = {self.early_stop_counter}")

                    if self.early_stop_counter >= self.args.patience:
                        print("Early stopping triggered!")
                        return

    def evaluate(self, epoch):
        self.model.eval()
        imgs, txts = [], []
        with torch.no_grad():
            for batch in self.val_loader:
                if self.args.dataset == "ccrcc":
                    z_img = self.model.image_embed(batch["ct"].to(self.device))
                    z_txt = self.model.text_embed(batch["clingen"].to(self.device))
                else:
                    z_img = self.model.image_embed(batch["image"].to(self.device))
                    z_txt = self.model.text_embed(
                        batch["input_ids"].to(self.device),
                        batch["attention_mask"].to(self.device),
                    )
                imgs.append(z_img)
                txts.append(z_txt)

        z_img = torch.cat(imgs)
        z_txt = torch.cat(txts)
        metrics = retrieval_accuracy(z_img, z_txt, topk=(1, 5))
        print(f"[Epoch {epoch}] Validation retrieval: {metrics}")

        score = 0.5 * (metrics["i2t@1"] + metrics["t2i@1"])
        return score, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anchored/GW MM-SSL Pretraining")

    # Model
    parser.add_argument("--vision_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--text_name", type=str, default="bert-base-uncased")
    parser.add_argument("--shared_dim", type=int, default=256)

    # Dataset
    parser.add_argument("--dataset", type=str, choices=["flickr", "coco", "ccrcc"], default="flickr")
    parser.add_argument("--ccrcc_root", type=str, default=None,
                        help="Root directory for ccRCC data (expects clinical+genomic_split.csv and 'CT Features/' folder)")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--paired_fraction", type=float, default=0.2)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    # Loss weights
    parser.add_argument("--lambda_clip", type=float, default=1.0)
    parser.add_argument("--lambda_ot", type=float, default=0.5)
    parser.add_argument("--lambda_mlm", type=float, default=1.0)
    parser.add_argument("--lambda_mae", type=float, default=1.0)

    # OT Variants
    parser.add_argument("--use_anchored_ot", action="store_true")
    parser.add_argument("--alpha_anchor", type=float, default=0.1)
    parser.add_argument("--use_gw_ot", action="store_true")

    # Mask ratios for ccRCC feature MLM/MAE
    parser.add_argument("--mlm_mask_ratio", type=float, default=0.25,
                        help="Mask ratio for clingen feature MLM (ccRCC only)")
    parser.add_argument("--mae_mask_ratio", type=float, default=0.25,
                        help="Mask ratio for CT feature MAE (ccRCC only)")

    # Early stopping
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()
