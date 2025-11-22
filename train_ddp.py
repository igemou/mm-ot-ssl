import os
import argparse
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch import optim
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import trange

from models.mm_model import MultiModalModel
from data import FlickrMultiModalDataset, COCOMultiModalDataset
from utils.losses import (
    clip_contrastive_loss,
    sinkhorn_ot_loss,
    anchored_ot_loss,
    gromov_wasserstein_loss, 
    mlm_loss,
    MAE_loss,
)
from utils.ddpmetric import SmoothedValue, retrieval_accuracy

def setup(model):
    print(f"Num Avail Devices: {torch.accelerator.device_count()}")
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend)
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.accelerator.device_count()
    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id])
    return model, device_id

def cleanup():
    dist.destroy_process_group() 


class Trainer:
    """Unified trainer for Anchored / GW Multimodal SSL pretraining."""

    def __init__(self, args):
        self.args = args
        model = MultiModalModel(
            vision_name=args.vision_name,
            text_name=args.text_name,
            shared_dim=args.shared_dim,
        )
        self.model, self.device = setup(model)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-4)
        self.mlm_loss_fn = mlm_loss(model_name=args.text_name)
        self.mae_loss_fn = MAE_loss()

        print(f"Dataset: {args.dataset}")

        data_fns = {"flickr": self._build_flickr_datasets,
                    "coco": self._build_coco_datasets}
        data_fns[args.dataset]()
        os.makedirs(args.save_dir, exist_ok=True)

        self.best_score = -1
        self.early_stop_counter = 0

    def _build_flickr_datasets(self, ddp=True):
        """Build datasets, skipping empty splits safely."""
        print("Building Flickr30k datasets...")

        requested = ["paired", "unpaired", "text_only", "image_only"]
        self.train_loaders = {}
        self.iters = {}
        self.samplers = {}

        for mode in requested:
            dataset = FlickrMultiModalDataset(
                split=mode,
                paired_fraction=self.args.paired_fraction,
                train=True,
            )
            sampler = DistributedSampler(dataset) if ddp else None

            if len(dataset) == 0:
                print(f"[WARN] '{mode}' split is empty — skipping.")
                continue

            loader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=not ddp,
                num_workers=4,
                drop_last=True,
                sampler=sampler,
            )

            self.train_loaders[mode] = (loader, sampler)
            self.iters[mode] = iter(loader)
            self.samplers[mode] = sampler

        if "paired" not in self.train_loaders:
            raise RuntimeError("ERROR: 'paired' dataset cannot be empty.")

        # Validation split
        dataset = FlickrMultiModalDataset(split="paired", paired_fraction=0.1, train=False),
        self.val_loader = DataLoader(dataset,
                                     batch_size=self.args.batch_size,
                                     shuffle=False,
                                     num_workers=4)

        print("Flickr30k loaded splits:", list(self.train_loaders.keys()))

    def _build_coco_datasets(self, ddp=True):
        """Build datasets, skipping empty splits safely."""
        print("Building COCO datasets...")

        requested = ["paired", "unpaired", "text_only", "image_only"]
        self.train_loaders = {}
        self.iters = {}
        self.samplers = {}

        for mode in requested:
            dataset = COCOMultiModalDataset(
                split=mode,
                paired_fraction=self.args.paired_fraction,
                train=True,
            )

            sampler = DistributedSampler(dataset) if ddp else None

            if len(dataset) == 0:
                print(f"[WARN] '{mode}' split is empty — skipping.")
                continue

            loader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=not ddp,
                num_workers=4,
                drop_last=True,
                sampler=sampler,
            )

            self.train_loaders[mode] = loader
            self.iters[mode] = iter(loader)
            self.samplers[mode] = sampler

        if "paired" not in self.train_loaders:
            raise RuntimeError("ERROR: 'paired' dataset cannot be empty.")

        # Validation split
        dataset = COCOMultiModalDataset(split="paired", paired_fraction=0.1, train=False)
        self.val_sampler = DistributedSampler(dataset) if ddp else None
        self.val_loader = DataLoader(dataset,
                                      batch_size=self.args.batch_size,
                                      shuffle=False,
                                      num_workers=4,
                                      sampler=sampler)

        print("COCO loaded splits:", list(self.train_loaders.keys()))

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
            # epoch_losses = {k: 0.0 for k in λ}
            steps = len(self.train_loaders["paired"])

            for mode in self.samplers: self.samplers[mode].set_epoch(epoch)
            self.val_sampler.set_epoch(epoch)

            metrics = {k: SmoothedValue() for k in λ}
        
            pbar = trange(steps, desc=f"Epoch {epoch}") if (int(os.environ["LOCAL_RANK"]) == 0) else range(steps) 
            for _ in pbar:
                self.opt.zero_grad(set_to_none=True)
                total_loss = 0.0

                # Paired CLIP loss
                b = self._next("paired")
                out = self.model(
                    images=b["image"].to(self.device),
                    input_ids=b["input_ids"].to(self.device),
                    attention_mask=b["attention_mask"].to(self.device),
                )
                loss_clip = clip_contrastive_loss(out["z_img"], out["z_txt"])
                total_loss += λ["clip"] * loss_clip
                metrics["clip"].update(loss_clip.item())

                # Unpaired OT 
                if "unpaired" in self.train_loaders and λ["ot"] > 0:
                    b = self._next("unpaired")
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
                    metrics["ot"].update(loss_ot.item())

                # Text-only MLM 
                if "text_only" in self.train_loaders and λ["mlm"] > 0:
                    b = self._next("text_only")
                    loss_mlm = self.mlm_loss_fn(
                        input_ids=b["input_ids"].to(self.device),
                        attention_mask=b["attention_mask"].to(self.device),
                        labels=b["input_ids"].to(self.device),
                    )
                    total_loss += λ["mlm"] * loss_mlm
                    metrics["mlm"].update(loss_mlm.item())

                # Image-only MAE 
                if "image_only" in self.train_loaders and λ["mae"] > 0:
                    b = self._next("image_only")
                    out = self.model(images=b["image"].to(self.device))
                    loss_mae = self.mae_loss_fn(out["pred_patches"], out["target_patches"], out["mask"])
                    total_loss += λ["mae"] * loss_mae
                    metrics["mae"].update(loss_mae.item())

                total_loss.backward()
                self.opt.step()
                if (int(os.environ["LOCAL_RANK"]) == 0):
                    pbar.set_postfix({"mae": loss_mae.item(),
                                    "clip": loss_clip.item() if λ["ot"] > 0 else 0.0,
                                    "ot": loss_ot.item() if λ["mlm"] > 0 else 0.0,
                                    "mlm": loss_mlm.item() if λ["mae"] > 0 else 0.0,})
                
                dist.barrier(device_ids = [self.device])


            # avg = {k: v / steps for k, v in epoch_losses.items()}
            for _, meter in metrics.items():
                meter.synchronize_between_processes(device_ids=[self.device])

            avg = {k: meter.global_avg for k, meter in metrics.items()}

            print(f"Epoch {epoch} | Losses: {avg}")

            if (epoch % self.args.eval_every == 0):
                score, metrics = self.evaluate(epoch) 

                if (int(os.environ["LOCAL_RANK"]) == 0):
                    # Save best model
                    if score > self.best_score:
                        self.best_score = score
                        self.early_stop_counter = 0
                        best_path = os.path.join(self.args.save_dir, "best.pt")
                        torch.save(self.model.state_dict(), best_path)
                        print(f"New best model saved at epoch {epoch} (score {score:.4f})")

                        with open(os.path.join(self.args.desc_dir, "results.txt"), "w") as f:
                            f.write(args.desc)
                            f.write(f"Epoch: {epoch}, metrics: {metrics}")

                    else:
                        self.early_stop_counter += 1
                        print(f"No improvement. Early stop counter = {self.early_stop_counter}")

                        if self.early_stop_counter >= self.args.patience:
                            print("Early stopping triggered!")
                            return
            
            dist.barrier(device_ids = [self.device])

        cleanup()

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
        if (int(os.environ["LOCAL_RANK"]) == 0):
            print(f"\n[Epoch {epoch}] Validation retrieval: {metrics}\n")

        score = 0.5 * (metrics["i2t@1"] + metrics["t2i@1"])
        return score, metrics


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser(description="Anchored/GW MultiModal SSL Pretraining (Flickr30k)")

    # Model
    parser.add_argument("--vision_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--text_name", type=str, default="bert-base-uncased")
    parser.add_argument("--shared_dim", type=int, default=256)

    #Dataset
    parser.add_argument("--dataset", type=str, choices=["flickr", "coco"], default="flickr")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
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

    # OT Variants
    parser.add_argument("--use_anchored_ot", action="store_true")
    parser.add_argument("--alpha_anchor", type=float, default=0.1)
    parser.add_argument("--use_gw_ot", action="store_true")

     # Early stopping
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--desc_dir", type=str)
    parser.add_argument("--desc", type=str)

    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()
