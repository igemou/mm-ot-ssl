import os
import json
import optuna

import torch
import torch.distributed as dist 

from optuna.pruners import MedianPruner
from train_ddp import Trainer
from train_ddp import argparse as train_argparse

def ddp_setup():
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    print(f"Start running DDP on rank {rank}")
    print(f"Num Avail Devices: {torch.accelerator.device_count()}")

    device = rank % torch.accelerator.device_count()

    return device


def build_args(lambda_clip, lambda_ot, lambda_mlm, lambda_mae, save_path, desc):
    parser = train_argparse.ArgumentParser()

    parser.add_argument("--vision_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--text_name", type=str, default="bert-base-uncased")
    parser.add_argument("--shared_dim", type=int, default=256)
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--paired_fraction", type=float, default=0.2)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default=save_path)
    parser.add_argument("--desc_dir", type=str, default="/users/bjoo2/scratch/anchor/optuna/")
    parser.add_argument("--desc", type=str, default=desc)
    parser.add_argument("--write", type=bool, default=False)

    parser.add_argument("--lambda_clip", type=float, default=lambda_clip)
    parser.add_argument("--lambda_ot", type=float, default=lambda_ot)
    parser.add_argument("--lambda_mlm", type=float, default=lambda_mlm)
    parser.add_argument("--lambda_mae", type=float, default=lambda_mae)

    parser.add_argument("--use_anchored_ot", action="store_false")
    parser.add_argument("--alpha_anchor", type=float, default=0.0)

    parser.add_argument("--use_gw_ot", action="store_true")
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args([])
    return args


def objective_alignment(trial, device):
    trial = optuna.integration.TorchDistributedTrial(trial)

    lambda_clip = trial.suggest_float("lambda_clip", 0.3, 2.0, log=True)
    lambda_ot   = trial.suggest_float("lambda_ot", 0.05, 1.0, log=True)

    save_path = f"results/alignment/trial_{trial.number}"
    os.makedirs(save_path, exist_ok=True)
    desc = f"gw_b32_clip{lambda_clip}_ot{lambda_ot}_mlm0_mae0"
    args = build_args(lambda_clip, lambda_ot, 0.0, 0.0, save_path, desc)

    trainer = Trainer(args, device=device)
    trainer.train(trial=trial)

    score, _ = trainer.evaluate(epoch=args.epochs)
    # trial.report(score, step=0)
    # if trial.should_prune():
    #     raise optuna.TrialPruned()

    return score


def objective_unimodal(trial, best_align, device):
    trial = optuna.integration.TorchDistributedTrial(trial)

    lambda_mlm = trial.suggest_float("lambda_mlm", 0.1, 1.0)
    lambda_mae = trial.suggest_float("lambda_mae", 0.1, 1.0)

    lambda_clip = best_align["lambda_clip"]
    lambda_ot   = best_align["lambda_ot"]

    save_path = f"results/unimodal/trial_{trial.number}"
    os.makedirs(save_path, exist_ok=True)
    desc = f"gw_b32_clip{lambda_clip}_ot{lambda_ot}_mlm{lambda_mlm}_mae{lambda_mae}"
    args = build_args(lambda_clip, lambda_ot, lambda_mlm, lambda_mae, save_path, desc)

    trainer = Trainer(args, device=device)
    trainer.train(trial=trial)

    score, _ = trainer.evaluate(epoch=args.epochs)
    # trial.report(score, step=0)
    # if trial.should_prune():
    #     raise optuna.TrialPruned()

    return score


if __name__ == "__main__":
    os.makedirs("~/scratch/anchor/optuna/alignment", exist_ok=True)
    os.makedirs("~/scratch/anchor/optuna/unimodal", exist_ok=True)

    device = ddp_setup()
    rank = dist.get_rank()
    
    n_trials=20

    if rank == 0:
        pruner = MedianPruner(n_startup_trials=3)
        study1 = optuna.create_study(direction="maximize", pruner=pruner)
        study1.optimize(lambda t: objective_alignment(t, device), n_trials=n_trials)
        best_align = study1.best_params
        with open("~/scratch/anchor/optuna/best_alignment.json", "w") as f:
            json.dump(best_align, f, indent=4)
    else:
        study = None
        for _ in range(n_trials):
            try:
                objective_alignment(None, device)
            except optuna.TrialPruned:
                pass

        best_align = None

    #broadcast best params from rank 0 to all other ranks
    b_list = [best_align]
    dist.broadcast_object_list(b_list, src = 0)
    #grab best params from list
    best_align = b_list[0]

    if rank == 0:
        study2 = optuna.create_study(direction="maximize", pruner=pruner)
        study2.optimize(lambda t: objective_unimodal(t, best_align, device), n_trials=n_trials)
        best_unimodal = study2.best_params

        with open("results/best_unimodal.json", "w") as f:
            json.dump(best_unimodal, f, indent=4)
    else:
        study = None
        for _ in range(n_trials):
            try:
                objective_unimodal(None, best_align, device)
            except optuna.TrialPruned:
                pass


