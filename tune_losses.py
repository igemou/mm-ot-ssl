import os
import json
import optuna

from optuna.pruners import MedianPruner
from train import Trainer
from train import argparse as train_argparse


def build_args(lambda_clip, lambda_ot, lambda_mlm, lambda_mae, save_path):
    parser = train_argparse.ArgumentParser()

    parser.add_argument("--vision_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--text_name", type=str, default="bert-base-uncased")
    parser.add_argument("--shared_dim", type=int, default=256)
    parser.add_argument("--dataset", type=str, default="flickr")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--paired_fraction", type=float, default=0.2)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default=save_path)

    parser.add_argument("--lambda_clip", type=float, default=lambda_clip)
    parser.add_argument("--lambda_ot", type=float, default=lambda_ot)
    parser.add_argument("--lambda_mlm", type=float, default=lambda_mlm)
    parser.add_argument("--lambda_mae", type=float, default=lambda_mae)

    parser.add_argument("--use_anchored_ot", action="store_false")
    parser.add_argument("--alpha_anchor", type=float, default=0.0)

    parser.add_argument("--use_gw_ot", action="store_true")
    parser.add_argument("--patience", type=int, default=3)

    args = parser.parse_args([])
    return args


def objective_alignment(trial):
    lambda_clip = trial.suggest_float("lambda_clip", 0.3, 2.0, log=True)
    lambda_ot   = trial.suggest_float("lambda_ot", 0.05, 1.0, log=True)

    save_path = f"results/alignment/trial_{trial.number}"
    args = build_args(lambda_clip, lambda_ot, 0.0, 0.0, save_path)

    os.makedirs(args.save_dir, exist_ok=True)
    trainer = Trainer(args)
    trainer.train()

    score, _ = trainer.evaluate(epoch=args.epochs)
    trial.report(score, step=0)

    if trial.should_prune():
        raise optuna.TrialPruned()

    return score


def objective_unimodal(trial, best_align):
    lambda_mlm = trial.suggest_float("lambda_mlm", 0.1, 1.0)
    lambda_mae = trial.suggest_float("lambda_mae", 0.1, 1.0)

    lambda_clip = best_align["lambda_clip"]
    lambda_ot   = best_align["lambda_ot"]

    save_path = f"results/unimodal/trial_{trial.number}"
    args = build_args(lambda_clip, lambda_ot, lambda_mlm, lambda_mae, save_path)

    os.makedirs(args.save_dir, exist_ok=True)
    trainer = Trainer(args)
    trainer.train()

    score, _ = trainer.evaluate(epoch=args.epochs)
    trial.report(score, step=0)

    if trial.should_prune():
        raise optuna.TrialPruned()

    return score


if __name__ == "__main__":
    os.makedirs("results/alignment", exist_ok=True)
    os.makedirs("results/unimodal", exist_ok=True)

    pruner = MedianPruner(n_startup_trials=3)

    study1 = optuna.create_study(direction="maximize", pruner=pruner)
    study1.optimize(objective_alignment, n_trials=20)
    best_align = study1.best_params

    with open("results/best_alignment.json", "w") as f:
        json.dump(best_align, f, indent=4)

    study2 = optuna.create_study(direction="maximize", pruner=pruner)
    study2.optimize(lambda t: objective_unimodal(t, best_align), n_trials=20)
    best_unimodal = study2.best_params

    with open("results/best_unimodal.json", "w") as f:
        json.dump(best_unimodal, f, indent=4)
