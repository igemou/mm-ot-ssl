#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1 
#SBATCH --output=logs/coco_ssl_%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH --mem=50g
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G



echo "=== Starting COCO SSL pretraining ==="
#  Load a CUDA module
module load cuda
module load miniconda3/23.11.0s
source /gpfs/runtime/opt/miniconda/4.12.0/etc/profile.d/conda.sh

conda activate anchor

# Run training
python train.py \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --eval_every 1 \
    --save_dir ~/scratch/coco/checkpoints/pretrain \
    --paired_fraction 0.2 \
    --lambda_clip 1.0 \
    --lambda_ot 0.5 \
    --lambda_mlm 1.0 \
    --lambda_mae 1.0 \
    --use_anchored_ot \
    --alpha_anchor 0.1 \
    --dataset coco

echo "=== Done! Check logs/flickr_ssl_${SLURM_JOB_ID}.out ==="
