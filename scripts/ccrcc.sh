#!/bin/bash
#SBATCH -p rsingh47-gcondo
#SBATCH --job-name=ccRCC
#SBATCH --output=logs/ccrcc_%j.out
#SBATCH --error=logs/ccrcc_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1             
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00

echo "=== Starting Flickr SSL pretraining ==="
module load cuda/12.1
source /oscar/home/igemou/anchored-mmSSL/.env/bin/activate

export HF_HOME=/oscar/scratch/igemou/hf_home
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=$HF_HOME

python train.py \
  --dataset ccrcc \
  --ccrcc_root data \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-4 \
  --eval_every 1 \
  --lambda_clip 1.0 \
  --lambda_ot 1.0 \
  --lambda_mlm 1.0 \
  --lambda_mae 1.0 \
  --mlm_mask_ratio 0.25 \
  --mae_mask_ratio 0.25 \
  --use_gw_ot \
  --save_dir checkpoints/ccrcc_gw_ot_allequal
