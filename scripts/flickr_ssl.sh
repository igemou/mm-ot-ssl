#!/bin/bash
#SBATCH -p rsingh47-gcondo
#SBATCH --job-name=FlickrSSL
#SBATCH --output=logs/flickr_ssl_%j.out
#SBATCH --error=logs/flickr_ssl_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1             
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

echo "=== Starting Flickr SSL pretraining ==="
module load cuda/12.1
source /oscar/home/igemou/anchored-mmSSL/.env/bin/activate

export HF_HOME=/oscar/scratch/igemou/hf_home
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=$HF_HOME

# Run training
python train.py \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-4 \
    --eval_every 1 \
    --save_dir checkpoints/flickr_pretrain \
    --paired_fraction 0.2 \
    --lambda_clip 1.0 \
    --lambda_ot 0.5 \
    --lambda_mlm 1.0 \
    --lambda_mae 1.0 \
    --use_anchored_ot \
    --alpha_anchor 0.1

echo "=== Done! Check logs/flickr_ssl_${SLURM_JOB_ID}.out ==="
