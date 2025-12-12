#!/bin/bash
#SBATCH -p 3090-gcondo
#SBATCH --job-name=ccRCCSSL
#SBATCH -o test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00

echo "=== Starting ccRCC SSL pretraining ==="
module load cuda/12.1
source /oscar/home/igemou/anchored-mmSSL/.env/bin/activate

export HF_HOME=/oscar/scratch/igemou/hf_home
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=$HF_HOME

clip="$1"
ot="$2"
mlm="$3"
mae="$4"

python train.py \
  --dataset ccrcc \
  --ccrcc_root data \
  --epochs 30 --batch_size 32 --lr 1e-4 --eval_every 1 \
  --lambda_clip ${clip} --lambda_ot ${ot} --lambda_mlm ${mlm} --lambda_mae ${mae} \
  --mlm_mask_ratio 0.25 \
  --mae_mask_ratio 0.25 \
  --use_gw_ot \
  --save_dir checkpoints/ccrcc_gw_ot/${clip}_${ot}_${mlm}_${mae}
