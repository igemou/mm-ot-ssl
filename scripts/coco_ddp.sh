#!/bin/bash

#SBATCH --partition=gpu --gres=gpu:1  
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --output=logs/coco_ssl_ddp_%j.out
#SBATCH -t 03:00:00
#SBATCH --mem=16g

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1) 
export MASTER_PORT=12345 

echo "=== Starting COCO SSL pretraining ==="
#  Load a CUDA module
module load cuda
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

conda activate anchor

cd /users/bjoo2/code/anchor/

# Run training
#echo "Running Plain OT on COCO"
#srun torchrun \
#  --nnodes=2 \
#  --nproc_per_node=1 \
#  --rdzv_id=100 \
#  --rdzv_backend=c10d \
#  --rdzv_endpoint=$MASTER_ADDR:29400 \
#  train_ddp.py \
#  --dataset coco\
#  --epochs 30 \
#  --batch_size 16 \
#  --lr 1e-4 \
#  --eval_every 1 \
#  --paired_fraction 0.2 \
#  --lambda_clip 1.0 --lambda_ot 0.5 --lambda_mlm 1.0  --lambda_mae 1.0 \
#  --save_dir /users/bjoo2/scratch/anchor/checkpoints/coco_plain_ot\
#  --desc_dir /users/bjoo2/scratch/anchor/\
#  --desc plain_ot
#
#echo "Running Anchor OT on COCO"
#srun torchrun \
#  --nnodes=2 \
#  --nproc_per_node=1 \
#  --rdzv_id=100 \
#  --rdzv_backend=c10d \
#  --rdzv_endpoint=$MASTER_ADDR:29400 \
#  train_ddp.py \
#    --dataset coco\
#    --epochs 30 \
#    --batch_size 16 \
#    --lr 1e-4 \
#    --eval_every 1 \
#    --paired_fraction 0.2 \
#    --lambda_clip 1.0 --lambda_ot 0.2 --lambda_mlm 1.0 --lambda_mae 1.0 \
#    --use_anchored_ot \
#    --alpha_anchor 0.8\
#    --save_dir /users/bjoo2/scratch/anchor/checkpoints/coco_anchored_ot \
#    --desc_dir /users/bjoo2/scratch/anchor/\
#    --desc anchored_ot
#
#echo "Running CLIP Alignment on COCO with 20% Paired"
#srun torchrun \
# --nnodes=2 \
# --nproc_per_node=1 \
# --rdzv_id=100 \
# --rdzv_backend=c10d \
# --rdzv_endpoint=$MASTER_ADDR:29400 \
# train_ddp.py \
# --dataset coco\
# --epochs 30 --batch_size 32 --lr 1e-4 --eval_every 1 \
# --paired_fraction 0.2 \
# --lambda_clip 1.0 --lambda_ot 0.0 --lambda_mlm 1.0 --lambda_mae 1.0 \
# --save_dir /users/bjoo2/scratch/anchor/checkpoints/coco_clip_20p \
# --desc_dir /users/bjoo2/scratch/anchor/\
# --desc clip_only_20p_b32

#echo "Running CLIP Alignment on COCO with 100% Paired"
#srun torchrun \
#  --nnodes=2 \
#  --nproc_per_node=1 \
#  --rdzv_id=100 \
#  --rdzv_backend=c10d \
#  --rdzv_endpoint=$MASTER_ADDR:29400 \
#  train_ddp.py \
#  --dataset coco\
#  --epochs 30 --batch_size 16 --lr 1e-4 --eval_every 1 \
#  --paired_fraction 1.0 \
#  --lambda_clip 1.0 --lambda_ot 0.0 --lambda_mlm 1.0 --lambda_mae 1.0 \
#  --save_dir /users/bjoo2/scratch/anchor/checkpoints/coco_clip_100p \
#  --desc_dir /users/bjoo2/scratch/anchor/\
#  --desc clip_only_100p

echo "Running GW OT on COCO"
srun torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --rdzv_id=100 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:29400 \
  train_ddp.py \
    --dataset coco\
    --epochs 30 --batch_size 32 --lr 1e-4 --eval_every 1 \
    --paired_fraction 0.2 \
    --lambda_clip .75 --lambda_ot 1.0 --lambda_mlm 0.75 --lambda_mae 1.0 \
    --use_gw_ot \
    --save_dir /users/bjoo2/scratch/anchor/checkpoints/coco_gw_ot_20p \
    --desc_dir /users/bjoo2/scratch/anchor/\
    --desc gw_b32_clip.75_ot1_mlm.75_mae1

echo "=== Done! Check logs/coco_ssl_ddp_${SLURM_JOB_ID}.out ==="
