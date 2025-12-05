#!/bin/bash

#SBATCH --partition=gpu --gres=gpu:1  
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --output=logs/optuna_coco_%j.out
#SBATCH -t 30:00:00
#SBATCH --mem=16g

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1) 
export MASTER_PORT=12345 

echo "=== Starting COCO Optuna Search ==="
#  Load a CUDA module
module load cuda
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

conda activate anchor

cd /users/bjoo2/code/anchor/

echo "Running Optuna on GW OT"
srun torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --rdzv_id=100 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:29400 \
  tune_losses_ddp.py

echo "=== Done! Check logs/optuna_coco_${SLURM_JOB_ID}.out ==="
