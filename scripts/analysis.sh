#!/bin/bash
#SBATCH -p rsingh47-gcondo
#SBATCH --job-name=Analysis
#SBATCH --output=logs/Analysis_%j.out
#SBATCH --error=logs/Analysis_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1             
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

module load cuda/12.1
source /oscar/home/igemou/anchored-mmSSL/.env/bin/activate

export HF_HOME=/oscar/scratch/igemou/hf_home
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=$HF_HOME

python evaluate.py --checkpoint checkpoints/flickr_plain_ot/epoch_30.pt
python analysis.py --checkpoint checkpoints/flickr_plain_ot/epoch_30.pt

python evaluate.py --checkpoint checkpoints/flickr_anchored_ot/epoch_30.pt 
python analysis.py --checkpoint checkpoints/flickr_anchored_ot/epoch_30.pt

python evaluate.py --checkpoint checkpoints/flickr_clip_20p/epoch_20.pt
python analysis.py --checkpoint checkpoints/flickr_clip_20p/epoch_20.pt

python evaluate.py --checkpoint checkpoints/flickr_clip_100p/epoch_20.pt
python analysis.py --checkpoint checkpoints/flickr_clip_100p/epoch_20.pt

python evaluate.py --checkpoint checkpoints/flickr_gw_ot_20p/epoch_20.pt
python analysis.py --checkpoint checkpoints/flickr_gw_ot_20p/epoch_20.pt


python qualitative_plots.py \
  --checkpoints \
    checkpoints/flickr_plain_ot/epoch_30.pt \
    checkpoints/flickr_anchored_ot/epoch_30.pt \
    checkpoints/flickr_clip_20p/epoch_20.pt \
    checkpoints/flickr_clip_100p/epoch_20.pt \
    checkpoints/flickr_gw_ot_20p/epoch_20.pt

echo "=== Done! Check logs/flickr_ssl_${SLURM_JOB_ID}.out ==="
