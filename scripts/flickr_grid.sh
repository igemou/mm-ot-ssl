#!/bin/bash
#SBATCH -p 3090-gcondo
#SBATCH --job-name=FlickrSSL
#SBATCH -o test
#SBATCH --gres=gpu:1             
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00

echo "=== Starting Flickr SSL pretraining ==="
module load anaconda/2023.09-0-7nso27y
source activate STProj

clip="$1"
ot="$2"
mlm="$3"
mae="$4"


python ../train.py \
  --epochs 30 --batch_size 32 --lr 1e-4 --eval_every 1 \
  --paired_fraction 0.2 \
  --lambda_clip ${clip} --lambda_ot ${ot} --lambda_mlm ${mlm} --lambda_mae ${mae} \
  --use_gw_ot \
  --save_dir /oscar/scratch/ajain59/SSL_Project/flickr_ckpt/${clip}_${ot}_${mlm}_${mae} 


