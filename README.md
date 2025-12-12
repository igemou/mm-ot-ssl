## Intro 
mm-ot-ssl is a self-supervised multimodal learning framework for training joint representations from partially paired and fully unpaired data. It combines CLIP-style contrastive alignment, masked reconstruction (MLM and MAE), and Gromov–Wasserstein optimal transport to align modalities at both the instance and distribution level. The framework is modality-agnostic and works with raw inputs or precomputed features, and we evaluate it on standard image–text benchmarks (Flickr30k, MSCOCO) as well as a medical dataset (ccRCC) with CT and clinical–genomic features.


## How to Use

### Setup
```bash
git clone https://github.com/igemou/mm-ot-ssl.git
cd mm-ot-ssl
pip install -r requirements.txt
```

### Datasets

Supported datasets:
- Flickr30k / MSCOCO: image–text benchmarks with partial pairing
- ccRCC: precomputed CT features paired with clinical–genomic vectors

### Training
Example: Flickr dataset
```bash
python train.py \
  --epochs 30 --batch_size 16 --lr 1e-4 --eval_every 1 \
  --paired_fraction 0.2 \
  --lambda_clip 1.0 --lambda_ot 1.0 --lambda_mlm 1.0 --lambda_mae 1.0 \
  --use_gw_ot \
  --save_dir checkpoints/flickr_gw_ot
```
