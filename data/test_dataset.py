from coco_dataset import COCOMultiModalDataset
from flickr_dataset import FlickrMultiModalDataset
from torch.utils.data import DataLoader

from datasets import config
print("Cache path:", config.HF_DATASETS_CACHE)

def test_dataset():
    for mode in ["paired", "unpaired", "image_only", "text_only"]:
        print(f"\n--- Mode: {mode} ---")
        ds = FlickrMultiModalDataset(split="test", mode=mode, paired_fraction=0.1)
        dl = DataLoader(ds, batch_size=2, shuffle=True)
        batch = next(iter(dl))
        for k, v in batch.items():
            print(f"{k}: {v.shape if hasattr(v, 'shape') else type(v)}")
        print("Dataset length:", len(ds))

if __name__ == "__main__":
    test_dataset()
