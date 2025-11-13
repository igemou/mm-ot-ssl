from coco_dataset import COCOMultiModalDataset
from torch.utils.data import DataLoader
def test_dataset():
    for mode in ["paired", "unpaired", "image_only", "text_only"]:
        print(f"\n mode: {mode} ---")
        ds = COCOMultiModalDataset(split="train", mode=mode, paired_fraction=0.1)
        dl = DataLoader(ds, batch_size=2, shuffle=True)

        batch = next(iter(dl))
        for k, v in batch.items():
            if hasattr(v, "shape"):
                print(f"{k}: {tuple(v.shape)}")
            else:
                print(f"{k}: {type(v)}")
        print(f"Length of {mode} dataset:", len(ds))

if __name__ == "__main__":
    test_dataset()
