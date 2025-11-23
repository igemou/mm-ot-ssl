from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from transformers import AutoTokenizer
import random

class COCOMultiModalDataset(Dataset):
    """
    COCO multimodal dataset for SSL.
    Modes:
      - paired: matched image-caption pairs
      - unpaired: mismatched image & caption
      - image_only / text_only: single modality
    """

    def __init__(
        self,
        split="paired",
        tokenizer_name="bert-base-uncased",
        image_size=224,
        max_length=32,
        paired_fraction=0.2,
        seed=42,
        train=True,
        val_ratio=0.1,
    ):
        super().__init__()
        self.mode = split
        self.paired_fraction = paired_fraction
        random.seed(seed)

        # Load the dataset

        data_path = "/users/bjoo2/scratch/coco"
        split_string = "train" if train else "val"
        
        self.dataset = datasets.CocoCaptions(root=f"{data_path}/{split_string}2014", 
                                             annFile=f"{data_path}/annotations/captions_{split_string}2014.json")

        # Create paired/unpaired index partitions
        n_total = len(self.dataset)
        n_paired = int(n_total * paired_fraction)
        indices = list(range(n_total))
        random.shuffle(indices)
        self.paired_idx = indices[:n_paired]
        self.unpaired_idx = indices[n_paired:]
        half = len(self.unpaired_idx) // 2
        self.unpaired_A = self.unpaired_idx[:half]
        self.unpaired_B = self.unpaired_idx[half:]

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        if self.mode == "paired":
            return len(self.paired_idx)
        elif self.mode == "unpaired":
            return min(len(self.unpaired_A), len(self.unpaired_B))
        elif self.mode in ["image_only", "text_only"]:
            return len(self.dataset)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _tokenize(self, caption):
        t = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        return t["input_ids"].squeeze(0), t["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        if self.mode == "paired":
            img, cap = self.dataset[self.paired_idx[idx]]
            image = self.transform(img)
            caption = random.choice(cap).strip()

        elif self.mode == "unpaired":
            img, _ = self.dataset[self.unpaired_A[idx]]
            _, cap = self.dataset[self.unpaired_B[idx]]
            image = self.transform(img)
            caption = random.choice(cap).strip()

        elif self.mode == "image_only":
            img, _ = self.dataset[idx]
            image = self.transform(img)
            return {"image": image}

        elif self.mode == "text_only":
            _, cap = self.dataset[idx]
            caption = random.choice(cap).strip()
            ids, mask = self._tokenize(caption)
            return {"input_ids": ids, "attention_mask": mask}

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        ids, mask = self._tokenize(caption)
        return {"image": image, "input_ids": ids, "attention_mask": mask}
