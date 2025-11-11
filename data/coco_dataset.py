
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from pycocotools.coco import COCO


class COCOMultiModalDataset(Dataset):
    """
    COCO dataset for multimodal SSL:
      - paired:  image-caption pairs (20%)
      - unpaired: mismatched images & captions
      - image_only: image samples only
      - text_only: caption samples only
    """

    def __init__(
        self,
        root,
        captions_file,
        split="paired",
        tokenizer_name="bert-base-uncased",
        image_size=224,
        max_length=32,
        paired_fraction=0.2,
        seed=42,
    ):
        super().__init__()
        self.coco = COCO(captions_file)
        self.root = root
        self.split = split
        self.paired_fraction = paired_fraction
        self.seed = seed

        self.image_ids = sorted(list(self.coco.imgs.keys()))
        self.ann_ids = sorted(list(self.coco.anns.keys()))
        random.seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        n_total = len(self.image_ids)
        n_paired = int(n_total * paired_fraction)
        shuffled = self.image_ids.copy()
        random.shuffle(shuffled)

        self.paired_ids = shuffled[:n_paired]
        self.unpaired_img_ids = shuffled[n_paired:]
        half = len(self.unpaired_img_ids) // 2
        self.unpaired_img_A = self.unpaired_img_ids[:half]
        self.unpaired_img_B = self.unpaired_img_ids[half:]

    def __len__(self):
        if self.split == "paired":
            return len(self.paired_ids)
        elif self.split == "unpaired":
            return min(len(self.unpaired_img_A), len(self.unpaired_img_B))
        elif self.split == "image_only":
            return len(self.image_ids)
        elif self.split == "text_only":
            return len(self.ann_ids)
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __getitem__(self, idx):
        if self.split == "paired":
            return self._get_paired(idx)
        elif self.split == "unpaired":
            return self._get_unpaired(idx)
        elif self.split == "image_only":
            return self._get_image_only(idx)
        elif self.split == "text_only":
            return self._get_text_only(idx)
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def _load_image(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.root, img_info["file_name"])
        image = Image.open(path).convert("RGB")
        return self.transform(image)

    def _tokenize_caption(self, caption):
        t = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        return t["input_ids"].squeeze(0), t["attention_mask"].squeeze(0)

    def _get_paired(self, idx):
        img_id = self.paired_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        caption = self.coco.loadAnns(random.choice(ann_ids))[0]["caption"]

        image = self._load_image(img_id)
        ids, mask = self._tokenize_caption(caption)
        return {"image": image, "input_ids": ids, "attention_mask": mask}

    def _get_unpaired(self, idx):
        # mismatched image-caption pairs from disjoint halves
        img_id = self.unpaired_img_A[idx]
        txt_img_id = self.unpaired_img_B[idx]
        ann_ids = self.coco.getAnnIds(imgIds=txt_img_id)
        caption = self.coco.loadAnns(random.choice(ann_ids))[0]["caption"]

        image = self._load_image(img_id)
        ids, mask = self._tokenize_caption(caption)
        return {"image": image, "input_ids": ids, "attention_mask": mask}

    def _get_image_only(self, idx):
        img_id = self.image_ids[idx]
        image = self._load_image(img_id)
        return {"image": image}

    def _get_text_only(self, idx):
        ann_id = self.ann_ids[idx % len(self.ann_ids)]
        caption = self.coco.loadAnns(ann_id)[0]["caption"]
        ids, mask = self._tokenize_caption(caption)
        return {"input_ids": ids, "attention_mask": mask}
