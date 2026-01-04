import os
import random
from typing import Dict, Any, List, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CCRCCMultiModalDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "paired",
        subset: str = "train",
        seed: int = 42,
        return_labels: bool = True,
        clingen_feature_cols: Optional[List[str]] = None,
    ):
        super().__init__()
        assert split in {"paired", "unpaired", "image_only", "text_only"}
        assert subset in {"train", "test"}

        self.root = root
        self.mode = split
        self.subset = subset
        self.return_labels = return_labels
        random.seed(seed)
        np.random.seed(seed)

        cg_path = os.path.join(root, "clinical+genomic_split.csv")
        if not os.path.exists(cg_path):
            raise FileNotFoundError(cg_path)
        df = pd.read_csv(cg_path)

        id_col = "case_id"
        split_col = "Split"
        label_col = "vital_status_12"

        if id_col not in df.columns:
            raise ValueError(f"{id_col} not in {cg_path}")
        if split_col not in df.columns:
            raise ValueError(f"{split_col} not in {cg_path}")
        if self.return_labels and label_col not in df.columns:
            raise ValueError(f"{label_col} not in {cg_path}")

        mask = df[split_col].astype(str).str.lower() == subset.lower()
        df = df[mask].copy()

        if clingen_feature_cols is None:
            drop_cols = {id_col, split_col}
            if self.return_labels:
                drop_cols.add(label_col)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_cols = [c for c in numeric_cols if c not in drop_cols]
        else:
            feature_cols = clingen_feature_cols

        if not feature_cols:
            raise ValueError("No clingen feature columns inferred.")

        self.clingen_ids: List[str] = df[id_col].astype(str).tolist()
        feats = df[feature_cols].fillna(0.0).values.astype(np.float32)
        self.clingen_feats: Dict[str, np.ndarray] = {
            cid: feats[i] for i, cid in enumerate(self.clingen_ids)
        }

        if self.return_labels:
            self.labels: Dict[str, float] = {}
            for _, row in df.iterrows():
                cid = str(row[id_col])
                self.labels[cid] = float(row[label_col])
        else:
            self.labels = {}

        self.ct_dir = os.path.join(root, "CT_Features")
        if not os.path.isdir(self.ct_dir):
            raise ValueError(f"CT_Features directory not found at {self.ct_dir}")

        self.case_to_npz = defaultdict(list)
        for f in os.listdir(self.ct_dir):
            if not f.endswith(".npz"):
                continue
            base = os.path.splitext(f)[0]           # e.g. "C3L-00610-1"
            parts = base.split("-")
            if len(parts) >= 2:
                case_id = "-".join(parts[:2])       # "C3L-00610"
            else:
                case_id = parts[0]
            path = os.path.join(self.ct_dir, f)
            self.case_to_npz[case_id].append(path)

        self.ct_ids: List[str] = [
            cid for cid in self.clingen_ids if cid in self.case_to_npz
        ]

        self.paired_ids = self.ct_ids
        self.clingen_only_ids = [cid for cid in self.clingen_ids if cid not in self.ct_ids]
        self.image_only_ids = self.ct_ids

        self.unpaired_ct_ids = self.ct_ids.copy()
        self.unpaired_clingen_ids = self.clingen_ids.copy()
        random.shuffle(self.unpaired_ct_ids)
        random.shuffle(self.unpaired_clingen_ids)

    def _load_ct_feature(self, case_id: str) -> np.ndarray:
        paths = self.case_to_npz.get(case_id, [])
        if not paths:
            raise FileNotFoundError(f"No CT npz files for case_id={case_id} in {self.ct_dir}")

        arrs = []
        for p in paths:
            npz = np.load(p)
            arr = npz[npz.files[0]]
            arr = arr.reshape(-1).astype(np.float32)
            arrs.append(arr)

        if len(arrs) == 1:
            feat = arrs[0]
        else:
            feat = np.stack(arrs, axis=0).mean(axis=0)
        return feat

    def __len__(self) -> int:
        if self.mode == "paired":
            return len(self.paired_ids)
        if self.mode == "unpaired":
            return min(len(self.unpaired_ct_ids), len(self.unpaired_clingen_ids))
        if self.mode == "image_only":
            return len(self.image_only_ids)
        if self.mode == "text_only":
            return len(self.clingen_ids)
        raise ValueError(self.mode)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.mode == "paired":
            cid = self.paired_ids[idx]
            ct = self._load_ct_feature(cid)
            cl = self.clingen_feats[cid]
            sample = {
                "case_id": cid,
                "ct": torch.from_numpy(ct),
                "clingen": torch.from_numpy(cl),
            }
            if self.return_labels:
                sample["label"] = torch.tensor(self.labels[cid], dtype=torch.float32)
            return sample

        if self.mode == "unpaired":
            cid_ct = self.unpaired_ct_ids[idx]
            cid_cl = self.unpaired_clingen_ids[idx]
            ct = self._load_ct_feature(cid_ct)
            cl = self.clingen_feats[cid_cl]
            sample = {
                "case_id_ct": cid_ct,
                "case_id_clingen": cid_cl,
                "ct": torch.from_numpy(ct),
                "clingen": torch.from_numpy(cl),
            }
            if self.return_labels:
                sample["label_ct"] = torch.tensor(self.labels.get(cid_ct, 0.0))
                sample["label_clingen"] = torch.tensor(self.labels.get(cid_cl, 0.0))
            return sample

        if self.mode == "image_only":
            cid = self.image_only_ids[idx]
            ct = self._load_ct_feature(cid)
            sample = {
                "case_id": cid,
                "ct": torch.from_numpy(ct),
            }
            if self.return_labels:
                sample["label"] = torch.tensor(self.labels[cid], dtype=torch.float32)
            return sample

        if self.mode == "text_only":
            cid = self.clingen_ids[idx]
            cl = self.clingen_feats[cid]
            sample = {
                "case_id": cid,
                "clingen": torch.from_numpy(cl),
            }
            if self.return_labels:
                sample["label"] = torch.tensor(self.labels[cid], dtype=torch.float32)
            return sample

        raise ValueError(self.mode)
