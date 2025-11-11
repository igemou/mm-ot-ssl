import random
from torch.utils.data import Sampler


class MultiStreamBatchSampler(Sampler):
    """
    Cycles through multiple datasets (paired, unpaired, image-only, text-only)
    according to a specified ratio.

    Args:
        datasets (dict): {"paired": paired_ds, "unpaired": unpaired_ds, ...}
        ratio (dict): sampling ratios for each dataset, e.g.
                      {"paired": 0.5, "unpaired": 0.3, "image_only": 0.1, "text_only": 0.1}
        seed (int): random seed
    """

    def __init__(self, datasets, ratio, seed=42):
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())
        self.ratio = ratio
        self.seed = seed

        total = sum(ratio.values())
        self.weights = [ratio[k] / total for k in self.dataset_names]
        self.min_len = min(len(ds) for ds in datasets.values())
        random.seed(seed)

    def __iter__(self):
        n_total = self.min_len
        for _ in range(n_total):
            split = random.choices(self.dataset_names, weights=self.weights, k=1)[0]
            idx = random.randint(0, len(self.datasets[split]) - 1)
            yield split, idx

    def __len__(self):
        return self.min_len
