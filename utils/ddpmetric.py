import builtins
import datetime
import os
import time
from collections import defaultdict, deque, ChainMap
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


class SmoothedValue(object):
    """
    ####
    # This code was directly repurposed from 
    # https://github.com/facebookresearch/mae
    ####
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value

    def synchronize_between_processes(self, device_ids):
        """
        Warning: does not synchronize the deque!
        """
        if not dist.is_available() or not dist.is_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier(device_ids=device_ids)
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

def ddp_cos_sim(z_img, z_txt, device, order = 2, eps = 1e-12):
    #get the total number of images and texts
    count = torch.tensor([len(z_img), len(z_txt)], device=z_img.device, dtype=torch.int).to('cuda')
    dist.barrier(device_ids=[device])
    dist.all_reduce(count)
    num_imgs, num_txts = count.tolist()

    z_img = F.normalize(z_img, dim=-1)
    z_txt = F.normalize(z_txt, dim=-1)

    #gather all text embs
    all_txt = torch.zeros((num_txts, ) + z_txt.shape[1:], dtype=z_txt.dtype, device='cuda')
    dist.all_gather_into_tensor(all_txt, z_txt)
    all_txt = all_txt.to(device)
    
    #do device_img <-> all_txt similarity
    partial_sim = z_img @ all_txt.t() #device_img, all_txt

    #gather all similarities
    all_sim = torch.zeros((num_imgs, num_txts), dtype=partial_sim.dtype, device='cuda')
    dist.all_gather_into_tensor(all_sim, partial_sim) #all_img, all_txt

    all_sim = all_sim.to(device)

    return all_sim

def _ddp_cos_sim(z_img, z_txt, device, order = 2, eps = 1e-12):
    raw = []
    normed = []

    print("getting stats")
    count = torch.tensor([len(z_img), len(z_txt)], device=z_img.device, dtype=torch.int).to('cuda')
    dist.barrier(device_ids=[device])
    dist.all_reduce(count)
    num_imgs, num_txts = count.tolist()
    print(f"Note: Gathered num imgs: {num_imgs}, local num imgs: {len(z_img)}, Gathered num txt: {num_txts}, local num imgs: {len(z_txt)}")

    print("gathering all img zs")
    all_img = torch.zeros((num_imgs, ) + z_img.shape[1:], dtype=z_img.dtype, device='cuda')
    dist.all_gather_into_tensor(all_img, z_img)
    raw.append(all_img)
    
    print("gathering all txt zs")
    all_txt = torch.zeros((num_txts, ) + z_txt.shape[1:], dtype=z_txt.dtype, device='cuda')
    dist.all_gather_into_tensor(all_txt, z_txt)
    raw.append(all_txt)

    z_img = F.normalize(z_img, dim=-1)
    z_txt = F.normalize(z_txt, dim=-1)

    print("gathering normed img zs")
    all_img = torch.zeros((num_imgs, ) + z_img.shape[1:], dtype=z_img.dtype, device='cuda')
    dist.all_gather_into_tensor(all_img, z_img)
    normed.append(all_img)

    print("gathering normed txt zs")
    all_txt = torch.zeros((num_txts, ) + z_txt.shape[1:], dtype=z_txt.dtype, device='cuda')
    dist.all_gather_into_tensor(all_txt, z_txt)
    normed.append(all_txt)
    
    #do device_img <-> all_txt similarity
    partial_sim = z_img @ all_txt.t() #device_img, all_txt

    #gather all similarities
    print("gathering sims")
    all_sim = torch.zeros((num_imgs, num_txts), dtype=partial_sim.dtype, device='cuda')
    dist.all_gather_into_tensor(all_sim, partial_sim) #all_img, all_txt

    all_sim = all_sim.to(device)

    return all_sim, raw, normed


# Retrieval Accuracy (Image→Text)
def i2t_retrieval_accuracy(sim, topk=(1, 5)):
    """
    Computes retrieval accuracy for image→text and text→image directions.
    """
    labels = torch.arange(sim.size(0), device=sim.device)
    results = {}

    # Image → Text retrieval
    rank_i = sim.argsort(dim=-1, descending=True)
    for k in topk:
        correct_i = (rank_i[:, :k] == labels.unsqueeze(1)).any(dim=1).float().mean()
        results[f"i2t@{k}"] = correct_i.item()

    return results

# Retrieval Accuracy (Text→Image)
def t2i_retrieval_accuracy(sim, topk=(1, 5)):
    """
    Computes retrieval accuracy for image→text and text→image directions.
    """
    labels = torch.arange(sim.size(0), device=sim.device)
    results = {}

    # Text → Image retrieval
    rank_t = sim.t().argsort(dim=-1, descending=True)
    for k in topk:
        correct_t = (rank_t[:, :k] == labels.unsqueeze(1)).any(dim=1).float().mean()
        results[f"t2i@{k}"] = correct_t.item()

    return results

def gather_dicts(device_dict):
    #theres a funny way to do this by using a sparse tensor and all gathering it across ranks
    #where the ith value in the sparse tensor is the topi retrieval value
    gathered_dicts = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_dicts, device_dict)

    ret = dict(ChainMap(*gathered_dicts))

    return ret