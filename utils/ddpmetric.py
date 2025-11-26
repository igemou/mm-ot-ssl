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
    # as well as the img emb normalization denom
    #allows us to use 1 less an allreduce
    count_denom = torch.cat((
        torch.tensor([[len(z_img), len(z_txt)]]),
        z_img.pow(order).sum(dim=0, keepdim = True) #1, emb_sz
    ), dim=1).to('cuda')
    dist.barrier(device_ids=[device])
    dist.all_reduce(count_denom)
    num_imgs, num_txts = int(count_denom[0,0].item()), int(count_denom[0,1].item())
    
    #manually normalize over the last dimension by gathering the normalization factor 
    #for all elts in the last dim, divide by max(eps, sum(x^ord)^(1/ord))
    img_denom = count_denom[:, 2:]
    img_denom = img_denom.pow(1.0 / float(order)).clamp(min=eps)

    #normalize
    z_img = z_img / img_denom

    #gather all text embs
    all_txt = torch.zeros((num_txts, ) + z_txt.shape[1:], dtype=z_txt.dtype, device='cuda')
    dist.all_gather_into_tensor(all_txt, z_txt)
    all_txt = all_txt.to(device)
    all_txt = F.normalize(all_txt, dim=-1)
    
    #do device_img <-> all_txt similarity
    partial_sim = z_img @ all_txt.t() #device_img, all_txt

    #gather all similarities
    all_sim = torch.zeros((num_imgs, num_txts), dtype=partial_sim.dtype, device='cuda')
    dist.all_gather_into_tensor(all_sim, partial_sim) #all_img, all_txt

    all_sim = all_sim.to(device)

    return all_sim


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