####
# This code was directly repurposed from 
# https://github.com/facebookresearch/mae
####

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist

from utils.metrics import cosine_similarity


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
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


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

# Retrieval Accuracy (Image→Text and Text→Image)
def retrieval_accuracy(z_img, z_txt, device, topk=(1, 5)):
    """
    Computes retrieval accuracy for image→text and text→image directions.
    """
    sim = cosine_similarity(z_img, z_txt)
    labels = torch.arange(z_img.size(0), device=z_img.device)
    results = {}
    for k in topk:
        results[f"i2t@{k}"] = SmoothedValue()
        results[f"t2i@{k}"] = SmoothedValue()


    # Image → Text retrieval
    rank_i = sim.argsort(dim=-1, descending=True)
    for k in topk:
        correct_i = (rank_i[:, :k] == labels.unsqueeze(1)).any(dim=1).float()
        results[f"i2t@{k}"].update(correct_i.sum(), n=len(z_img))

    # Text → Image retrieval
    rank_t = sim.t().argsort(dim=-1, descending=True)
    for k in topk:
        correct_t = (rank_t[:, :k] == labels.unsqueeze(1)).any(dim=1).float()
        results[f"t2i@{k}"].update(correct_t.sum(), n=len(z_img))


    for _, meter in results.items():
        meter.synchronize_between_processes(device_ids=[device])

    avg = {k: meter.global_avg for k, meter in results.items()}

    return avg