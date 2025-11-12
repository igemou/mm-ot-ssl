# TODO: Add pretraining pipeline after fixing MAE decoder and its utilities
# And make they are integrated in mm_model\
import os
from argparse import ArgumentParser

from tqdm import trange

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch import optim
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP

def get_args_parser():
    parser = ArgumentParser('MM Anchor pretraining', add_help=False)
    
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--data_path', default="/users/bjoo2/data/bjoo2/mae",
                        help="directory to which the data will be downloaded")

    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    # Model parameters
    parser.add_argument('--model', default="base", type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    
    parser.add_argument('--save_path', default="/users/bjoo2/scratch/mae/weights",
                        help="directory to which the pretrained model weights should be saved")

    return parser

def setup(model):
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend)
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.accelerator.device_count()
    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id])
    return model, device_id

def cleanup():
    dist.destroy_process_group() 

def get_loader(batch_size: int = 32, ddp: bool = False):
    dataset = None
    sampler = DistributedSampler(dataset) if ddp else None
    samplers = [sampler] if dist else []

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler)
    return loader, *samplers

def main(args):
    model = None
    model, device_id = setup(model)

    lr = args.lr * float(args.batch_size) * float(os.environ["WORLD_SIZE"]) / 256.0
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    pbar = trange(0, args.epochs, desc="Training Epochs", postfix={})
    for epoch in pbar:
        train_stats = train_one_epoch(model, train_loader, train_sampler,
                                      optimizer, device_id, epoch, loss_scaler,
                                      args=args)
        test_stats = test(model, test_loader, test_sampler, device_id, args=args)

        postfix = {**train_stats, **test_stats}
        pbar.set_postfix(postfix)