"""
Train the U-Net speech enhancement model.

Usage:
    python scripts/train.py --config configs/default.yaml --run-name baseline
    python scripts/train.py --config configs/default.yaml --resume checkpoints/last.pt
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim

from noiseremover.config import load_config
from noiseremover.dataset import make_dataloaders
from noiseremover.model import UNet
from noiseremover.losses import SpectralLoss
from noiseremover.trainer import Trainer
from noiseremover.utils import get_device


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    device = get_device(args.gpu)
    print(f"Using device: {device}")

    model = UNet(base_channels=cfg.base_channels, depth=cfg.depth).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)
    loss_fn = SpectralLoss()

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    train_loader, val_loader = make_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    trainer = Trainer(model, optimizer, scheduler, loss_fn, train_loader, val_loader, cfg, run_name=args.run_name)
    trainer.fit()


if __name__ == "__main__":
    main()
