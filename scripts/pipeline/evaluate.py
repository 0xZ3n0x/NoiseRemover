"""
Evaluate a trained model on the test split.

Usage:
    python scripts/evaluate.py --config configs/default.yaml
    python scripts/evaluate.py --config configs/default.yaml --split val --checkpoint checkpoints/last.pt
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from noiseremover.config import load_config
from noiseremover.data.dataset import NoisyCleanDataset
from noiseremover.evaluator import evaluate
from noiseremover.utils import get_device, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.gpu)
    model, ckpt = load_model(cfg, args.checkpoint, device)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    manifest = os.path.join(cfg.processed_root, args.split, "manifest.csv")
    ds = NoisyCleanDataset(manifest, cfg)
    loader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    df = evaluate(model, loader, cfg, output_dir=args.output_dir)
    print(f"\nResults saved to {args.output_dir}/metrics.csv")


if __name__ == "__main__":
    main()
