"""
TensorBoard loglarından training/val loss curve çıkarır.

Usage:
    python scripts/plot_loss.py
    python scripts/plot_loss.py --runs runs/full_train_v6 --output-dir outputs/figures
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
}
plt.rcParams.update(STYLE)


def load_scalars(run_path: str, tag: str):
    ea = EventAccumulator(run_path)
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return [], []
    events = ea.Scalars(tag)
    steps  = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir",   default="runs")
    parser.add_argument("--output-dir", default="outputs/figures")
    parser.add_argument("--best-epoch", type=int, default=28, help="Epoch of best checkpoint")
    parser.add_argument("--best-val",   type=float, default=75.3971, help="Val loss of best checkpoint")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    runs = sorted([
        r for r in os.listdir(args.runs_dir)
        if os.path.isdir(os.path.join(args.runs_dir, r))
        and r.startswith("full_train")
        and r != "full_train"  # ilk run hatalı config ile yapıldı
    ])

    # -----------------------------------------------------------------------
    # 1. Best run: full loss curve (train + val)
    # -----------------------------------------------------------------------
    best_run = "full_train_v6"
    run_path = os.path.join(args.runs_dir, best_run)

    train_steps, train_vals = load_scalars(run_path, "train/loss_epoch")
    val_steps,   val_vals   = load_scalars(run_path, "val/loss_epoch")

    if train_vals and val_vals:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(train_steps, train_vals, label="Train loss", color="#3498db", linewidth=2)
        ax.plot(val_steps,   val_vals,   label="Val loss",   color="#e74c3c", linewidth=2)

        # Best checkpoint marker
        best_val_in_run = min(val_vals)
        best_ep         = val_steps[val_vals.index(best_val_in_run)]
        ax.axvline(best_ep, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.scatter([best_ep], [best_val_in_run], color="black", zorder=5,
                   label=f"Best checkpoint (epoch {best_ep}, val={best_val_in_run:.2f})")

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss (MSE + 0.1×L1)", fontsize=12)
        ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        plt.tight_layout()
        path = os.path.join(args.output_dir, "loss_curve.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # -----------------------------------------------------------------------
    # 2. Learning rate curve
    # -----------------------------------------------------------------------
    run_path = os.path.join(args.runs_dir, best_run)
    lr_steps, lr_vals = load_scalars(run_path, "train/lr")

    if lr_vals:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(lr_steps, lr_vals, color="#9b59b6", linewidth=2)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title("Learning Rate Schedule (CosineAnnealingLR)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(args.output_dir, "lr_curve.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    print(f"\nAll loss curves saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
