"""
Analiz sonuçlarını görselleştirir.

Üretilen grafikler:
  1. Per noise-category SNRi bar chart (tüm 50 kategori)
  2. Per noise-category PESQ bar chart
  3. Per SNR-level SNRi + PESQ çift eksen
  4. SNRi dağılımı histogram
  5. Best/worst 5 örnek spektrogramları (noisy vs denoised vs clean)
  6. Scatter: SNRi vs PESQ

Usage:
    python scripts/visualize.py
    python scripts/visualize.py --analysis-dir outputs/analysis --output-dir outputs/figures
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from noiseremover.config import load_config
from noiseremover.audio_utils import mel_mask_to_wav_cfg
from noiseremover.utils import load_model

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
}
plt.rcParams.update(STYLE)

COLOR_GOOD = "#2ecc71"
COLOR_BAD  = "#e74c3c"
COLOR_MID  = "#3498db"


# ---------------------------------------------------------------------------
# 1. Per noise-category bar charts
# ---------------------------------------------------------------------------

def plot_by_category(by_cat: pd.DataFrame, output_dir: str):
    for metric, label, color in [
        ("snri",    "SNRi (dB)",  COLOR_MID),
        ("pesq_wb", "PESQ",       "#9b59b6"),
    ]:
        if metric not in by_cat.columns:
            continue

        df = by_cat.sort_values(metric, ascending=True)
        fig, ax = plt.subplots(figsize=(10, 14))
        bars = ax.barh(df.index, df[metric], color=color, alpha=0.85, edgecolor="white")

        # Değerleri bar üzerine yaz
        for bar, val in zip(bars, df[metric]):
            ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=8.5)

        ax.set_xlabel(label, fontsize=12)
        ax.set_title(f"Per Noise Category — {label}", fontsize=14, fontweight="bold")
        ax.axvline(df[metric].mean(), color="black", linestyle="--", linewidth=1, alpha=0.5, label=f"Mean: {df[metric].mean():.2f}")
        ax.legend(fontsize=10)
        plt.tight_layout()
        path = os.path.join(output_dir, f"by_category_{metric}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 2. Per SNR-level chart
# ---------------------------------------------------------------------------

def plot_by_snr(by_snr: pd.DataFrame, output_dir: str):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    snr_levels = by_snr.index.astype(int)
    x = np.arange(len(snr_levels))
    width = 0.35

    bars1 = ax1.bar(x - width/2, by_snr["snri"], width, label="SNRi (dB)", color=COLOR_MID, alpha=0.85)
    ax1.set_ylabel("SNRi (dB)", color=COLOR_MID, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=COLOR_MID)

    if "pesq_wb" in by_snr.columns:
        bars2 = ax2.bar(x + width/2, by_snr["pesq_wb"], width, label="PESQ", color="#e67e22", alpha=0.85)
        ax2.set_ylabel("PESQ", color="#e67e22", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="#e67e22")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{v} dB" for v in snr_levels])
    ax1.set_xlabel("Input SNR Level", fontsize=12)
    ax1.set_title("Per SNR Level — SNRi and PESQ", fontsize=14, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels() if "pesq_wb" in by_snr.columns else ([], [])
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, "by_snr_level.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 3. SNRi dağılım histogramı
# ---------------------------------------------------------------------------

def plot_snri_histogram(df: pd.DataFrame, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["snri"], bins=50, color=COLOR_MID, alpha=0.8, edgecolor="white")
    ax.axvline(df["snri"].mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"Mean: {df['snri'].mean():.2f} dB")
    ax.axvline(0, color=COLOR_BAD, linestyle="-", linewidth=1.5, alpha=0.7, label="SNRi = 0 (no gain)")
    ax.set_xlabel("SNRi (dB)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("SNRi Distribution — Test Set", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "snri_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 4. SNRi vs PESQ scatter
# ---------------------------------------------------------------------------

def plot_scatter(df: pd.DataFrame, output_dir: str):
    if "pesq_wb" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(df["snri"], df["pesq_wb"], alpha=0.3, s=12, c=df["snr_db"],
                    cmap="RdYlGn", vmin=-5, vmax=15)
    plt.colorbar(sc, ax=ax, label="Input SNR (dB)")
    ax.set_xlabel("SNRi (dB)", fontsize=12)
    ax.set_ylabel("PESQ", fontsize=12)
    ax.set_title("SNRi vs PESQ — Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "snri_vs_pesq.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 5. Spectrogram: noisy vs denoised vs clean
# ---------------------------------------------------------------------------

def plot_spectrogram_panel(noisy_mel, enhanced_mel, clean_mel,
                           title: str, path: str):
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

    vmin = min(noisy_mel.min(), clean_mel.min())
    vmax = max(noisy_mel.max(), clean_mel.max())

    for i, (mel, label) in enumerate([
        (noisy_mel,    "Noisy Input"),
        (enhanced_mel, "Enhanced (Model Output)"),
        (clean_mel,    "Clean Reference"),
    ]):
        ax = fig.add_subplot(gs[i])
        im = ax.imshow(mel, origin="lower", aspect="auto", cmap="magma",
                       vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time frames")
        if i == 0:
            ax.set_ylabel("Mel bins")
        else:
            ax.set_yticks([])

    fig.colorbar(im, ax=fig.get_axes(), shrink=0.8, label="dB")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_spectrograms(df: pd.DataFrame, model, cfg, output_dir: str):
    device = torch.device("cpu")

    def process(row):
        d = np.load(row["npz_path"])
        noisy_mel = d["noisy_mel"]
        clean_mel = d["clean_mel"]
        phase     = d["noisy_phase"]
        mel_t = torch.from_numpy(noisy_mel).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            mask = model(mel_t)[0, 0].numpy()
        ones = np.ones_like(mask)
        enhanced_mel_amp = noisy_mel.copy()
        import librosa
        amp = librosa.db_to_amplitude(noisy_mel) * mask
        enhanced_mel = librosa.amplitude_to_db(amp, ref=np.max)
        return noisy_mel, enhanced_mel, clean_mel

    spec_dir = os.path.join(output_dir, "spectrograms")
    os.makedirs(spec_dir, exist_ok=True)

    # Best 5 by SNRi
    for rank, (_, row) in enumerate(df.nlargest(5, "snri").iterrows(), 1):
        noisy_mel, enhanced_mel, clean_mel = process(row)
        snri_val = row["snri"]
        pesq_val = row.get("pesq_wb", float("nan"))
        cat = row["noise_category"]
        snr = int(row["snr_db"])
        title = f"Best #{rank} | {cat} | input SNR={snr} dB | SNRi={snri_val:.1f} dB | PESQ={pesq_val:.2f}"
        path = os.path.join(spec_dir, f"best_{rank:02d}_{cat}_snr{snr}.png")
        plot_spectrogram_panel(noisy_mel, enhanced_mel, clean_mel, title, path)

    # Worst 5 by SNRi
    for rank, (_, row) in enumerate(df.nsmallest(5, "snri").iterrows(), 1):
        noisy_mel, enhanced_mel, clean_mel = process(row)
        snri_val = row["snri"]
        pesq_val = row.get("pesq_wb", float("nan"))
        cat = row["noise_category"]
        snr = int(row["snr_db"])
        title = f"Worst #{rank} | {cat} | input SNR={snr} dB | SNRi={snri_val:.1f} dB | PESQ={pesq_val:.2f}"
        path = os.path.join(spec_dir, f"worst_{rank:02d}_{cat}_snr{snr}.png")
        plot_spectrogram_panel(noisy_mel, enhanced_mel, clean_mel, title, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="configs/default.yaml")
    parser.add_argument("--checkpoint",   default="checkpoints/best.pt")
    parser.add_argument("--analysis-dir", default="outputs/analysis")
    parser.add_argument("--output-dir",   default="outputs/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df      = pd.read_csv(os.path.join(args.analysis_dir, "metrics_full.csv"))
    by_cat  = pd.read_csv(os.path.join(args.analysis_dir, "by_noise_category.csv"), index_col=0)
    by_snr  = pd.read_csv(os.path.join(args.analysis_dir, "by_snr_level.csv"), index_col=0)

    print(f"Loaded {len(df)} samples")

    plot_by_category(by_cat, args.output_dir)
    plot_by_snr(by_snr, args.output_dir)
    plot_snri_histogram(df, args.output_dir)
    plot_scatter(df, args.output_dir)

    cfg = load_config(args.config)
    model, _ = load_model(cfg, args.checkpoint, torch.device("cpu"))
    plot_spectrograms(df, model, cfg, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
