"""
Per noise-category ve per SNR-level metrik analizi.

Test setindeki her örnek için SNRi ve PESQ hesaplar,
sonuçları CSV olarak kaydeder ve breakdown tabloları üretir.

Usage:
    python scripts/analyze.py
    python scripts/analyze.py --split test --output-dir outputs/analysis
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from noiseremover.config import load_config
from noiseremover.data import mel_mask_to_wav_cfg
from noiseremover.evaluator import compute_snr
from noiseremover.utils import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--split",      default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default="outputs/analysis")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (for quick test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cpu")
    model, ckpt = load_model(cfg, args.checkpoint, device)
    print(f"Checkpoint: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    try:
        from pesq import pesq as pesq_fn
        has_pesq = True
    except ImportError:
        has_pesq = False
        print("Warning: pesq not installed, skipping PESQ metric")

    manifest = pd.read_csv(os.path.join(cfg.processed_root, args.split, "manifest.csv"))
    if args.max_samples:
        manifest = manifest.sample(n=min(args.max_samples, len(manifest)), random_state=42)

    print(f"Evaluating {len(manifest)} samples from {args.split} split...")
    os.makedirs(args.output_dir, exist_ok=True)

    records = []
    with torch.no_grad():
        for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
            d = np.load(row["npz_path"])
            noisy_mel = d["noisy_mel"]
            clean_mel = d["clean_mel"]
            phase     = d["noisy_phase"]

            mel_t = torch.from_numpy(noisy_mel).unsqueeze(0).unsqueeze(0)
            mask  = model(mel_t)[0, 0].numpy()

            ones        = np.ones_like(mask)
            enhanced    = mel_mask_to_wav_cfg(noisy_mel, mask,  phase, cfg)
            clean_wav   = mel_mask_to_wav_cfg(clean_mel, ones,  phase, cfg)
            noisy_wav   = mel_mask_to_wav_cfg(noisy_mel, ones,  phase, cfg)

            snr_in  = compute_snr(clean_wav, noisy_wav)
            snr_out = compute_snr(clean_wav, enhanced)
            snri    = snr_out - snr_in

            rec = {
                "npz_path":       row["npz_path"],
                "noise_category": row["noise_category"],
                "snr_db":         row["snr_db"],
                "snr_input":      snr_in,
                "snr_output":     snr_out,
                "snri":           snri,
            }

            if has_pesq:
                try:
                    rec["pesq_wb"] = pesq_fn(cfg.sample_rate, clean_wav, enhanced, "wb")
                except Exception:
                    rec["pesq_wb"] = float("nan")

            records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.output_dir, "metrics_full.csv"), index=False)
    print(f"\nSaved {len(df)} records → {args.output_dir}/metrics_full.csv")

    cols = ["snri", "pesq_wb"] if has_pesq else ["snri"]

    # --- Overall ---
    print("\n=== Overall ===")
    print(df[cols].describe().round(3).to_string())

    # --- Per noise category ---
    print("\n=== Per Noise Category (mean) ===")
    by_cat = df.groupby("noise_category")[cols].mean().round(3).sort_values("snri", ascending=False)
    print(by_cat.to_string())
    by_cat.to_csv(os.path.join(args.output_dir, "by_noise_category.csv"))

    # --- Per SNR level ---
    print("\n=== Per SNR Level (mean) ===")
    by_snr = df.groupby("snr_db")[cols].mean().round(3)
    print(by_snr.to_string())
    by_snr.to_csv(os.path.join(args.output_dir, "by_snr_level.csv"))

    # --- Best / worst samples by SNRi ---
    print("\n=== Best 5 Samples (SNRi) ===")
    print(df.nlargest(5, "snri")[["noise_category", "snr_db", "snri"] + (["pesq_wb"] if has_pesq else [])].to_string(index=False))

    print("\n=== Worst 5 Samples (SNRi) ===")
    print(df.nsmallest(5, "snri")[["noise_category", "snr_db", "snri"] + (["pesq_wb"] if has_pesq else [])].to_string(index=False))

    # --- Best / worst by PESQ ---
    if has_pesq:
        print("\n=== Best 5 Samples (PESQ) ===")
        print(df.nlargest(5, "pesq_wb")[["noise_category", "snr_db", "snri", "pesq_wb"]].to_string(index=False))

        print("\n=== Worst 5 Samples (PESQ) ===")
        print(df.nsmallest(5, "pesq_wb")[["noise_category", "snr_db", "snri", "pesq_wb"]].to_string(index=False))

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
