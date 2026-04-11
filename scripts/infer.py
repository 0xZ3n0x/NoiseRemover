"""
Single-file inference script with two modes.

  sample mode — pick a test .npz and produce noisy + denoised WAVs
  long   mode — take a LibriSpeech FLAC, mix with ESC-50 noise, denoise

Usage:
    python scripts/infer.py sample
    python scripts/infer.py sample --npz dataset/processed/test/npz/xxx.npz

    python scripts/infer.py long
    python scripts/infer.py long --noise rain --snr 0
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from noiseremover.config import load_config
from noiseremover.audio_utils import mel_mask_to_wav_cfg, load_audio, save_audio, mix_at_snr
from noiseremover.inference import denoise_file
from noiseremover.utils import get_device, load_model


# ---------------------------------------------------------------------------
# sample mode
# ---------------------------------------------------------------------------

def run_sample(args, cfg, model):
    import librosa
    device = next(model.parameters()).device

    npz_path = args.npz or str(random.choice(
        list(Path(cfg.processed_root, "test", "npz").glob("*.npz"))
    ))
    print(f"Sample: {npz_path}")
    d = np.load(npz_path)
    noisy_mel = d["noisy_mel"]
    phase = d["noisy_phase"]
    print(f"SNR: {d['snr_db']} dB | Noise: {d['noise_category']}")

    # Noisy waveform (identity mask)
    noisy_wav = mel_mask_to_wav_cfg(noisy_mel, np.ones_like(noisy_mel[:1]), phase, cfg)

    # Denoised waveform
    mel_t = torch.from_numpy(noisy_mel).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        mask = model(mel_t)[0, 0].cpu().numpy()
    denoised_wav = mel_mask_to_wav_cfg(noisy_mel, mask, phase, cfg)

    os.makedirs(args.output_dir, exist_ok=True)
    sf.write(f"{args.output_dir}/noisy.wav", noisy_wav, cfg.sample_rate)
    sf.write(f"{args.output_dir}/denoised.wav", denoised_wav, cfg.sample_rate)
    print(f"Saved:\n  {args.output_dir}/noisy.wav\n  {args.output_dir}/denoised.wav")


# ---------------------------------------------------------------------------
# long mode
# ---------------------------------------------------------------------------

def run_long(args, cfg, model):
    # Find longest LibriSpeech FLAC
    flac_files = sorted(
        Path(cfg.dataset_root, "librispeech").rglob("*.flac"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    clean_path = str(flac_files[0])
    print(f"Clean: {clean_path}")

    # Find noise file (exact category match first, then partial)
    meta = pd.read_csv(os.path.join(cfg.dataset_root, "esc50", "meta", "esc50.csv"))
    noise_rows = meta[meta["category"] == args.noise]
    if noise_rows.empty:
        noise_rows = meta[meta["category"].str.contains(args.noise, case=False)]
    if noise_rows.empty:
        raise ValueError(f"Noise category '{args.noise}' not found")
    noise_file = os.path.join(cfg.dataset_root, "esc50", "audio", noise_rows.iloc[0]["filename"])
    print(f"Noise: {noise_file} ({noise_rows.iloc[0]['category']})")

    clean_wav = load_audio(clean_path, cfg.sample_rate)
    noise_wav = load_audio(noise_file, cfg.sample_rate)

    if len(noise_wav) < len(clean_wav):
        noise_wav = np.tile(noise_wav, len(clean_wav) // len(noise_wav) + 1)
    noise_wav = noise_wav[:len(clean_wav)]

    noisy_wav = mix_at_snr(clean_wav, noise_wav, args.snr)
    print(f"Duration: {len(clean_wav)/cfg.sample_rate:.1f}s | SNR: {args.snr} dB")

    os.makedirs(args.output_dir, exist_ok=True)
    noisy_path    = os.path.join(args.output_dir, "long_noisy.wav")
    clean_path_out = os.path.join(args.output_dir, "long_clean.wav")
    denoised_path = os.path.join(args.output_dir, "long_denoised.wav")

    save_audio(noisy_path, noisy_wav, cfg.sample_rate)
    save_audio(clean_path_out, clean_wav, cfg.sample_rate)

    print("Denoising...")
    denoise_file(noisy_path, denoised_path, model, cfg)

    print(f"Saved:\n  {clean_path_out}\n  {noisy_path}\n  {denoised_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["sample", "long"])
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device (default: cpu)")
    # sample-mode options
    parser.add_argument("--npz",        default=None, help="Specific .npz file (sample mode)")
    # long-mode options
    parser.add_argument("--noise",      default="rain", help="ESC-50 category (long mode)")
    parser.add_argument("--snr",        type=float, default=0, help="SNR in dB (long mode)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    model, ckpt = load_model(cfg, args.checkpoint, device)
    print(f"Checkpoint: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    if args.mode == "sample":
        run_sample(args, cfg, model)
    else:
        run_long(args, cfg, model)


if __name__ == "__main__":
    main()
