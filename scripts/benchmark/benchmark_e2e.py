"""
End-to-end inference pipeline benchmark.

Her aşamayı ayrı ayrı ölçer:
  1. WAV yükleme
  2. STFT → Mel (librosa, per chunk)
  3. Model forward pass (CPU)
  4. Mask → iSTFT (librosa, per chunk)
  5. Overlap-add

Usage:
    python scripts/benchmark_e2e.py
    python scripts/benchmark_e2e.py --duration 10 --n-runs 20
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from noiseremover.config import load_config
from noiseremover.data.audio_utils import load_audio, wav_to_mel_spectrogram, mel_mask_to_wav
from noiseremover.utils import load_model

N_RUNS = 10


def fmt(mean_ms, std_ms):
    return f"{mean_ms:6.1f} ms  ±{std_ms:.1f}"


def benchmark_e2e(model, cfg, wav, n_runs=N_RUNS):
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    chunk_len = cfg.clip_samples
    overlap = 0.5
    hop = int(chunk_len * (1 - overlap))

    pad = chunk_len - (len(wav) - chunk_len) % hop if len(wav) > chunk_len else chunk_len - len(wav)
    wav_padded = np.concatenate([wav, np.zeros(pad, dtype=np.float32)])
    starts = list(range(0, len(wav_padded) - chunk_len + 1, hop))
    n_chunks = len(starts)

    times_stft  = []
    times_model = []
    times_istft = []
    times_total = []

    window = np.hanning(chunk_len).astype(np.float32)

    # Warmup — model + librosa cache ısıtma (3 tam pipeline geçişi)
    for _ in range(3):
        chunk = wav_padded[:chunk_len]
        mel, phase = wav_to_mel_spectrogram(
            chunk, cfg.sample_rate, cfg.n_fft, cfg.hop_length,
            cfg.win_length, cfg.n_mels, cfg.fmin, cfg.fmax,
        )
        mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            mask = model(mel_t)[0, 0].numpy()
        mel_mask_to_wav(mel, mask, phase, cfg.n_fft, cfg.hop_length,
                        cfg.win_length, cfg.n_mels, cfg.sample_rate, cfg.fmin, cfg.fmax)

    for _ in range(n_runs):
        out    = np.zeros_like(wav_padded)
        weight = np.zeros_like(wav_padded)
        t_stft = t_model = t_istft = 0.0

        t0_total = time.perf_counter()

        for start in starts:
            chunk = wav_padded[start: start + chunk_len]

            # 1. STFT → Mel
            t0 = time.perf_counter()
            mel, phase = wav_to_mel_spectrogram(
                chunk, cfg.sample_rate, cfg.n_fft, cfg.hop_length,
                cfg.win_length, cfg.n_mels, cfg.fmin, cfg.fmax,
            )
            t_stft += time.perf_counter() - t0

            # 2. Model forward
            t0 = time.perf_counter()
            mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                mask = model(mel_t)[0, 0].numpy()
            t_model += time.perf_counter() - t0

            # 3. Mask → iSTFT
            t0 = time.perf_counter()
            enhanced = mel_mask_to_wav(
                mel, mask, phase,
                cfg.n_fft, cfg.hop_length, cfg.win_length,
                cfg.n_mels, cfg.sample_rate, cfg.fmin, cfg.fmax,
            )
            enhanced = enhanced[:chunk_len]
            if len(enhanced) < chunk_len:
                enhanced = np.pad(enhanced, (0, chunk_len - len(enhanced)))
            t_istft += time.perf_counter() - t0

            out[start: start + chunk_len]    += enhanced * window
            weight[start: start + chunk_len] += window

        # 4. Overlap-add normalize (ihmal edilebilir, yine de ölçüyoruz)
        weight = np.where(weight > 1e-8, weight, 1.0)
        _ = out / weight

        t_total = time.perf_counter() - t0_total

        times_stft.append(t_stft * 1000)
        times_model.append(t_model * 1000)
        times_istft.append(t_istft * 1000)
        times_total.append(t_total * 1000)

    return {
        "n_chunks":   n_chunks,
        "stft":  (np.mean(times_stft),  np.std(times_stft)),
        "model": (np.mean(times_model), np.std(times_model)),
        "istft": (np.mean(times_istft), np.std(times_istft)),
        "total": (np.mean(times_total), np.std(times_total)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--duration",   type=float, default=10.0, help="Synthetic audio duration in seconds")
    parser.add_argument("--n-runs",     type=int,   default=N_RUNS)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, ckpt = load_model(cfg, args.checkpoint, torch.device("cpu"))
    print(f"Checkpoint : epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    print(f"Audio      : {args.duration:.1f}s synthetic white noise")
    print(f"Chunk      : {cfg.clip_duration_s}s  |  Overlap: 50%")
    print(f"Runs       : {args.n_runs}")
    print("=" * 55)

    wav = np.random.randn(int(cfg.sample_rate * args.duration)).astype(np.float32)
    r = benchmark_e2e(model, cfg, wav, n_runs=args.n_runs)

    n   = r["n_chunks"]
    tot = r["total"][0]

    audio_ms  = args.duration * 1000
    effective = cfg.clip_duration_s * 0.5 * 1000

    print(f"Chunks processed : {n}  ({args.duration:.0f}s audio, 50% overlap)")
    print(f"{'Stage':<20} {'Total (ms)':>12}   {'Per chunk':>10}   {'%':>6}")
    print("-" * 55)
    for key in ["stft", "model", "istft"]:
        m, s = r[key]
        pct = m / tot * 100
        print(f"  {key.upper():<18} {fmt(m, s)}   {m/n:>7.1f} ms   {pct:>5.1f}%")
    print("-" * 55)
    total_m, total_s = r["total"]
    print(f"  {'TOTAL':<18} {fmt(total_m, total_s)}   {total_m/n:>7.1f} ms   100.0%")
    print("=" * 55)
    print(f"Realtime ratio   : {audio_ms/total_m:.1f}x")
    print(f"Per-chunk budget : {effective:.0f} ms  →  actual {total_m/n:.1f} ms/chunk")


if __name__ == "__main__":
    main()
