"""
GPU-accelerated spectrogram pair generation.

Pipeline (single-threaded CPU, GPU for all heavy compute):
  1. Preload ESC-50 noise clips into RAM
  2. Loop clean files: load → chunk → mix at SNR
  3. Batch chunks → GPU (STFT + mel)
  4. Save .npz to disk

Usage:
    python scripts/prepare_data.py --config configs/default.yaml
    python scripts/prepare_data.py --config configs/default.yaml --batch-size 512
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm

from noiseremover.config import load_config
from noiseremover.data.audio_utils import make_gpu_transforms, wav_batch_to_mel, mix_at_snr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def glob_audio(root, exts=(".flac", ".wav")):
    return sorted(str(p) for p in Path(root).rglob("*") if p.suffix in exts)


def speaker_id(path):
    # LibriSpeech: .../speaker/chapter/file.flac
    return Path(path).parts[-3]


def load_audio_sf(path: str, target_sr: int) -> np.ndarray:
    """Load audio with soundfile + resample with torchaudio if needed."""
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
        wav = wav_t.squeeze(0).numpy()
    return wav.astype(np.float32)


def preload_noise(noise_files: list, sr: int) -> dict:
    """Load all ESC-50 clips into RAM (~640 MB). Done once."""
    print(f"Preloading {len(noise_files)} noise clips into RAM...")
    cache = {}
    for path in tqdm(noise_files, desc="Noise"):
        try:
            cache[path] = load_audio_sf(path, sr)
        except Exception:
            pass
    return cache


def save_npz(npz_path: str, noisy_mel, clean_mel, noisy_phase, snr_db, noise_cat):
    np.savez_compressed(
        npz_path,
        noisy_mel=noisy_mel,
        clean_mel=clean_mel,
        noisy_phase=noisy_phase,
        snr_db=np.float32(snr_db),
        noise_category=noise_cat,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Number of chunks to process per GPU batch")
    parser.add_argument("--max-files-per-speaker", type=int, default=None,
                        help="Max FLAC files to process per speaker (None = all)")
    parser.add_argument("--max-chunks-per-file", type=int, default=None,
                        help="Max 4-second chunks to take per FLAC file (None = all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device: {device} ({gpu_name})")

    transforms = make_gpu_transforms(cfg, device)
    chunk_samples = int(cfg.sample_rate * cfg.clip_duration_s)

    # --- Index clean files ---
    libri_root = os.path.join(cfg.dataset_root, "librispeech")
    esc50_root = os.path.join(cfg.dataset_root, "esc50")

    print("Indexing LibriSpeech...")
    clean_files = glob_audio(libri_root)

    # Group files by speaker and optionally limit per speaker
    from collections import defaultdict
    speaker_files = defaultdict(list)
    for f in clean_files:
        speaker_files[speaker_id(f)].append(f)

    if args.max_files_per_speaker is not None:
        for spk in speaker_files:
            rng.shuffle(speaker_files[spk])
            speaker_files[spk] = speaker_files[spk][:args.max_files_per_speaker]

    clean_files = [f for files in speaker_files.values() for f in files]
    print(f"  {len(clean_files)} files across {len(speaker_files)} speakers")

    speakers = sorted(speaker_files.keys())
    rng.shuffle(speakers)

    n_val  = max(1, int(len(speakers) * cfg.val_split))
    n_test = max(1, int(len(speakers) * cfg.test_split))
    val_speakers  = set(speakers[:n_val])
    test_speakers = set(speakers[n_val:n_val + n_test])

    def get_split(path):
        s = speaker_id(path)
        if s in val_speakers:  return "val"
        if s in test_speakers: return "test"
        return "train"

    # --- Index & preload noise ---
    print("Indexing ESC-50...")
    meta = pd.read_csv(os.path.join(esc50_root, "meta", "esc50.csv"))
    noise_files = [os.path.join(esc50_root, "audio", r["filename"]) for _, r in meta.iterrows()]
    noise_cats  = {os.path.join(esc50_root, "audio", r["filename"]): r["category"]
                   for _, r in meta.iterrows()}
    noise_cache = preload_noise(noise_files, cfg.sample_rate)
    available_noise = [p for p in noise_files if p in noise_cache]

    # --- Create output dirs ---
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(cfg.processed_root, split, "npz"), exist_ok=True)

    # --- Batch state ---
    batch_clean = []
    batch_noisy = []
    batch_meta  = []   # (npz_path, snr_db, noise_cat, split)
    manifests   = {"train": [], "val": [], "test": []}

    def flush():
        if not batch_clean:
            return
        mel_clean, _      = wav_batch_to_mel(batch_clean, transforms, cfg, device)
        mel_noisy, phases = wav_batch_to_mel(batch_noisy, transforms, cfg, device)

        for i, (npz_path, snr_db, noise_cat, split) in enumerate(batch_meta):
            save_npz(npz_path, mel_noisy[i], mel_clean[i], phases[i], snr_db, noise_cat)
            manifests[split].append({
                "npz_path":       npz_path,
                "snr_db":         snr_db,
                "noise_category": noise_cat,
            })

        batch_clean.clear()
        batch_noisy.clear()
        batch_meta.clear()

    # --- Main loop ---
    print(f"Processing {len(clean_files)} files  |  batch_size={args.batch_size}")
    for clean_path in tqdm(clean_files, desc="Files"):
        split = get_split(clean_path)

        try:
            clean_wav = load_audio_sf(clean_path, cfg.sample_rate)
        except Exception:
            continue

        chunks = [
            clean_wav[i:i + chunk_samples]
            for i in range(0, len(clean_wav) - chunk_samples + 1, chunk_samples)
        ]
        if args.max_chunks_per_file is not None:
            chunks = chunks[:args.max_chunks_per_file]

        stem   = Path(clean_path).stem
        npz_dir = os.path.join(cfg.processed_root, split, "npz")

        for chunk_idx, clean_chunk in enumerate(chunks):
            for snr_db in cfg.snr_levels_db:
                npz_path = os.path.join(npz_dir, f"{stem}_c{chunk_idx}_snr{snr_db}.npz")

                noise_path  = rng.choice(available_noise)
                noise_wav   = noise_cache[noise_path]

                if len(noise_wav) < chunk_samples:
                    noise_wav = np.tile(noise_wav, chunk_samples // len(noise_wav) + 1)
                start       = rng.randint(0, len(noise_wav) - chunk_samples)
                noise_chunk = noise_wav[start:start + chunk_samples]

                noisy_chunk = mix_at_snr(clean_chunk, noise_chunk, snr_db)

                batch_clean.append(clean_chunk)
                batch_noisy.append(noisy_chunk)
                batch_meta.append((npz_path, snr_db, noise_cats[noise_path], split))

                if len(batch_clean) >= args.batch_size:
                    flush()

    flush()  # leftover chunks

    # --- Write manifests ---
    for split, records in manifests.items():
        out_path = os.path.join(cfg.processed_root, split, "manifest.csv")
        pd.DataFrame(records).to_csv(out_path, index=False)
        print(f"{split}: {len(records)} samples  →  {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
