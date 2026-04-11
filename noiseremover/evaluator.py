import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .data import mel_mask_to_wav_cfg


def compute_snr(clean: np.ndarray, enhanced: np.ndarray) -> float:
    noise = clean - enhanced
    return float(10 * np.log10(np.sum(clean ** 2) / (np.sum(noise ** 2) + 1e-8)))


def evaluate(model, test_loader: DataLoader, cfg: Config, output_dir: str = "outputs") -> pd.DataFrame:
    """
    Run evaluation on test_loader.
    Returns a DataFrame with per-batch metrics (SNRi, PESQ).
    """
    try:
        from pesq import pesq
        has_pesq = True
    except ImportError:
        has_pesq = False

    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()

    records = []
    with torch.no_grad():
        for noisy, clean, phase in tqdm(test_loader, desc="Evaluating"):
            noisy_t = noisy.to(device)
            mask = model(noisy_t).cpu()

            B = noisy.shape[0]
            for b in range(B):
                noisy_mel = noisy[b, 0].numpy()
                clean_mel = clean[b, 0].numpy()
                mask_np = mask[b, 0].numpy()
                phase_np = phase[b].numpy()

                ones = np.ones_like(mask_np)
                enhanced_wav = mel_mask_to_wav_cfg(noisy_mel, mask_np, phase_np, cfg)
                clean_wav    = mel_mask_to_wav_cfg(clean_mel, ones, phase_np, cfg)
                noisy_wav    = mel_mask_to_wav_cfg(noisy_mel, ones, phase_np, cfg)

                snr_in = compute_snr(clean_wav, noisy_wav)
                snr_out = compute_snr(clean_wav, enhanced_wav)
                snri = snr_out - snr_in

                row = {"snr_input": snr_in, "snr_output": snr_out, "snri": snri}

                if has_pesq:
                    try:
                        row["pesq_wb"] = pesq(cfg.sample_rate, clean_wav, enhanced_wav, "wb")
                    except Exception:
                        row["pesq_wb"] = float("nan")

                records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    print("\n=== Evaluation Results ===")
    print(df[["snri", "pesq_wb"] if "pesq_wb" in df.columns else ["snri"]].describe())
    return df
