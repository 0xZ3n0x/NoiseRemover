import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .config import Config


class NoisyCleanDataset(Dataset):
    def __init__(self, manifest_csv: str, cfg: Config, snr_filter=None):
        df = pd.read_csv(manifest_csv)
        if snr_filter is not None:
            df = df[df["snr_db"].isin(snr_filter)]
        if cfg.max_samples is not None:
            df = df.sample(n=min(cfg.max_samples, len(df)), random_state=42)
        self.records = df["npz_path"].tolist()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        data = np.load(self.records[idx])
        noisy = torch.from_numpy(data["noisy_mel"])   # (n_mels, T)
        clean = torch.from_numpy(data["clean_mel"])   # (n_mels, T)
        phase = torch.from_numpy(data["noisy_phase"]) # (n_fft//2+1, T)

        noisy = noisy.unsqueeze(0)  # (1, n_mels, T)
        clean = clean.unsqueeze(0)  # (1, n_mels, T)
        return noisy, clean, phase


def make_dataloaders(cfg: Config):
    train_manifest = os.path.join(cfg.processed_root, "train", "manifest.csv")
    val_manifest = os.path.join(cfg.processed_root, "val", "manifest.csv")

    train_ds = NoisyCleanDataset(train_manifest, cfg)
    val_ds = NoisyCleanDataset(val_manifest, cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
