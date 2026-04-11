import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from noiseremover.dataset import NoisyCleanDataset
from noiseremover.config import Config


def make_dummy_dataset(tmp_dir, n_samples=4):
    npz_dir = os.path.join(tmp_dir, "npz")
    os.makedirs(npz_dir)
    records = []
    for i in range(n_samples):
        noisy_mel = np.random.randn(128, 248).astype(np.float32)
        clean_mel = np.random.randn(128, 248).astype(np.float32)
        noisy_phase = np.random.randn(513, 248).astype(np.float32)
        path = os.path.join(npz_dir, f"sample_{i}.npz")
        np.savez_compressed(path, noisy_mel=noisy_mel, clean_mel=clean_mel, noisy_phase=noisy_phase, snr_db=np.float32(0))
        records.append({"npz_path": path, "snr_db": 0, "noise_category": "test"})
    manifest = os.path.join(tmp_dir, "manifest.csv")
    pd.DataFrame(records).to_csv(manifest, index=False)
    return manifest


def test_dataset_length():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = make_dummy_dataset(tmp, n_samples=4)
        cfg = Config()
        ds = NoisyCleanDataset(manifest, cfg)
        assert len(ds) == 4


def test_dataset_item_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = make_dummy_dataset(tmp, n_samples=2)
        cfg = Config()
        ds = NoisyCleanDataset(manifest, cfg)
        noisy, clean, phase = ds[0]
        assert noisy.shape == (1, 128, 248)
        assert clean.shape == (1, 128, 248)
        assert phase.shape == (513, 248)


def test_dataset_types():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = make_dummy_dataset(tmp, n_samples=2)
        cfg = Config()
        ds = NoisyCleanDataset(manifest, cfg)
        noisy, clean, phase = ds[0]
        assert isinstance(noisy, torch.Tensor)
        assert isinstance(clean, torch.Tensor)
        assert isinstance(phase, torch.Tensor)
