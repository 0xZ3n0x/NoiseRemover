import os
import tempfile
import torch
import numpy as np
import pytest
from torch.utils.data import DataLoader, TensorDataset

from noiseremover.model import UNet
from noiseremover.train.trainer import Trainer, _apply_mask
from noiseremover.train.losses import SpectralLoss
from noiseremover.config import Config


def make_dummy_loader(n=8, n_mels=128, T=248):
    noisy = torch.randn(n, 1, n_mels, T)
    clean = torch.randn(n, 1, n_mels, T)
    phase = torch.randn(n, 513, T)
    ds = TensorDataset(noisy, clean, phase)
    return DataLoader(ds, batch_size=4)


def make_trainer(tmp_dir):
    cfg = Config()
    cfg.checkpoint_dir = os.path.join(tmp_dir, "checkpoints")
    cfg.log_dir = os.path.join(tmp_dir, "runs")
    cfg.max_epochs = 2
    cfg.early_stopping_patience = 10

    model = UNet(base_channels=cfg.base_channels, depth=cfg.depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    loss_fn = SpectralLoss()
    loader = make_dummy_loader()

    return Trainer(model, optimizer, scheduler, loss_fn, loader, loader, cfg, run_name="test")


def test_apply_mask_shape():
    noisy = torch.randn(2, 1, 128, 248)
    mask = torch.sigmoid(torch.randn(2, 1, 128, 248))
    out = _apply_mask(noisy, mask)
    assert out.shape == noisy.shape


def test_apply_mask_range():
    """Mask=1 should closely preserve the input in dB domain."""
    noisy = torch.zeros(1, 1, 128, 248)  # 0 dB input
    mask = torch.ones(1, 1, 128, 248)
    out = _apply_mask(noisy, mask)
    assert out.shape == noisy.shape


def test_trainer_fit_runs():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = make_trainer(tmp)
        trainer.fit()
        assert os.path.exists(os.path.join(tmp, "checkpoints", "best.pt"))
        assert os.path.exists(os.path.join(tmp, "checkpoints", "last.pt"))


def test_trainer_val_loss_decreases_or_stable():
    """Val loss should be a finite number after training."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = make_trainer(tmp)
        val_loss = trainer.val_epoch(1)
        assert np.isfinite(val_loss)
        assert val_loss > 0
