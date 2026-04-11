import torch
import pytest
from noiseremover.model import UNet


def test_unet_output_shape():
    model = UNet(base_channels=16, depth=3)
    x = torch.randn(2, 1, 128, 248)
    out = model(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_unet_mask_range():
    model = UNet(base_channels=16, depth=3)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 1, 128, 248)
        mask = model(x)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0


def test_unet_different_time_lengths():
    model = UNet(base_channels=16, depth=3)
    model.eval()
    with torch.no_grad():
        for T in [100, 200, 248, 300]:
            x = torch.randn(1, 1, 128, T)
            out = model(x)
            assert out.shape == x.shape, f"Shape mismatch for T={T}"


def test_unet_parameter_count():
    model = UNet(base_channels=32, depth=5)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 1_000_000, "Model seems too small"
    assert n_params < 50_000_000, "Model seems too large"
