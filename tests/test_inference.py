import os
import tempfile
import numpy as np
import torch
import soundfile as sf
import pytest

from noiseremover.inference import denoise_file
from noiseremover.model import UNet
from noiseremover.config import Config


def make_dummy_model(cfg):
    model = UNet(base_channels=cfg.base_channels, depth=cfg.depth)
    model.eval()
    return model


def make_wav(path, duration=4.0, sr=16000):
    wav = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1
    sf.write(path, wav, sr)


def test_denoise_file_output_exists():
    cfg = Config()
    model = make_dummy_model(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "noisy.wav")
        out_path = os.path.join(tmp, "denoised.wav")
        make_wav(in_path, sr=cfg.sample_rate)
        denoise_file(in_path, out_path, model, cfg)
        assert os.path.exists(out_path)


def test_denoise_file_output_length():
    cfg = Config()
    model = make_dummy_model(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "noisy.wav")
        out_path = os.path.join(tmp, "denoised.wav")
        duration = 8.0
        make_wav(in_path, duration=duration, sr=cfg.sample_rate)
        denoise_file(in_path, out_path, model, cfg)
        out_wav, sr = sf.read(out_path)
        assert sr == cfg.sample_rate
        assert len(out_wav) > 0


def test_denoise_file_short_clip():
    """Clips shorter than one chunk should still be processed."""
    cfg = Config()
    model = make_dummy_model(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "short.wav")
        out_path = os.path.join(tmp, "denoised.wav")
        make_wav(in_path, duration=2.0, sr=cfg.sample_rate)
        denoise_file(in_path, out_path, model, cfg)
        assert os.path.exists(out_path)
