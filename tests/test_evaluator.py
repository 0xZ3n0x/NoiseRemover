import numpy as np
import pytest

from noiseremover.evaluator import compute_snr


def test_compute_snr_perfect():
    """If enhanced == clean, SNR should be very high."""
    clean = np.random.randn(16000).astype(np.float32)
    snr = compute_snr(clean, clean)
    assert snr > 60.0


def test_compute_snr_zero_noise():
    """SNR should increase as noise decreases."""
    clean = np.random.randn(16000).astype(np.float32)
    noise = np.random.randn(16000).astype(np.float32) * 0.01
    enhanced = clean + noise
    snr = compute_snr(clean, enhanced)
    assert snr > 30.0


def test_compute_snr_high_noise():
    """High noise should give low SNR."""
    clean = np.random.randn(16000).astype(np.float32) * 0.01
    noise = np.random.randn(16000).astype(np.float32)
    enhanced = clean + noise
    snr = compute_snr(clean, enhanced)
    assert snr < 0.0


def test_compute_snr_returns_float():
    clean = np.ones(1000, dtype=np.float32)
    enhanced = clean + np.random.randn(1000).astype(np.float32) * 0.1
    result = compute_snr(clean, enhanced)
    assert isinstance(result, float)
