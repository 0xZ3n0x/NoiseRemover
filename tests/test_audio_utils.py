import numpy as np
import pytest
from noiseremover.audio_utils import wav_to_mel_spectrogram, mel_mask_to_wav, mix_at_snr


SR = 16000
N_FFT = 1024
HOP = 256
WIN = 1024
N_MELS = 128
FMIN = 0.0
FMAX = 8000.0
DURATION = 4


def make_sine(freq=440, duration=DURATION, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_wav_to_mel_shape():
    wav = make_sine()
    mel, phase = wav_to_mel_spectrogram(wav, SR, N_FFT, HOP, WIN, N_MELS, FMIN, FMAX)
    assert mel.shape[0] == N_MELS
    assert phase.shape[0] == N_FFT // 2 + 1
    assert mel.shape[1] == phase.shape[1]


def test_mel_mask_to_wav_length():
    wav = make_sine()
    mel, phase = wav_to_mel_spectrogram(wav, SR, N_FFT, HOP, WIN, N_MELS, FMIN, FMAX)
    mask = np.ones_like(mel)
    out = mel_mask_to_wav(mel, mask, phase, N_FFT, HOP, WIN, N_MELS, SR, FMIN, FMAX)
    assert isinstance(out, np.ndarray)
    assert len(out) > 0


def test_mix_at_snr():
    clean = make_sine(freq=440)
    noise = make_sine(freq=1000)
    noisy = mix_at_snr(clean, noise, snr_db=0)
    assert noisy.shape == clean.shape
    # At 0 dB SNR, clean and noise RMS should be approximately equal
    clean_rms = np.sqrt(np.mean(clean ** 2))
    noise_in_mix = noisy - clean
    noise_rms = np.sqrt(np.mean(noise_in_mix ** 2))
    ratio_db = 20 * np.log10(clean_rms / (noise_rms + 1e-8))
    assert abs(ratio_db) < 1.0  # within 1 dB


def test_identity_mask():
    wav = make_sine()
    mel, phase = wav_to_mel_spectrogram(wav, SR, N_FFT, HOP, WIN, N_MELS, FMIN, FMAX)
    mask = np.ones_like(mel)
    out = mel_mask_to_wav(mel, mask, phase, N_FFT, HOP, WIN, N_MELS, SR, FMIN, FMAX)
    assert len(out) > 0
