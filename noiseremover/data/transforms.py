import numpy as np
import torch
import torchaudio.functional as F_audio


# ---------------------------------------------------------------------------
# GPU-ready helpers (used by prepare_data batch pipeline)
# ---------------------------------------------------------------------------

def make_gpu_transforms(cfg, device: torch.device) -> dict:
    """
    Pre-build mel filterbank and hann window on *device*.
    Returns a dict consumed by wav_batch_to_mel().
    """
    window = torch.hann_window(cfg.win_length).to(device)
    mel_fb = F_audio.melscale_fbanks(
        n_freqs=cfg.n_fft // 2 + 1,
        f_min=cfg.fmin,
        f_max=cfg.fmax,
        n_mels=cfg.n_mels,
        sample_rate=cfg.sample_rate,
        norm="slaney",
        mel_scale="slaney",
    ).T.to(device)  # (n_mels, n_freqs)
    return {"window": window, "mel_fb": mel_fb}


def wav_batch_to_mel(
    wavs: list,          # list of (N,) numpy float32 arrays, all same length
    transforms: dict,    # from make_gpu_transforms()
    cfg,
    device: torch.device,
) -> tuple:
    """
    GPU-batched STFT + mel for a list of waveform chunks.

    Returns
    -------
    mel_db : (B, n_mels, T) float32 numpy  — log-mel spectrogram in dB
    phase  : (B, n_fft//2+1, T) float32 numpy — STFT phase
    """
    window = transforms["window"]
    mel_fb = transforms["mel_fb"]      # (n_mels, n_freqs)

    batch = torch.from_numpy(np.stack(wavs)).to(device)  # (B, N)

    stft = torch.stft(
        batch,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        return_complex=True,
    )  # (B, n_freqs, T)

    magnitude = stft.abs()   # (B, n_freqs, T)
    phase = stft.angle()     # (B, n_freqs, T)

    # Mel: (n_mels, n_freqs) x (B, n_freqs, T) -> (B, n_mels, T)
    mel = torch.einsum("mf,bft->bmt", mel_fb, magnitude)

    # Amplitude to dB with ref=max (matches librosa default)
    amin = 1e-5
    mel = mel.clamp(min=amin)
    ref = mel.amax(dim=(-2, -1), keepdim=True).clamp(min=amin)
    mel_db = 20.0 * torch.log10(mel / ref)

    return (
        mel_db.cpu().numpy().astype(np.float32),
        phase.cpu().numpy().astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Scalar helpers (used by inference / evaluation — CPU path)
# ---------------------------------------------------------------------------

def wav_to_mel_spectrogram(
    wav: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
):
    """
    Single-sample CPU fallback (used at inference / evaluation time).

    Returns
    -------
    mel_db : (n_mels, T) float32
    phase  : (n_fft//2+1, T) float32
    """
    import librosa
    stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_db = librosa.amplitude_to_db(mel_basis @ magnitude, ref=np.max)
    return mel_db.astype(np.float32), phase.astype(np.float32)


def mel_mask_to_wav(
    noisy_mel_db: np.ndarray,
    mask: np.ndarray,
    noisy_phase: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    sample_rate: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Apply mask and reconstruct waveform via iSTFT (inference)."""
    import librosa
    mel_amplitude = librosa.db_to_amplitude(noisy_mel_db)
    enhanced_amplitude = mel_amplitude * mask
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    magnitude = np.maximum(np.linalg.pinv(mel_basis) @ enhanced_amplitude, 0.0)
    stft = magnitude * np.exp(1j * noisy_phase)
    return librosa.istft(stft, hop_length=hop_length, win_length=win_length).astype(np.float32)


def mel_mask_to_wav_cfg(
    noisy_mel_db: np.ndarray,
    mask: np.ndarray,
    noisy_phase: np.ndarray,
    cfg,
) -> np.ndarray:
    """Convenience wrapper around mel_mask_to_wav that takes a Config object."""
    return mel_mask_to_wav(
        noisy_mel_db, mask, noisy_phase,
        cfg.n_fft, cfg.hop_length, cfg.win_length,
        cfg.n_mels, cfg.sample_rate, cfg.fmin, cfg.fmax,
    )
