import numpy as np
import torch

from .config import Config
from .data import wav_to_mel_spectrogram, mel_mask_to_wav, load_audio, save_audio


def denoise_file(input_path: str, output_path: str, model: torch.nn.Module, cfg: Config, overlap: float = 0.5) -> np.ndarray:
    """
    Denoise an arbitrary-length audio file using sliding window + overlap-add.

    Args:
        overlap: fraction of chunk to overlap (0.5 = 50%)

    Returns:
        denoised waveform as numpy array
    """
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    wav = load_audio(input_path, cfg.sample_rate)
    chunk_len = cfg.clip_samples
    hop = int(chunk_len * (1 - overlap))

    # Pad so we cover the full signal
    pad = chunk_len - (len(wav) - chunk_len) % hop if len(wav) > chunk_len else chunk_len - len(wav)
    wav_padded = np.concatenate([wav, np.zeros(pad, dtype=np.float32)])

    window = np.hanning(chunk_len).astype(np.float32)
    out = np.zeros_like(wav_padded)
    weight = np.zeros_like(wav_padded)

    starts = range(0, len(wav_padded) - chunk_len + 1, hop)
    for start in starts:
        chunk = wav_padded[start: start + chunk_len]
        mel, phase = wav_to_mel_spectrogram(
            chunk, cfg.sample_rate, cfg.n_fft, cfg.hop_length,
            cfg.win_length, cfg.n_mels, cfg.fmin, cfg.fmax,
        )

        mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,n_mels,T)
        with torch.no_grad():
            mask = model(mel_t)[0, 0].cpu().numpy()

        enhanced = mel_mask_to_wav(
            mel, mask, phase,
            cfg.n_fft, cfg.hop_length, cfg.win_length,
            cfg.n_mels, cfg.sample_rate, cfg.fmin, cfg.fmax,
        )

        # Trim/pad to chunk_len samples (iSTFT may differ slightly)
        enhanced = enhanced[:chunk_len]
        if len(enhanced) < chunk_len:
            enhanced = np.pad(enhanced, (0, chunk_len - len(enhanced)))

        out[start: start + chunk_len] += enhanced * window
        weight[start: start + chunk_len] += window

    weight = np.where(weight > 1e-8, weight, 1.0)
    out = out / weight
    result = out[:len(wav)]

    save_audio(output_path, result, cfg.sample_rate)
    return result
