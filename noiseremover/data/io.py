import numpy as np
import torch
import torchaudio
import soundfile as sf


def mix_at_snr(clean_wav: np.ndarray, noise_wav: np.ndarray, snr_db: float) -> np.ndarray:
    clean_rms = np.sqrt(np.mean(clean_wav ** 2) + 1e-8)
    noise_rms = np.sqrt(np.mean(noise_wav ** 2) + 1e-8)
    scale = clean_rms / (noise_rms * 10 ** (snr_db / 20.0))
    return (clean_wav + scale * noise_wav).astype(np.float32)


def load_audio(path: str, sr: int) -> np.ndarray:
    wav, orig_sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if orig_sr != sr:
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, orig_sr, sr)
        wav = wav_t.squeeze(0).numpy()
    return wav.astype(np.float32)


def save_audio(path: str, wav: np.ndarray, sr: int) -> None:
    sf.write(path, wav, sr)
