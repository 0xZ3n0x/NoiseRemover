"""
Gradio demo for real-time speech denoising.

Usage:
    python scripts/demo.py
    python scripts/demo.py --checkpoint checkpoints/best.pt --config configs/default.yaml
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from noiseremover.config import load_config
from noiseremover.inference import denoise_file
from noiseremover.data import wav_to_mel_spectrogram
from noiseremover.utils import load_model

SAMPLES_DIR = Path(__file__).parent / "demo_samples"

PRESETS = {
    "Siren — 0 dB  (best case)":       "siren_0dB",
    "Clock alarm — 0 dB  (best case)":  "clock_alarm_0dB",
    "Rain — 0 dB":                       "rain_0dB",
    "Coughing — 5 dB":                   "coughing_5dB",
    "Insects — 15 dB  (worst case)":    "insects_15dB",
    "Train — 15 dB  (worst case)":      "train_15dB",
}


def three_panel_figure(noisy_mel, clean_mel, enhanced_mel):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    vmin = min(noisy_mel.min(), clean_mel.min(), enhanced_mel.min())
    vmax = max(noisy_mel.max(), clean_mel.max(), enhanced_mel.max())

    for ax, mel, title in zip(axes,
                               [noisy_mel, clean_mel, enhanced_mel],
                               ["Noisy Input", "Clean Reference", "Enhanced Output"]):
        ax.imshow(mel, origin="lower", aspect="auto", cmap="magma",
                  vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Mel bin")

    plt.tight_layout()
    return fig


def compute_metrics(clean_wav, enhanced_wav, noisy_wav, sample_rate):
    min_len = min(len(clean_wav), len(enhanced_wav), len(noisy_wav))
    c = clean_wav[:min_len]
    e = enhanced_wav[:min_len]
    n = noisy_wav[:min_len]

    eps = 1e-10
    noise_in  = n - c
    noise_out = e - c

    snr_in  = 10 * np.log10(np.sum(c**2) / (np.sum(noise_in**2)  + eps) + eps)
    snr_out = 10 * np.log10(np.sum(c**2) / (np.sum(noise_out**2) + eps) + eps)
    snri    = snr_out - snr_in

    try:
        from pesq import pesq
        pesq_score = pesq(sample_rate, c, e, "wb")
    except Exception:
        pesq_score = None

    return snri, pesq_score


def build_process_fn(model, cfg):
    def process(input_audio, clean_audio):
        if input_audio is None:
            return None, None, "SNRi: —  ·  PESQ: —"

        sr, wav = input_audio
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
            sf.write(f_in.name, wav, sr)
            enhanced = denoise_file(f_in.name, f_out.name, model, cfg)

        clip = slice(0, cfg.clip_samples)
        noisy_mel, _    = wav_to_mel_spectrogram(wav[clip],      cfg.sample_rate, cfg.n_fft, cfg.hop_length, cfg.win_length, cfg.n_mels, cfg.fmin, cfg.fmax)
        enhanced_mel, _ = wav_to_mel_spectrogram(enhanced[clip], cfg.sample_rate, cfg.n_fft, cfg.hop_length, cfg.win_length, cfg.n_mels, cfg.fmin, cfg.fmax)

        # Clean reference (from preset or fallback to noisy)
        if clean_audio is not None:
            _, clean_wav = clean_audio
            if clean_wav.dtype != np.float32:
                clean_wav = clean_wav.astype(np.float32) / np.iinfo(clean_wav.dtype).max
        else:
            clean_wav = wav

        clean_mel, _ = wav_to_mel_spectrogram(
            clean_wav[clip], cfg.sample_rate, cfg.n_fft, cfg.hop_length,
            cfg.win_length, cfg.n_mels, cfg.fmin, cfg.fmax,
        )

        fig = three_panel_figure(noisy_mel, clean_mel, enhanced_mel)

        snri, pesq_score = compute_metrics(clean_wav, enhanced, wav, cfg.sample_rate)
        pesq_str = f"{pesq_score:.2f}" if pesq_score is not None else "N/A"
        metrics = f"SNRi: **{snri:+.2f} dB**  ·  PESQ: **{pesq_str}**"

        return (cfg.sample_rate, enhanced), fig, metrics

    return process


def load_preset(preset_name):
    if not preset_name or preset_name not in PRESETS:
        return None, None
    key = PRESETS[preset_name]
    noisy_path = SAMPLES_DIR / f"{key}_noisy.wav"
    clean_path = SAMPLES_DIR / f"{key}_clean.wav"
    if not noisy_path.exists():
        return None, None
    noisy_wav, sr = sf.read(str(noisy_path))
    clean_wav, _  = sf.read(str(clean_path))
    return (sr, noisy_wav.astype(np.float32)), (sr, clean_wav.astype(np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--share",      action="store_true")
    parser.add_argument("--device",     default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device(args.device)
    model, ckpt = load_model(cfg, args.checkpoint, device)
    print(f"Loaded checkpoint — epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    process = build_process_fn(model, cfg)

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif !important; }
    """

    with gr.Blocks(title="NoiseRemover") as demo:
        gr.Markdown(
            """
            # NoiseRemover — Speech Enhancement
            Select a preset or upload your own noisy audio, then click **Denoise**.
            """
        )

        # Hidden state for clean reference (used by presets)
        clean_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                preset_dd = gr.Dropdown(
                    choices=list(PRESETS.keys()),
                    label="Preset examples",
                    value=None,
                )
            with gr.Column(scale=3):
                pass

        with gr.Row():
            input_audio  = gr.Audio(sources=["upload", "microphone"], type="numpy",
                                    label="Noisy Input", interactive=True,
                                    buttons=["download"], editable=False)
            with gr.Column():
                output_audio = gr.Audio(label="Enhanced Output", type="numpy",
                                        interactive=False, buttons=["download"], editable=False)
                metrics_md = gr.Markdown("SNRi: —  ·  PESQ: —")

        btn = gr.Button("Denoise", variant="primary", size="lg")

        spec_plot = gr.Plot(label="Spectrograms: Noisy / Clean Reference / Enhanced")

        # Preset → fill noisy audio + stash clean reference
        preset_dd.change(
            fn=load_preset,
            inputs=preset_dd,
            outputs=[input_audio, clean_state],
        )

        btn.click(
            fn=process,
            inputs=[input_audio, clean_state],
            outputs=[output_audio, spec_plot, metrics_md],
        )

    demo.launch(share=args.share, theme=gr.themes.Soft(), css=css)


if __name__ == "__main__":
    main()
