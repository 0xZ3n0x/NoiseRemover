"""
Gradio demo for real-time speech denoising.

Usage:
    python scripts/demo.py
    python scripts/demo.py --checkpoint checkpoints/best.pt --config configs/default.yaml
"""
import argparse
import tempfile

import numpy as np
import torch
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from noiseremover.config import load_config
from noiseremover.inference import denoise_file
from noiseremover.data.audio_utils import wav_to_mel_spectrogram
from noiseremover.utils import load_model, get_device


def spectrogram_figure(mel_db: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel bin")
    plt.tight_layout()
    return fig


def build_process_fn(model, cfg):
    def process(input_audio):
        if input_audio is None:
            return None, None, None

        sr, wav = input_audio
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
            import soundfile as sf
            sf.write(f_in.name, wav, sr)
            enhanced = denoise_file(f_in.name, f_out.name, model, cfg)

        noisy_mel, _ = wav_to_mel_spectrogram(
            wav[:cfg.clip_samples] if len(wav) >= cfg.clip_samples else wav,
            cfg.sample_rate, cfg.n_fft, cfg.hop_length,
            cfg.win_length, cfg.n_mels, cfg.fmin, cfg.fmax,
        )
        enhanced_mel, _ = wav_to_mel_spectrogram(
            enhanced[:cfg.clip_samples] if len(enhanced) >= cfg.clip_samples else enhanced,
            cfg.sample_rate, cfg.n_fft, cfg.hop_length,
            cfg.win_length, cfg.n_mels, cfg.fmin, cfg.fmax,
        )

        fig_noisy = spectrogram_figure(noisy_mel, "Noisy Input")
        fig_enhanced = spectrogram_figure(enhanced_mel, "Enhanced Output")

        return (cfg.sample_rate, enhanced), fig_noisy, fig_enhanced

    return process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device (default: cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    model, _ = load_model(cfg, args.checkpoint, device)
    print(f"Loaded model from {args.checkpoint}")

    process = build_process_fn(model, cfg)

    with gr.Blocks(title="NoiseRemover — Speech Enhancement") as demo:
        gr.Markdown("## NoiseRemover\nUpload or record noisy audio to get a denoised version.")
        with gr.Row():
            input_audio = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Noisy Input")
            output_audio = gr.Audio(label="Enhanced Output", type="numpy")
        with gr.Row():
            fig_noisy = gr.Plot(label="Noisy Spectrogram")
            fig_enhanced = gr.Plot(label="Enhanced Spectrogram")
        btn = gr.Button("Denoise", variant="primary")
        btn.click(fn=process, inputs=input_audio, outputs=[output_audio, fig_noisy, fig_enhanced])

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
