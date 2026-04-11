"""
CPU ve GPU inference sürelerini ölçer, realtime kullanım için değerlendirir.

Usage:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --checkpoint checkpoints/best_pruned.pt
"""
import argparse
import time
import numpy as np
import torch
from noiseremover.config import load_config
from noiseremover.model import UNet

N_RUNS = 50


def benchmark(model, dummy_input, device, n_runs=N_RUNS):
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    return np.mean(times), np.std(times), np.min(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)

    T = int(cfg.clip_samples / cfg.hop_length) + 1
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input shape: (1, 1, {cfg.n_mels}, {T})")
    print(f"Chunk duration: {cfg.clip_duration_s}s | Realtime budget: {cfg.clip_duration_s * 1000:.0f} ms")
    print("=" * 55)

    results = {}

    # GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        model = UNet(base_channels=cfg.base_channels, depth=cfg.depth).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        dummy = torch.randn(1, 1, cfg.n_mels, T).to(device)
        mean, std, mn = benchmark(model, dummy, device)
        results["GPU"] = mean
        print(f"GPU  | mean: {mean:.1f} ms | std: {std:.1f} ms | min: {mn:.1f} ms")
        del model, dummy
        torch.cuda.empty_cache()
    else:
        print("GPU  | not available")

    # CPU
    device = torch.device("cpu")
    model = UNet(base_channels=cfg.base_channels, depth=cfg.depth).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    dummy = torch.randn(1, 1, cfg.n_mels, T)
    mean, std, mn = benchmark(model, dummy, device, n_runs=10)
    results["CPU"] = mean
    print(f"CPU  | mean: {mean:.1f} ms | std: {std:.1f} ms | min: {mn:.1f} ms")

    print("=" * 55)
    budget_ms = cfg.clip_duration_s * 1000
    for dev, t in results.items():
        ratio = budget_ms / t
        rt = "✓ realtime" if ratio >= 1.0 else "✗ not realtime"
        print(f"{dev}  | {ratio:.1f}x realtime  →  {rt}")

    print()
    print("Not: Gerçek realtime için overlap (50%) ile efektif")
    print(f"     pencere her {cfg.clip_duration_s * 0.5 * 1000:.0f} ms'de bir işlenmeli.")


if __name__ == "__main__":
    main()
