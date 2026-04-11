from dataclasses import dataclass, field
from typing import List
import yaml


@dataclass
class Config:
    # Audio
    sample_rate: int = 16000
    clip_duration_s: float = 4.0
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float = 8000.0

    # Dataset
    snr_levels_db: List[int] = field(default_factory=lambda: [-5, 0, 5, 10, 15])
    val_split: float = 0.05
    test_split: float = 0.05
    dataset_root: str = "dataset"
    processed_root: str = "dataset/processed"
    max_samples: int = None  # None = use all; set to limit dataset size for benchmarking

    # Model
    base_channels: int = 32
    depth: int = 5

    # Training
    batch_size: int = 16
    num_workers: int = 4
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stopping_patience: int = 10

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"

    @property
    def clip_samples(self) -> int:
        return int(self.sample_rate * self.clip_duration_s)


def load_config(yaml_path: str) -> Config:
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}
    cfg = Config()
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
