import torch
from .model import UNet
from .config import Config


def get_device(gpu: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def load_model(cfg: Config, checkpoint_path: str, device: torch.device):
    """Load UNet from checkpoint. Returns (model, checkpoint_dict)."""
    model = UNet(base_channels=cfg.base_channels, depth=cfg.depth).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt
