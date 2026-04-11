import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralLoss(nn.Module):
    """
    MSE + weighted L1 on log-mel spectrograms.
    L1 term prevents over-smoothed outputs and preserves sharp formant peaks.
    """

    def __init__(self, l1_weight: float = 0.1):
        super().__init__()
        self.l1_weight = l1_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        l1 = F.l1_loss(pred, target)
        return mse + self.l1_weight * l1
