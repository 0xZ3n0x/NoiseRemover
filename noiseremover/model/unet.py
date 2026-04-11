import torch
import torch.nn as nn
from .blocks import EncoderBlock, DecoderBlock, BottleneckBlock


class UNet(nn.Module):
    """
    U-Net encoder-decoder for noise mask prediction.

    Input:  (B, 1, n_mels, T) — noisy log-mel spectrogram
    Output: (B, 1, n_mels, T) — mask M in [0, 1]
    Clean estimate: noisy * mask
    """

    def __init__(self, base_channels: int = 32, depth: int = 5):
        super().__init__()
        # Channel sizes: [1, 32, 64, 128, 256, 512] for depth=5
        ch = [1] + [base_channels * (2 ** i) for i in range(depth)]

        self.encoders = nn.ModuleList([
            EncoderBlock(ch[i], ch[i + 1]) for i in range(depth)
        ])
        self.bottleneck = BottleneckBlock(ch[depth])

        # Bottleneck outputs ch[depth]*2; decoder[0] takes that as input
        dec_in = [ch[depth] * 2] + [ch[i] for i in range(depth, 0, -1)]
        dec_out = [ch[i] for i in range(depth, 0, -1)]
        self.decoders = nn.ModuleList([
            DecoderBlock(dec_in[i], dec_out[i]) for i in range(depth)
        ])

        self.head = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.head(x)
