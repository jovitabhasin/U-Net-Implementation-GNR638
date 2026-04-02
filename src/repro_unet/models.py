from __future__ import annotations

import math

import torch
from torch import nn


def _he_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
        std = math.sqrt(2.0 / fan_in)
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DoubleConvValid(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
        )
        self.apply(_he_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNetOriginal(nn.Module):
    """Original valid-convolution U-Net from the 2015 paper."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        feature_channels: tuple[int, ...] = (64, 128, 256, 512, 1024),
    ) -> None:
        super().__init__()

        enc_in = (in_channels,) + feature_channels[:-1]
        self.encoder_blocks = nn.ModuleList(
            DoubleConvValid(inp, out) for inp, out in zip(enc_in, feature_channels)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        decoder_in = feature_channels[::-1][:-1]
        decoder_out = feature_channels[::-1][1:]
        self.upconvs = nn.ModuleList(
            nn.ConvTranspose2d(inp, out, kernel_size=2, stride=2)
            for inp, out in zip(decoder_in, decoder_out)
        )
        self.decoder_blocks = nn.ModuleList(
            DoubleConvValid(out * 2, out) for out in decoder_out
        )
        self.head = nn.Conv2d(feature_channels[0], num_classes, kernel_size=1)
        _he_init(self.head)

    @staticmethod
    def center_crop(skip: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        _, _, h, w = skip.shape
        target_h, target_w = target_hw
        top = (h - target_h) // 2
        left = (w - target_w) // 2
        return skip[:, :, top : top + target_h, left : left + target_w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []

        for index, block in enumerate(self.encoder_blocks):
            x = block(x)
            if index != len(self.encoder_blocks) - 1:
                skips.append(x)
                x = self.pool(x)

        for upconv, block, skip in zip(self.upconvs, self.decoder_blocks, reversed(skips)):
            x = upconv(x)
            skip = self.center_crop(skip, x.shape[-2:])
            x = torch.cat([skip, x], dim=1)
            x = block(x)

        return self.head(x)
