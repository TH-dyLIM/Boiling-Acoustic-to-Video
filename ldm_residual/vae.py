from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalGaussianDistribution:
    def __init__(self, moments: torch.Tensor) -> None:
        self.mean, self.logvar = torch.chunk(moments, 2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)

    def sample(self) -> torch.Tensor:
        std = torch.exp(0.5 * self.logvar)
        return self.mean + std * torch.randn_like(std)

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        return 0.5 * (self.mean.pow(2) + self.logvar.exp() - 1.0 - self.logvar)


class ConvBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class ResidualFrameVAE(nn.Module):
    """Small frame-wise VAE trained from scratch on residual pseudo-images.

    This class mimics the tiny subset of the diffusers VAE API used by the LDM
    scripts: ``encode(...).latent_dist.sample/mode`` and ``decode(...).sample``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
        downsample_factor: int = 8,
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if int(downsample_factor) != 8:
            raise ValueError("ResidualFrameVAE currently supports downsample_factor=8 only.")
        self.config = SimpleNamespace(
            latent_channels=int(latent_channels),
            scaling_factor=float(scaling_factor),
            downsample_factor=int(downsample_factor),
            in_channels=int(in_channels),
            base_channels=int(base_channels),
        )
        c = int(base_channels)
        zc = int(latent_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, padding=1),
            ConvBlock2d(c, c),
            nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1),
            ConvBlock2d(c * 2, c * 2),
            nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1),
            ConvBlock2d(c * 4, c * 4),
            nn.Conv2d(c * 4, c * 4, kernel_size=4, stride=2, padding=1),
            ConvBlock2d(c * 4, c * 4),
            nn.GroupNorm(min(8, c * 4), c * 4),
            nn.SiLU(),
            nn.Conv2d(c * 4, zc * 2, kernel_size=3, padding=1),
        )
        self.decoder_in = nn.Conv2d(zc, c * 4, kernel_size=3, padding=1)
        self.decoder_blocks = nn.Sequential(
            ConvBlock2d(c * 4, c * 4),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c * 4, c * 4, kernel_size=3, padding=1),
            ConvBlock2d(c * 4, c * 2),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c * 2, c * 2, kernel_size=3, padding=1),
            ConvBlock2d(c * 2, c),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            ConvBlock2d(c, c),
            nn.GroupNorm(min(8, c), c),
            nn.SiLU(),
            nn.Conv2d(c, in_channels, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> SimpleNamespace:
        dtype = next(self.parameters()).dtype
        moments = self.encoder(x.to(dtype=dtype))
        return SimpleNamespace(latent_dist=DiagonalGaussianDistribution(moments))

    def decode(self, z: torch.Tensor, num_frames: int | None = None) -> SimpleNamespace:
        dtype = next(self.parameters()).dtype
        h = self.decoder_in(z.to(dtype=dtype))
        sample = torch.tanh(self.decoder_blocks(h))
        return SimpleNamespace(sample=sample)


def make_residual_frame_vae(
    latent_channels: int = 4,
    base_channels: int = 64,
    downsample_factor: int = 8,
    scaling_factor: float = 1.0,
) -> ResidualFrameVAE:
    return ResidualFrameVAE(
        in_channels=3,
        latent_channels=latent_channels,
        base_channels=base_channels,
        downsample_factor=downsample_factor,
        scaling_factor=scaling_factor,
    )
