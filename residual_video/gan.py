from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def _sn(module: nn.Module, enabled: bool = True) -> nn.Module:
    return spectral_norm(module) if enabled else module


class ResidualPatchDiscriminator3D(nn.Module):
    """Small 3D PatchGAN discriminator for residual video clips.

    The discriminator judges local space-time patches of residual = video - bg,
    conditioned on the static background and ROI. It intentionally sees residuals
    rather than RGB frames so that the adversarial signal focuses on bubble shape
    and contour detail instead of the easy static background.
    """

    def __init__(
        self,
        base_channels: int = 32,
        num_layers: int = 4,
        use_spectral_norm: bool = True,
        roi_residual_weight: float = 0.5,
        include_edges: bool = False,
    ) -> None:
        super().__init__()
        self.roi_residual_weight = float(roi_residual_weight)
        self.include_edges = bool(include_edges)
        in_channels = 3 + 3 + 1  # residual, background, ROI
        if self.include_edges:
            in_channels += 3  # spatial edge magnitude of residual
        layers: list[nn.Module] = []
        channels = int(base_channels)
        layers.extend(
            [
                _sn(
                    nn.Conv3d(
                        in_channels,
                        channels,
                        kernel_size=(3, 4, 4),
                        stride=(1, 2, 2),
                        padding=(1, 1, 1),
                    ),
                    use_spectral_norm,
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )
        prev = channels
        for layer_index in range(1, int(num_layers)):
            channels = min(int(base_channels) * (2**layer_index), 256)
            stride = (1, 2, 2) if layer_index < int(num_layers) - 1 else (1, 1, 1)
            layers.extend(
                [
                    _sn(
                        nn.Conv3d(
                            prev,
                            channels,
                            kernel_size=(3, 4, 4),
                            stride=stride,
                            padding=(1, 1, 1),
                            bias=False,
                        ),
                        use_spectral_norm,
                    ),
                    nn.GroupNorm(min(8, channels), channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            prev = channels
        layers.append(
            _sn(
                nn.Conv3d(prev, 1, kernel_size=(3, 3, 3), stride=1, padding=1),
                use_spectral_norm,
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        video: torch.Tensor,
        background: torch.Tensor,
        roi: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        # video: B,T,3,H,W, background: B,3,H,W, roi: B,1,H,W
        residual = video - background.unsqueeze(1)
        if self.roi_residual_weight != 0.0:
            roi_t = roi.unsqueeze(1).to(residual.dtype)
            residual = residual * (1.0 + self.roi_residual_weight * roi_t)
        residual = residual.permute(0, 2, 1, 3, 4).contiguous()
        bg = background.unsqueeze(2).expand(-1, -1, video.shape[1], -1, -1)
        roi_map = roi.unsqueeze(2).expand(-1, -1, video.shape[1], -1, -1)
        parts = [residual, bg.to(residual.dtype), roi_map.to(residual.dtype)]
        if self.include_edges:
            parts.append(spatial_sobel_magnitude_3d(residual))
        x = torch.cat(parts, dim=1)
        if not return_features:
            return self.net(x)

        features: list[torch.Tensor] = []
        h = x
        for layer in self.net:
            h = layer(h)
            if isinstance(layer, nn.LeakyReLU):
                features.append(h)
        return h, features


class MultiScaleResidualPatchDiscriminator3D(nn.Module):
    """Multi-scale residual PatchGAN with optional edge channels.

    This version is more useful than simply increasing GAN weight: the full
    scale discriminator sees bubble boundaries, while the half-scale branch
    checks whether the generated plume statistics remain plausible.
    """

    def __init__(
        self,
        scales: tuple[float, ...] = (1.0, 0.5),
        base_channels: int = 32,
        num_layers: int = 4,
        use_spectral_norm: bool = True,
        roi_residual_weight: float = 0.5,
        include_edges: bool = True,
    ) -> None:
        super().__init__()
        self.scales = tuple(float(scale) for scale in scales)
        self.discriminators = nn.ModuleList(
            [
                ResidualPatchDiscriminator3D(
                    base_channels=base_channels,
                    num_layers=num_layers,
                    use_spectral_norm=use_spectral_norm,
                    roi_residual_weight=roi_residual_weight,
                    include_edges=include_edges,
                )
                for _ in self.scales
            ]
        )

    def forward(
        self,
        video: torch.Tensor,
        background: torch.Tensor,
        roi: torch.Tensor,
        return_features: bool = False,
    ) -> list[torch.Tensor] | list[tuple[torch.Tensor, list[torch.Tensor]]]:
        outputs = []
        for scale, discriminator in zip(self.scales, self.discriminators):
            v, b, r = resize_condition_triplet(video, background, roi, scale)
            outputs.append(discriminator(v, b, r, return_features=return_features))
        return outputs


def resize_condition_triplet(
    video: torch.Tensor,
    background: torch.Tensor,
    roi: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if abs(float(scale) - 1.0) < 1e-6:
        return video, background, roi
    height = max(8, int(round(video.shape[-2] * float(scale))))
    width = max(8, int(round(video.shape[-1] * float(scale))))
    bsz, frames, channels, _, _ = video.shape
    v = video.reshape(bsz * frames, channels, video.shape[-2], video.shape[-1])
    v = F.interpolate(v, size=(height, width), mode="bilinear", align_corners=False)
    v = v.reshape(bsz, frames, channels, height, width)
    b = F.interpolate(background, size=(height, width), mode="bilinear", align_corners=False)
    r = F.interpolate(roi, size=(height, width), mode="nearest")
    return v, b, r


def spatial_sobel_magnitude_3d(x: torch.Tensor) -> torch.Tensor:
    # x: B,C,T,H,W. Apply spatial Sobel per channel and frame.
    bsz, channels, frames, height, width = x.shape
    flat = x.permute(0, 2, 1, 3, 4).reshape(bsz * frames * channels, 1, height, width)
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(flat, kx, padding=1)
    gy = F.conv2d(flat, ky, padding=1)
    edge = torch.sqrt(gx * gx + gy * gy + 1e-8)
    return edge.reshape(bsz, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()


def logits_from_output(output: object) -> list[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, tuple):
        return [output[0]]
    if isinstance(output, list):
        logits: list[torch.Tensor] = []
        for item in output:
            logits.extend(logits_from_output(item))
        return logits
    raise TypeError(f"Unsupported discriminator output type: {type(output)!r}")


def features_from_output(output: object) -> list[list[torch.Tensor]]:
    if isinstance(output, tuple):
        return [output[1]]
    if isinstance(output, list):
        groups: list[list[torch.Tensor]] = []
        for item in output:
            groups.extend(features_from_output(item))
        return groups
    return []


def hinge_discriminator_loss(real_logits: object, fake_logits: object) -> torch.Tensor:
    real_list = logits_from_output(real_logits)
    fake_list = logits_from_output(fake_logits)
    losses = [
        F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
        for real, fake in zip(real_list, fake_list)
    ]
    return torch.stack(losses).mean()


def hinge_generator_loss(fake_logits: object) -> torch.Tensor:
    losses = [-fake.mean() for fake in logits_from_output(fake_logits)]
    return torch.stack(losses).mean()


def discriminator_feature_matching_loss(real_output: object, fake_output: object) -> torch.Tensor:
    real_groups = features_from_output(real_output)
    fake_groups = features_from_output(fake_output)
    losses: list[torch.Tensor] = []
    for real_features, fake_features in zip(real_groups, fake_groups):
        for real, fake in zip(real_features, fake_features):
            losses.append(F.l1_loss(fake, real.detach()))
    if not losses:
        logits = logits_from_output(fake_output)
        return logits[0].new_tensor(0.0)
    return torch.stack(losses).mean()


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(requires_grad)
