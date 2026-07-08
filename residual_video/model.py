from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Dropout3d(float(dropout)),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AudioResidualUNet(nn.Module):
    """Predict video residual frames over a static background.

    The model is deterministic by design: the same bg/roi/audio input produces
    the same residual movie. This matches the scientific-use requirement better
    than stochastic SVD sampling for the current dataset scale.
    """

    def __init__(
        self,
        num_frames: int = 8,
        audio_channels: int = 32,
        base_channels: int = 32,
        residual_scale: float = 1.0,
        dropout: float = 0.0,
        use_mask_head: bool = False,
        mask_bias_init: float = 0.0,
        physics_dim: int = 3,
        physics_channels: int = 8,
        audio_feature_dim: int = 6,
        audio_feature_channels: int = 8,
        use_previous_frame_condition: bool = False,
    ) -> None:
        super().__init__()
        self.num_frames = int(num_frames)
        self.audio_channels = int(audio_channels)
        self.residual_scale = float(residual_scale)
        self.use_mask_head = bool(use_mask_head)
        self.physics_dim = int(physics_dim)
        self.physics_channels = int(physics_channels)
        self.audio_feature_dim = int(audio_feature_dim)
        self.audio_feature_channels = int(audio_feature_channels)
        self.use_previous_frame_condition = bool(use_previous_frame_condition)

        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=4, padding=4),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=4, padding=4),
            nn.SiLU(),
            nn.Conv1d(64, self.audio_channels, kernel_size=9, stride=4, padding=4),
            nn.SiLU(),
        )

        if self.physics_dim > 0 and self.physics_channels > 0:
            self.physics_encoder = nn.Sequential(
                nn.Linear(self.physics_dim, self.physics_channels),
                nn.SiLU(),
                nn.Linear(self.physics_channels, self.physics_channels),
                nn.SiLU(),
            )
        else:
            self.physics_encoder = None
        if self.audio_feature_dim > 0 and self.audio_feature_channels > 0:
            self.audio_feature_encoder = nn.Sequential(
                nn.Linear(self.audio_feature_dim, self.audio_feature_channels),
                nn.SiLU(),
                nn.Linear(self.audio_feature_channels, self.audio_feature_channels),
                nn.SiLU(),
            )
        else:
            self.audio_feature_encoder = None

        in_channels = 3 + 1 + self.audio_channels + 1
        if self.use_previous_frame_condition:
            in_channels += 3
        if self.physics_encoder is not None:
            in_channels += self.physics_channels
        if self.audio_feature_encoder is not None:
            in_channels += self.audio_feature_channels
        c = int(base_channels)
        self.enc1 = ConvBlock3d(in_channels, c, dropout)
        self.down1 = nn.Conv3d(c, c * 2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
        self.enc2 = ConvBlock3d(c * 2, c * 2, dropout)
        self.down2 = nn.Conv3d(c * 2, c * 4, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
        self.enc3 = ConvBlock3d(c * 4, c * 4, dropout)
        self.mid = ConvBlock3d(c * 4, c * 4, dropout)

        self.up2 = nn.Conv3d(c * 4, c * 2, kernel_size=1)
        self.dec2 = ConvBlock3d(c * 4, c * 2, dropout)
        self.up1 = nn.Conv3d(c * 2, c, kernel_size=1)
        self.dec1 = ConvBlock3d(c * 2, c, dropout)
        self.out = nn.Conv3d(c, 3, kernel_size=3, padding=1)
        if self.use_mask_head:
            self.mask_out = nn.Conv3d(c, 1, kernel_size=3, padding=1)
        else:
            self.mask_out = None

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        if self.mask_out is not None:
            nn.init.zeros_(self.mask_out.weight)
            nn.init.constant_(self.mask_out.bias, float(mask_bias_init))

    def forward(
        self,
        background: torch.Tensor,
        roi: torch.Tensor,
        audio: torch.Tensor,
        physics: torch.Tensor | None = None,
        audio_features: torch.Tensor | None = None,
        previous_frame: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        bsz, _, height, width = background.shape
        frames = self.num_frames

        audio_feat = self.audio_encoder(audio.float())
        audio_feat = F.adaptive_avg_pool1d(audio_feat, frames).transpose(1, 2)
        audio_map = audio_feat.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        audio_map = audio_map.expand(bsz, self.audio_channels, frames, height, width)

        bg = background.unsqueeze(2).expand(bsz, 3, frames, height, width)
        roi_map = roi.unsqueeze(2).expand(bsz, 1, frames, height, width)
        t = torch.linspace(-1.0, 1.0, frames, device=background.device, dtype=background.dtype)
        t = t.view(1, 1, frames, 1, 1).expand(bsz, 1, frames, height, width)

        cond_maps = [bg, roi_map, audio_map.to(background.dtype), t]
        if self.use_previous_frame_condition:
            if previous_frame is None:
                previous_frame = background
            prev_map = previous_frame.unsqueeze(2).expand(bsz, 3, frames, height, width)
            cond_maps.append(prev_map.to(background.dtype))
        if self.physics_encoder is not None:
            if physics is None:
                physics = background.new_zeros((bsz, self.physics_dim))
            physics_feat = self.physics_encoder(physics.float()).to(background.dtype)
            physics_map = physics_feat.view(bsz, self.physics_channels, 1, 1, 1).expand(
                bsz, self.physics_channels, frames, height, width
            )
            cond_maps.append(physics_map)
        if self.audio_feature_encoder is not None:
            if audio_features is None:
                audio_features = background.new_zeros((bsz, self.audio_feature_dim))
            audio_scalar_feat = self.audio_feature_encoder(audio_features.float()).to(background.dtype)
            audio_scalar_map = audio_scalar_feat.view(bsz, self.audio_feature_channels, 1, 1, 1).expand(
                bsz, self.audio_feature_channels, frames, height, width
            )
            cond_maps.append(audio_scalar_map)

        x = torch.cat(cond_maps, dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        mid = self.mid(e3)

        y = F.interpolate(mid, size=e2.shape[-3:], mode="trilinear", align_corners=False)
        y = self.up2(y)
        y = self.dec2(torch.cat([y, e2], dim=1))
        y = F.interpolate(y, size=e1.shape[-3:], mode="trilinear", align_corners=False)
        y = self.up1(y)
        y = self.dec1(torch.cat([y, e1], dim=1))
        residual = torch.tanh(self.out(y)) * self.residual_scale
        residual = residual.permute(0, 2, 1, 3, 4).contiguous()
        if self.mask_out is None:
            return residual

        mask_logits = self.mask_out(y).permute(0, 2, 1, 3, 4).contiguous()
        mask = torch.sigmoid(mask_logits)
        return {
            "residual": residual,
            "mask_logits": mask_logits,
            "mask": mask,
            "effective_residual": residual * mask,
        }
