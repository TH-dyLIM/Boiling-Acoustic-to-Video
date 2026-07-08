from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_residual.model import (
    AudioEncoder,
    CrossAttentionAudio,
    Down3d,
    ResBlock3d,
    Up3d,
    sinusoidal_embedding,
)


class LatentConditionalUNet(nn.Module):
    """3D UNet denoiser for latent residual-video diffusion.

    The model is intentionally close to ``FlowResidualUNet`` so the ablation is
    mostly objective/space: pixel-space flow matching versus latent-space DDPM.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_mult: tuple[int, ...] = (1, 2, 4),
        audio_dim: int = 96,
        audio_tokens: int = 24,
        scalar_feature_dim: int = 6,
        physics_dim: int = 3,
        time_dim: int = 192,
        cond_dim: int = 384,
        dropout: float = 0.05,
        use_attention_at: tuple[int, ...] = (1, 2),
        use_prev_frame: bool = True,
        use_stft: bool = False,
        stft_freq_bins: int = 64,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.use_prev_frame = bool(use_prev_frame)
        self.use_stft = bool(use_stft)
        self.time_dim = int(time_dim)

        self.audio_encoder = AudioEncoder(
            audio_dim=audio_dim,
            num_tokens=audio_tokens,
            scalar_feature_dim=scalar_feature_dim,
            use_stft=self.use_stft,
            stft_freq_bins=int(stft_freq_bins),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.physics_mlp = nn.Sequential(
            nn.Linear(physics_dim + scalar_feature_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        spatial_in = self.latent_channels + 1 + 1 + (self.latent_channels if self.use_prev_frame else 0)
        in_channels = self.latent_channels + spatial_in
        chs = [int(base_channels * m) for m in channel_mult]
        self.in_proj = nn.Conv3d(in_channels, chs[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev_c = chs[0]
        skip_channels: list[int] = []
        for level, c in enumerate(chs):
            blocks = nn.ModuleList(
                [
                    ResBlock3d(prev_c, c, cond_dim, audio_dim, dropout),
                    ResBlock3d(c, c, cond_dim, audio_dim, dropout),
                ]
            )
            self.down_blocks.append(blocks)
            self.down_attn.append(CrossAttentionAudio(c, audio_dim) if level in use_attention_at else nn.Identity())
            skip_channels.append(c)
            self.downs.append(Down3d(c) if level < len(chs) - 1 else nn.Identity())
            prev_c = c

        self.mid_block1 = ResBlock3d(prev_c, prev_c, cond_dim, audio_dim, dropout)
        self.mid_attn = CrossAttentionAudio(prev_c, audio_dim)
        self.mid_block2 = ResBlock3d(prev_c, prev_c, cond_dim, audio_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.ups = nn.ModuleList()
        for level in reversed(range(len(chs))):
            c = chs[level]
            self.ups.append(Up3d(prev_c) if level < len(chs) - 1 else nn.Identity())
            blocks = nn.ModuleList(
                [
                    ResBlock3d(prev_c + skip_channels[level], c, cond_dim, audio_dim, dropout),
                    ResBlock3d(c, c, cond_dim, audio_dim, dropout),
                ]
            )
            self.up_blocks.append(blocks)
            self.up_attn.append(CrossAttentionAudio(c, audio_dim) if level in use_attention_at else nn.Identity())
            prev_c = c

        self.out_norm = nn.GroupNorm(min(8, prev_c), prev_c)
        self.out_conv = nn.Conv3d(prev_c, self.latent_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _build_input(
        self,
        noisy_latents: torch.Tensor,
        background_latents: torch.Tensor,
        roi: torch.Tensor,
        prior: torch.Tensor,
        prev_latents: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, frames, _, _, _ = noisy_latents.shape
        x = noisy_latents.permute(0, 2, 1, 3, 4)
        bg = background_latents.unsqueeze(2).expand(-1, -1, frames, -1, -1)
        roi_map = roi.unsqueeze(2).expand(-1, -1, frames, -1, -1)
        prior_map = prior.unsqueeze(2).expand(-1, -1, frames, -1, -1)
        parts = [x, bg, roi_map, prior_map]
        if self.use_prev_frame:
            if prev_latents is None:
                prev_latents = background_latents
            parts.append(prev_latents.unsqueeze(2).expand(-1, -1, frames, -1, -1))
        return torch.cat(parts, dim=1)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        background_latents: torch.Tensor,
        roi_latent: torch.Tensor,
        prior_latent: torch.Tensor,
        prev_latents: torch.Tensor | None,
        audio: torch.Tensor,
        scalar_features: torch.Tensor,
        physics: torch.Tensor,
        cond_dropout_mask: torch.Tensor | None = None,
        audio_stft: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, frames, _, _, _ = noisy_latents.shape
        audio_seq, audio_tokens = self.audio_encoder(
            audio,
            scalar_features,
            target_frames=frames,
            audio_stft=audio_stft if self.use_stft else None,
        )
        if cond_dropout_mask is not None:
            mask = cond_dropout_mask.float().view(bsz, 1, 1)
            audio_seq = audio_seq * mask
            audio_tokens = audio_tokens * mask

        time_emb = sinusoidal_embedding(timesteps.float(), self.time_dim)
        cond = self.time_mlp(time_emb)
        cond = cond + self.physics_mlp(torch.cat([physics.float(), scalar_features.float()], dim=-1))

        x = self.in_proj(self._build_input(noisy_latents, background_latents, roi_latent, prior_latent, prev_latents))
        skips: list[torch.Tensor] = []
        for blocks, attn, down in zip(self.down_blocks, self.down_attn, self.downs):
            for block in blocks:
                x = block(x, cond, audio_seq)
            x = attn(x, audio_tokens) if isinstance(attn, CrossAttentionAudio) else attn(x)
            skips.append(x)
            x = down(x)

        x = self.mid_block1(x, cond, audio_seq)
        x = self.mid_attn(x, audio_tokens)
        x = self.mid_block2(x, cond, audio_seq)

        for blocks, attn, up in zip(self.up_blocks, self.up_attn, self.ups):
            x = up(x)
            skip = skips.pop()
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                x = block(x, cond, audio_seq)
            x = attn(x, audio_tokens) if isinstance(attn, CrossAttentionAudio) else attn(x)

        out = self.out_conv(F.silu(self.out_norm(x)))
        return out.permute(0, 2, 1, 3, 4).contiguous()
