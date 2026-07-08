from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=t.device) / max(1, half - 1)
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class FiLM3d(nn.Module):
    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]
        return x * (1.0 + scale) + shift


class TemporalFiLM3d(nn.Module):
    """Per-frame FiLM modulation from a (B,T,Ca) audio embedding."""

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, audio_seq: torch.Tensor) -> torch.Tensor:
        bsz, channels, frames, height, width = x.shape
        audio_resampled = F.adaptive_avg_pool1d(audio_seq.transpose(1, 2), frames).transpose(1, 2)
        params = self.proj(audio_resampled)
        scale, shift = params.chunk(2, dim=-1)
        scale = scale.transpose(1, 2)[:, :, :, None, None]
        shift = shift.transpose(1, 2)[:, :, :, None, None]
        return x * (1.0 + scale) + shift


class ResBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, audio_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.film = FiLM3d(out_channels, cond_dim)
        self.audio_film = TemporalFiLM3d(out_channels, audio_dim)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout3d(float(dropout))
        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, audio_seq: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.film(h, cond)
        h = self.audio_film(h, audio_seq)
        h = self.dropout(F.silu(self.norm2(h)))
        h = self.conv2(h)
        return h + self.skip(x)


class CrossAttentionAudio(nn.Module):
    """Spatial-temporal tokens attend to audio token sequence."""

    def __init__(self, channels: int, audio_token_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        head_dim = max(1, channels // num_heads)
        self.num_heads = int(num_heads)
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.to_q = nn.Conv3d(channels, self.inner_dim, kernel_size=1)
        self.to_k = nn.Linear(audio_token_dim, self.inner_dim)
        self.to_v = nn.Linear(audio_token_dim, self.inner_dim)
        self.to_out = nn.Conv3d(self.inner_dim, channels, kernel_size=1)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x: torch.Tensor, audio_tokens: torch.Tensor) -> torch.Tensor:
        bsz, channels, frames, height, width = x.shape
        h = self.norm(x)
        q = self.to_q(h)
        q = q.view(bsz, self.num_heads, self.head_dim, frames * height * width).transpose(-1, -2)
        k = self.to_k(audio_tokens).view(bsz, audio_tokens.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(audio_tokens).view(bsz, audio_tokens.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(-1, -2).reshape(bsz, self.inner_dim, frames, height, width)
        return x + self.to_out(attn)


class Down3d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv3d(channels, channels, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Up3d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.ConvTranspose3d(channels, channels, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class STFTEncoder(nn.Module):
    """Encode a log-frequency log-magnitude STFT into per-frame + token features.

    Inputs:
      stft: (B, 1, n_freq_bins, n_time_bins) log-frequency log-magnitude
        STFT computed in the dataloader.

    Returns:
      seq: (B, T_frames, audio_dim)
      tokens: (B, num_tokens, audio_dim)
    """

    def __init__(self, freq_bins: int, audio_dim: int, num_tokens: int) -> None:
        super().__init__()
        self.audio_dim = int(audio_dim)
        self.num_tokens = int(num_tokens)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.SiLU(),
            nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.SiLU(),
            nn.Conv2d(96, self.audio_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.SiLU(),
        )

    def forward(self, stft: torch.Tensor, target_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(stft.float())
        feat = feat.mean(dim=2)
        seq = F.adaptive_avg_pool1d(feat, target_frames).transpose(1, 2)
        tokens = F.adaptive_avg_pool1d(feat, self.num_tokens).transpose(1, 2)
        return seq, tokens


class AudioEncoder(nn.Module):
    """Encode 1 MHz audio into per-frame embedding sequence + token sequence.

    Optionally fuses a 2D STFT branch (log-frequency log-magnitude) so the
    model receives explicit time-frequency information in addition to the
    raw waveform features.

    Returns:
      audio_seq: (B, T_frames, audio_dim) for FiLM modulation.
      audio_tokens: (B, num_tokens, audio_dim) for cross-attention.
    """

    def __init__(
        self,
        audio_dim: int = 64,
        num_tokens: int = 16,
        scalar_feature_dim: int = 6,
        use_stft: bool = False,
        stft_freq_bins: int = 64,
    ) -> None:
        super().__init__()
        self.audio_dim = int(audio_dim)
        self.num_tokens = int(num_tokens)
        self.use_stft = bool(use_stft)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=4, padding=7),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=15, stride=4, padding=7),
            nn.SiLU(),
            nn.Conv1d(64, 96, kernel_size=11, stride=4, padding=5),
            nn.SiLU(),
            nn.Conv1d(96, 128, kernel_size=9, stride=4, padding=4),
            nn.SiLU(),
            nn.Conv1d(128, self.audio_dim, kernel_size=7, stride=2, padding=3),
            nn.SiLU(),
        )
        self.token_proj = nn.Linear(self.audio_dim + scalar_feature_dim, self.audio_dim)
        if self.use_stft:
            self.stft_encoder = STFTEncoder(
                freq_bins=int(stft_freq_bins),
                audio_dim=self.audio_dim,
                num_tokens=self.num_tokens,
            )
            self.stft_seq_gate = nn.Parameter(torch.zeros(1))
            self.stft_tok_gate = nn.Parameter(torch.zeros(1))
        else:
            self.stft_encoder = None

    def forward(
        self,
        audio: torch.Tensor,
        scalar_features: torch.Tensor,
        target_frames: int,
        audio_stft: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(audio.float())
        seq = F.adaptive_avg_pool1d(feat, target_frames).transpose(1, 2)
        token_feat = F.adaptive_avg_pool1d(feat, self.num_tokens).transpose(1, 2)
        scalar = scalar_features.unsqueeze(1).expand(-1, self.num_tokens, -1)
        tokens = self.token_proj(torch.cat([token_feat, scalar.float()], dim=-1))
        if self.use_stft and self.stft_encoder is not None and audio_stft is not None:
            stft_seq, stft_tokens = self.stft_encoder(audio_stft, target_frames)
            seq = seq + self.stft_seq_gate * stft_seq
            tokens = tokens + self.stft_tok_gate * stft_tokens
        return seq, tokens


class FlowResidualUNet(nn.Module):
    """Flow-matching velocity predictor for residual chunks.

    Inputs:
      noisy_residual: (B, T, 3, H, W)  (already in [-residual_scale, +residual_scale])
      time:           (B,)             flow time in [0,1]
      background:     (B, 3, H, W)
      roi:            (B, 1, H, W)
      nucleation_prior: (B, 1, H, W)
      prev_last_frame:  (B, 3, H, W)
      audio:          (B, 1, L)
      scalar_features:(B, 6)
      physics:        (B, 3)
    Returns:
      velocity: (B, T, 3, H, W)
    """

    def __init__(
        self,
        chunk_frames: int = 12,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 4),
        audio_dim: int = 64,
        audio_tokens: int = 16,
        scalar_feature_dim: int = 6,
        physics_dim: int = 3,
        time_dim: int = 128,
        cond_dim: int = 256,
        dropout: float = 0.0,
        use_attention_at: tuple[int, ...] = (2,),
        residual_scale: float = 0.5,
        use_prev_frame: bool = True,
        use_stft: bool = False,
        stft_freq_bins: int = 64,
    ) -> None:
        super().__init__()
        self.chunk_frames = int(chunk_frames)
        self.residual_scale = float(residual_scale)
        self.use_prev_frame = bool(use_prev_frame)
        self.use_stft = bool(use_stft)

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
        self.cond_dim = int(cond_dim)
        self.time_dim = int(time_dim)
        self.audio_dim = int(audio_dim)

        spatial_in = 3 + 1 + 1 + (3 if self.use_prev_frame else 0)
        in_channels = 3 + spatial_in
        cm = list(channel_mult)
        chs = [int(base_channels * m) for m in cm]
        self.in_proj = nn.Conv3d(in_channels, chs[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev_c = chs[0]
        skip_channels: list[int] = []
        for level, c in enumerate(chs):
            block = nn.ModuleList(
                [
                    ResBlock3d(prev_c, c, cond_dim, audio_dim, dropout),
                    ResBlock3d(c, c, cond_dim, audio_dim, dropout),
                ]
            )
            self.down_blocks.append(block)
            attn = CrossAttentionAudio(c, audio_dim) if level in use_attention_at else nn.Identity()
            self.down_attn.append(attn)
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
            block = nn.ModuleList(
                [
                    ResBlock3d(prev_c + skip_channels[level], c, cond_dim, audio_dim, dropout),
                    ResBlock3d(c, c, cond_dim, audio_dim, dropout),
                ]
            )
            self.up_blocks.append(block)
            attn = CrossAttentionAudio(c, audio_dim) if level in use_attention_at else nn.Identity()
            self.up_attn.append(attn)
            prev_c = c

        self.out_norm = nn.GroupNorm(min(8, prev_c), prev_c)
        self.out_conv = nn.Conv3d(prev_c, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _build_input(
        self,
        noisy: torch.Tensor,
        background: torch.Tensor,
        roi: torch.Tensor,
        prior: torch.Tensor,
        prev_last: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, frames, _, height, width = noisy.shape
        noisy_3d = noisy.permute(0, 2, 1, 3, 4)
        bg = background.unsqueeze(2).expand(-1, -1, frames, -1, -1)
        roi_map = roi.unsqueeze(2).expand(-1, -1, frames, -1, -1)
        prior_map = prior.unsqueeze(2).expand(-1, -1, frames, -1, -1)
        cond_maps = [noisy_3d, bg, roi_map, prior_map]
        if self.use_prev_frame:
            if prev_last is None:
                prev_last = background
            prev_map = prev_last.unsqueeze(2).expand(-1, -1, frames, -1, -1)
            cond_maps.append(prev_map)
        return torch.cat(cond_maps, dim=1)

    def forward(
        self,
        noisy_residual: torch.Tensor,
        time: torch.Tensor,
        background: torch.Tensor,
        roi: torch.Tensor,
        nucleation_prior: torch.Tensor,
        prev_last_frame: torch.Tensor | None,
        audio: torch.Tensor,
        scalar_features: torch.Tensor,
        physics: torch.Tensor,
        cond_dropout_mask: torch.Tensor | None = None,
        audio_stft: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, frames, _, _, _ = noisy_residual.shape
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
        time_emb = sinusoidal_embedding(time, self.time_dim)
        time_cond = self.time_mlp(time_emb)
        physics_cond = self.physics_mlp(torch.cat([physics.float(), scalar_features.float()], dim=-1))
        cond = time_cond + physics_cond

        x = self.in_proj(self._build_input(noisy_residual, background, roi, nucleation_prior, prev_last_frame))
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
            if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                x = block(x, cond, audio_seq)
            x = attn(x, audio_tokens) if isinstance(attn, CrossAttentionAudio) else attn(x)

        velocity = self.out_conv(F.silu(self.out_norm(x)))
        return velocity.permute(0, 2, 1, 3, 4).contiguous()
