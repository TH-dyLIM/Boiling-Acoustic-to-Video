from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from flow_residual.model import AudioEncoder, sinusoidal_embedding


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * float(mlp_ratio))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FactorizedVideoTokenBlock(nn.Module):
    """Movie-Gen-style video-token block with separate temporal/spatial attention.

    Full spacetime attention over all tokens is expensive for 128x128 video. This
    block keeps the video-token formulation but factorizes attention into
    temporal attention over T and spatial attention over HxW patch tokens.
    """

    def __init__(self, dim: int, cond_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim))
        self.norm_temporal = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, dropout=float(dropout), batch_first=True)
        self.norm_spatial = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, dropout=float(dropout), batch_first=True)
        self.norm_audio = nn.LayerNorm(dim)
        self.audio_attn = nn.MultiheadAttention(dim, num_heads, dropout=float(dropout), batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, audio_tokens: torch.Tensor) -> torch.Tensor:
        # x: B,T,N,D where N is spatial patch count.
        bsz, frames, spatial_tokens, dim = x.shape
        cond_token = self.cond_proj(cond).view(bsz, 1, 1, dim)
        x = x + cond_token

        y = self.norm_temporal(x)
        y = y.permute(0, 2, 1, 3).reshape(bsz * spatial_tokens, frames, dim)
        y, _ = self.temporal_attn(y, y, y, need_weights=False)
        y = y.reshape(bsz, spatial_tokens, frames, dim).permute(0, 2, 1, 3)
        x = x + y

        y = self.norm_spatial(x).reshape(bsz * frames, spatial_tokens, dim)
        y, _ = self.spatial_attn(y, y, y, need_weights=False)
        y = y.reshape(bsz, frames, spatial_tokens, dim)
        x = x + y

        y = self.norm_audio(x).reshape(bsz, frames * spatial_tokens, dim)
        y, _ = self.audio_attn(y, audio_tokens, audio_tokens, need_weights=False)
        y = y.reshape(bsz, frames, spatial_tokens, dim)
        x = x + y

        return x + self.mlp(self.norm_mlp(x))


class VideoTokenFlowTransformer(nn.Module):
    """Audio-conditioned residual video flow matching with video tokens.

    This model follows the same training/sampling contract as FlowResidualUNet
    but replaces the 3D U-Net with a video-token Transformer. It is intended as
    a compact Movie-Gen-inspired experiment: patchify residual video chunks,
    condition on raw audio/background/ROI/previous frame, and predict the flow
    velocity in residual space.
    """

    def __init__(
        self,
        chunk_frames: int = 8,
        resolution: int = 128,
        patch_size: int = 16,
        hidden_dim: int = 384,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        audio_dim: int = 96,
        audio_tokens: int = 24,
        scalar_feature_dim: int = 6,
        physics_dim: int = 3,
        time_dim: int = 192,
        cond_dim: int = 384,
        dropout: float = 0.0,
        residual_scale: float = 2.0,
        use_prev_frame: bool = True,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.chunk_frames = int(chunk_frames)
        self.resolution = int(resolution)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.use_prev_frame = bool(use_prev_frame)
        self.use_checkpoint = bool(use_checkpoint)
        if self.resolution % self.patch_size != 0:
            raise ValueError(f"resolution {self.resolution} must be divisible by patch_size {self.patch_size}")

        self.grid = self.resolution // self.patch_size
        self.num_spatial_tokens = self.grid * self.grid
        spatial_in = 3 + 1 + 1 + (3 if self.use_prev_frame else 0)
        in_channels = 3 + spatial_in
        self.patch_embed = nn.Conv3d(
            in_channels,
            self.hidden_dim,
            kernel_size=(1, self.patch_size, self.patch_size),
            stride=(1, self.patch_size, self.patch_size),
        )
        self.temporal_pos = nn.Parameter(torch.zeros(1, self.chunk_frames, 1, self.hidden_dim))
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, self.num_spatial_tokens, self.hidden_dim))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)

        self.audio_encoder = AudioEncoder(
            audio_dim=int(audio_dim),
            num_tokens=int(audio_tokens),
            scalar_feature_dim=int(scalar_feature_dim),
        )
        self.audio_proj = nn.Linear(int(audio_dim), self.hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(int(time_dim), int(cond_dim)),
            nn.SiLU(),
            nn.Linear(int(cond_dim), int(cond_dim)),
        )
        self.physics_mlp = nn.Sequential(
            nn.Linear(int(physics_dim) + int(scalar_feature_dim), int(cond_dim)),
            nn.SiLU(),
            nn.Linear(int(cond_dim), int(cond_dim)),
        )
        self.time_dim = int(time_dim)
        self.blocks = nn.ModuleList(
            [
                FactorizedVideoTokenBlock(
                    self.hidden_dim,
                    int(cond_dim),
                    int(num_heads),
                    float(mlp_ratio),
                    float(dropout),
                )
                for _ in range(int(depth))
            ]
        )
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, 3 * self.patch_size * self.patch_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _build_input(
        self,
        noisy: torch.Tensor,
        background: torch.Tensor,
        roi: torch.Tensor,
        prior: torch.Tensor,
        prev_last: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, frames, _, _, _ = noisy.shape
        noisy_3d = noisy.permute(0, 2, 1, 3, 4)
        bg = background.unsqueeze(2).expand(-1, -1, frames, -1, -1)
        roi_map = roi.unsqueeze(2).expand(-1, -1, frames, -1, -1)
        prior_map = prior.unsqueeze(2).expand(-1, -1, frames, -1, -1)
        cond_maps = [noisy_3d, bg, roi_map, prior_map]
        if self.use_prev_frame:
            if prev_last is None:
                prev_last = background
            cond_maps.append(prev_last.unsqueeze(2).expand(-1, -1, frames, -1, -1))
        return torch.cat(cond_maps, dim=1)

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, frames, spatial_tokens, channels = tokens.shape
        if spatial_tokens != self.num_spatial_tokens:
            raise ValueError(f"Expected {self.num_spatial_tokens} spatial tokens, got {spatial_tokens}")
        patches = self.out_proj(self.out_norm(tokens))
        p = self.patch_size
        patches = patches.view(bsz, frames, self.grid, self.grid, 3, p, p)
        video = patches.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            bsz, frames, 3, self.resolution, self.resolution
        )
        return video.contiguous()

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
        del audio_stft
        bsz, frames, _, _, _ = noisy_residual.shape
        if frames != self.chunk_frames:
            raise ValueError(f"Expected {self.chunk_frames} frames, got {frames}")

        audio_seq, audio_tokens = self.audio_encoder(audio, scalar_features, target_frames=frames)
        del audio_seq
        audio_tokens = self.audio_proj(audio_tokens)
        if cond_dropout_mask is not None:
            mask = cond_dropout_mask.float().view(bsz, 1, 1)
            audio_tokens = audio_tokens * mask

        time_cond = self.time_mlp(sinusoidal_embedding(time, self.time_dim))
        physics_cond = self.physics_mlp(torch.cat([physics.float(), scalar_features.float()], dim=-1))
        cond = time_cond + physics_cond

        x = self.patch_embed(
            self._build_input(noisy_residual, background, roi, nucleation_prior, prev_last_frame)
        )
        # B,D,T,H',W' -> B,T,N,D
        x = x.permute(0, 2, 3, 4, 1).reshape(bsz, frames, self.num_spatial_tokens, self.hidden_dim)
        x = x + self.temporal_pos[:, :frames] + self.spatial_pos
        for block in self.blocks:
            if self.use_checkpoint and self.training and x.requires_grad:
                x = checkpoint(block, x, cond, audio_tokens, use_reentrant=False)
            else:
                x = block(x, cond, audio_tokens)
        return self._unpatchify(x)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class AdaLNZeroVideoTokenBlock(nn.Module):
    """DiT-style factorized video-token block.

    The block keeps the same factorized temporal/spatial/audio attention layout
    as ``VideoTokenFlowTransformer`` but uses adaptive LayerNorm-Zero gates from
    the global flow-time/condition vector. This is closer to DiT practice than
    additive condition tokens and usually gives more stable flow training when
    starting from randomly initialized video-token Transformers.
    """

    def __init__(self, dim: int, cond_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm_temporal = nn.LayerNorm(dim, elementwise_affine=False)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, dropout=float(dropout), batch_first=True)
        self.norm_spatial = nn.LayerNorm(dim, elementwise_affine=False)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, dropout=float(dropout), batch_first=True)
        self.norm_audio = nn.LayerNorm(dim, elementwise_affine=False)
        self.audio_attn = nn.MultiheadAttention(dim, num_heads, dropout=float(dropout), batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 12 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, audio_tokens: torch.Tensor) -> torch.Tensor:
        bsz, frames, spatial_tokens, dim = x.shape
        chunks = self.ada(cond).chunk(12, dim=-1)
        (
            shift_t,
            scale_t,
            gate_t,
            shift_s,
            scale_s,
            gate_s,
            shift_a,
            scale_a,
            gate_a,
            shift_m,
            scale_m,
            gate_m,
        ) = [c.view(bsz, 1, 1, dim) for c in chunks]

        y = _modulate(self.norm_temporal(x), shift_t, scale_t)
        y = y.permute(0, 2, 1, 3).reshape(bsz * spatial_tokens, frames, dim)
        y, _ = self.temporal_attn(y, y, y, need_weights=False)
        y = y.reshape(bsz, spatial_tokens, frames, dim).permute(0, 2, 1, 3)
        x = x + gate_t * y

        y = _modulate(self.norm_spatial(x), shift_s, scale_s)
        y = y.reshape(bsz * frames, spatial_tokens, dim)
        y, _ = self.spatial_attn(y, y, y, need_weights=False)
        y = y.reshape(bsz, frames, spatial_tokens, dim)
        x = x + gate_s * y

        y = _modulate(self.norm_audio(x), shift_a, scale_a)
        y = y.reshape(bsz, frames * spatial_tokens, dim)
        y, _ = self.audio_attn(y, audio_tokens, audio_tokens, need_weights=False)
        y = y.reshape(bsz, frames, spatial_tokens, dim)
        x = x + gate_a * y

        y = _modulate(self.norm_mlp(x), shift_m, scale_m)
        return x + gate_m * self.mlp(y)


class DiTVideoTokenFlowTransformer(nn.Module):
    """DiT-style audio-conditioned residual video flow matching backbone.

    This is intended for the c128-rawamp comparison: it preserves the same
    flow-matching target, residual pixel space, background/ROI/no-prior
    conditioning, and raw-amplitude audio path as the 3D U-Net model, while
    replacing the backbone with a patch-token Transformer.
    """

    def __init__(
        self,
        chunk_frames: int = 8,
        resolution: int = 128,
        patch_size: int = 8,
        hidden_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        audio_dim: int = 128,
        audio_tokens: int = 32,
        scalar_feature_dim: int = 6,
        physics_dim: int = 3,
        time_dim: int = 256,
        cond_dim: int = 512,
        dropout: float = 0.0,
        residual_scale: float = 2.0,
        use_prev_frame: bool = True,
        use_checkpoint: bool = False,
        use_refine_head: bool = False,
    ) -> None:
        super().__init__()
        self.chunk_frames = int(chunk_frames)
        self.resolution = int(resolution)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.use_prev_frame = bool(use_prev_frame)
        self.use_checkpoint = bool(use_checkpoint)
        self.use_refine_head = bool(use_refine_head)
        if self.resolution % self.patch_size != 0:
            raise ValueError(f"resolution {self.resolution} must be divisible by patch_size {self.patch_size}")

        self.grid = self.resolution // self.patch_size
        self.num_spatial_tokens = self.grid * self.grid
        spatial_in = 3 + 1 + 1 + (3 if self.use_prev_frame else 0)
        in_channels = 3 + spatial_in
        self.patch_embed = nn.Conv3d(
            in_channels,
            self.hidden_dim,
            kernel_size=(1, self.patch_size, self.patch_size),
            stride=(1, self.patch_size, self.patch_size),
        )
        self.temporal_pos = nn.Parameter(torch.zeros(1, self.chunk_frames, 1, self.hidden_dim))
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, self.num_spatial_tokens, self.hidden_dim))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)

        self.audio_encoder = AudioEncoder(
            audio_dim=int(audio_dim),
            num_tokens=int(audio_tokens),
            scalar_feature_dim=int(scalar_feature_dim),
        )
        self.audio_proj = nn.Linear(int(audio_dim), self.hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(int(time_dim), int(cond_dim)),
            nn.SiLU(),
            nn.Linear(int(cond_dim), int(cond_dim)),
        )
        self.physics_mlp = nn.Sequential(
            nn.Linear(int(physics_dim) + int(scalar_feature_dim), int(cond_dim)),
            nn.SiLU(),
            nn.Linear(int(cond_dim), int(cond_dim)),
        )
        self.time_dim = int(time_dim)
        self.blocks = nn.ModuleList(
            [
                AdaLNZeroVideoTokenBlock(
                    self.hidden_dim,
                    int(cond_dim),
                    int(num_heads),
                    float(mlp_ratio),
                    float(dropout),
                )
                for _ in range(int(depth))
            ]
        )
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(int(cond_dim), 2 * self.hidden_dim))
        self.out_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
        self.out_proj = nn.Linear(self.hidden_dim, 3 * self.patch_size * self.patch_size)
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        if self.use_refine_head:
            # Pixel-space refinement that smooths DiT patch seams. The two 3x3
            # spatial convs (depthwise + pointwise) act on the unpatchified
            # 3-channel velocity volume. The pointwise layer is zero-init so the
            # block starts as identity and only learns to mix neighbouring
            # patches once it sees a gradient signal.
            self.refine_dw = nn.Conv3d(3, 3, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=3)
            self.refine_pw = nn.Conv3d(3, 3, kernel_size=1)
            nn.init.zeros_(self.refine_pw.weight)
            nn.init.zeros_(self.refine_pw.bias)

    def _build_input(
        self,
        noisy: torch.Tensor,
        background: torch.Tensor,
        roi: torch.Tensor,
        prior: torch.Tensor,
        prev_last: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, frames, _, _, _ = noisy.shape
        noisy_3d = noisy.permute(0, 2, 1, 3, 4)
        bg = background.unsqueeze(2).expand(-1, -1, frames, -1, -1)
        roi_map = roi.unsqueeze(2).expand(-1, -1, frames, -1, -1)
        prior_map = prior.unsqueeze(2).expand(-1, -1, frames, -1, -1)
        cond_maps = [noisy_3d, bg, roi_map, prior_map]
        if self.use_prev_frame:
            if prev_last is None:
                prev_last = background
            cond_maps.append(prev_last.unsqueeze(2).expand(-1, -1, frames, -1, -1))
        return torch.cat(cond_maps, dim=1)

    def _unpatchify(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        bsz, frames, spatial_tokens, dim = tokens.shape
        if spatial_tokens != self.num_spatial_tokens:
            raise ValueError(f"Expected {self.num_spatial_tokens} spatial tokens, got {spatial_tokens}")
        shift, scale = self.final_ada(cond).chunk(2, dim=-1)
        tokens = _modulate(self.out_norm(tokens), shift.view(bsz, 1, 1, dim), scale.view(bsz, 1, 1, dim))
        patches = self.out_proj(tokens)
        p = self.patch_size
        patches = patches.view(bsz, frames, self.grid, self.grid, 3, p, p)
        video = patches.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            bsz, frames, 3, self.resolution, self.resolution
        ).contiguous()
        if self.use_refine_head:
            # B,T,C,H,W -> B,C,T,H,W for Conv3d, then back.
            v3d = video.permute(0, 2, 1, 3, 4)
            delta = self.refine_pw(F.silu(self.refine_dw(v3d)))
            v3d = v3d + delta
            video = v3d.permute(0, 2, 1, 3, 4).contiguous()
        return video

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
        del audio_stft
        bsz, frames, _, _, _ = noisy_residual.shape
        if frames != self.chunk_frames:
            raise ValueError(f"Expected {self.chunk_frames} frames, got {frames}")

        audio_seq, audio_tokens = self.audio_encoder(audio, scalar_features, target_frames=frames)
        del audio_seq
        audio_tokens = self.audio_proj(audio_tokens)
        if cond_dropout_mask is not None:
            mask = cond_dropout_mask.float().view(bsz, 1, 1)
            audio_tokens = audio_tokens * mask

        time_cond = self.time_mlp(sinusoidal_embedding(time, self.time_dim))
        physics_cond = self.physics_mlp(torch.cat([physics.float(), scalar_features.float()], dim=-1))
        cond = time_cond + physics_cond

        x = self.patch_embed(
            self._build_input(noisy_residual, background, roi, nucleation_prior, prev_last_frame)
        )
        x = x.permute(0, 2, 3, 4, 1).reshape(bsz, frames, self.num_spatial_tokens, self.hidden_dim)
        x = x + self.temporal_pos[:, :frames] + self.spatial_pos
        for block in self.blocks:
            if self.use_checkpoint and self.training and x.requires_grad:
                x = checkpoint(block, x, cond, audio_tokens, use_reentrant=False)
            else:
                x = block(x, cond, audio_tokens)
        return self._unpatchify(x, cond)
