from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioROIPhysicsProjector(nn.Module):
    """Project audio, ROI, and scalar physics into SVD cross-attention tokens.

    This is the first practical stage of the requested Audio ControlNet/IP-Adapter
    idea. It keeps the SVD backbone frozen except LoRA layers, and feeds compact
    condition tokens next to the CLIP image token.
    """

    def __init__(
        self,
        cross_attention_dim: int = 1024,
        num_audio_tokens: int = 8,
        num_roi_tokens: int = 4,
        num_prior_tokens: int = 0,
        physics_dim: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        audio_condition_mode: str = "waveform",
        audio_stft_n_fft: int = 1024,
        audio_stft_hop_length: int = 256,
        audio_stft_freq_bins: int = 128,
        audio_stft_time_bins: int = 128,
    ):
        super().__init__()
        self.cross_attention_dim = int(cross_attention_dim)
        self.num_audio_tokens = int(num_audio_tokens)
        self.num_roi_tokens = int(num_roi_tokens)
        self.num_prior_tokens = int(num_prior_tokens)
        self.audio_condition_mode = str(audio_condition_mode).lower()
        if self.audio_condition_mode not in {"waveform", "stft_image"}:
            raise ValueError(f"audio_condition_mode must be waveform or stft_image, got {audio_condition_mode!r}")
        self.audio_stft_n_fft = int(audio_stft_n_fft)
        self.audio_stft_hop_length = int(audio_stft_hop_length)
        self.audio_stft_freq_bins = int(audio_stft_freq_bins)
        self.audio_stft_time_bins = int(audio_stft_time_bins)

        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, stride=4, padding=4),
            nn.SiLU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=4, padding=4),
            nn.SiLU(),
            nn.Conv1d(128, hidden_dim, kernel_size=9, stride=4, padding=4),
            nn.SiLU(),
        )
        self.audio_stft_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.cross_attention_dim),
        )

        self.roi_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.roi_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.cross_attention_dim),
        )
        if self.num_prior_tokens > 0:
            self.prior_encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
            )
            self.prior_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, self.cross_attention_dim),
            )
            self.latent_prior_encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, 4, kernel_size=3, padding=1),
            )
        else:
            self.prior_encoder = None
            self.prior_proj = None
            self.latent_prior_encoder = None

        self.physics_proj = nn.Sequential(
            nn.Linear(physics_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.cross_attention_dim),
        )
        self.latent_roi_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 4, kernel_size=3, padding=1),
        )
        self.latent_audio_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4),
        )
        self.latent_physics_proj = nn.Sequential(
            nn.Linear(physics_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.type_embed = nn.Parameter(
            torch.zeros(
                1,
                1 + self.num_audio_tokens + self.num_roi_tokens + self.num_prior_tokens,
                self.cross_attention_dim,
            )
        )
        self.dropout = nn.Dropout(float(dropout))

        self._init_weights()
        nn.init.zeros_(self.latent_roi_encoder[-1].weight)
        nn.init.zeros_(self.latent_roi_encoder[-1].bias)
        nn.init.zeros_(self.latent_audio_proj[-1].weight)
        nn.init.zeros_(self.latent_audio_proj[-1].bias)
        nn.init.zeros_(self.latent_physics_proj[-1].weight)
        nn.init.zeros_(self.latent_physics_proj[-1].bias)
        if self.latent_prior_encoder is not None:
            nn.init.zeros_(self.latent_prior_encoder[-1].weight)
            nn.init.zeros_(self.latent_prior_encoder[-1].bias)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _stft_image(self, audio: torch.Tensor) -> torch.Tensor:
        y = audio.float().view(audio.shape[0], -1)
        n_fft = max(16, min(int(self.audio_stft_n_fft), int(y.shape[-1])))
        hop = max(1, min(int(self.audio_stft_hop_length), n_fft))
        window = torch.hann_window(n_fft, device=y.device, dtype=torch.float32)
        spec = torch.stft(
            y,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        ).abs()
        spec = torch.log1p(spec)
        spec = spec - spec.amin(dim=(-2, -1), keepdim=True)
        spec = spec / spec.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        spec = spec.unsqueeze(1)
        return F.interpolate(
            spec,
            size=(int(self.audio_stft_freq_bins), int(self.audio_stft_time_bins)),
            mode="bilinear",
            align_corners=False,
        )

    def _audio_feature_map(self, audio: torch.Tensor) -> torch.Tensor:
        if self.audio_condition_mode == "stft_image":
            return self.audio_stft_encoder(self._stft_image(audio))
        return self.audio_encoder(audio.float())

    def _audio_tokens(self, audio: torch.Tensor) -> torch.Tensor:
        feat = self._audio_feature_map(audio)
        if feat.ndim == 4:
            rows = max(1, int(self.num_audio_tokens**0.5))
            cols = max(1, int((self.num_audio_tokens + rows - 1) // rows))
            tokens = F.adaptive_avg_pool2d(feat, (rows, cols)).flatten(2).transpose(1, 2)
            if tokens.shape[1] < self.num_audio_tokens:
                pad = tokens[:, -1:].repeat(1, self.num_audio_tokens - tokens.shape[1], 1)
                tokens = torch.cat([tokens, pad], dim=1)
            return self.audio_proj(tokens[:, : self.num_audio_tokens])
        tokens = F.adaptive_avg_pool1d(feat, self.num_audio_tokens).transpose(1, 2)
        return self.audio_proj(tokens)

    def _audio_frame_features(self, audio: torch.Tensor, frames: int) -> torch.Tensor:
        feat = self._audio_feature_map(audio)
        if feat.ndim == 4:
            feat = feat.mean(dim=2)
        feat = F.adaptive_avg_pool1d(feat, int(frames)).transpose(1, 2)
        return feat

    def forward(
        self,
        audio: torch.Tensor,
        roi: torch.Tensor,
        physics: torch.Tensor,
        nucleation_prior: torch.Tensor | None = None,
        drop_audio: bool = False,
        drop_roi: bool = False,
        drop_physics: bool = False,
        drop_prior: bool = False,
    ) -> torch.Tensor:
        if drop_roi:
            roi = torch.zeros_like(roi)
        if drop_physics:
            physics = torch.zeros_like(physics)
        if nucleation_prior is None:
            nucleation_prior = roi.new_zeros((roi.shape[0], 1, roi.shape[-2], roi.shape[-1]))
        if drop_prior:
            nucleation_prior = torch.zeros_like(nucleation_prior)

        bsz = audio.shape[0]
        if drop_audio:
            a = audio.new_zeros((bsz, self.num_audio_tokens, self.cross_attention_dim), dtype=torch.float32)
        else:
            a = self._audio_tokens(audio)

        r = self.roi_encoder(roi.float())
        side = max(1, int(self.num_roi_tokens**0.5))
        r = F.adaptive_avg_pool2d(r, (side, side)).flatten(2).transpose(1, 2)
        if r.shape[1] < self.num_roi_tokens:
            pad = r[:, -1:].repeat(1, self.num_roi_tokens - r.shape[1], 1)
            r = torch.cat([r, pad], dim=1)
        r = r[:, : self.num_roi_tokens]
        r = self.roi_proj(r)

        p = self.physics_proj(physics.float()).unsqueeze(1)
        token_parts = [p, a, r]
        if self.prior_encoder is not None and self.prior_proj is not None:
            q = self.prior_encoder(nucleation_prior.float())
            side = max(1, int(self.num_prior_tokens**0.5))
            q = F.adaptive_avg_pool2d(q, (side, side)).flatten(2).transpose(1, 2)
            if q.shape[1] < self.num_prior_tokens:
                pad = q[:, -1:].repeat(1, self.num_prior_tokens - q.shape[1], 1)
                q = torch.cat([q, pad], dim=1)
            q = q[:, : self.num_prior_tokens]
            q = self.prior_proj(q)
            token_parts.append(q)
        tokens = torch.cat(token_parts, dim=1)
        tokens = tokens + self.type_embed[:, : tokens.shape[1]].to(tokens.dtype)
        return self.dropout(tokens)

    def latent_residual(
        self,
        audio: torch.Tensor,
        roi: torch.Tensor,
        physics: torch.Tensor,
        frames: int,
        height: int,
        width: int,
        nucleation_prior: torch.Tensor | None = None,
        drop_audio: bool = False,
        drop_roi: bool = False,
        drop_physics: bool = False,
        drop_prior: bool = False,
    ) -> torch.Tensor:
        bsz = audio.shape[0]
        if drop_roi:
            roi = torch.zeros_like(roi)
        if drop_physics:
            physics = torch.zeros_like(physics)
        if nucleation_prior is None:
            nucleation_prior = roi.new_zeros((roi.shape[0], 1, roi.shape[-2], roi.shape[-1]))
        if drop_prior:
            nucleation_prior = torch.zeros_like(nucleation_prior)

        if drop_audio:
            audio_bias = audio.new_zeros((bsz, int(frames), 4, 1, 1), dtype=torch.float32)
        else:
            audio_feat = self._audio_frame_features(audio, int(frames))
            audio_bias = self.latent_audio_proj(audio_feat).view(bsz, int(frames), 4, 1, 1)

        roi_latent = self.latent_roi_encoder(roi.float())
        roi_latent = F.interpolate(roi_latent, size=(int(height), int(width)), mode="bilinear", align_corners=False)
        if self.latent_prior_encoder is not None:
            prior_latent = self.latent_prior_encoder(nucleation_prior.float())
            prior_latent = F.interpolate(prior_latent, size=(int(height), int(width)), mode="bilinear", align_corners=False)
        else:
            prior_latent = torch.zeros_like(roi_latent)
        physics_bias = self.latent_physics_proj(physics.float()).view(bsz, 1, 4, 1, 1)
        return roi_latent.unsqueeze(1) + prior_latent.unsqueeze(1) + audio_bias + physics_bias
