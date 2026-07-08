"""Audio feature extraction for AC-RVFM.

This module exposes:

- LogFreqSTFT: log-frequency-binned STFT magnitude for a single-channel
  acoustic chunk. Computed on CPU in the dataloader and passed to the model
  as a 2D conditioning tensor of shape (1, n_freq_bins, n_time_bins).

The STFT preserves absolute amplitude information (no per-chunk peak
normalization), which is essential for boiling regime / heat-flux inference
from external pool-boiling acoustics (Lim & Bang, IJHMT 2024).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class LogFreqSTFT:
    """Compute a log-frequency-binned log-magnitude STFT.

    Linear-frequency STFT bins are mapped onto ``n_freq_bins`` log-spaced
    bins between ``fmin`` and ``fmax``. The final tensor is
    ``log1p(magnitude)`` so absolute amplitude is preserved (no per-chunk
    normalization) and the response is smooth at zero.
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 2048,
        hop_length: int = 1024,
        n_freq_bins: int = 64,
        fmin: float = 100.0,
        fmax: float | None = None,
        log_compress: bool = True,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_freq_bins = int(n_freq_bins)
        self.fmin = float(fmin)
        self.fmax = float(fmax) if fmax is not None else float(sample_rate) / 2.0
        self.log_compress = bool(log_compress)

        self.window = torch.hann_window(self.n_fft, dtype=torch.float32)
        n_stft = self.n_fft // 2 + 1
        freqs = torch.linspace(0.0, float(self.sample_rate) / 2.0, n_stft)
        eps = 1e-6
        log_lo = math.log(max(self.fmin, eps))
        log_hi = math.log(max(self.fmax, self.fmin * 2.0))
        edges = torch.exp(torch.linspace(log_lo, log_hi, self.n_freq_bins + 1))

        weights = torch.zeros(self.n_freq_bins, n_stft, dtype=torch.float32)
        for bin_idx in range(self.n_freq_bins):
            lo = float(edges[bin_idx].item())
            hi = float(edges[bin_idx + 1].item())
            mask = (freqs >= lo) & (freqs < hi)
            count = int(mask.sum().item())
            if count > 0:
                weights[bin_idx, mask] = 1.0 / float(count)
            else:
                nearest = int(torch.argmin((freqs - 0.5 * (lo + hi)).abs()).item())
                weights[bin_idx, nearest] = 1.0
        self.weights = weights

    def n_time_bins(self, audio_len: int) -> int:
        return 1 + max(0, int(audio_len)) // self.hop_length

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute the log-frequency log-magnitude STFT for one chunk.

        Args:
            audio: ``(1, L)`` or ``(L,)`` waveform tensor.

        Returns:
            Tensor of shape ``(1, n_freq_bins, n_time_bins)``.
        """

        y = audio.detach().float().view(-1)
        if y.numel() < self.n_fft:
            pad = self.n_fft - int(y.numel())
            y = torch.cat([y, torch.zeros(pad, dtype=y.dtype)], dim=0)

        spec = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(y.dtype),
            center=True,
            return_complex=True,
            normalized=False,
        )
        mag = spec.abs()
        if self.log_compress:
            mag = torch.log1p(mag)
        binned = self.weights.to(mag.dtype) @ mag
        return binned.unsqueeze(0).contiguous()


def resample_stft_time(stft: torch.Tensor, target_bins: int) -> torch.Tensor:
    """Resample the time axis of a precomputed STFT to ``target_bins``.

    Args:
        stft: ``(B, 1, F, T)`` or ``(1, F, T)``.
        target_bins: desired number of time bins.

    Returns:
        Tensor with the time axis interpolated to ``target_bins``.
    """

    squeeze = False
    if stft.ndim == 3:
        stft = stft.unsqueeze(0)
        squeeze = True
    out = F.interpolate(stft, size=(stft.shape[-2], int(target_bins)), mode="bilinear", align_corners=False)
    return out.squeeze(0) if squeeze else out
