from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from flow_residual.audio_features import LogFreqSTFT
from residual_video.dataset import (
    _image_01,
    _read_video_rgb,
    _rgb_frame_to_01,
    audio_scalar_features,
    audio_segment_from_cached,
    read_audio_segment,
    read_jsonl,
    safe_cache_stem,
    tensor_u8_to_01,
)


def _physics_key(physics_raw: dict[str, Any]) -> str:
    regime = int(round(float(physics_raw.get("regime", 0.0))))
    hf = float(physics_raw.get("heat_flux", 0.0))
    bin_index = int(min(7, max(0, math.floor(hf / 100.0))))
    return f"r{regime}_hf{bin_index}"


def _resize_01(tensor: torch.Tensor, resolution: int, mode: str) -> torch.Tensor:
    if int(tensor.shape[-1]) == int(resolution) and int(tensor.shape[-2]) == int(resolution):
        return tensor
    x = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor
    if mode == "nearest":
        out = F.interpolate(x.float(), size=(resolution, resolution), mode="nearest")
    else:
        out = F.interpolate(x.float(), size=(resolution, resolution), mode="bilinear", align_corners=False)
    return out.squeeze(0).clamp(0.0, 1.0)


class ChunkResidualDataset(Dataset):
    """Chunked dataset: returns T-frame clips for flow-matching diffusion.

    Returned tensors:
      pixel_values: T,3,H,W in [0,1]
      background:   3,H,W in [0,1]
      residual:     T,3,H,W (= pixel_values - background)
      roi:          1,H,W in [0,1]
      nucleation_prior: 1,H,W in [0,1]   spatial nucleation site prior
      prev_last_frame:  3,H,W in [0,1]   last frame of preceding chunk (or background)
      audio:        1,L in [-1,1]        full chunk audio
      audio_features: 6                  scalar shape descriptors
      physics:      3                    [hf_norm, htc_norm, regime_norm]
    """

    def __init__(
        self,
        manifest_path: str | Path,
        resolution: int = 128,
        chunk_frames: int = 12,
        frame_stride: int = 1,
        audio_sample_rate: int = 1_000_000,
        video_fps: int = 100,
        random_start: bool = True,
        cache_dir: str | Path | None = None,
        prior_path: str | Path | None = None,
        prior_mode: str = "per_class",
        roi_mode: str = "normal",
        load_audio: bool = True,
        fixed_starts_per_clip: list[int] | None = None,
        audio_normalize: bool = True,
        audio_clamp_value: float | None = None,
        use_stft: bool = False,
        stft_n_fft: int = 2048,
        stft_hop_length: int = 1024,
        stft_n_freq_bins: int = 64,
        stft_fmin: float = 100.0,
        stft_fmax: float | None = None,
    ) -> None:
        self.rows = read_jsonl(manifest_path)
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest_path}")
        self.resolution = int(resolution)
        self.chunk_frames = int(chunk_frames)
        self.frame_stride = max(1, int(frame_stride))
        self.audio_sample_rate = int(audio_sample_rate)
        self.video_fps = int(video_fps)
        self.random_start = bool(random_start)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.prior_mode = str(prior_mode)
        if self.prior_mode not in {"per_class", "global", "none"}:
            raise ValueError(f"prior_mode must be per_class, global, or none, got {self.prior_mode!r}")
        self.roi_mode = str(roi_mode)
        if self.roi_mode not in {"normal", "none"}:
            raise ValueError(f"roi_mode must be normal or none, got {self.roi_mode!r}")
        self.load_audio = bool(load_audio)
        self.fixed_starts_per_clip = [int(x) for x in fixed_starts_per_clip] if fixed_starts_per_clip else []
        self.audio_normalize = bool(audio_normalize)
        self.audio_clamp_value = float(audio_clamp_value) if audio_clamp_value is not None else None
        self.use_stft = bool(use_stft)
        self.stft_n_fft = int(stft_n_fft)
        self.stft_hop_length = int(stft_hop_length)
        self.stft_n_freq_bins = int(stft_n_freq_bins)
        self.stft_fmin = float(stft_fmin)
        self.stft_fmax = float(stft_fmax) if stft_fmax is not None else None
        self.stft = (
            LogFreqSTFT(
                sample_rate=self.audio_sample_rate,
                n_fft=self.stft_n_fft,
                hop_length=self.stft_hop_length,
                n_freq_bins=self.stft_n_freq_bins,
                fmin=self.stft_fmin,
                fmax=self.stft_fmax,
            )
            if self.use_stft
            else None
        )

        if self.fixed_starts_per_clip:
            self.samples = [
                (row_index, start) for row_index in range(len(self.rows)) for start in self.fixed_starts_per_clip
            ]
        else:
            self.samples = [(row_index, None) for row_index in range(len(self.rows))]

        duration = (self.chunk_frames * self.frame_stride) / float(self.video_fps)
        self.audio_len = int(round(duration * self.audio_sample_rate))

        self.priors: dict[str, torch.Tensor] = {}
        self.global_prior: torch.Tensor | None = None
        if prior_path is not None and self.prior_mode != "none":
            blob = torch.load(Path(prior_path), map_location="cpu")
            for key, value in blob.get("per_class", {}).items():
                self.priors[str(key)] = _resize_01(value.float(), self.resolution, "bilinear").contiguous()
            global_prior = blob.get("global", None)
            if global_prior is not None:
                self.global_prior = _resize_01(global_prior.float(), self.resolution, "bilinear").contiguous()

    def __len__(self) -> int:
        return len(self.samples)

    def _select_prior(self, physics_raw: dict[str, Any]) -> torch.Tensor:
        if self.prior_mode == "none":
            return torch.zeros(1, self.resolution, self.resolution, dtype=torch.float32)
        if self.prior_mode == "per_class" and self.priors:
            key = _physics_key(physics_raw)
            if key in self.priors:
                return self.priors[key].clone()
            regime = int(round(float(physics_raw.get("regime", 0.0))))
            for k, v in self.priors.items():
                if k.startswith(f"r{regime}_"):
                    return v.clone()
        if self.global_prior is not None:
            return self.global_prior.clone()
        return torch.zeros(1, self.resolution, self.resolution, dtype=torch.float32)

    def _cache_path(self, row_index: int, row: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        stem = row.get("stem", Path(row.get("video", "")).stem)
        return self.cache_dir / f"{safe_cache_stem(row_index, stem)}.pt"

    def _load_cache(self, row_index: int, row: dict[str, Any]) -> dict[str, Any] | None:
        path = self._cache_path(row_index, row)
        if path is None or not path.exists():
            return None
        return torch.load(path, map_location="cpu")

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index, fixed_start = self.samples[index]
        row = self.rows[row_index]
        cache = self._load_cache(row_index, row)
        if cache is not None:
            return self._from_cache(row_index, row, fixed_start, cache)
        return self._from_disk(row_index, row, fixed_start)

    def _from_cache(
        self, row_index: int, row: dict[str, Any], fixed_start: int | None, cache: dict[str, Any]
    ) -> dict[str, Any]:
        frames_u8 = cache["frames_u8"]
        fps = float(cache.get("fps", self.video_fps) or self.video_fps)
        total = int(frames_u8.shape[0])
        needed = (self.chunk_frames - 1) * self.frame_stride + 1
        max_start = max(0, total - needed)
        if fixed_start is None:
            start = random.randint(0, max_start) if self.random_start and max_start > 0 else 0
        else:
            start = max(0, min(int(fixed_start), max_start))
        idx_np = (start + np.arange(self.chunk_frames) * self.frame_stride).clip(0, total - 1)
        idx_t = torch.from_numpy(idx_np).long()
        frames_chunk = tensor_u8_to_01(frames_u8[idx_t])
        frames_chunk = self._resize_video(frames_chunk)

        background = self._resize_image(tensor_u8_to_01(cache["background_u8"]), is_mask=False)
        roi = self._resize_image(tensor_u8_to_01(cache["roi_u8"]), is_mask=True)
        if start > 0:
            prev_index = max(0, int(start) - self.frame_stride)
            prev_last = self._resize_image(tensor_u8_to_01(frames_u8[prev_index]), is_mask=False)
        else:
            prev_last = background.clone()

        cached_audio = cache.get("audio")
        audio_sr = int(cache.get("audio_sr", self.audio_sample_rate))
        if cached_audio is not None and self.load_audio and audio_sr == self.audio_sample_rate:
            audio = audio_segment_from_cached(
                cached_audio,
                start_sec=start / float(fps),
                duration_sec=(self.chunk_frames * self.frame_stride) / float(fps),
                target_sr=self.audio_sample_rate,
                target_len=self.audio_len,
                normalize=self.audio_normalize,
                clamp_value=self.audio_clamp_value,
            )
        elif self.load_audio:
            audio = read_audio_segment(
                row["audio"],
                start_sec=start / float(fps),
                duration_sec=(self.chunk_frames * self.frame_stride) / float(fps),
                target_sr=self.audio_sample_rate,
                target_len=self.audio_len,
                normalize=self.audio_normalize,
                clamp_value=self.audio_clamp_value,
            )
        else:
            audio = torch.zeros(1, self.audio_len, dtype=torch.float32)

        return self._pack(row, start, frames_chunk, background, roi, prev_last, audio)

    def _from_disk(self, row_index: int, row: dict[str, Any], fixed_start: int | None) -> dict[str, Any]:
        frames_rgb, fps = _read_video_rgb(row["video"])
        total = int(frames_rgb.shape[0])
        needed = (self.chunk_frames - 1) * self.frame_stride + 1
        max_start = max(0, total - needed)
        if fixed_start is None:
            start = random.randint(0, max_start) if self.random_start and max_start > 0 else 0
        else:
            start = max(0, min(int(fixed_start), max_start))
        idx = (start + np.arange(self.chunk_frames) * self.frame_stride).clip(0, total - 1)
        frames = torch.stack([_rgb_frame_to_01(frames_rgb[int(i)], self.resolution) for i in idx], dim=0)

        bg_path = row.get("background", "")
        if bg_path and Path(bg_path).exists():
            background = _image_01(bg_path, self.resolution, "RGB")
        else:
            background = frames[0].clone()
        roi_path = row.get("roi", "")
        if roi_path and Path(roi_path).exists():
            roi = _image_01(roi_path, self.resolution, "L")
        else:
            roi = torch.zeros(1, self.resolution, self.resolution, dtype=torch.float32)
        if start > 0:
            prev_index = max(0, int(start) - self.frame_stride)
            prev_last = _rgb_frame_to_01(frames_rgb[prev_index], self.resolution)
        else:
            prev_last = background.clone()
        if self.load_audio:
            audio = read_audio_segment(
                row["audio"],
                start_sec=start / float(fps),
                duration_sec=(self.chunk_frames * self.frame_stride) / float(fps),
                target_sr=self.audio_sample_rate,
                target_len=self.audio_len,
                normalize=self.audio_normalize,
                clamp_value=self.audio_clamp_value,
            )
        else:
            audio = torch.zeros(1, self.audio_len, dtype=torch.float32)
        return self._pack(row, start, frames, background, roi, prev_last, audio)

    def _resize_image(self, tensor: torch.Tensor, is_mask: bool) -> torch.Tensor:
        return _resize_01(tensor, self.resolution, "nearest" if is_mask else "bilinear")

    def _resize_video(self, video: torch.Tensor) -> torch.Tensor:
        if int(video.shape[-1]) == self.resolution and int(video.shape[-2]) == self.resolution:
            return video
        out = F.interpolate(video.float(), size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        return out.clamp(0.0, 1.0)

    def _pack(
        self,
        row: dict[str, Any],
        start: int,
        frames: torch.Tensor,
        background: torch.Tensor,
        roi: torch.Tensor,
        prev_last: torch.Tensor,
        audio: torch.Tensor,
    ) -> dict[str, Any]:
        if self.roi_mode == "none":
            roi = torch.zeros_like(roi)
        residual = frames - background.unsqueeze(0)
        physics_raw = row.get("physics_raw", {})
        prior = self._select_prior(physics_raw)
        physics = torch.tensor(row.get("physics_norm", [0.0, 0.0, 0.0]), dtype=torch.float32)
        audio_stft = (
            self.stft(audio)
            if self.use_stft and self.stft is not None
            else torch.zeros(1, self.stft_n_freq_bins, 1, dtype=torch.float32)
        )
        return {
            "pixel_values": frames,
            "background": background,
            "residual": residual,
            "roi": roi,
            "nucleation_prior": prior,
            "prev_last_frame": prev_last,
            "audio": audio,
            "audio_features": audio_scalar_features(audio),
            "audio_stft": audio_stft,
            "physics": physics,
            "physics_raw": physics_raw,
            "stem": row.get("stem", Path(row["video"]).stem),
            "split": row.get("split", ""),
            "start_frame": int(start),
            "condition_source": row.get("condition_source", ""),
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = [
        "pixel_values",
        "background",
        "residual",
        "roi",
        "nucleation_prior",
        "prev_last_frame",
        "audio",
        "audio_features",
        "audio_stft",
        "physics",
    ]
    out: dict[str, Any] = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
    for key in ["stem", "split", "start_frame", "physics_raw", "condition_source"]:
        out[key] = [item[key] for item in batch]
    return out


def write_prior_blob(
    out_path: str | Path,
    per_class: dict[str, torch.Tensor],
    global_prior: torch.Tensor,
    meta: dict[str, Any] | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"per_class": per_class, "global": global_prior, "meta": meta or {}}
    torch.save(blob, out_path)


def physics_key_for_row(row: dict[str, Any]) -> str:
    return _physics_key(row.get("physics_raw", {}))
