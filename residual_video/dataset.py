from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from svd_audio_control.video_io import pil_to_tensor


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_cache_stem(index: int, stem: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(stem)).strip("._")
    if not safe:
        safe = "sample"
    return f"{int(index):05d}_{safe}"


def _read_video_rgb(video_path: str | Path) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"Failed to decode frames: {video_path}")
    if fps <= 0:
        fps = 100.0
    return np.stack(frames, axis=0), fps


def _extract_frames(
    frames: np.ndarray,
    num_frames: int,
    frame_stride: int,
    start_frame: int | None,
    random_start: bool,
) -> tuple[np.ndarray, int]:
    frame_stride = max(1, int(frame_stride))
    total = int(frames.shape[0])
    needed_span = (int(num_frames) - 1) * frame_stride + 1
    max_start = max(0, total - needed_span)
    if start_frame is None:
        start = random.randint(0, max_start) if random_start and max_start > 0 else 0
    else:
        start = max(0, min(int(start_frame), max_start))
    idx = start + np.arange(int(num_frames)) * frame_stride
    return frames[np.clip(idx, 0, total - 1)], start


def _resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if x.size == 0 or int(src_sr) == int(dst_sr):
        return x.astype(np.float32)
    g = math.gcd(int(src_sr), int(dst_sr))
    return resample_poly(x, int(dst_sr) // g, int(src_sr) // g).astype(np.float32)


def _read_csv_waveform(csv_path: str | Path, default_sr: int = 1_000_000) -> tuple[np.ndarray, int]:
    """Read oscilloscope-style CSV audio files with Time,Voltage columns."""
    path = Path(csv_path)
    try:
        data = np.loadtxt(str(path), delimiter=",", skiprows=1, dtype=np.float32)
    except Exception:
        data = np.genfromtxt(str(path), delimiter=",", skip_header=1, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.size == 0:
        return np.zeros((0,), dtype=np.float32), int(default_sr)

    if data.shape[1] >= 2:
        time = data[:, 0].astype(np.float64)
        x = data[:, 1].astype(np.float32)
        finite_time = time[np.isfinite(time)]
        diffs = np.diff(finite_time[: min(finite_time.size, 20000)])
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            median_dt = float(np.median(diffs))
            src_sr = int(round(1.0 / median_dt)) if median_dt > 0 else int(default_sr)
        else:
            src_sr = int(default_sr)
    else:
        x = data[:, 0].astype(np.float32)
        src_sr = int(default_sr)
    x = x[np.isfinite(x)]
    return x.astype(np.float32, copy=False), int(src_sr)


def read_full_audio_file(audio_path: str | Path, target_sr: int) -> tuple[torch.Tensor, int]:
    path = Path(audio_path)
    if path.suffix.lower() == ".csv":
        x, src_sr = _read_csv_waveform(path, default_sr=int(target_sr))
    else:
        with sf.SoundFile(str(path), "r") as f:
            src_sr = int(f.samplerate)
            data = f.read(dtype="float32", always_2d=True)
        x = data[:, 0].astype(np.float32)
    x = _resample(x, src_sr, int(target_sr))
    return torch.from_numpy(x.astype(np.float32, copy=False)), int(target_sr)


def read_audio_segment(
    wav_path: str | Path,
    start_sec: float,
    duration_sec: float,
    target_sr: int,
    target_len: int,
    normalize: bool = True,
    clamp_value: float | None = None,
) -> torch.Tensor:
    """Read an audio chunk from disk.

    Args:
        normalize: When True (legacy), divide the chunk by its own peak so
            ``|max| == 1``. When False, preserve absolute amplitude so peak/
            RMS/crest features reflect the original acoustic intensity, which
            is critical for boiling regime / heat-flux inference.
        clamp_value: If set, hard-clip the output into ``[-clamp_value,
            clamp_value]`` to bound activations. Ignored when None.
    """
    audio_path = Path(wav_path)
    if audio_path.suffix.lower() == ".csv":
        full_audio, _ = read_full_audio_file(audio_path, int(target_sr))
        return audio_segment_from_cached(
            full_audio,
            start_sec=start_sec,
            duration_sec=duration_sec,
            target_sr=int(target_sr),
            target_len=int(target_len),
            normalize=normalize,
            clamp_value=clamp_value,
        )

    with sf.SoundFile(str(wav_path), "r") as f:
        src_sr = int(f.samplerate)
        raw_start = int(round(float(start_sec) * src_sr))
        pad_left_src = max(0, -raw_start)
        start = max(0, raw_start)
        frames = max(1, int(round(float(duration_sec) * src_sr)))
        f.seek(start)
        read_frames = max(1, frames - pad_left_src)
        data = f.read(frames=read_frames, dtype="float32", always_2d=True)
    x = data[:, 0].astype(np.float32)
    if pad_left_src > 0:
        x = np.concatenate([np.zeros((pad_left_src,), dtype=np.float32), x], axis=0)
    if x.size:
        x = x - float(x.mean())
    x = _resample(x, src_sr, int(target_sr))
    y = np.zeros((int(target_len),), dtype=np.float32)
    n = min(y.size, x.size)
    if n > 0:
        y[:n] = x[:n]
    if normalize:
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 1e-6:
            y = y / peak
        y = np.clip(y, -1.0, 1.0)
    elif clamp_value is not None:
        y = np.clip(y, -float(clamp_value), float(clamp_value))
    return torch.from_numpy(y.astype(np.float32, copy=False)).view(1, -1)


def audio_segment_from_cached(
    audio: torch.Tensor,
    start_sec: float,
    duration_sec: float,
    target_sr: int,
    target_len: int,
    normalize: bool = True,
    clamp_value: float | None = None,
) -> torch.Tensor:
    """Slice a chunk from a pre-loaded full-clip waveform.

    Args:
        normalize: When True (legacy), per-chunk peak normalization. When
            False, return the raw amplitude so absolute intensity is
            preserved.
        clamp_value: If set, hard-clip the output into ``[-clamp_value,
            clamp_value]`` to bound activations. Ignored when None.
    """
    y_src = audio.detach().float().view(-1)
    raw_start = int(round(float(start_sec) * int(target_sr)))
    frames = max(1, int(round(float(duration_sec) * int(target_sr))))
    pad_left = max(0, -raw_start)
    start = max(0, raw_start)
    end = max(start, start + frames - pad_left)
    chunk = y_src[start:end]
    if pad_left > 0:
        chunk = torch.cat([torch.zeros(pad_left, dtype=chunk.dtype), chunk], dim=0)
    if chunk.numel() > 0:
        chunk = chunk - chunk.mean()
    out = torch.zeros(int(target_len), dtype=torch.float32)
    n = min(out.numel(), chunk.numel())
    if n > 0:
        out[:n] = chunk[:n]
    if normalize:
        peak = float(out.abs().max().item()) if out.numel() else 0.0
        if peak > 1e-6:
            out = out / peak
        return out.clamp(-1.0, 1.0).view(1, -1)
    if clamp_value is not None:
        out = out.clamp(-float(clamp_value), float(clamp_value))
    return out.view(1, -1)


def audio_scalar_features(audio: torch.Tensor) -> torch.Tensor:
    y = audio.detach().float().view(-1)
    eps = 1e-12
    abs_y = y.abs()
    rms = torch.sqrt(torch.mean(y * y) + eps)
    peak = torch.max(abs_y) if y.numel() else torch.tensor(0.0)
    mean_abs = torch.mean(abs_y) if y.numel() else torch.tensor(0.0)
    std = torch.std(y, unbiased=False) if y.numel() else torch.tensor(0.0)
    if y.numel() > 1:
        zcr = torch.mean((torch.signbit(y[1:]) != torch.signbit(y[:-1])).float())
    else:
        zcr = torch.tensor(0.0)
    crest = torch.clamp(peak / torch.clamp(rms, min=eps), max=20.0) / 20.0

    def log_amp(value: torch.Tensor) -> torch.Tensor:
        return torch.clamp((torch.log10(torch.clamp(value, min=eps)) + 8.0) / 8.0, 0.0, 1.0)

    return torch.stack(
        [
            log_amp(rms),
            log_amp(peak),
            log_amp(mean_abs),
            log_amp(std),
            torch.clamp(zcr, 0.0, 1.0),
            crest,
        ]
    ).float()


def _image_01(path: str | Path, resolution: int, mode: str = "RGB") -> torch.Tensor:
    tensor = pil_to_tensor(Image.open(path), resolution, resolution, mode)
    if mode == "L":
        return tensor
    return ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)


def _rgb_frame_to_01(frame_rgb: np.ndarray, resolution: int) -> torch.Tensor:
    img = Image.fromarray(frame_rgb.astype(np.uint8))
    tensor = pil_to_tensor(img, resolution, resolution, "RGB")
    return ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)


def tensor_01_to_u8(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.detach().float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def tensor_u8_to_01(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().div(255.0).clamp(0.0, 1.0)


class ResidualVideoManifestDataset(Dataset):
    """Dataset for predicting only foreground residual over a known background.

    Returned tensors:
      pixel_values: F,3,H,W in [0,1]
      background: 3,H,W in [0,1]
      residual: F,3,H,W where residual = pixel_values - background
      roi: 1,H,W in [0,1]
      audio: 1,L in [-1,1]
      previous_frame: 3,H,W in [0,1] for autoregressive next-frame prediction
      physics: 3 normalized scalars [heat_flux, htc, regime]
      audio_features: scalar audio energy/shape features
    """

    def __init__(
        self,
        manifest_path: str | Path,
        resolution: int = 256,
        num_frames: int = 8,
        frame_stride: int = 1,
        audio_sample_rate: int = 1_000_000,
        video_fps: int = 100,
        random_start: bool = True,
        load_audio: bool = True,
        fixed_start_frames: list[int] | None = None,
        rollout_steps: int = 1,
        audio_context_frames: int = 1,
        audio_context_future_frames: int = 0,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.rows = read_jsonl(manifest_path)
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest_path}")
        self.resolution = int(resolution)
        self.num_frames = int(num_frames)
        self.frame_stride = int(frame_stride)
        self.audio_sample_rate = int(audio_sample_rate)
        self.video_fps = int(video_fps)
        self.random_start = bool(random_start)
        self.load_audio = bool(load_audio)
        self.rollout_steps = max(1, int(rollout_steps))
        self.audio_context_frames = max(1, int(audio_context_frames))
        self.audio_context_future_frames = max(0, min(int(audio_context_future_frames), self.audio_context_frames - 1))
        self.audio_context_past_frames = self.audio_context_frames - self.audio_context_future_frames - 1
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.fixed_start_frames = [int(x) for x in fixed_start_frames] if fixed_start_frames else []
        if self.fixed_start_frames:
            self.samples = [
                (row_index, start_frame)
                for row_index in range(len(self.rows))
                for start_frame in self.fixed_start_frames
            ]
        else:
            self.samples = [(row_index, None) for row_index in range(len(self.rows))]
        duration = (self.num_frames * self.frame_stride) / float(self.video_fps)
        self.audio_len = int(round(duration * self.audio_sample_rate))
        self.step_audio_len = int(
            round((self.audio_context_frames * self.frame_stride / float(self.video_fps)) * self.audio_sample_rate)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index, fixed_start_frame = self.samples[index]
        row = self.rows[row_index]
        cache = self._load_cache(row_index, row)
        if cache is not None:
            return self._getitem_from_cache(row_index, row, fixed_start_frame, cache)

        frames_rgb, fps = _read_video_rgb(row["video"])
        target_count = max(self.num_frames, self.rollout_steps)
        clip, start_frame = _extract_frames(
            frames_rgb,
            target_count,
            self.frame_stride,
            fixed_start_frame if fixed_start_frame is not None else row.get("start_frame"),
            self.random_start if fixed_start_frame is None else False,
        )

        frames = []
        for frame_rgb in clip[: self.num_frames]:
            frames.append(_rgb_frame_to_01(frame_rgb, self.resolution))
        pixel_values = torch.stack(frames, dim=0)

        bg_path = row.get("background", "")
        if bg_path and Path(bg_path).exists():
            background = _image_01(bg_path, self.resolution, "RGB")
        else:
            background = pixel_values[0].clone()

        if start_frame > 0:
            prev_index = max(0, int(start_frame) - self.frame_stride)
            previous_frame = _rgb_frame_to_01(frames_rgb[prev_index], self.resolution)
        else:
            previous_frame = background.clone()

        roi_path = row.get("roi", "")
        if roi_path and Path(roi_path).exists():
            roi = _image_01(roi_path, self.resolution, "L")
        else:
            roi = torch.zeros(1, self.resolution, self.resolution, dtype=torch.float32)

        start_sec = start_frame / float(fps)
        duration_sec = (self.num_frames * self.frame_stride) / float(fps)
        if self.load_audio:
            audio = read_audio_segment(
                row["audio"],
                start_sec=start_sec,
                duration_sec=duration_sec,
                target_sr=self.audio_sample_rate,
                target_len=self.audio_len,
            )
        else:
            audio = torch.zeros(1, self.audio_len, dtype=torch.float32)

        rollout_frames = []
        rollout_previous_frames = []
        rollout_audio = []
        rollout_audio_features = []
        step_duration_sec = self.audio_context_frames * self.frame_stride / float(fps)
        for step, frame_rgb in enumerate(clip[: self.rollout_steps]):
            target_index = start_frame + step * self.frame_stride
            rollout_frames.append(_rgb_frame_to_01(frame_rgb, self.resolution))
            if target_index > 0:
                prev_index = max(0, int(target_index) - self.frame_stride)
                rollout_previous_frames.append(_rgb_frame_to_01(frames_rgb[prev_index], self.resolution))
            else:
                rollout_previous_frames.append(background.clone())
            if self.load_audio:
                audio_start_index = target_index - self.audio_context_past_frames * self.frame_stride
                step_audio = read_audio_segment(
                    row["audio"],
                    start_sec=audio_start_index / float(fps),
                    duration_sec=step_duration_sec,
                    target_sr=self.audio_sample_rate,
                    target_len=self.step_audio_len,
                )
            else:
                step_audio = torch.zeros(1, self.step_audio_len, dtype=torch.float32)
            rollout_audio.append(step_audio)
            rollout_audio_features.append(audio_scalar_features(step_audio))

        residual = pixel_values - background.unsqueeze(0)
        physics = torch.tensor(row.get("physics_norm", [0.0, 0.0, 0.0]), dtype=torch.float32)
        return {
            "pixel_values": pixel_values,
            "background": background,
            "residual": residual,
            "roi": roi,
            "audio": audio,
            "previous_frame": previous_frame,
            "rollout_pixel_values": torch.stack(rollout_frames, dim=0),
            "rollout_previous_frames": torch.stack(rollout_previous_frames, dim=0),
            "rollout_audio": torch.stack(rollout_audio, dim=0),
            "rollout_audio_features": torch.stack(rollout_audio_features, dim=0),
            "audio_features": audio_scalar_features(audio),
            "physics": physics,
            "stem": row.get("stem", Path(row["video"]).stem),
            "split": row.get("split", ""),
            "start_frame": start_frame,
            "physics_raw": row.get("physics_raw", {}),
            "condition_source": row.get("condition_source", ""),
        }

    def _cache_path(self, row_index: int, row: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{safe_cache_stem(row_index, row.get('stem', Path(row.get('video', '')).stem))}.pt"

    def _load_cache(self, row_index: int, row: dict[str, Any]) -> dict[str, Any] | None:
        path = self._cache_path(row_index, row)
        if path is None or not path.exists():
            return None
        return torch.load(path, map_location="cpu")

    def _getitem_from_cache(
        self,
        row_index: int,
        row: dict[str, Any],
        fixed_start_frame: int | None,
        cache: dict[str, Any],
    ) -> dict[str, Any]:
        frames_u8 = cache["frames_u8"]
        fps = float(cache.get("fps", self.video_fps) or self.video_fps)
        total = int(frames_u8.shape[0])
        target_count = max(self.num_frames, self.rollout_steps)
        needed_span = (target_count - 1) * self.frame_stride + 1
        max_start = max(0, total - needed_span)
        if fixed_start_frame is None:
            row_start = row.get("start_frame")
            if row_start is None:
                start_frame = random.randint(0, max_start) if self.random_start and max_start > 0 else 0
            else:
                start_frame = max(0, min(int(row_start), max_start))
        else:
            start_frame = max(0, min(int(fixed_start_frame), max_start))

        idx = start_frame + torch.arange(target_count, dtype=torch.long) * self.frame_stride
        idx = idx.clamp(0, total - 1)
        selected = tensor_u8_to_01(frames_u8[idx])
        pixel_values = selected[: self.num_frames]
        background = tensor_u8_to_01(cache["background_u8"])
        roi = tensor_u8_to_01(cache["roi_u8"])

        if start_frame > 0:
            prev_index = max(0, int(start_frame) - self.frame_stride)
            previous_frame = tensor_u8_to_01(frames_u8[prev_index])
        else:
            previous_frame = background.clone()

        cached_audio = cache.get("audio")
        audio_sr = int(cache.get("audio_sr", self.audio_sample_rate))
        if cached_audio is not None and self.load_audio and audio_sr == self.audio_sample_rate:
            audio = audio_segment_from_cached(
                cached_audio,
                start_sec=start_frame / float(fps),
                duration_sec=(self.num_frames * self.frame_stride) / float(fps),
                target_sr=self.audio_sample_rate,
                target_len=self.audio_len,
            )
        elif self.load_audio:
            audio = read_audio_segment(
                row["audio"],
                start_sec=start_frame / float(fps),
                duration_sec=(self.num_frames * self.frame_stride) / float(fps),
                target_sr=self.audio_sample_rate,
                target_len=self.audio_len,
            )
        else:
            audio = torch.zeros(1, self.audio_len, dtype=torch.float32)

        rollout_frames = []
        rollout_previous_frames = []
        rollout_audio = []
        rollout_audio_features = []
        step_duration_sec = self.audio_context_frames * self.frame_stride / float(fps)
        for step in range(self.rollout_steps):
            target_index = start_frame + step * self.frame_stride
            frame_index = min(max(0, target_index), total - 1)
            rollout_frames.append(tensor_u8_to_01(frames_u8[frame_index]))
            if target_index > 0:
                prev_index = max(0, int(target_index) - self.frame_stride)
                rollout_previous_frames.append(tensor_u8_to_01(frames_u8[prev_index]))
            else:
                rollout_previous_frames.append(background.clone())
            audio_start_index = target_index - self.audio_context_past_frames * self.frame_stride
            if cached_audio is not None and self.load_audio and audio_sr == self.audio_sample_rate:
                step_audio = audio_segment_from_cached(
                    cached_audio,
                    start_sec=audio_start_index / float(fps),
                    duration_sec=step_duration_sec,
                    target_sr=self.audio_sample_rate,
                    target_len=self.step_audio_len,
                )
            elif self.load_audio:
                step_audio = read_audio_segment(
                    row["audio"],
                    start_sec=audio_start_index / float(fps),
                    duration_sec=step_duration_sec,
                    target_sr=self.audio_sample_rate,
                    target_len=self.step_audio_len,
                )
            else:
                step_audio = torch.zeros(1, self.step_audio_len, dtype=torch.float32)
            rollout_audio.append(step_audio)
            rollout_audio_features.append(audio_scalar_features(step_audio))

        residual = pixel_values - background.unsqueeze(0)
        physics = torch.tensor(row.get("physics_norm", [0.0, 0.0, 0.0]), dtype=torch.float32)
        return {
            "pixel_values": pixel_values,
            "background": background,
            "residual": residual,
            "roi": roi,
            "audio": audio,
            "previous_frame": previous_frame,
            "rollout_pixel_values": torch.stack(rollout_frames, dim=0),
            "rollout_previous_frames": torch.stack(rollout_previous_frames, dim=0),
            "rollout_audio": torch.stack(rollout_audio, dim=0),
            "rollout_audio_features": torch.stack(rollout_audio_features, dim=0),
            "audio_features": audio_scalar_features(audio),
            "physics": physics,
            "stem": row.get("stem", Path(row["video"]).stem),
            "split": row.get("split", ""),
            "start_frame": start_frame,
            "physics_raw": row.get("physics_raw", {}),
            "condition_source": row.get("condition_source", ""),
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = [
        "pixel_values",
        "background",
        "residual",
        "roi",
        "audio",
        "previous_frame",
        "rollout_pixel_values",
        "rollout_previous_frames",
        "rollout_audio",
        "rollout_audio_features",
        "audio_features",
        "physics",
    ]
    out: dict[str, Any] = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
    for key in ["stem", "split", "start_frame", "physics_raw", "condition_source"]:
        out[key] = [item[key] for item in batch]
    return out
