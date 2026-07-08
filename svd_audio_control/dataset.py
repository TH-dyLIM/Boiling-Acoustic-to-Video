from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .video_io import find_with_ext, pil_to_tensor


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
    needed_span = (num_frames - 1) * frame_stride + 1
    max_start = max(0, total - needed_span)
    if start_frame is None:
        start = random.randint(0, max_start) if random_start and max_start > 0 else 0
    else:
        start = max(0, min(int(start_frame), max_start))
    idx = start + np.arange(num_frames) * frame_stride
    idx = np.clip(idx, 0, total - 1)
    return frames[idx], start


def _resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if x.size == 0 or src_sr == dst_sr:
        return x.astype(np.float32)
    g = math.gcd(src_sr, dst_sr)
    return resample_poly(x, dst_sr // g, src_sr // g).astype(np.float32)


def _read_audio_segment(
    wav_path: str | Path,
    start_sec: float,
    duration_sec: float,
    target_sr: int,
    target_len: int,
) -> torch.Tensor:
    with sf.SoundFile(str(wav_path), "r") as f:
        src_sr = int(f.samplerate)
        start = max(0, int(round(start_sec * src_sr)))
        frames = max(1, int(round(duration_sec * src_sr)))
        f.seek(start)
        data = f.read(frames=frames, dtype="float32", always_2d=True)
    x = data[:, 0].astype(np.float32)
    x = x - float(x.mean()) if x.size else x
    x = _resample(x, src_sr, target_sr)
    y = np.zeros((target_len,), dtype=np.float32)
    n = min(target_len, x.size)
    if n > 0:
        y[:n] = x[:n]
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-6:
        y = y / peak
    return torch.from_numpy(np.clip(y, -1.0, 1.0)).view(1, -1)


class BoilingSVDManifestDataset(Dataset):
    """Manifest dataset for SVD i2v LoRA training.

    Each item returns:
      pixel_values: F,3,H,W in [-1,1]
      conditioning_image: 3,H,W in [-1,1]
      roi: 1,H,W in [0,1]
      audio: 1,L in [-1,1]
      physics: 3 normalized scalars
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
        conditioning_image_source: str = "background",
    ):
        self.rows = _read_jsonl(manifest_path)
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest_path}")
        self.resolution = int(resolution)
        self.num_frames = int(num_frames)
        self.frame_stride = int(frame_stride)
        self.audio_sample_rate = int(audio_sample_rate)
        self.video_fps = int(video_fps)
        self.random_start = bool(random_start)
        self.load_audio = bool(load_audio)
        self.conditioning_image_source = str(conditioning_image_source)
        duration = (self.num_frames * self.frame_stride) / float(self.video_fps)
        self.audio_len = int(round(duration * self.audio_sample_rate))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        frames_rgb, fps = _read_video_rgb(row["video"])
        clip, start_frame = _extract_frames(
            frames_rgb,
            self.num_frames,
            self.frame_stride,
            row.get("start_frame"),
            self.random_start,
        )
        frame_t = torch.from_numpy(clip).permute(0, 3, 1, 2).float()
        resized = []
        for frame in frame_t:
            img = Image.fromarray(frame.permute(1, 2, 0).byte().numpy())
            resized.append(pil_to_tensor(img, self.resolution, self.resolution, "RGB"))
        pixel_values = torch.stack(resized, dim=0)

        bg_path = row.get("background")
        if self.conditioning_image_source == "first_frame":
            conditioning_image = pixel_values[0].clone()
        elif bg_path and Path(bg_path).exists():
            conditioning_image = pil_to_tensor(
                Image.open(bg_path), self.resolution, self.resolution, "RGB"
            )
        else:
            conditioning_image = pixel_values[0].clone()

        roi_path = row.get("roi")
        if roi_path and Path(roi_path).exists():
            roi = pil_to_tensor(Image.open(roi_path), self.resolution, self.resolution, "L")
        else:
            roi = torch.zeros(1, self.resolution, self.resolution, dtype=torch.float32)

        start_sec = start_frame / float(fps)
        duration_sec = (self.num_frames * self.frame_stride) / float(fps)
        if self.load_audio:
            audio = _read_audio_segment(
                row["audio"],
                start_sec=start_sec,
                duration_sec=duration_sec,
                target_sr=self.audio_sample_rate,
                target_len=self.audio_len,
            )
        else:
            audio = torch.zeros(1, self.audio_len, dtype=torch.float32)

        physics = torch.tensor(row.get("physics_norm", [0.0, 0.0, 0.0]), dtype=torch.float32)
        return {
            "pixel_values": pixel_values,
            "conditioning_image": conditioning_image,
            "roi": roi,
            "audio": audio,
            "physics": physics,
            "stem": row.get("stem", Path(row["video"]).stem),
            "split": row.get("split", ""),
            "start_frame": start_frame,
            "physics_raw": row.get("physics_raw", {}),
            "condition_source": row.get("condition_source", ""),
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = ["pixel_values", "conditioning_image", "roi", "audio", "physics"]
    out: dict[str, Any] = {k: torch.stack([item[k] for item in batch], dim=0) for k in tensor_keys}
    for key in ["stem", "split", "start_frame", "physics_raw", "condition_source"]:
        out[key] = [item[key] for item in batch]
    return out
