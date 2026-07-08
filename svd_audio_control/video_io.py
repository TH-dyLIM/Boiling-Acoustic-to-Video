from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def find_with_ext(root: str | Path, stem: str, exts: Iterable[str]) -> Path | None:
    root = Path(root)
    for ext in exts:
        path = root / f"{stem}{ext}"
        if path.exists():
            return path
    for ext in exts:
        hits = list(root.rglob(f"{stem}{ext}"))
        if hits:
            return hits[0]
    return None


def resize_pad_bchw(img: torch.Tensor, height: int, width: int, mode: str = "bicubic") -> torch.Tensor:
    """Resize with aspect ratio preserved and zero pad to B,C,H,W."""
    old_h, old_w = int(img.shape[-2]), int(img.shape[-1])
    ratio = min(height / old_h, width / old_w)
    new_h = max(1, int(round(old_h * ratio)))
    new_w = max(1, int(round(old_w * ratio)))
    if mode == "nearest":
        out = F.interpolate(img, size=(new_h, new_w), mode="nearest")
    else:
        out = F.interpolate(img, size=(new_h, new_w), mode=mode, align_corners=False)
    pad_h = height - new_h
    pad_w = width - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return F.pad(out, (left, right, top, bottom))


def pil_to_tensor(img: Image.Image, height: int, width: int, mode: str = "RGB") -> torch.Tensor:
    if mode == "L":
        img = img.convert("L")
        arr = np.asarray(img, dtype=np.float32)[None, None]
        tensor = torch.from_numpy(arr)
        tensor = resize_pad_bchw(tensor, height, width, "nearest").squeeze(0)
        return (tensor > 127.0).float()
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None]
    tensor = torch.from_numpy(arr)
    tensor = resize_pad_bchw(tensor, height, width, "bicubic").squeeze(0)
    return tensor / 127.5 - 1.0


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    """Convert C,H,W tensor in [-1,1] or [0,1] to PIL RGB."""
    img = img.detach().float().cpu()
    if img.ndim != 3:
        raise ValueError(f"Expected C,H,W tensor, got shape {tuple(img.shape)}")
    if img.min() < -0.01:
        img = (img + 1.0) / 2.0
    img = img.clamp(0.0, 1.0)
    arr = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
    """Convert F,C,H,W or B,F,C,H,W tensor to uint8 F,H,W,C."""
    frames = frames.detach().float().cpu()
    if frames.ndim == 5:
        frames = frames[0]
    if frames.ndim != 4:
        raise ValueError(f"Expected F,C,H,W frames, got {tuple(frames.shape)}")
    if frames.min() < -0.01:
        frames = (frames + 1.0) / 2.0
    frames = frames.clamp(0.0, 1.0)
    return (frames.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)


def save_video_gif(frames: torch.Tensor, path: str | Path, fps: int = 10) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = frames_to_uint8(frames)
    duration = 1.0 / max(1, int(fps))
    imageio.mimsave(path, list(arr), duration=duration)


def save_video_mp4(frames: torch.Tensor, path: str | Path, fps: int = 10) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = frames_to_uint8(frames)
    imageio.mimsave(path, list(arr), fps=fps, macro_block_size=1)

