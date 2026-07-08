from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from residual_video.dataset import (
    _image_01,
    _read_video_rgb,
    _rgb_frame_to_01,
    read_jsonl,
    safe_cache_stem,
    tensor_u8_to_01,
)
from flow_residual.dataset import _physics_key


def _accumulate_clip_residual(
    frames: torch.Tensor,
    background: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    residual = (frames - background.unsqueeze(0)).abs().mean(dim=1)
    mask = (residual > float(threshold)).float()
    return mask.mean(dim=0)


def _load_clip(
    row_index: int, row: dict[str, Any], cache_dir: Path | None, resolution: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_dir is not None:
        stem = row.get("stem", Path(row.get("video", "")).stem)
        path = cache_dir / f"{safe_cache_stem(row_index, stem)}.pt"
        if path.exists():
            cache = torch.load(path, map_location="cpu")
            frames = tensor_u8_to_01(cache["frames_u8"]).float()
            bg = tensor_u8_to_01(cache["background_u8"]).float()
            return frames, bg
    frames_rgb, _ = _read_video_rgb(row["video"])
    frames = torch.stack([_rgb_frame_to_01(f, resolution) for f in frames_rgb], dim=0)
    bg_path = row.get("background", "")
    if bg_path and Path(bg_path).exists():
        bg = _image_01(bg_path, resolution, "RGB")
    else:
        bg = frames[0].clone()
    return frames, bg


def compute_priors(
    manifest_path: str | Path,
    cache_dir: str | Path | None,
    resolution: int,
    foreground_threshold: float = 0.04,
    smoothing_sigma: float = 0.0,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, int]]:
    rows = read_jsonl(manifest_path)
    cache_dir_path = Path(cache_dir) if cache_dir else None
    per_class_acc: dict[str, list[torch.Tensor]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    global_acc: list[torch.Tensor] = []
    for row_index, row in enumerate(rows):
        frames, bg = _load_clip(row_index, row, cache_dir_path, resolution)
        if frames.shape[-1] != resolution:
            frames = torch.nn.functional.interpolate(frames, size=(resolution, resolution), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
            bg = torch.nn.functional.interpolate(bg.unsqueeze(0), size=(resolution, resolution), mode="bilinear", align_corners=False).squeeze(0).clamp(0.0, 1.0)
        prior = _accumulate_clip_residual(frames, bg, foreground_threshold)
        key = _physics_key(row.get("physics_raw", {}))
        per_class_acc[key].append(prior)
        counts[key] += 1
        global_acc.append(prior)

    per_class: dict[str, torch.Tensor] = {}
    for key, items in per_class_acc.items():
        stacked = torch.stack(items, dim=0).mean(dim=0)
        per_class[key] = _post_process(stacked.unsqueeze(0), smoothing_sigma)
    if global_acc:
        global_prior = _post_process(torch.stack(global_acc, dim=0).mean(dim=0).unsqueeze(0), smoothing_sigma)
    else:
        global_prior = torch.zeros(1, resolution, resolution, dtype=torch.float32)
    return per_class, global_prior, dict(counts)


def _post_process(prior: torch.Tensor, sigma: float) -> torch.Tensor:
    out = prior.float()
    if sigma and float(sigma) > 0.0:
        out = _gaussian_blur(out, float(sigma))
    peak = float(out.max().item())
    if peak > 1e-6:
        out = out / peak
    return out.clamp(0.0, 1.0)


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel_2d.view(1, 1, *kernel_2d.shape)
    x = image.unsqueeze(0)
    x = torch.nn.functional.pad(x, (radius, radius, radius, radius), mode="reflect")
    return torch.nn.functional.conv2d(x, kernel).squeeze(0).clamp(0.0, 1.0)
