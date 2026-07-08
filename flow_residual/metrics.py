from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def compose_video(
    background: torch.Tensor,
    residual: torch.Tensor,
    clamp: bool = True,
) -> torch.Tensor:
    out = background.unsqueeze(1) + residual
    return out.clamp(0.0, 1.0) if clamp else out


def foreground_mask(residual: torch.Tensor, threshold: float = 0.04) -> torch.Tensor:
    strength = residual.detach().abs().mean(dim=2)
    return (strength > float(threshold)).float()


def void_fraction(mask: torch.Tensor) -> torch.Tensor:
    return mask.mean(dim=(2, 3))


def _connected_components(binary: np.ndarray) -> int:
    if not binary.any():
        return 0
    visited = np.zeros_like(binary, dtype=bool)
    height, width = binary.shape
    count = 0
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or visited[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    return count


def nucleation_count_per_frame(mask: torch.Tensor, downsample: int = 4) -> torch.Tensor:
    bsz, frames, _, _ = mask.shape
    if downsample > 1:
        mask = F.avg_pool2d(mask.unsqueeze(2).reshape(bsz * frames, 1, *mask.shape[-2:]), downsample)
        mask = (mask > 0.25).float().reshape(bsz, frames, *mask.shape[-2:])
    counts = torch.zeros(bsz, frames, dtype=torch.float32)
    array = mask.detach().cpu().numpy().astype(bool)
    for b in range(bsz):
        for t in range(frames):
            counts[b, t] = float(_connected_components(array[b, t]))
    return counts


def departure_frequency(mask: torch.Tensor) -> torch.Tensor:
    """Per-pixel transitions 0->1 across time, summed over T-1 transitions."""
    if mask.shape[1] < 2:
        return torch.zeros(mask.shape[0], dtype=torch.float32)
    transitions = ((mask[:, 1:] > 0.5) & (mask[:, :-1] <= 0.5)).float()
    return transitions.sum(dim=(1, 2, 3))


def ks_statistic(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().cpu().numpy().reshape(-1)
    b = b.detach().cpu().numpy().reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    grid = np.unique(np.concatenate([a_sorted, b_sorted]))
    cdf_a = np.searchsorted(a_sorted, grid, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, grid, side="right") / b_sorted.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def spatial_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def spatial_density_map(mask: torch.Tensor) -> torch.Tensor:
    return mask.mean(dim=1)


def video_metrics(
    pred_video: torch.Tensor,
    target_video: torch.Tensor,
    background: torch.Tensor,
    foreground_threshold: float = 0.04,
) -> dict[str, float]:
    pred = pred_video.detach().float().clamp(0.0, 1.0)
    target = target_video.detach().float().clamp(0.0, 1.0)
    bg = background.detach().float().clamp(0.0, 1.0)
    mse = ((pred - target) ** 2).mean().item()
    mae = (pred - target).abs().mean().item()
    psnr = 99.0 if mse <= 1e-12 else float(-10.0 * np.log10(mse))

    pred_residual = pred - bg.unsqueeze(1)
    target_residual = target - bg.unsqueeze(1)
    pred_mask = foreground_mask(pred_residual, foreground_threshold)
    target_mask = foreground_mask(target_residual, foreground_threshold)

    pred_vf = void_fraction(pred_mask)
    target_vf = void_fraction(target_mask)
    pred_counts = nucleation_count_per_frame(pred_mask)
    target_counts = nucleation_count_per_frame(target_mask)
    pred_dep = departure_frequency(pred_mask)
    target_dep = departure_frequency(target_mask)
    pred_density = spatial_density_map(pred_mask)
    target_density = spatial_density_map(target_mask)

    out = {
        "mse_video": float(mse),
        "mae_video": float(mae),
        "psnr_video": float(psnr),
        "pred_void_fraction_mean": float(pred_vf.mean().item()),
        "target_void_fraction_mean": float(target_vf.mean().item()),
        "void_fraction_mae": float((pred_vf - target_vf).abs().mean().item()),
        "void_fraction_ks": ks_statistic(pred_vf, target_vf),
        "pred_nucleation_count_mean": float(pred_counts.mean().item()),
        "target_nucleation_count_mean": float(target_counts.mean().item()),
        "nucleation_count_mae": float((pred_counts - target_counts).abs().mean().item()),
        "nucleation_count_ks": ks_statistic(pred_counts, target_counts),
        "pred_departure_freq_mean": float(pred_dep.mean().item()),
        "target_departure_freq_mean": float(target_dep.mean().item()),
        "departure_freq_mae": float((pred_dep - target_dep).abs().mean().item()),
        "spatial_density_l1": spatial_l1(pred_density, target_density),
    }
    return out


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row})
    out: dict[str, float] = {}
    for key in keys:
        values = [row[key] for row in rows if key in row and not (isinstance(row[key], float) and np.isnan(row[key]))]
        if values:
            out[f"{key}_mean"] = float(sum(values) / len(values))
    return out


def merge_dict(*dicts: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for d in dicts:
        out.update(d)
    return out
