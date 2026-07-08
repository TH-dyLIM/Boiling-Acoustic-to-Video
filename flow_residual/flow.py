from __future__ import annotations

from typing import Callable

import torch


def sample_flow_time(batch_size: int, device: torch.device, schedule: str = "logit_normal") -> torch.Tensor:
    if schedule == "uniform":
        return torch.rand(batch_size, device=device).clamp(1e-4, 1.0 - 1e-4)
    if schedule == "logit_normal":
        u = torch.randn(batch_size, device=device)
        return torch.sigmoid(u).clamp(1e-4, 1.0 - 1e-4)
    raise ValueError(f"Unknown time schedule: {schedule!r}")


def make_noised(
    target_residual_scaled: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear flow matching path: x_t = (1 - t) * x_0 + t * eps.

    Velocity target is v* = eps - x_0.
    """
    noise = torch.randn_like(target_residual_scaled)
    t_view = time.view(-1, 1, 1, 1, 1).to(target_residual_scaled.dtype)
    noisy = (1.0 - t_view) * target_residual_scaled + t_view * noise
    velocity_target = noise - target_residual_scaled
    return noisy, velocity_target


def flow_matching_loss(
    pred_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    foreground_weight_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    diff_sq = (pred_velocity - target_velocity) ** 2
    if foreground_weight_map is None:
        loss = diff_sq.mean()
        weighted_loss = loss
    else:
        weights = foreground_weight_map.to(diff_sq.dtype)
        loss = diff_sq.mean()
        weighted_loss = (diff_sq * weights).sum() / weights.sum().clamp_min(1e-6)
    parts = {
        "fm_mse": float(loss.detach().cpu().item()),
        "fm_weighted": float(weighted_loss.detach().cpu().item()),
    }
    return weighted_loss, parts


@torch.no_grad()
def euler_sample(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    shape: tuple[int, ...],
    device: torch.device,
    num_steps: int = 25,
    cfg_scale: float = 0.0,
    uncond_velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(seed))
        x = torch.randn(*shape, device=device, generator=gen)
    else:
        x = torch.randn(*shape, device=device)
    times = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    for i in range(num_steps):
        t = times[i].expand(shape[0])
        dt = (times[i + 1] - times[i])
        v = velocity_fn(x, t)
        if cfg_scale > 0.0 and uncond_velocity_fn is not None:
            v_u = uncond_velocity_fn(x, t)
            v = v_u + (1.0 + cfg_scale) * (v - v_u)
        x = x + v * dt
    return x


def patch_edge_smoothness(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Squared-difference penalty across DiT patch seams.

    For a video tensor ``x`` of shape ``(B, T, C, H, W)`` and patch grid stride
    ``patch_size``, this returns the mean L2 squared difference between pixel
    columns/rows that straddle a patch boundary. Useful as an auxiliary loss to
    discourage visible 8x8 (or whatever) tiling in DiT-style decoders.
    """

    p = int(patch_size)
    if p <= 1:
        return x.new_zeros(())
    H, W = x.shape[-2:]
    parts: list[torch.Tensor] = []
    if W > p:
        cols = torch.arange(p, W, p, device=x.device)
        if cols.numel() > 0:
            h_diff = (x.index_select(-1, cols) - x.index_select(-1, cols - 1)) ** 2
            parts.append(h_diff.mean())
    if H > p:
        rows = torch.arange(p, H, p, device=x.device)
        if rows.numel() > 0:
            v_diff = (x.index_select(-2, rows) - x.index_select(-2, rows - 1)) ** 2
            parts.append(v_diff.mean())
    if not parts:
        return x.new_zeros(())
    return torch.stack(parts).mean()


def foreground_weight(
    target_residual: torch.Tensor,
    roi: torch.Tensor | None,
    threshold: float = 0.04,
    base_weight: float = 1.0,
    fg_weight: float = 5.0,
    roi_weight: float = 1.0,
) -> torch.Tensor:
    strength = target_residual.detach().abs().mean(dim=2, keepdim=True)
    fg = (strength > float(threshold)).float()
    weights = base_weight + fg_weight * fg
    if roi is not None:
        weights = weights + roi_weight * roi.unsqueeze(1)
    return weights
