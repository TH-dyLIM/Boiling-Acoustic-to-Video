from __future__ import annotations

import torch
import torch.nn.functional as F


def effective_residual(
    residual: torch.Tensor,
    mask: torch.Tensor | None = None,
    residual_target_scale: float = 1.0,
) -> torch.Tensor:
    out = residual
    if mask is not None:
        out = out * mask
    scale = max(float(residual_target_scale), 1e-6)
    return out / scale


def compose_video(
    background: torch.Tensor,
    residual: torch.Tensor,
    clamp: bool = True,
    mask: torch.Tensor | None = None,
    residual_target_scale: float = 1.0,
    base_frame: torch.Tensor | None = None,
) -> torch.Tensor:
    base = background if base_frame is None else base_frame
    video = base.unsqueeze(1) + effective_residual(residual, mask, residual_target_scale)
    return video.clamp(0.0, 1.0) if clamp else video


def foreground_mask(target_residual: torch.Tensor, threshold: float = 0.035) -> torch.Tensor:
    strength = target_residual.detach().abs().mean(dim=2, keepdim=True)
    return (strength > float(threshold)).float()


def masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if mask.shape[2] == 1 and value.shape[2] != 1:
        mask = mask.expand(-1, -1, value.shape[2], -1, -1)
    return (value * mask).sum() / mask.sum().clamp_min(float(eps))


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    intersection = (pred * target).sum(dim=(1, 2, 3, 4))
    denom = pred.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    return (1.0 - (2.0 * intersection + eps) / (denom + eps)).mean()


def video_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    background: torch.Tensor | None = None,
    previous_frame: torch.Tensor | None = None,
    pred_mask: torch.Tensor | None = None,
    foreground_threshold: float = 0.035,
    change_threshold: float | None = None,
) -> dict[str, float]:
    pred = pred.detach().float().clamp(0.0, 1.0).cpu()
    target = target.detach().float().clamp(0.0, 1.0).cpu()
    mse = torch.mean((pred - target) ** 2).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    psnr = 99.0 if mse <= 1e-12 else float(-10.0 * torch.log10(torch.tensor(mse)).item())
    out = {"mse_video": float(mse), "mae_video": float(mae), "psnr_video": float(psnr)}
    out["edge_mae_video"] = float(F.l1_loss(sobel_edges(pred), sobel_edges(target)).item())

    if background is not None:
        bg = background.detach().float().clamp(0.0, 1.0).cpu()
        target_residual = target - bg.unsqueeze(1)
        pred_residual = pred - bg.unsqueeze(1)
        mask = foreground_mask(target_residual, foreground_threshold)
        fg_pixels = float(mask.sum().item())
        out["residual_l1_video"] = float(F.l1_loss(pred_residual, target_residual).item())
        out["fg_pixel_count"] = fg_pixels
        out["fg_coverage"] = float(mask.mean().item())
        out["fg_mse_video"] = float(masked_mean((pred - target) ** 2, mask).item()) if fg_pixels > 0 else 0.0
        out["fg_mae_video"] = float(masked_mean(torch.abs(pred - target), mask).item()) if fg_pixels > 0 else 0.0
        if pred_mask is not None:
            pm = pred_mask.detach().float().cpu()
            pred_binary = (pm > 0.5).float()
        else:
            pred_binary = foreground_mask(pred_residual, foreground_threshold)
        union = ((pred_binary + mask) > 0).float().sum().item()
        inter = (pred_binary * mask).sum().item()
        out["bubble_mask_iou"] = float(inter / union) if union > 0 else 1.0
        out["pred_fg_coverage"] = float(pred_binary.mean().item())
    if previous_frame is not None:
        prev = previous_frame.detach().float().clamp(0.0, 1.0).cpu()
        target_delta = target - prev.unsqueeze(1)
        pred_delta = pred - prev.unsqueeze(1)
        threshold = foreground_threshold if change_threshold is None else float(change_threshold)
        change_mask = foreground_mask(target_delta, threshold)
        change_pixels = float(change_mask.sum().item())
        out["change_coverage"] = float(change_mask.mean().item())
        out["change_l1_video"] = float(F.l1_loss(pred_delta, target_delta).item())
        out["change_fg_mae_video"] = (
            float(masked_mean(torch.abs(pred_delta - target_delta), change_mask).item()) if change_pixels > 0 else 0.0
        )
        out["change_edge_mae_video"] = float(F.l1_loss(sobel_edges(pred_delta), sobel_edges(target_delta)).item())
    return out


def sobel_edges(video: torch.Tensor) -> torch.Tensor:
    bsz, frames, channels, height, width = video.shape
    x = video.reshape(bsz * frames * channels, 1, height, width)
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=video.device,
        dtype=video.dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=video.device,
        dtype=video.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8).reshape(bsz, frames, channels, height, width)


def laplacian_edges(video: torch.Tensor) -> torch.Tensor:
    bsz, frames, channels, height, width = video.shape
    x = video.reshape(bsz * frames * channels, 1, height, width)
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=video.device,
        dtype=video.dtype,
    ).view(1, 1, 3, 3)
    return F.conv2d(x, kernel, padding=1).reshape(bsz, frames, channels, height, width)


def residual_loss(
    pred_residual: torch.Tensor,
    target_residual: torch.Tensor,
    background: torch.Tensor,
    target_video: torch.Tensor,
    roi: torch.Tensor,
    previous_frame: torch.Tensor | None = None,
    pred_mask: torch.Tensor | None = None,
    pred_mask_logits: torch.Tensor | None = None,
    prediction_base: str = "background",
    l1_weight: float = 0.5,
    residual_weight: float = 0.5,
    temporal_weight: float = 0.2,
    edge_weight: float = 0.05,
    laplacian_weight: float = 0.0,
    foreground_weight: float = 4.0,
    foreground_loss_weight: float = 0.0,
    foreground_threshold: float = 0.035,
    roi_weight: float = 1.0,
    residual_target_scale: float = 1.0,
    mask_bce_weight: float = 0.0,
    mask_dice_weight: float = 0.0,
    change_loss_weight: float = 0.0,
    change_threshold: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if prediction_base not in {"background", "previous_frame"}:
        raise ValueError(f"prediction_base must be 'background' or 'previous_frame', got {prediction_base!r}")
    base_frame = background
    if prediction_base == "previous_frame":
        if previous_frame is None:
            raise ValueError("previous_frame is required when prediction_base='previous_frame'")
        base_frame = previous_frame
        target_residual = target_video - previous_frame.detach().unsqueeze(1)

    target_scaled_residual = target_residual * float(residual_target_scale)
    pred_effective_residual = pred_residual if pred_mask is None else pred_residual * pred_mask
    pred_video = compose_video(
        background,
        pred_residual,
        clamp=False,
        mask=pred_mask,
        residual_target_scale=residual_target_scale,
        base_frame=base_frame,
    )
    target_mask = foreground_mask(target_residual, foreground_threshold)
    fg = target_residual.detach().abs().mean(dim=2, keepdim=True).clamp(0.0, 0.25) * 4.0
    weights = 1.0 + float(foreground_weight) * (target_mask + fg) + float(roi_weight) * roi.unsqueeze(1)

    mse = ((pred_video - target_video) ** 2 * weights).mean()
    l1 = (torch.abs(pred_video - target_video) * weights).mean()
    residual_l1 = F.l1_loss(pred_effective_residual, target_scaled_residual)
    fg_mse = masked_mean((pred_video - target_video) ** 2, target_mask)
    fg_l1 = masked_mean(torch.abs(pred_video - target_video), target_mask)
    if target_video.shape[1] > 1:
        temporal = F.l1_loss(pred_video[:, 1:] - pred_video[:, :-1], target_video[:, 1:] - target_video[:, :-1])
    else:
        temporal = pred_video.new_tensor(0.0)
    edge = F.l1_loss(sobel_edges(pred_video), sobel_edges(target_video))
    laplacian = F.l1_loss(laplacian_edges(pred_video), laplacian_edges(target_video))
    threshold = foreground_threshold if change_threshold is None else float(change_threshold)
    change_mask = foreground_mask(target_residual, threshold)
    change_l1 = masked_mean(torch.abs(pred_effective_residual - target_scaled_residual), change_mask)
    change_edge = F.l1_loss(sobel_edges(pred_effective_residual), sobel_edges(target_scaled_residual))
    if pred_mask_logits is not None:
        bce = F.binary_cross_entropy_with_logits(pred_mask_logits, target_mask)
        dice = dice_loss_from_logits(pred_mask_logits, target_mask)
    else:
        bce = pred_video.new_tensor(0.0)
        dice = pred_video.new_tensor(0.0)
    loss = (
        mse
        + float(l1_weight) * l1
        + float(residual_weight) * residual_l1
        + float(foreground_loss_weight) * (fg_mse + fg_l1)
        + float(temporal_weight) * temporal
        + float(edge_weight) * edge
        + float(laplacian_weight) * laplacian
        + float(mask_bce_weight) * bce
        + float(mask_dice_weight) * dice
        + float(change_loss_weight) * (change_l1 + 0.25 * change_edge)
    )
    parts = {
        "loss": float(loss.detach().cpu().item()),
        "mse": float(mse.detach().cpu().item()),
        "l1": float(l1.detach().cpu().item()),
        "residual_l1": float(residual_l1.detach().cpu().item()),
        "fg_mse": float(fg_mse.detach().cpu().item()),
        "fg_l1": float(fg_l1.detach().cpu().item()),
        "temporal": float(temporal.detach().cpu().item()),
        "edge": float(edge.detach().cpu().item()),
        "laplacian": float(laplacian.detach().cpu().item()),
        "mask_bce": float(bce.detach().cpu().item()),
        "mask_dice": float(dice.detach().cpu().item()),
        "fg_coverage": float(target_mask.detach().mean().cpu().item()),
        "change_l1": float(change_l1.detach().cpu().item()),
        "change_edge": float(change_edge.detach().cpu().item()),
        "change_coverage": float(change_mask.detach().mean().cpu().item()),
    }
    return loss, parts
