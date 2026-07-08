from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from residual_video.dataset import ResidualVideoManifestDataset, collate_fn
from residual_video.gan import (
    MultiScaleResidualPatchDiscriminator3D,
    ResidualPatchDiscriminator3D,
    discriminator_feature_matching_loss,
    hinge_discriminator_loss,
    hinge_generator_loss,
    logits_from_output,
    set_requires_grad,
)
from residual_video.metrics import compose_video, residual_loss, video_metrics
from residual_video.model import AudioResidualUNet


def load_json_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def apply_config(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    cli_keys = {item.lstrip("-").replace("-", "_") for item in sys.argv[1:] if item.startswith("--")}
    for key, value in config.items():
        if hasattr(args, key) and key not in cli_keys:
            setattr(args, key, value)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--train_manifest", default=str(PROJECT_ROOT / "manifests" / "train.jsonl"))
    parser.add_argument("--val_manifest", default=str(PROJECT_ROOT / "manifests" / "val.jsonl"))
    parser.add_argument("--train_cache_dir", default="")
    parser.add_argument("--val_cache_dir", default="")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "residual_video"))
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--video_fps", type=int, default=100)
    parser.add_argument("--audio_sample_rate", type=int, default=1_000_000)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=30000)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--audio_channels", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--residual_target_scale", type=float, default=1.0)
    parser.add_argument("--use_mask_head", action="store_true")
    parser.add_argument("--mask_bias_init", type=float, default=0.0)
    parser.add_argument("--physics_dim", type=int, default=3)
    parser.add_argument("--physics_channels", type=int, default=8)
    parser.add_argument("--audio_feature_dim", type=int, default=6)
    parser.add_argument("--audio_feature_channels", type=int, default=8)
    parser.add_argument("--use_previous_frame_condition", action="store_true")
    parser.add_argument("--prediction_base", choices=["background", "previous_frame"], default="background")
    parser.add_argument("--train_rollout_steps", type=int, default=1)
    parser.add_argument("--val_rollout_steps", type=int, default=1)
    parser.add_argument("--val_start_frames", default="")
    parser.add_argument("--audio_context_frames", type=int, default=1)
    parser.add_argument("--audio_context_future_frames", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--l1_weight", type=float, default=0.5)
    parser.add_argument("--residual_weight", type=float, default=0.5)
    parser.add_argument("--temporal_weight", type=float, default=0.2)
    parser.add_argument("--edge_weight", type=float, default=0.05)
    parser.add_argument("--laplacian_weight", type=float, default=0.0)
    parser.add_argument("--foreground_weight", type=float, default=4.0)
    parser.add_argument("--foreground_loss_weight", type=float, default=0.0)
    parser.add_argument("--foreground_threshold", type=float, default=0.035)
    parser.add_argument("--roi_weight", type=float, default=1.0)
    parser.add_argument("--mask_bce_weight", type=float, default=0.0)
    parser.add_argument("--mask_dice_weight", type=float, default=0.0)
    parser.add_argument("--change_loss_weight", type=float, default=0.0)
    parser.add_argument("--change_threshold", type=float, default=0.02)
    parser.add_argument("--patchgan_weight", type=float, default=0.0)
    parser.add_argument("--patchgan_lr", type=float, default=1e-4)
    parser.add_argument("--patchgan_base_channels", type=int, default=32)
    parser.add_argument("--patchgan_layers", type=int, default=4)
    parser.add_argument("--patchgan_warmup_steps", type=int, default=1000)
    parser.add_argument("--patchgan_train_every", type=int, default=1)
    parser.add_argument("--patchgan_roi_residual_weight", type=float, default=0.5)
    parser.add_argument("--patchgan_no_spectral_norm", action="store_true")
    parser.add_argument("--patchgan_multiscale", action="store_true")
    parser.add_argument("--patchgan_scales", default="1.0,0.5")
    parser.add_argument("--patchgan_include_edges", action="store_true")
    parser.add_argument("--patchgan_feature_matching_weight", type=float, default=0.0)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--num_val_batches", type=int, default=8)
    parser.add_argument("--best_metric_name", default="val_mse_video")
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    return apply_config(args, load_json_config(args.config))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(args: argparse.Namespace) -> AudioResidualUNet:
    return AudioResidualUNet(
        num_frames=args.num_frames,
        audio_channels=args.audio_channels,
        base_channels=args.base_channels,
        residual_scale=args.residual_scale,
        dropout=args.dropout,
        use_mask_head=getattr(args, "use_mask_head", False),
        mask_bias_init=getattr(args, "mask_bias_init", 0.0),
        physics_dim=getattr(args, "physics_dim", 3),
        physics_channels=getattr(args, "physics_channels", 8),
        audio_feature_dim=getattr(args, "audio_feature_dim", 6),
        audio_feature_channels=getattr(args, "audio_feature_channels", 8),
        use_previous_frame_condition=getattr(args, "use_previous_frame_condition", False),
    )


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
    best_metric: float,
    stale_validations: int = 0,
    discriminator: torch.nn.Module | None = None,
    d_optimizer: torch.optim.Optimizer | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "args": vars(args),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_metric": float(best_metric),
        "stale_validations": int(stale_validations),
    }
    if discriminator is not None:
        payload["discriminator"] = discriminator.state_dict()
    if d_optimizer is not None:
        payload["d_optimizer"] = d_optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    discriminator: torch.nn.Module | None = None,
    d_optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys in {path}: {unexpected}")
    if optimizer is not None and "optimizer" in ckpt and not missing:
        optimizer.load_state_dict(ckpt["optimizer"])
    if discriminator is not None and "discriminator" in ckpt:
        d_missing, d_unexpected = discriminator.load_state_dict(ckpt["discriminator"], strict=False)
        if d_unexpected:
            raise RuntimeError(f"Unexpected discriminator checkpoint keys in {path}: {d_unexpected}")
        if d_optimizer is not None and "d_optimizer" in ckpt and not d_missing:
            d_optimizer.load_state_dict(ckpt["d_optimizer"])
    return ckpt


def split_model_output(output: torch.Tensor | dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if isinstance(output, dict):
        return output["residual"], output.get("mask"), output.get("mask_logits")
    return output, None, None


def parse_start_frames(value: str) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    return values or (1.0,)


def compose_prediction(
    args: argparse.Namespace,
    background: torch.Tensor,
    pred_residual: torch.Tensor,
    pred_mask: torch.Tensor | None,
    previous_frame: torch.Tensor | None,
    clamp: bool,
) -> torch.Tensor:
    base_frame = previous_frame if args.prediction_base == "previous_frame" else None
    return compose_video(
        background,
        pred_residual,
        clamp=clamp,
        mask=pred_mask,
        residual_target_scale=args.residual_target_scale,
        base_frame=base_frame,
    )


def compute_batch(
    args: argparse.Namespace,
    model: AudioResidualUNet,
    batch: dict[str, Any],
    device: torch.device,
    use_amp: bool,
    training: bool,
    return_tensors: bool = False,
) -> tuple[torch.Tensor, dict[str, float]] | tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    background = batch["background"].to(device)
    roi = batch["roi"].to(device)
    audio = batch["audio"].to(device)
    previous_frame = batch["previous_frame"].to(device)
    audio_features = batch["audio_features"].to(device)
    physics = batch["physics"].to(device)
    target_video = batch["pixel_values"].to(device)
    target_residual = batch["residual"].to(device)
    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        pred_residual, pred_mask, pred_mask_logits = split_model_output(
            model(
                background,
                roi,
                audio,
                physics=physics,
                audio_features=audio_features,
                previous_frame=previous_frame,
            )
        )
        loss, parts = residual_loss(
            pred_residual,
            target_residual,
            background,
            target_video,
            roi,
            previous_frame=previous_frame,
            pred_mask=pred_mask,
            pred_mask_logits=pred_mask_logits,
            prediction_base=args.prediction_base,
            l1_weight=args.l1_weight,
            residual_weight=args.residual_weight,
            temporal_weight=args.temporal_weight,
            edge_weight=args.edge_weight,
            laplacian_weight=args.laplacian_weight,
            foreground_weight=args.foreground_weight,
            foreground_loss_weight=args.foreground_loss_weight,
            foreground_threshold=args.foreground_threshold,
            roi_weight=args.roi_weight,
            residual_target_scale=args.residual_target_scale,
            mask_bce_weight=args.mask_bce_weight,
            mask_dice_weight=args.mask_dice_weight,
            change_loss_weight=args.change_loss_weight,
            change_threshold=args.change_threshold,
        )
        pred_video_for_gan = compose_prediction(
            args,
            background,
            pred_residual,
            pred_mask,
            previous_frame,
            clamp=False,
        )
    if not training:
        pred_video = pred_video_for_gan.clamp(0.0, 1.0)
        parts.update(
            video_metrics(
                pred_video,
                target_video,
                background=background,
                previous_frame=previous_frame,
                pred_mask=pred_mask,
                foreground_threshold=args.foreground_threshold,
                change_threshold=args.change_threshold,
            )
        )
    if return_tensors:
        return loss, parts, {
            "pred_video": pred_video_for_gan,
            "target_video": target_video,
            "background": background,
            "roi": roi,
        }
    return loss, parts


def compute_rollout_batch(
    args: argparse.Namespace,
    model: AudioResidualUNet,
    batch: dict[str, Any],
    device: torch.device,
    use_amp: bool,
    training: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    steps = int(args.train_rollout_steps if training else args.val_rollout_steps)
    if steps <= 1:
        return compute_batch(args, model, batch, device, use_amp, training)

    background = batch["background"].to(device)
    roi = batch["roi"].to(device)
    physics = batch["physics"].to(device)
    target_seq = batch["rollout_pixel_values"][:, :steps].to(device)
    audio_seq = batch["rollout_audio"][:, :steps].to(device)
    audio_feature_seq = batch["rollout_audio_features"][:, :steps].to(device)
    previous_frame = batch["rollout_previous_frames"][:, 0].to(device)
    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16

    losses = []
    part_rows: list[dict[str, float]] = []
    pred_frames = []
    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        for step in range(steps):
            target_video = target_seq[:, step : step + 1]
            target_residual = target_video - background.unsqueeze(1)
            pred_residual, pred_mask, pred_mask_logits = split_model_output(
                model(
                    background,
                    roi,
                    audio_seq[:, step],
                    physics=physics,
                    audio_features=audio_feature_seq[:, step],
                    previous_frame=previous_frame,
                )
            )
            loss, parts = residual_loss(
                pred_residual,
                target_residual,
                background,
                target_video,
                roi,
                previous_frame=previous_frame,
                pred_mask=pred_mask,
                pred_mask_logits=pred_mask_logits,
                prediction_base=args.prediction_base,
                l1_weight=args.l1_weight,
                residual_weight=args.residual_weight,
                temporal_weight=args.temporal_weight,
                edge_weight=args.edge_weight,
                laplacian_weight=args.laplacian_weight,
                foreground_weight=args.foreground_weight,
                foreground_loss_weight=args.foreground_loss_weight,
                foreground_threshold=args.foreground_threshold,
                roi_weight=args.roi_weight,
                residual_target_scale=args.residual_target_scale,
                mask_bce_weight=args.mask_bce_weight,
                mask_dice_weight=args.mask_dice_weight,
                change_loss_weight=args.change_loss_weight,
                change_threshold=args.change_threshold,
            )
            pred_video = compose_prediction(args, background, pred_residual, pred_mask, previous_frame, clamp=not training)
            pred_frame = pred_video[:, 0]
            losses.append(loss)
            part_rows.append(parts)
            pred_frames.append(pred_frame)
            previous_frame = pred_frame

    total_loss = torch.stack(losses).mean()
    keys = sorted({key for row in part_rows for key in row})
    out = {
        key: float(sum(row[key] for row in part_rows if key in row) / sum(1 for row in part_rows if key in row))
        for key in keys
    }
    out["loss"] = float(total_loss.detach().cpu().item())
    if not training:
        pred_seq = torch.stack(pred_frames, dim=1)
        out.update(
            video_metrics(
                pred_seq,
                target_seq,
                background=background,
                pred_mask=None,
                foreground_threshold=args.foreground_threshold,
                change_threshold=args.change_threshold,
            )
        )
    return total_loss, out


@torch.no_grad()
def validate(
    args: argparse.Namespace,
    model: AudioResidualUNet,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    rows = []
    for i, batch in enumerate(loader):
        if int(args.num_val_batches) > 0 and i >= int(args.num_val_batches):
            break
        _, parts = compute_rollout_batch(args, model, batch, device, use_amp, training=False)
        rows.append(parts)
    model.train()
    if not rows:
        return {"val_loss": math.nan, "val_mse_video": math.nan, "val_psnr_video": math.nan}
    keys = sorted({key for row in rows for key in row})
    out = {}
    for key in keys:
        vals = [row[key] for row in rows if key in row]
        out[f"val_{key}"] = float(sum(vals) / len(vals))
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.mixed_precision != "no"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume_checkpoint:
        for stale_name in ("metrics.jsonl", "best.pt", "last.pt"):
            stale_path = output_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()
    (output_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_ds = ResidualVideoManifestDataset(
        args.train_manifest,
        resolution=args.resolution,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        audio_sample_rate=args.audio_sample_rate,
        video_fps=args.video_fps,
        random_start=True,
        load_audio=True,
        rollout_steps=max(1, int(args.train_rollout_steps)),
        audio_context_frames=args.audio_context_frames,
        audio_context_future_frames=args.audio_context_future_frames,
        cache_dir=args.train_cache_dir or None,
    )
    val_ds = ResidualVideoManifestDataset(
        args.val_manifest,
        resolution=args.resolution,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        audio_sample_rate=args.audio_sample_rate,
        video_fps=args.video_fps,
        random_start=False,
        load_audio=True,
        fixed_start_frames=parse_start_frames(args.val_start_frames),
        rollout_steps=max(1, int(args.val_rollout_steps)),
        audio_context_frames=args.audio_context_frames,
        audio_context_future_frames=args.audio_context_future_frames,
        cache_dir=args.val_cache_dir or None,
    )
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = make_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.adam_weight_decay)
    use_patchgan = float(getattr(args, "patchgan_weight", 0.0)) > 0.0
    if use_patchgan and int(args.train_rollout_steps) > 1:
        raise ValueError("PatchGAN training currently supports train_rollout_steps=1. Use chunk/frame training, not AR rollout.")
    discriminator = None
    d_optimizer = None
    if use_patchgan:
        scales = parse_float_list(args.patchgan_scales)
        if bool(args.patchgan_multiscale) or len(scales) > 1:
            discriminator = MultiScaleResidualPatchDiscriminator3D(
                scales=scales,
                base_channels=args.patchgan_base_channels,
                num_layers=args.patchgan_layers,
                use_spectral_norm=not bool(args.patchgan_no_spectral_norm),
                roi_residual_weight=args.patchgan_roi_residual_weight,
                include_edges=bool(args.patchgan_include_edges),
            ).to(device)
        else:
            discriminator = ResidualPatchDiscriminator3D(
                base_channels=args.patchgan_base_channels,
                num_layers=args.patchgan_layers,
                use_spectral_norm=not bool(args.patchgan_no_spectral_norm),
                roi_residual_weight=args.patchgan_roi_residual_weight,
                include_edges=bool(args.patchgan_include_edges),
            ).to(device)
        d_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=float(args.patchgan_lr),
            betas=(0.5, 0.999),
            weight_decay=0.0,
        )
    global_step = 0
    best_metric = math.inf
    stale_validations = 0
    if args.resume_checkpoint:
        ckpt = load_checkpoint(args.resume_checkpoint, model, optimizer, discriminator, d_optimizer)
        global_step = int(ckpt.get("step", 0))
        best_metric = float(ckpt.get("best_metric", math.inf))
        stale_validations = int(ckpt.get("stale_validations", 0))

    scaler = GradScaler(device="cuda", enabled=use_amp and args.mixed_precision == "fp16")
    d_scaler = GradScaler(device="cuda", enabled=use_amp and args.mixed_precision == "fp16")
    progress = tqdm(total=args.max_train_steps, initial=global_step)
    model.train()
    if discriminator is not None:
        discriminator.train()
    while global_step < int(args.max_train_steps):
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if use_patchgan:
                assert discriminator is not None and d_optimizer is not None
                loss, parts, tensors = compute_batch(
                    args,
                    model,
                    batch,
                    device,
                    use_amp,
                    training=True,
                    return_tensors=True,
                )
                gan_active = global_step >= int(args.patchgan_warmup_steps)
                should_train_d = gan_active and (global_step % max(1, int(args.patchgan_train_every)) == 0)
                if should_train_d:
                    d_optimizer.zero_grad(set_to_none=True)
                    set_requires_grad(discriminator, True)
                    with autocast(
                        device_type="cuda",
                        dtype=torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16,
                        enabled=use_amp,
                    ):
                        real_logits = discriminator(
                            tensors["target_video"],
                            tensors["background"],
                            tensors["roi"],
                        )
                        fake_logits = discriminator(
                            tensors["pred_video"].detach(),
                            tensors["background"],
                            tensors["roi"],
                        )
                        d_loss = hinge_discriminator_loss(real_logits, fake_logits)
                    d_scaler.scale(d_loss).backward()
                    d_scaler.step(d_optimizer)
                    d_scaler.update()
                    parts["patchgan_d_loss"] = float(d_loss.detach().cpu().item())
                    parts["patchgan_real"] = float(
                        torch.stack([item.detach().mean() for item in logits_from_output(real_logits)]).mean().cpu().item()
                    )
                    parts["patchgan_fake"] = float(
                        torch.stack([item.detach().mean() for item in logits_from_output(fake_logits)]).mean().cpu().item()
                    )

                if gan_active:
                    set_requires_grad(discriminator, False)
                    with autocast(
                        device_type="cuda",
                        dtype=torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16,
                        enabled=use_amp,
                    ):
                        return_features = float(args.patchgan_feature_matching_weight) > 0.0
                        fake_logits_for_g = discriminator(
                            tensors["pred_video"],
                            tensors["background"],
                            tensors["roi"],
                            return_features=return_features,
                        )
                        adv_loss = hinge_generator_loss(fake_logits_for_g)
                        if return_features:
                            real_logits_for_fm = discriminator(
                                tensors["target_video"].detach(),
                                tensors["background"],
                                tensors["roi"],
                                return_features=True,
                            )
                            fm_loss = discriminator_feature_matching_loss(real_logits_for_fm, fake_logits_for_g)
                        else:
                            fm_loss = adv_loss.new_tensor(0.0)
                        total_loss = (
                            loss
                            + float(args.patchgan_weight) * adv_loss
                            + float(args.patchgan_feature_matching_weight) * fm_loss
                        )
                    parts["recon_loss"] = float(loss.detach().cpu().item())
                    parts["patchgan_g_loss"] = float(adv_loss.detach().cpu().item())
                    parts["patchgan_fm_loss"] = float(fm_loss.detach().cpu().item())
                    parts["loss"] = float(total_loss.detach().cpu().item())
                    loss = total_loss
                    set_requires_grad(discriminator, True)
                else:
                    parts["recon_loss"] = float(loss.detach().cpu().item())
                    parts["patchgan_g_loss"] = 0.0
                    parts["patchgan_d_loss"] = 0.0
                    parts["patchgan_fm_loss"] = 0.0
            else:
                loss, parts = compute_rollout_batch(args, model, batch, device, use_amp, training=True)
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{parts['loss']:.4f}", mse=f"{parts['mse']:.4f}")

            if global_step % int(args.checkpointing_steps) == 0:
                save_checkpoint(
                    output_dir / "checkpoints" / f"checkpoint-{global_step:06d}.pt",
                    model,
                    optimizer,
                    args,
                    global_step,
                    best_metric,
                    stale_validations,
                    discriminator=discriminator,
                    d_optimizer=d_optimizer,
                )

            if global_step % int(args.validation_steps) == 0:
                val = validate(args, model, val_loader, device, use_amp)
                metric = float(val.get(args.best_metric_name, val.get("val_loss", math.inf)))
                row = {"step": global_step, **parts, **val}
                with (output_dir / "metrics.jsonl").open("a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                improved = metric < best_metric - float(args.early_stopping_min_delta)
                if improved:
                    best_metric = metric
                    stale_validations = 0
                    save_checkpoint(
                        output_dir / "best.pt",
                        model,
                        optimizer,
                        args,
                        global_step,
                        best_metric,
                        stale_validations,
                        discriminator=discriminator,
                        d_optimizer=d_optimizer,
                    )
                else:
                    stale_validations += 1

                if int(args.early_stopping_patience) > 0 and stale_validations >= int(args.early_stopping_patience):
                    row = {
                        "step": global_step,
                        "early_stop": True,
                        "best_metric_name": args.best_metric_name,
                        "best_metric": best_metric,
                        "stale_validations": stale_validations,
                    }
                    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8", newline="\n") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    global_step = int(args.max_train_steps)
                    break

            if global_step >= int(args.max_train_steps):
                break

    save_checkpoint(
        output_dir / "last.pt",
        model,
        optimizer,
        args,
        global_step,
        best_metric,
        stale_validations,
        discriminator=discriminator,
        d_optimizer=d_optimizer,
    )
    progress.close()


if __name__ == "__main__":
    main()
