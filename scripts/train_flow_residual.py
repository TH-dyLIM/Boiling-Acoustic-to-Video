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

from flow_residual.dataset import ChunkResidualDataset, collate_fn
from flow_residual.flow import (
    euler_sample,
    flow_matching_loss,
    foreground_weight,
    make_noised,
    patch_edge_smoothness,
    sample_flow_time,
)
from flow_residual.metrics import compose_video, video_metrics, aggregate_metrics
from flow_residual.model import FlowResidualUNet
from flow_residual.token_model import DiTVideoTokenFlowTransformer, VideoTokenFlowTransformer


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
    parser.add_argument("--prior_path", default="")
    parser.add_argument("--prior_mode", choices=["per_class", "global", "none"], default="per_class")
    parser.add_argument("--roi_mode", choices=["normal", "none"], default="normal")
    parser.add_argument("--background_mode", choices=["normal", "none"], default="normal")
    parser.add_argument("--prediction_mode", choices=["residual", "video"], default="residual")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "flow_residual_chunk"))
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--chunk_frames", type=int, default=12)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--video_fps", type=int, default=100)
    parser.add_argument("--audio_sample_rate", type=int, default=1_000_000)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=40000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--model_type", choices=["unet3d", "token_transformer", "dit_token_transformer"], default="unet3d")
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--audio_dim", type=int, default=64)
    parser.add_argument("--audio_tokens", type=int, default=16)
    parser.add_argument("--scalar_feature_dim", type=int, default=6)
    parser.add_argument("--physics_dim", type=int, default=3)
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--cond_dim", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--transformer_dim", type=int, default=384)
    parser.add_argument("--transformer_depth", type=int, default=8)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--transformer_checkpoint", action="store_true")
    parser.add_argument(
        "--use_refine_head",
        type=lambda v: str(v).lower() not in {"false", "0", "no"},
        default=False,
        help="DiT-only: add a zero-init Conv3d refinement head after unpatchify to smooth patch seams.",
    )
    parser.add_argument(
        "--patch_tv_weight",
        type=float,
        default=0.0,
        help="DiT-only: weight on patch-boundary squared-diff loss applied to predicted velocity.",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--residual_scale", type=float, default=2.0)
    parser.add_argument("--use_prev_frame", action="store_true")
    parser.add_argument("--cond_dropout_prob", type=float, default=0.1)
    parser.add_argument("--time_schedule", choices=["uniform", "logit_normal"], default="logit_normal")
    parser.add_argument("--foreground_threshold", type=float, default=0.04)
    parser.add_argument("--fg_weight", type=float, default=4.0)
    parser.add_argument("--roi_weight", type=float, default=1.0)
    parser.add_argument("--audio_normalize", type=lambda v: str(v).lower() not in {"false", "0", "no"}, default=True)
    parser.add_argument("--audio_clamp_value", type=float, default=0.0)
    parser.add_argument("--use_stft", type=lambda v: str(v).lower() not in {"false", "0", "no"}, default=False)
    parser.add_argument("--stft_n_fft", type=int, default=2048)
    parser.add_argument("--stft_hop_length", type=int, default=1024)
    parser.add_argument("--stft_n_freq_bins", type=int, default=64)
    parser.add_argument("--stft_fmin", type=float, default=100.0)
    parser.add_argument("--stft_fmax", type=float, default=0.0)
    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    parser.add_argument("--validation_steps", type=int, default=1000)
    parser.add_argument("--num_val_batches", type=int, default=4)
    parser.add_argument("--val_inference_steps", type=int, default=20)
    parser.add_argument("--val_cfg_scale", type=float, default=0.0)
    parser.add_argument("--best_metric_name", default="val_void_fraction_mae_mean")
    parser.add_argument("--distribution_void_weight", type=float, default=1.0)
    parser.add_argument("--distribution_spatial_weight", type=float, default=0.0)
    parser.add_argument("--distribution_nucleation_weight", type=float, default=0.0)
    parser.add_argument("--distribution_departure_weight", type=float, default=0.0)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    return apply_config(args, load_json_config(args.config))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(args: argparse.Namespace) -> FlowResidualUNet:
    if getattr(args, "model_type", "unet3d") == "dit_token_transformer":
        return DiTVideoTokenFlowTransformer(
            chunk_frames=args.chunk_frames,
            resolution=args.resolution,
            patch_size=args.patch_size,
            hidden_dim=args.transformer_dim,
            depth=args.transformer_depth,
            num_heads=args.transformer_heads,
            mlp_ratio=args.transformer_mlp_ratio,
            audio_dim=args.audio_dim,
            audio_tokens=args.audio_tokens,
            scalar_feature_dim=args.scalar_feature_dim,
            physics_dim=args.physics_dim,
            time_dim=args.time_dim,
            cond_dim=args.cond_dim,
            dropout=args.dropout,
            residual_scale=args.residual_scale,
            use_prev_frame=args.use_prev_frame,
            use_checkpoint=args.transformer_checkpoint,
            use_refine_head=bool(getattr(args, "use_refine_head", False)),
        )
    if getattr(args, "model_type", "unet3d") == "token_transformer":
        return VideoTokenFlowTransformer(
            chunk_frames=args.chunk_frames,
            resolution=args.resolution,
            patch_size=args.patch_size,
            hidden_dim=args.transformer_dim,
            depth=args.transformer_depth,
            num_heads=args.transformer_heads,
            mlp_ratio=args.transformer_mlp_ratio,
            audio_dim=args.audio_dim,
            audio_tokens=args.audio_tokens,
            scalar_feature_dim=args.scalar_feature_dim,
            physics_dim=args.physics_dim,
            time_dim=args.time_dim,
            cond_dim=args.cond_dim,
            dropout=args.dropout,
            residual_scale=args.residual_scale,
            use_prev_frame=args.use_prev_frame,
            use_checkpoint=args.transformer_checkpoint,
        )
    return FlowResidualUNet(
        chunk_frames=args.chunk_frames,
        base_channels=args.base_channels,
        audio_dim=args.audio_dim,
        audio_tokens=args.audio_tokens,
        scalar_feature_dim=args.scalar_feature_dim,
        physics_dim=args.physics_dim,
        time_dim=args.time_dim,
        cond_dim=args.cond_dim,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
        use_prev_frame=args.use_prev_frame,
        use_stft=bool(getattr(args, "use_stft", False)),
        stft_freq_bins=int(getattr(args, "stft_n_freq_bins", 64)),
    )


def model_background(args: argparse.Namespace, background: torch.Tensor) -> torch.Tensor:
    if getattr(args, "background_mode", "normal") == "none":
        return torch.zeros_like(background)
    return background


def flow_target(args: argparse.Namespace, target_video: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    scale = float(args.residual_scale)
    if getattr(args, "prediction_mode", "residual") == "video":
        return target_video * scale
    return (target_video - background.unsqueeze(1)) * scale


def compose_prediction(args: argparse.Namespace, sampled: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    scale = max(float(args.residual_scale), 1e-6)
    if getattr(args, "prediction_mode", "residual") == "video":
        return (sampled / scale).clamp(0.0, 1.0)
    return compose_video(background, sampled / scale, clamp=True)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
    best_metric: float,
    stale_validations: int = 0,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "args": vars(args),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_metric": float(best_metric),
            "stale_validations": int(stale_validations),
        },
        path,
    )


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys in {path}: {unexpected}")
    if optimizer is not None and "optimizer" in ckpt and not missing:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def compute_train_loss(
    args: argparse.Namespace,
    model: FlowResidualUNet,
    batch: dict[str, Any],
    device: torch.device,
    use_amp: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    background = batch["background"].to(device)
    background_cond = model_background(args, background)
    roi = batch["roi"].to(device)
    prior = batch["nucleation_prior"].to(device)
    prev_last = batch["prev_last_frame"].to(device)
    audio = batch["audio"].to(device)
    scalar_features = batch["audio_features"].to(device)
    physics = batch["physics"].to(device)
    target_video = batch["pixel_values"].to(device)
    target_residual_unscaled = batch["residual"].to(device)
    target = flow_target(args, target_video, background)
    audio_stft = batch["audio_stft"].to(device) if bool(getattr(args, "use_stft", False)) else None

    bsz = target.shape[0]
    time = sample_flow_time(bsz, device, args.time_schedule)
    noisy, velocity_target = make_noised(target, time)
    if args.cond_dropout_prob > 0.0:
        cond_dropout_mask = (torch.rand(bsz, device=device) > float(args.cond_dropout_prob)).float()
    else:
        cond_dropout_mask = None

    fg_weights = foreground_weight(
        target_residual_unscaled,
        roi,
        threshold=args.foreground_threshold,
        base_weight=1.0,
        fg_weight=args.fg_weight,
        roi_weight=args.roi_weight,
    )

    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        pred_velocity = model(
            noisy,
            time,
            background_cond,
            roi,
            prior,
            prev_last if args.use_prev_frame else None,
            audio,
            scalar_features,
            physics,
            cond_dropout_mask=cond_dropout_mask,
            audio_stft=audio_stft,
        )
        loss, parts = flow_matching_loss(pred_velocity, velocity_target, foreground_weight_map=fg_weights)
        tv_weight = float(getattr(args, "patch_tv_weight", 0.0))
        if tv_weight > 0.0:
            edge_loss = patch_edge_smoothness(pred_velocity, int(args.patch_size))
            loss = loss + tv_weight * edge_loss
            parts["patch_edge_tv"] = float(edge_loss.detach().cpu().item())
            parts["patch_edge_tv_weighted"] = float((tv_weight * edge_loss).detach().cpu().item())
    return loss, parts


def distribution_score(args: argparse.Namespace, metrics: dict[str, float]) -> float:
    """Composite validation score for choosing checkpoints by distribution quality.

    Raw count/frequency errors are normalized by target means so they can be
    mixed with void fraction and spatial density errors.
    """

    void_mae = float(metrics.get("void_fraction_mae_mean", math.inf))
    spatial_l1 = float(metrics.get("spatial_density_l1_mean", 0.0))
    nuc_mae = float(metrics.get("nucleation_count_mae_mean", 0.0))
    nuc_target = max(float(metrics.get("target_nucleation_count_mean_mean", 1.0)), 1e-6)
    dep_mae = float(metrics.get("departure_freq_mae_mean", 0.0))
    dep_target = max(float(metrics.get("target_departure_freq_mean_mean", 1.0)), 1e-6)
    return (
        float(args.distribution_void_weight) * void_mae
        + float(args.distribution_spatial_weight) * spatial_l1
        + float(args.distribution_nucleation_weight) * (nuc_mae / nuc_target)
        + float(args.distribution_departure_weight) * (dep_mae / dep_target)
    )


@torch.no_grad()
def validate(
    args: argparse.Namespace,
    model: FlowResidualUNet,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    use_stft = bool(getattr(args, "use_stft", False))
    for i, batch in enumerate(loader):
        if int(args.num_val_batches) > 0 and i >= int(args.num_val_batches):
            break
        background = batch["background"].to(device)
        background_cond = model_background(args, background)
        roi = batch["roi"].to(device)
        prior = batch["nucleation_prior"].to(device)
        prev_last = batch["prev_last_frame"].to(device)
        audio = batch["audio"].to(device)
        scalar_features = batch["audio_features"].to(device)
        physics = batch["physics"].to(device)
        target_video = batch["pixel_values"].to(device)
        audio_stft = batch["audio_stft"].to(device) if use_stft else None
        bsz = background.shape[0]
        shape = (bsz, args.chunk_frames, 3, args.resolution, args.resolution)

        def velocity_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                return model(
                    x,
                    t,
                    background_cond,
                    roi,
                    prior,
                    prev_last if args.use_prev_frame else None,
                    audio,
                    scalar_features,
                    physics,
                    audio_stft=audio_stft,
                ).float()

        if float(args.val_cfg_scale) > 0.0:
            zero_mask = torch.zeros(bsz, device=device)

            def uncond_velocity_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    return model(
                        x,
                        t,
                        background_cond,
                        roi,
                        prior,
                        prev_last if args.use_prev_frame else None,
                        audio,
                        scalar_features,
                        physics,
                        cond_dropout_mask=zero_mask,
                        audio_stft=audio_stft,
                    ).float()
        else:
            uncond_velocity_fn = None

        sampled = euler_sample(
            velocity_fn,
            shape,
            device,
            num_steps=int(args.val_inference_steps),
            cfg_scale=float(args.val_cfg_scale),
            uncond_velocity_fn=uncond_velocity_fn,
            seed=int(args.seed) + i,
        )
        pred_video = compose_prediction(args, sampled, background)
        rows.append(
            video_metrics(
                pred_video,
                target_video,
                background=background,
                foreground_threshold=args.foreground_threshold,
            )
        )

    train_eval_loss, _ = compute_train_loss(args, model, next(iter(loader)), device, use_amp)
    out = aggregate_metrics(rows)
    out["val_fm_loss"] = float(train_eval_loss.detach().cpu().item())
    out["val_void_fraction_mae_mean"] = float(out.get("void_fraction_mae_mean", math.inf))
    out["val_distribution_score"] = float(distribution_score(args, out))
    model.train()
    return {f"val_{k}" if not k.startswith("val_") else k: v for k, v in out.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.mixed_precision != "no"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume_checkpoint:
        for stale_name in ("metrics.jsonl", "best.pt", "last.pt"):
            stale = output_dir / stale_name
            if stale.exists():
                stale.unlink()
    (output_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    audio_kwargs = dict(
        audio_normalize=bool(getattr(args, "audio_normalize", True)),
        audio_clamp_value=(float(args.audio_clamp_value) if float(getattr(args, "audio_clamp_value", 0.0)) > 0.0 else None),
        use_stft=bool(getattr(args, "use_stft", False)),
        stft_n_fft=int(getattr(args, "stft_n_fft", 2048)),
        stft_hop_length=int(getattr(args, "stft_hop_length", 1024)),
        stft_n_freq_bins=int(getattr(args, "stft_n_freq_bins", 64)),
        stft_fmin=float(getattr(args, "stft_fmin", 100.0)),
        stft_fmax=(float(args.stft_fmax) if float(getattr(args, "stft_fmax", 0.0)) > 0.0 else None),
    )
    train_ds = ChunkResidualDataset(
        args.train_manifest,
        resolution=args.resolution,
        chunk_frames=args.chunk_frames,
        frame_stride=args.frame_stride,
        audio_sample_rate=args.audio_sample_rate,
        video_fps=args.video_fps,
        random_start=True,
        cache_dir=args.train_cache_dir or None,
        prior_path=args.prior_path or None,
        prior_mode=args.prior_mode,
        roi_mode=args.roi_mode,
        **audio_kwargs,
    )
    val_ds = ChunkResidualDataset(
        args.val_manifest,
        resolution=args.resolution,
        chunk_frames=args.chunk_frames,
        frame_stride=args.frame_stride,
        audio_sample_rate=args.audio_sample_rate,
        video_fps=args.video_fps,
        random_start=False,
        cache_dir=args.val_cache_dir or None,
        prior_path=args.prior_path or None,
        prior_mode=args.prior_mode,
        roi_mode=args.roi_mode,
        fixed_starts_per_clip=[0, 30, 60],
        **audio_kwargs,
    )
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = make_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.adam_weight_decay)
    global_step = 0
    best_metric = math.inf
    stale_validations = 0
    if args.resume_checkpoint:
        ckpt = load_checkpoint(args.resume_checkpoint, model, optimizer)
        global_step = int(ckpt.get("step", 0))
        best_metric = float(ckpt.get("best_metric", math.inf))
        stale_validations = int(ckpt.get("stale_validations", 0))

    scaler = GradScaler(device="cuda", enabled=use_amp and args.mixed_precision == "fp16")
    progress = tqdm(total=args.max_train_steps, initial=global_step)
    model.train()
    while global_step < int(args.max_train_steps):
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, parts = compute_train_loss(args, model, batch, device, use_amp)
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{parts['fm_weighted']:.4f}")

            if global_step % int(args.checkpointing_steps) == 0:
                save_checkpoint(
                    output_dir / "checkpoints" / f"checkpoint-{global_step:06d}.pt",
                    model,
                    optimizer,
                    args,
                    global_step,
                    best_metric,
                    stale_validations,
                )

            if global_step % int(args.validation_steps) == 0:
                val = validate(args, model, val_loader, device, use_amp)
                metric = float(val.get(args.best_metric_name, val.get("val_fm_loss", math.inf)))
                row = {"step": global_step, **parts, **val}
                with (output_dir / "metrics.jsonl").open("a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                improved = metric < best_metric - float(args.early_stopping_min_delta)
                if improved:
                    best_metric = metric
                    stale_validations = 0
                    save_checkpoint(output_dir / "best.pt", model, optimizer, args, global_step, best_metric, stale_validations)
                else:
                    stale_validations += 1
                if int(args.early_stopping_patience) > 0 and stale_validations >= int(args.early_stopping_patience):
                    row = {
                        "step": global_step,
                        "early_stop": True,
                        "best_metric_name": args.best_metric_name,
                        "best_metric": best_metric,
                    }
                    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8", newline="\n") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    global_step = int(args.max_train_steps)
                    break

            if global_step >= int(args.max_train_steps):
                break

    save_checkpoint(output_dir / "last.pt", model, optimizer, args, global_step, best_metric, stale_validations)
    progress.close()


if __name__ == "__main__":
    main()
