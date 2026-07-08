from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from flow_residual.dataset import ChunkResidualDataset, collate_fn
from flow_residual.metrics import aggregate_metrics, compose_video, video_metrics
from ldm_residual.vae import make_residual_frame_vae
from scripts.train_ldm_residual import target_to_pixels, to_01_pixels
from svd_audio_control.video_io import save_video_gif


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
    parser.add_argument("--train_manifest", default="manifests_csv_new/train.jsonl")
    parser.add_argument("--val_manifest", default="manifests_csv_new/val.jsonl")
    parser.add_argument("--train_cache_dir", default="cache/svd_csv8_residual_new/train")
    parser.add_argument("--val_cache_dir", default="cache/svd_csv8_residual_new/val")
    parser.add_argument("--output_dir", default="outputs/residual_vae_csv8_fromscratch_c128_rawamp")
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--chunk_frames", type=int, default=8)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--video_fps", type=int, default=100)
    parser.add_argument("--audio_sample_rate", type=int, default=1_000_000)
    parser.add_argument("--prior_mode", choices=["global", "per_class", "none"], default="none")
    parser.add_argument("--roi_mode", choices=["normal", "none"], default="normal")
    parser.add_argument("--audio_normalize", type=lambda v: str(v).lower() not in {"false", "0", "no"}, default=False)
    parser.add_argument("--audio_clamp_value", type=float, default=10.0)
    parser.add_argument("--residual_visual_scale", type=float, default=2.0)
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--vae_base_channels", type=int, default=64)
    parser.add_argument("--vae_downsample_factor", type=int, default=8)
    parser.add_argument("--vae_scaling_factor", type=float, default=1.0)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--max_train_steps", type=int, default=30000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--recon_l1_weight", type=float, default=1.0)
    parser.add_argument("--recon_mse_weight", type=float, default=0.25)
    parser.add_argument("--edge_weight", type=float, default=0.1)
    parser.add_argument("--kl_weight", type=float, default=1e-6)
    parser.add_argument("--foreground_threshold", type=float, default=0.04)
    parser.add_argument("--checkpointing_steps", type=int, default=2500)
    parser.add_argument("--validation_steps", type=int, default=1000)
    parser.add_argument("--num_val_batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=4234)
    args = parser.parse_args()
    return apply_config(args, load_json_config(args.config))


def dtype_from_precision(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def edge_map(x: torch.Tensor) -> torch.Tensor:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return F.pad(dx.abs(), (0, 1, 0, 0)) + F.pad(dy.abs(), (0, 0, 0, 1))


def make_model(args: argparse.Namespace) -> torch.nn.Module:
    return make_residual_frame_vae(
        latent_channels=args.latent_channels,
        base_channels=args.vae_base_channels,
        downsample_factor=args.vae_downsample_factor,
        scaling_factor=args.vae_scaling_factor,
    )


def compute_loss(args: argparse.Namespace, model: torch.nn.Module, batch: dict[str, Any], device: torch.device, use_amp: bool) -> tuple[torch.Tensor, dict[str, float]]:
    target = target_to_pixels(args, batch).to(device)
    bsz, frames, channels, height, width = target.shape
    flat = target.reshape(bsz * frames, channels, height, width)
    amp_dtype = dtype_from_precision(args.mixed_precision)
    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        dist = model.encode(flat).latent_dist
        z = dist.sample()
        recon = model.decode(z).sample
        l1 = (recon - flat).abs().mean()
        mse = F.mse_loss(recon, flat)
        edge = (edge_map(recon) - edge_map(flat)).abs().mean()
        kl = dist.kl().mean()
        loss = (
            float(args.recon_l1_weight) * l1
            + float(args.recon_mse_weight) * mse
            + float(args.edge_weight) * edge
            + float(args.kl_weight) * kl
        )
    parts = {
        "vae_l1": float(l1.detach().cpu().item()),
        "vae_mse": float(mse.detach().cpu().item()),
        "vae_edge": float(edge.detach().cpu().item()),
        "vae_kl": float(kl.detach().cpu().item()),
        "vae_loss": float(loss.detach().cpu().item()),
    }
    return loss, parts


@torch.no_grad()
def reconstruct_video(args: argparse.Namespace, model: torch.nn.Module, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    target = target_to_pixels(args, batch).to(device)
    bsz, frames, channels, height, width = target.shape
    flat = target.reshape(bsz * frames, channels, height, width)
    dist = model.encode(flat).latent_dist
    recon = model.decode(dist.mode()).sample.reshape(bsz, frames, channels, height, width)
    residual_vis = to_01_pixels(recon)
    residual = (residual_vis - 0.5) / max(float(args.residual_visual_scale), 1e-6)
    background = batch["background"].to(device)
    return compose_video(background, residual, clamp=True)


@torch.no_grad()
def validate(args: argparse.Namespace, model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    losses: list[dict[str, float]] = []
    first_layout: torch.Tensor | None = None
    for i, batch in enumerate(loader):
        if int(args.num_val_batches) > 0 and i >= int(args.num_val_batches):
            break
        loss, parts = compute_loss(args, model, batch, device, use_amp=False)
        parts["vae_loss"] = float(loss.detach().cpu().item())
        losses.append(parts)
        pred = reconstruct_video(args, model, batch, device)
        target = batch["pixel_values"].to(device)
        background = batch["background"].to(device)
        rows.append(video_metrics(pred, target, background=background, foreground_threshold=args.foreground_threshold))
        if first_layout is None:
            bg = background[0].unsqueeze(0).expand_as(target[0])
            first_layout = torch.cat([target[0].cpu(), pred[0].cpu(), bg.cpu()], dim=-1)
    out = aggregate_metrics(rows)
    if losses:
        for key in losses[0]:
            out[f"val_{key}"] = float(sum(row[key] for row in losses) / len(losses))
    model.train()
    return out, first_layout


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, args: argparse.Namespace, step: int, best_metric: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "args": vars(args),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_metric": float(best_metric),
        },
        path,
    )


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


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

    model = make_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.adam_weight_decay)
    scaler = GradScaler(enabled=(use_amp and args.mixed_precision == "fp16"))
    global_step = 0
    best_metric = math.inf
    if args.resume_checkpoint:
        ckpt = load_checkpoint(args.resume_checkpoint, model, optimizer)
        global_step = int(ckpt.get("step", 0))
        best_metric = float(ckpt.get("best_metric", math.inf))

    audio_kwargs = dict(
        audio_normalize=bool(getattr(args, "audio_normalize", False)),
        audio_clamp_value=(float(args.audio_clamp_value) if float(getattr(args, "audio_clamp_value", 0.0)) > 0.0 else None),
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
        prior_mode=args.prior_mode,
        roi_mode=args.roi_mode,
        fixed_starts_per_clip=[0, 30, 60],
        **audio_kwargs,
    )
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    train_iter = iter(train_loader)

    progress = tqdm(total=int(args.max_train_steps), initial=global_step)
    while global_step < int(args.max_train_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, parts = compute_loss(args, model, batch, device, use_amp=use_amp)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()

        global_step += 1
        if global_step % 10 == 0:
            progress.set_postfix(loss=f"{float(loss.detach().cpu().item()):.4f}")
        progress.update(1)

        if global_step % int(args.checkpointing_steps) == 0:
            save_checkpoint(output_dir / "checkpoints" / f"checkpoint-{global_step:06d}.pt", model, optimizer, args, global_step, best_metric)

        if global_step % int(args.validation_steps) == 0 or global_step == int(args.max_train_steps):
            val, layout = validate(args, model, val_loader, device)
            val.update({f"train_{k}": v for k, v in parts.items()})
            val["step"] = global_step
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(val, ensure_ascii=False) + "\n")
            if layout is not None:
                save_video_gif(layout, output_dir / "recon_preview.gif", fps=10)
            metric = float(val.get("val_vae_l1", val.get("vae_l1", math.inf)))
            if metric < best_metric:
                best_metric = metric
                save_checkpoint(output_dir / "best.pt", model, optimizer, args, global_step, best_metric)
            save_checkpoint(output_dir / "last.pt", model, optimizer, args, global_step, best_metric)

    progress.close()
    save_checkpoint(output_dir / "last.pt", model, optimizer, args, global_step, best_metric)


if __name__ == "__main__":
    main()
