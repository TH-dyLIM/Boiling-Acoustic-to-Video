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
from flow_residual.flow import foreground_weight
from flow_residual.metrics import aggregate_metrics, compose_video, video_metrics
from ldm_residual.model import LatentConditionalUNet


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
    parser.add_argument("--pretrained_vae_name_or_path", default="pretrained/stabilityai__stable-video-diffusion-img2vid-xt")
    parser.add_argument("--vae_subfolder", default="vae")
    parser.add_argument("--vae_type", choices=["pretrained", "custom"], default="pretrained")
    parser.add_argument("--custom_vae_checkpoint", default="")
    parser.add_argument("--vae_base_channels", type=int, default=64)
    parser.add_argument("--vae_downsample_factor", type=int, default=8)
    parser.add_argument("--vae_scaling_factor", type=float, default=1.0)
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--train_manifest", default="manifests_csv_new/train.jsonl")
    parser.add_argument("--val_manifest", default="manifests_csv_new/val.jsonl")
    parser.add_argument("--train_cache_dir", default="cache/svd_csv8_residual_new/train")
    parser.add_argument("--val_cache_dir", default="cache/svd_csv8_residual_new/val")
    parser.add_argument("--prior_path", default="")
    parser.add_argument("--prior_mode", choices=["global", "per_class", "none"], default="none")
    parser.add_argument("--roi_mode", choices=["normal", "none"], default="normal")
    parser.add_argument("--output_dir", default="outputs/ldm_residual_csv8_noprior_c128_rawamp_bg_roi")
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--chunk_frames", type=int, default=8)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--video_fps", type=int, default=100)
    parser.add_argument("--audio_sample_rate", type=int, default=1_000_000)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=55000)
    parser.add_argument("--learning_rate", type=float, default=7e-5)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=128)
    parser.add_argument("--audio_dim", type=int, default=96)
    parser.add_argument("--audio_tokens", type=int, default=24)
    parser.add_argument("--scalar_feature_dim", type=int, default=6)
    parser.add_argument("--physics_dim", type=int, default=3)
    parser.add_argument("--time_dim", type=int, default=192)
    parser.add_argument("--cond_dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--use_prev_frame", action="store_true")
    parser.add_argument("--use_stft", type=lambda v: str(v).lower() not in {"false", "0", "no"}, default=False)
    parser.add_argument("--stft_n_fft", type=int, default=2048)
    parser.add_argument("--stft_hop_length", type=int, default=1024)
    parser.add_argument("--stft_n_freq_bins", type=int, default=64)
    parser.add_argument("--stft_fmin", type=float, default=100.0)
    parser.add_argument("--stft_fmax", type=float, default=0.0)
    parser.add_argument("--audio_normalize", type=lambda v: str(v).lower() not in {"false", "0", "no"}, default=False)
    parser.add_argument("--audio_clamp_value", type=float, default=10.0)
    parser.add_argument("--residual_visual_scale", type=float, default=2.0)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", default="scaled_linear")
    parser.add_argument("--prediction_type", choices=["epsilon", "v_prediction"], default="epsilon")
    parser.add_argument("--cond_dropout_prob", type=float, default=0.1)
    parser.add_argument("--foreground_threshold", type=float, default=0.04)
    parser.add_argument("--fg_weight", type=float, default=6.0)
    parser.add_argument("--roi_weight", type=float, default=1.0)
    parser.add_argument("--checkpointing_steps", type=int, default=2500)
    parser.add_argument("--validation_steps", type=int, default=1000)
    parser.add_argument("--num_val_batches", type=int, default=39)
    parser.add_argument("--val_inference_steps", type=int, default=30)
    parser.add_argument("--best_metric_name", default="val_distribution_score")
    parser.add_argument("--distribution_void_weight", type=float, default=1.0)
    parser.add_argument("--distribution_spatial_weight", type=float, default=0.8)
    parser.add_argument("--distribution_nucleation_weight", type=float, default=0.4)
    parser.add_argument("--distribution_departure_weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=3234)
    args = parser.parse_args()
    return apply_config(args, load_json_config(args.config))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_precision(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def to_vae_pixels(x01: torch.Tensor) -> torch.Tensor:
    return x01.float().mul(2.0).sub(1.0).clamp(-1.0, 1.0)


def to_01_pixels(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().add(1.0).div(2.0).clamp(0.0, 1.0)


def target_to_pixels(args: argparse.Namespace, batch: dict[str, Any]) -> torch.Tensor:
    scale = max(float(args.residual_visual_scale), 1e-6)
    residual_vis = (batch["residual"].float() * scale + 0.5).clamp(0.0, 1.0)
    return to_vae_pixels(residual_vis)


def decoded_to_video(args: argparse.Namespace, decoded: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    scale = max(float(args.residual_visual_scale), 1e-6)
    residual_vis = to_01_pixels(decoded)
    residual = (residual_vis - 0.5) / scale
    return compose_video(background, residual, clamp=True)


def load_vae(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    if getattr(args, "vae_type", "pretrained") == "custom":
        if not getattr(args, "custom_vae_checkpoint", ""):
            raise ValueError("custom_vae_checkpoint is required when vae_type=custom")
        from ldm_residual.vae import make_residual_frame_vae

        ckpt = torch.load(args.custom_vae_checkpoint, map_location="cpu")
        vae_args = ckpt.get("args", {})
        vae = make_residual_frame_vae(
            latent_channels=int(vae_args.get("latent_channels", args.latent_channels)),
            base_channels=int(vae_args.get("vae_base_channels", args.vae_base_channels)),
            downsample_factor=int(vae_args.get("vae_downsample_factor", args.vae_downsample_factor)),
            scaling_factor=float(vae_args.get("vae_scaling_factor", args.vae_scaling_factor)),
        )
        vae.load_state_dict(ckpt["model"], strict=True)
        vae.requires_grad_(False)
        return vae.to(device=device, dtype=dtype)

    from diffusers import AutoencoderKLTemporalDecoder

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_vae_name_or_path,
        subfolder=args.vae_subfolder or None,
        torch_dtype=dtype,
        variant=args.variant or None,
    )
    vae.requires_grad_(False)
    return vae.to(device)


@torch.no_grad()
def encode_video_latents(vae: Any, pixel_values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    bsz, frames, channels, height, width = pixel_values.shape
    flat = pixel_values.to(dtype=dtype).reshape(bsz * frames, channels, height, width)
    latents = vae.encode(flat).latent_dist.sample()
    latents = latents * float(vae.config.scaling_factor)
    return latents.reshape(bsz, frames, latents.shape[1], latents.shape[2], latents.shape[3])


@torch.no_grad()
def encode_image_latents(vae: Any, images01: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    images = to_vae_pixels(images01).to(dtype=dtype)
    latents = vae.encode(images).latent_dist.mode()
    return latents * float(vae.config.scaling_factor)


@torch.no_grad()
def decode_latents(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    vae_dtype = next(vae.parameters()).dtype
    bsz, frames, channels, height, width = latents.shape
    flat = latents.reshape(bsz * frames, channels, height, width).to(dtype=vae_dtype)
    flat = flat / float(vae.config.scaling_factor)
    try:
        decoded = vae.decode(flat, num_frames=frames).sample
    except TypeError:
        decoded = vae.decode(flat).sample
    return decoded.reshape(bsz, frames, decoded.shape[1], decoded.shape[2], decoded.shape[3])


def make_model(args: argparse.Namespace) -> LatentConditionalUNet:
    return LatentConditionalUNet(
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        audio_dim=args.audio_dim,
        audio_tokens=args.audio_tokens,
        scalar_feature_dim=args.scalar_feature_dim,
        physics_dim=args.physics_dim,
        time_dim=args.time_dim,
        cond_dim=args.cond_dim,
        dropout=args.dropout,
        use_prev_frame=args.use_prev_frame,
        use_stft=bool(getattr(args, "use_stft", False)),
        stft_freq_bins=int(getattr(args, "stft_n_freq_bins", 64)),
    )


def make_schedulers(args: argparse.Namespace):
    from diffusers import DDIMScheduler, DDPMScheduler

    scheduler_kwargs = dict(
        num_train_timesteps=int(args.num_train_timesteps),
        beta_schedule=str(args.beta_schedule),
        prediction_type=str(args.prediction_type),
        clip_sample=False,
    )
    return DDPMScheduler(**scheduler_kwargs), DDIMScheduler(**scheduler_kwargs)


def scheduler_target(scheduler: Any, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    pred_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if pred_type == "v_prediction" and hasattr(scheduler, "get_velocity"):
        return scheduler.get_velocity(clean, noise, timesteps)
    return noise


def latent_condition_tensors(
    args: argparse.Namespace,
    vae: Any,
    batch: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    latent_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    background = batch["background"].to(device)
    prev_last = batch["prev_last_frame"].to(device)
    roi = batch["roi"].to(device)
    prior = batch["nucleation_prior"].to(device)
    background_latents = encode_image_latents(vae, background, dtype)
    prev_latents = encode_image_latents(vae, prev_last, dtype)
    roi_latent = F.interpolate(roi.float(), size=latent_hw, mode="nearest")
    prior_latent = F.interpolate(prior.float(), size=latent_hw, mode="bilinear", align_corners=False)
    if str(getattr(args, "prior_mode", "none")) == "none":
        prior_latent = torch.zeros_like(prior_latent)
    return background_latents, roi_latent, prior_latent, prev_latents


def latent_loss_weight(
    args: argparse.Namespace,
    batch: dict[str, Any],
    latent_hw: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    weights = foreground_weight(
        batch["residual"].to(device),
        batch["roi"].to(device),
        threshold=args.foreground_threshold,
        base_weight=1.0,
        fg_weight=args.fg_weight,
        roi_weight=args.roi_weight,
    )
    bsz, frames = weights.shape[:2]
    flat = weights.reshape(bsz * frames, 1, weights.shape[-2], weights.shape[-1])
    flat = F.interpolate(flat, size=latent_hw, mode="bilinear", align_corners=False)
    return flat.reshape(bsz, frames, 1, latent_hw[0], latent_hw[1]).clamp_min(1e-3)


def compute_train_loss(
    args: argparse.Namespace,
    vae: Any,
    model: LatentConditionalUNet,
    noise_scheduler: Any,
    batch: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    use_amp: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_pixels = target_to_pixels(args, batch).to(device)
    with torch.no_grad():
        clean_latents = encode_video_latents(vae, target_pixels, dtype).float()
    bsz = clean_latents.shape[0]
    latent_hw = (int(clean_latents.shape[-2]), int(clean_latents.shape[-1]))
    background_latents, roi_latent, prior_latent, prev_latents = latent_condition_tensors(
        args, vae, batch, device, dtype, latent_hw
    )
    audio = batch["audio"].to(device)
    scalar_features = batch["audio_features"].to(device)
    physics = batch["physics"].to(device)
    audio_stft = batch["audio_stft"].to(device) if bool(getattr(args, "use_stft", False)) else None

    noise = torch.randn_like(clean_latents)
    timesteps = torch.randint(
        0,
        int(noise_scheduler.config.num_train_timesteps),
        (bsz,),
        device=device,
        dtype=torch.long,
    )
    noisy = noise_scheduler.add_noise(clean_latents, noise, timesteps)
    target = scheduler_target(noise_scheduler, clean_latents, noise, timesteps)
    cond_mask = (
        (torch.rand(bsz, device=device) > float(args.cond_dropout_prob)).float()
        if float(args.cond_dropout_prob) > 0.0
        else None
    )
    weights = latent_loss_weight(args, batch, latent_hw, device).to(target.dtype)

    amp_dtype = dtype_from_precision(args.mixed_precision)
    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        pred = model(
            noisy,
            timesteps.float() / max(1, int(noise_scheduler.config.num_train_timesteps) - 1),
            background_latents.float(),
            roi_latent.float(),
            prior_latent.float(),
            prev_latents.float() if args.use_prev_frame else None,
            audio,
            scalar_features,
            physics,
            cond_dropout_mask=cond_mask,
            audio_stft=audio_stft,
        )
        diff_sq = (pred.float() - target.float()) ** 2
        loss_raw = diff_sq.mean()
        loss = (diff_sq * weights).sum() / weights.sum().clamp_min(1e-6)
    return loss, {"ldm_mse": float(loss_raw.detach().cpu().item()), "ldm_weighted": float(loss.detach().cpu().item())}


@torch.no_grad()
def sample_batch(
    args: argparse.Namespace,
    vae: Any,
    model: LatentConditionalUNet,
    sample_scheduler: Any,
    batch: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    num_steps: int,
    seed: int,
) -> torch.Tensor:
    background = batch["background"].to(device)
    # Encode the target only to infer latent shape; no target information is fed to the denoiser.
    dummy_pixels = target_to_pixels(args, batch).to(device)
    clean_shape = encode_video_latents(vae, dummy_pixels, dtype).shape
    latent_hw = (int(clean_shape[-2]), int(clean_shape[-1]))
    background_latents, roi_latent, prior_latent, prev_latents = latent_condition_tensors(
        args, vae, batch, device, dtype, latent_hw
    )
    audio = batch["audio"].to(device)
    scalar_features = batch["audio_features"].to(device)
    physics = batch["physics"].to(device)
    audio_stft = batch["audio_stft"].to(device) if bool(getattr(args, "use_stft", False)) else None
    gen = torch.Generator(device=device).manual_seed(int(seed))
    latents = torch.randn(clean_shape, generator=gen, device=device)
    sample_scheduler.set_timesteps(int(num_steps), device=device)
    for timestep in sample_scheduler.timesteps:
        t = timestep.expand(clean_shape[0]) if hasattr(timestep, "expand") else torch.full((clean_shape[0],), int(timestep), device=device)
        model_input = sample_scheduler.scale_model_input(latents, timestep)
        pred = model(
            model_input,
            t.float() / max(1, int(sample_scheduler.config.num_train_timesteps) - 1),
            background_latents.float(),
            roi_latent.float(),
            prior_latent.float(),
            prev_latents.float() if args.use_prev_frame else None,
            audio,
            scalar_features,
            physics,
            audio_stft=audio_stft,
        )
        latents = sample_scheduler.step(pred, timestep, latents).prev_sample
    decoded = decode_latents(vae, latents)
    return decoded_to_video(args, decoded, background)


def distribution_score(args: argparse.Namespace, metrics: dict[str, float]) -> float:
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
    vae: Any,
    model: LatentConditionalUNet,
    noise_scheduler: Any,
    sample_scheduler: Any,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    losses: list[float] = []
    for i, batch in enumerate(loader):
        if int(args.num_val_batches) > 0 and i >= int(args.num_val_batches):
            break
        loss, _ = compute_train_loss(args, vae, model, noise_scheduler, batch, device, dtype, use_amp=False)
        losses.append(float(loss.detach().cpu().item()))
        pred_video = sample_batch(
            args,
            vae,
            model,
            sample_scheduler,
            batch,
            device,
            dtype,
            num_steps=int(args.val_inference_steps),
            seed=int(args.seed) + i,
        )
        target_video = batch["pixel_values"].to(device).float()
        background = batch["background"].to(device).float()
        rows.append(video_metrics(pred_video, target_video, background=background, foreground_threshold=args.foreground_threshold))
    out = aggregate_metrics(rows)
    out["val_ldm_loss"] = float(sum(losses) / max(1, len(losses)))
    out["val_distribution_score"] = float(distribution_score(args, out))
    model.train()
    return {f"val_{k}" if not k.startswith("val_") else k: v for k, v in out.items()}


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
    dtype = dtype_from_precision(args.mixed_precision) if device.type == "cuda" else torch.float32
    use_amp = device.type == "cuda" and args.mixed_precision != "no"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume_checkpoint:
        for stale_name in ("metrics.jsonl", "best.pt", "last.pt"):
            stale = output_dir / stale_name
            if stale.exists():
                stale.unlink()
    (output_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    vae = load_vae(args, device, dtype)
    model = make_model(args).to(device)
    noise_scheduler, sample_scheduler = make_schedulers(args)
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
        loss, parts = compute_train_loss(args, vae, model, noise_scheduler, batch, device, dtype, use_amp=use_amp)
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
            val = validate(args, vae, model, noise_scheduler, sample_scheduler, val_loader, device, dtype)
            val.update({f"train_{k}": v for k, v in parts.items()})
            val["step"] = global_step
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(val, ensure_ascii=False) + "\n")
            metric = float(val.get(args.best_metric_name, val.get("val_distribution_score", math.inf)))
            if metric < best_metric:
                best_metric = metric
                save_checkpoint(output_dir / "best.pt", model, optimizer, args, global_step, best_metric)
            save_checkpoint(output_dir / "last.pt", model, optimizer, args, global_step, best_metric)

    progress.close()
    save_checkpoint(output_dir / "last.pt", model, optimizer, args, global_step, best_metric)


if __name__ == "__main__":
    main()
