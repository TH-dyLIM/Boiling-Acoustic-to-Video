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
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from flow_residual.dataset import ChunkResidualDataset, collate_fn
from flow_residual.metrics import aggregate_metrics, video_metrics
from scripts.train_svd_lora_audio_control import (
    add_time_ids,
    dtype_from_precision,
    encode_clip_image,
    encode_image_latents,
    encode_video_latents,
    get_cross_attention_dim,
    sample_scheduler_timesteps,
    velocity_target,
)
from svd_audio_control.conditioners import AudioROIPhysicsProjector
from svd_audio_control.lora_utils import add_unet_lora, load_checkpoint, save_checkpoint


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


def parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--pretrained_model_name_or_path", default="pretrained/stabilityai__stable-video-diffusion-img2vid-xt")
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--train_manifest", default=str(PROJECT_ROOT / "manifests" / "train.jsonl"))
    parser.add_argument("--val_manifest", default=str(PROJECT_ROOT / "manifests" / "val.jsonl"))
    parser.add_argument("--train_cache_dir", default="")
    parser.add_argument("--val_cache_dir", default="")
    parser.add_argument("--prior_path", default="")
    parser.add_argument("--prior_mode", choices=["global", "per_class", "none"], default="global")
    parser.add_argument("--roi_mode", choices=["normal", "none"], default="normal")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "svd_lora_stochastic_prior"))
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=14)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--video_fps", type=int, default=100)
    parser.add_argument("--audio_sample_rate", type=int, default=1_000_000)
    parser.add_argument("--conditioning_image_source", choices=["background", "prev_frame"], default="background")
    parser.add_argument("--target_representation", choices=["rgb", "residual"], default="residual")
    parser.add_argument("--residual_conditioning_image_source", choices=["zero", "background", "prev_frame"], default="zero")
    parser.add_argument("--residual_visual_scale", type=float, default=4.0)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=30000)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--num_audio_tokens", type=int, default=16)
    parser.add_argument("--num_roi_tokens", type=int, default=4)
    parser.add_argument("--num_prior_tokens", type=int, default=4)
    parser.add_argument("--conditioner_hidden_dim", type=int, default=384)
    parser.add_argument("--condition_dropout", type=float, default=0.1)
    parser.add_argument("--audio_condition_mode", choices=["waveform", "stft_image"], default="waveform")
    parser.add_argument("--audio_normalize", type=lambda v: str(v).lower() not in {"false", "0", "no"}, default=True)
    parser.add_argument("--audio_clamp_value", type=float, default=0.0)
    parser.add_argument("--audio_stft_n_fft", type=int, default=1024)
    parser.add_argument("--audio_stft_hop_length", type=int, default=256)
    parser.add_argument("--audio_stft_freq_bins", type=int, default=128)
    parser.add_argument("--audio_stft_time_bins", type=int, default=128)
    parser.add_argument("--latent_residual_scale", type=float, default=0.05)
    parser.add_argument("--disable_audio_condition", action="store_true")
    parser.add_argument("--disable_roi_condition", action="store_true")
    parser.add_argument("--disable_physics_condition", action="store_true")
    parser.add_argument("--disable_prior_condition", action="store_true")
    parser.add_argument("--noise_aug_strength", type=float, default=0.02)
    parser.add_argument("--motion_bucket_id", type=int, default=127)
    parser.add_argument("--fps_condition", type=int, default=7)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--validation_steps", type=int, default=1000)
    parser.add_argument("--num_val_batches", type=int, default=4)
    parser.add_argument("--val_start_frames", default="0,30,60")
    parser.add_argument("--train_fixed_starts", default="")
    parser.add_argument("--val_inference_steps", type=int, default=20)
    parser.add_argument("--val_cfg_scale", type=float, default=1.0)
    parser.add_argument("--best_metric_name", default="val_distribution_score")
    parser.add_argument("--foreground_threshold", type=float, default=0.04)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    return apply_config(args, load_json_config(args.config))


def to_svd_pixels(x01: torch.Tensor) -> torch.Tensor:
    return x01.float().mul(2.0).sub(1.0).clamp(-1.0, 1.0)


def to_01_pixels(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().add(1.0).div(2.0).clamp(0.0, 1.0)


def encode_clip_image_svd(pipe: Any, images: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    images01 = to_01_pixels(images).cpu()
    pil_images = []
    for img in images01:
        arr = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        pil_images.append(Image.fromarray(arr))
    clip = pipe.feature_extractor(images=pil_images, return_tensors="pt").pixel_values
    clip = clip.to(device=device, dtype=dtype)
    embeds = pipe.image_encoder(clip).image_embeds
    return embeds.unsqueeze(1)


def target_to_svd_pixels(args: argparse.Namespace, batch: dict[str, Any]) -> torch.Tensor:
    if args.target_representation == "rgb":
        return to_svd_pixels(batch["pixel_values"])
    scale = max(float(args.residual_visual_scale), 1e-6)
    residual_vis = (batch["residual"].float() * scale + 0.5).clamp(0.0, 1.0)
    return to_svd_pixels(residual_vis)


def decoded_to_video(args: argparse.Namespace, decoded: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    if args.target_representation == "rgb":
        return to_01_pixels(decoded)
    scale = max(float(args.residual_visual_scale), 1e-6)
    residual_vis = to_01_pixels(decoded)
    residual = (residual_vis - 0.5) / scale
    return (background.unsqueeze(1).float() + residual).clamp(0.0, 1.0)


def make_conditioning_image(args: argparse.Namespace, batch: dict[str, Any]) -> torch.Tensor:
    if args.target_representation == "residual":
        source = getattr(args, "residual_conditioning_image_source", "zero")
        if source == "background":
            return to_svd_pixels(batch["background"])
        if source == "prev_frame":
            return to_svd_pixels(batch["prev_last_frame"])
        # Zero residual is neutral gray after SVD pixel mapping.
        return torch.zeros_like(batch["background"].float())
    if args.conditioning_image_source == "prev_frame":
        return to_svd_pixels(batch["prev_last_frame"])
    return to_svd_pixels(batch["background"])


def condition_drop_flags(args: argparse.Namespace, training: bool) -> tuple[bool, bool, bool, bool]:
    drop = training and random.random() < float(args.condition_dropout)
    return (
        bool(args.disable_audio_condition) or drop,
        bool(args.disable_roi_condition) or drop,
        bool(args.disable_physics_condition) or drop,
        bool(args.disable_prior_condition) or drop,
    )


def build_condition(
    args: argparse.Namespace,
    pipe: Any,
    conditioner: AudioROIPhysicsProjector,
    batch: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cond_images = make_conditioning_image(args, batch).to(device)
    audio = batch["audio"].to(device)
    roi = batch["roi"].to(device)
    physics = batch["physics"].to(device)
    prior = batch["nucleation_prior"].to(device)
    bsz = cond_images.shape[0]
    frames = int(args.num_frames)
    drop_audio, drop_roi, drop_physics, drop_prior = condition_drop_flags(args, training)

    with torch.no_grad():
        image_latents = encode_image_latents(pipe.vae, cond_images, frames, args.noise_aug_strength, dtype)
        image_embeds = encode_clip_image_svd(pipe, cond_images, device, dtype)

    if float(args.latent_residual_scale) != 0.0:
        image_latents = image_latents + float(args.latent_residual_scale) * conditioner.latent_residual(
            audio,
            roi,
            physics,
            frames=frames,
            height=image_latents.shape[-2],
            width=image_latents.shape[-1],
            nucleation_prior=prior,
            drop_audio=drop_audio,
            drop_roi=drop_roi,
            drop_physics=drop_physics,
            drop_prior=drop_prior,
        ).to(dtype=dtype)

    extra_tokens = conditioner(
        audio,
        roi,
        physics,
        nucleation_prior=prior,
        drop_audio=drop_audio,
        drop_roi=drop_roi,
        drop_physics=drop_physics,
        drop_prior=drop_prior,
    ).to(dtype=dtype)
    encoder_hidden_states = torch.cat([image_embeds, extra_tokens], dim=1)
    added_time_ids = add_time_ids(args, bsz, device, dtype)
    return cond_images, image_latents, encoder_hidden_states, added_time_ids


def compute_loss(
    args: argparse.Namespace,
    pipe: Any,
    unet: torch.nn.Module,
    conditioner: AudioROIPhysicsProjector,
    scheduler: Any,
    batch: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    training: bool,
) -> torch.Tensor:
    pixel_values = target_to_svd_pixels(args, batch).to(device)
    bsz = pixel_values.shape[0]
    with torch.no_grad():
        latents = encode_video_latents(pipe.vae, pixel_values, dtype)
    _, image_latents, encoder_hidden_states, added_time_ids = build_condition(
        args, pipe, conditioner, batch, device, dtype, training=training
    )
    noise = torch.randn_like(latents)
    timesteps = sample_scheduler_timesteps(scheduler, bsz, device)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    model_input = torch.cat([noisy_latents, image_latents], dim=2).to(dtype=dtype)
    pred = unet(
        model_input,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        added_time_ids=added_time_ids,
        return_dict=False,
    )[0]
    target = velocity_target(scheduler, latents, noise, timesteps)
    return F.mse_loss(pred.float(), target.float(), reduction="mean")


def decode_latents(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    vae_dtype = next(vae.parameters()).dtype
    latents = latents.to(dtype=vae_dtype)
    bsz, frames, channels, height, width = latents.shape
    flat = latents.reshape(bsz * frames, channels, height, width) / float(vae.config.scaling_factor)
    try:
        decoded = vae.decode(flat, num_frames=frames).sample
    except TypeError:
        decoded = vae.decode(flat).sample
    return decoded.reshape(bsz, frames, decoded.shape[1], decoded.shape[2], decoded.shape[3])


@torch.no_grad()
def generate_batch(
    args: argparse.Namespace,
    pipe: Any,
    unet: torch.nn.Module,
    conditioner: AudioROIPhysicsProjector,
    batch: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    num_inference_steps: int,
    cfg_scale: float,
    seed: int,
) -> torch.Tensor:
    bsz = batch["pixel_values"].shape[0]
    _, image_latents, encoder_hidden_states, added_time_ids = build_condition(
        args, pipe, conditioner, batch, device, dtype, training=False
    )
    latent_channels = int(getattr(pipe.vae.config, "latent_channels", 4))
    shape = (bsz, int(args.num_frames), latent_channels, image_latents.shape[-2], image_latents.shape[-1])
    generator = torch.Generator(device=device).manual_seed(int(seed))
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    pipe.scheduler.set_timesteps(int(num_inference_steps), device=device)
    latents = latents * pipe.scheduler.init_noise_sigma

    uncond_states = None
    if float(cfg_scale) != 1.0:
        saved_disable = (
            args.disable_audio_condition,
            args.disable_roi_condition,
            args.disable_physics_condition,
            args.disable_prior_condition,
        )
        args.disable_audio_condition = True
        args.disable_roi_condition = True
        args.disable_physics_condition = True
        args.disable_prior_condition = True
        _, _, uncond_states, _ = build_condition(args, pipe, conditioner, batch, device, dtype, training=False)
        (
            args.disable_audio_condition,
            args.disable_roi_condition,
            args.disable_physics_condition,
            args.disable_prior_condition,
        ) = saved_disable

    for timestep in pipe.scheduler.timesteps:
        latent_model_input = pipe.scheduler.scale_model_input(latents, timestep)
        latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
        t = timestep.expand(bsz) if hasattr(timestep, "expand") else torch.tensor([timestep] * bsz, device=device)
        cond_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids,
            return_dict=False,
        )[0]
        if uncond_states is not None:
            uncond_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=uncond_states,
                added_time_ids=added_time_ids,
                return_dict=False,
            )[0]
            noise_pred = uncond_pred + float(cfg_scale) * (cond_pred - uncond_pred)
        else:
            noise_pred = cond_pred
        latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample
    return decode_latents(pipe.vae, latents)


def distribution_score(metrics: dict[str, float]) -> float:
    return float(
        metrics.get("void_fraction_mae_mean", 1.0)
        + 0.5 * metrics.get("spatial_density_l1_mean", 1.0)
        + 0.02 * metrics.get("nucleation_count_mae_mean", 50.0)
        + 0.001 * metrics.get("departure_freq_mae_mean", 500.0)
    )


@torch.no_grad()
def validate(
    args: argparse.Namespace,
    pipe: Any,
    unet: torch.nn.Module,
    conditioner: AudioROIPhysicsProjector,
    scheduler: Any,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    step: int,
) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    losses = []
    for i, batch in enumerate(loader):
        if int(args.num_val_batches) > 0 and i >= int(args.num_val_batches):
            break
        losses.append(compute_loss(args, pipe, unet, conditioner, scheduler, batch, device, dtype, training=False))
        pred = generate_batch(
            args,
            pipe,
            unet,
            conditioner,
            batch,
            device,
            dtype,
            num_inference_steps=int(args.val_inference_steps),
            cfg_scale=float(args.val_cfg_scale),
            seed=int(args.seed) + int(step) + i,
        )
        background = batch["background"].to(device).float()
        pred01 = decoded_to_video(args, pred, background)
        target01 = batch["pixel_values"].to(device).float()
        rows.append(video_metrics(pred01, target01, background=background, foreground_threshold=args.foreground_threshold))
    out = aggregate_metrics(rows)
    out["fm_loss_mean"] = float(torch.stack(losses).mean().detach().cpu().item()) if losses else math.inf
    out["distribution_score"] = distribution_score(out)
    return {f"val_{k}": v for k, v in out.items()}


def make_model(args: argparse.Namespace, pipe: Any) -> tuple[torch.nn.Module, AudioROIPhysicsProjector]:
    pipe.unet.requires_grad_(False)
    pipe.unet = add_unet_lora(pipe.unet, rank=args.lora_rank, alpha=args.lora_alpha)
    conditioner = AudioROIPhysicsProjector(
        cross_attention_dim=get_cross_attention_dim(pipe.unet),
        num_audio_tokens=args.num_audio_tokens,
        num_roi_tokens=args.num_roi_tokens,
        num_prior_tokens=args.num_prior_tokens,
        hidden_dim=args.conditioner_hidden_dim,
        audio_condition_mode=getattr(args, "audio_condition_mode", "waveform"),
        audio_stft_n_fft=getattr(args, "audio_stft_n_fft", 1024),
        audio_stft_hop_length=getattr(args, "audio_stft_hop_length", 256),
        audio_stft_freq_bins=getattr(args, "audio_stft_freq_bins", 128),
        audio_stft_time_bins=getattr(args, "audio_stft_time_bins", 128),
    )
    return pipe.unet, conditioner


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        from accelerate import Accelerator
        from diffusers import StableVideoDiffusionPipeline
    except Exception as exc:
        raise RuntimeError("Install requirements_svd_lora.txt before training.") from exc

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    )
    dtype = dtype_from_precision(args.mixed_precision)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume_checkpoint:
        for stale in ("metrics.jsonl", "best.pt", "last.pt"):
            p = output_dir / stale
            if p.exists():
                p.unlink()
    (output_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
        variant=args.variant or None,
    )
    pipe.vae.requires_grad_(False)
    pipe.image_encoder.requires_grad_(False)
    unet, conditioner = make_model(args, pipe)
    if args.gradient_checkpointing and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()
    global_step = 0
    if args.resume_checkpoint:
        ckpt = load_checkpoint(args.resume_checkpoint, unet, conditioner, map_location="cpu")
        global_step = int(ckpt.get("step", 0))

    train_ds = ChunkResidualDataset(
        args.train_manifest,
        resolution=args.resolution,
        chunk_frames=args.num_frames,
        frame_stride=args.frame_stride,
        audio_sample_rate=args.audio_sample_rate,
        video_fps=args.video_fps,
        random_start=True,
        cache_dir=args.train_cache_dir or None,
        prior_path=args.prior_path or None,
        prior_mode=args.prior_mode,
        roi_mode=getattr(args, "roi_mode", "normal"),
        audio_normalize=bool(getattr(args, "audio_normalize", True)),
        audio_clamp_value=(float(args.audio_clamp_value) if float(getattr(args, "audio_clamp_value", 0.0)) > 0.0 else None),
        fixed_starts_per_clip=parse_int_list(args.train_fixed_starts),
    )
    val_ds = ChunkResidualDataset(
        args.val_manifest,
        resolution=args.resolution,
        chunk_frames=args.num_frames,
        frame_stride=args.frame_stride,
        audio_sample_rate=args.audio_sample_rate,
        video_fps=args.video_fps,
        random_start=False,
        cache_dir=args.val_cache_dir or None,
        prior_path=args.prior_path or None,
        prior_mode=args.prior_mode,
        roi_mode=getattr(args, "roi_mode", "normal"),
        audio_normalize=bool(getattr(args, "audio_normalize", True)),
        audio_clamp_value=(float(args.audio_clamp_value) if float(getattr(args, "audio_clamp_value", 0.0)) > 0.0 else None),
        fixed_starts_per_clip=parse_int_list(args.val_start_frames),
    )
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    trainable_params = [p for p in unet.parameters() if p.requires_grad] + list(conditioner.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.adam_weight_decay)

    pipe.vae.to(accelerator.device, dtype=dtype)
    pipe.image_encoder.to(accelerator.device, dtype=dtype)
    unet, conditioner, optimizer, train_loader, val_loader = accelerator.prepare(
        unet, conditioner, optimizer, train_loader, val_loader
    )
    scheduler = pipe.scheduler

    best_metric = math.inf
    progress = tqdm(total=args.max_train_steps, initial=global_step, disable=not accelerator.is_local_main_process)
    unet.train()
    conditioner.train()
    while global_step < int(args.max_train_steps):
        for batch in train_loader:
            with accelerator.accumulate(unet):
                loss = compute_loss(args, pipe, unet, conditioner, scheduler, batch, accelerator.device, dtype, training=True)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{float(loss.detach().item()):.4f}")

                if accelerator.is_main_process and global_step % int(args.checkpointing_steps) == 0:
                    save_checkpoint(
                        output_dir / "checkpoints" / f"checkpoint-{global_step:06d}.pt",
                        accelerator.unwrap_model(unet),
                        accelerator.unwrap_model(conditioner),
                        vars(args),
                        global_step,
                    )

                if global_step % int(args.validation_steps) == 0:
                    unet.eval()
                    conditioner.eval()
                    val = validate(args, pipe, unet, conditioner, scheduler, val_loader, accelerator.device, dtype, global_step)
                    metric = float(val.get(args.best_metric_name, val.get("val_distribution_score", math.inf)))
                    if accelerator.is_main_process:
                        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8", newline="\n") as f:
                            f.write(json.dumps({"step": global_step, "train_loss": float(loss.detach().item()), **val}) + "\n")
                        if metric < best_metric:
                            best_metric = metric
                            save_checkpoint(
                                output_dir / "best.pt",
                                accelerator.unwrap_model(unet),
                                accelerator.unwrap_model(conditioner),
                                vars(args),
                                global_step,
                            )
                    unet.train()
                    conditioner.train()

                if global_step >= int(args.max_train_steps):
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(output_dir / "last.pt", accelerator.unwrap_model(unet), accelerator.unwrap_model(conditioner), vars(args), global_step)
    progress.close()


if __name__ == "__main__":
    main()
