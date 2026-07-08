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
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from svd_audio_control.conditioners import AudioROIPhysicsProjector
from svd_audio_control.dataset import BoilingSVDManifestDataset, collate_fn
from svd_audio_control.lora_utils import add_unet_lora, load_checkpoint, save_checkpoint
from svd_audio_control.video_io import tensor_to_pil


def load_json_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_config(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    cli_keys = {
        item.lstrip("-").replace("-", "_")
        for item in sys.argv[1:]
        if item.startswith("--")
    }
    for key, value in config.items():
        if hasattr(args, key) and key not in cli_keys:
            setattr(args, key, value)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/stable-video-diffusion-img2vid-xt")
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--train_manifest", default=str(PROJECT_ROOT / "manifests" / "train.jsonl"))
    parser.add_argument("--val_manifest", default=str(PROJECT_ROOT / "manifests" / "val.jsonl"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "svd_lora_audio_control"))
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--video_fps", type=int, default=100)
    parser.add_argument("--audio_sample_rate", type=int, default=1_000_000)
    parser.add_argument("--conditioning_image_source", choices=["background", "first_frame"], default="background")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--num_audio_tokens", type=int, default=8)
    parser.add_argument("--num_roi_tokens", type=int, default=4)
    parser.add_argument("--conditioner_hidden_dim", type=int, default=256)
    parser.add_argument("--condition_dropout", type=float, default=0.05)
    parser.add_argument("--latent_residual_scale", type=float, default=0.0)
    parser.add_argument("--disable_audio_condition", action="store_true")
    parser.add_argument("--disable_roi_condition", action="store_true")
    parser.add_argument("--disable_physics_condition", action="store_true")
    parser.add_argument("--noise_aug_strength", type=float, default=0.02)
    parser.add_argument("--motion_bucket_id", type=int, default=127)
    parser.add_argument("--fps_condition", type=int, default=7)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--validation_steps", type=int, default=1000)
    parser.add_argument("--num_val_batches", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    return apply_config(args, load_json_config(args.config))


def dtype_from_precision(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def get_cross_attention_dim(unet: torch.nn.Module) -> int:
    dim = getattr(unet.config, "cross_attention_dim", 1024)
    if isinstance(dim, (list, tuple)):
        dim = dim[0]
    return int(dim)


def encode_clip_image(pipe: Any, images: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    pil_images = [tensor_to_pil(img) for img in images]
    clip = pipe.feature_extractor(images=pil_images, return_tensors="pt").pixel_values
    clip = clip.to(device=device, dtype=dtype)
    embeds = pipe.image_encoder(clip).image_embeds
    return embeds.unsqueeze(1)


def encode_video_latents(vae: Any, pixel_values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    bsz, frames, channels, height, width = pixel_values.shape
    flat = pixel_values.to(dtype=dtype).reshape(bsz * frames, channels, height, width)
    latents = vae.encode(flat).latent_dist.sample()
    latents = latents * float(vae.config.scaling_factor)
    return latents.reshape(bsz, frames, latents.shape[1], latents.shape[2], latents.shape[3])


def encode_image_latents(vae: Any, images: torch.Tensor, frames: int, noise_aug_strength: float, dtype: torch.dtype) -> torch.Tensor:
    cond = images.to(dtype=dtype)
    if noise_aug_strength > 0:
        cond = cond + noise_aug_strength * torch.randn_like(cond)
        cond = cond.clamp(-1.0, 1.0)
    latents = vae.encode(cond).latent_dist.mode()
    latents = latents * float(vae.config.scaling_factor)
    return latents.unsqueeze(1).repeat(1, frames, 1, 1, 1)


def add_time_ids(args: argparse.Namespace, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = [float(args.fps_condition), float(args.motion_bucket_id), float(args.noise_aug_strength)]
    return torch.tensor([values], device=device, dtype=dtype).repeat(batch_size, 1)


def velocity_target(scheduler: Any, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    pred_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if pred_type == "epsilon":
        return noise
    if pred_type == "v_prediction" and hasattr(scheduler, "get_velocity"):
        return scheduler.get_velocity(latents, noise, timesteps)
    return noise


def sample_scheduler_timesteps(scheduler: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    if not hasattr(scheduler, "timesteps") or scheduler.timesteps is None or len(scheduler.timesteps) == 0:
        scheduler.set_timesteps(int(scheduler.config.num_train_timesteps), device=device)
    schedule = scheduler.timesteps.to(device)
    indices = torch.randint(0, int(schedule.shape[0]), (batch_size,), device=device)
    return schedule[indices]


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
    pixel_values = batch["pixel_values"].to(device)
    cond_images = batch["conditioning_image"].to(device)
    audio = batch["audio"].to(device)
    roi = batch["roi"].to(device)
    physics = batch["physics"].to(device)
    bsz, frames = pixel_values.shape[:2]

    with torch.no_grad():
        latents = encode_video_latents(pipe.vae, pixel_values, dtype)
        image_latents = encode_image_latents(pipe.vae, cond_images, frames, args.noise_aug_strength, dtype)
        image_embeds = encode_clip_image(pipe, cond_images, device, dtype)

    noise = torch.randn_like(latents)
    timesteps = sample_scheduler_timesteps(scheduler, bsz, device)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    model_input = torch.cat([noisy_latents, image_latents], dim=2).to(dtype=dtype)

    drop = training and random.random() < float(args.condition_dropout)
    if float(getattr(args, "latent_residual_scale", 0.0)) != 0.0:
        image_latents = image_latents + float(args.latent_residual_scale) * conditioner.latent_residual(
            audio,
            roi,
            physics,
            frames=frames,
            height=image_latents.shape[-2],
            width=image_latents.shape[-1],
            drop_audio=bool(args.disable_audio_condition) or drop,
            drop_roi=bool(args.disable_roi_condition) or drop,
            drop_physics=bool(args.disable_physics_condition) or drop,
        ).to(dtype=dtype)
        model_input = torch.cat([noisy_latents, image_latents], dim=2).to(dtype=dtype)

    extra_tokens = conditioner(
        audio,
        roi,
        physics,
        drop_audio=bool(args.disable_audio_condition) or drop,
        drop_roi=bool(args.disable_roi_condition) or drop,
        drop_physics=bool(args.disable_physics_condition) or drop,
    ).to(dtype=dtype)
    encoder_hidden_states = torch.cat([image_embeds, extra_tokens], dim=1)
    added_time_ids = add_time_ids(args, bsz, device, dtype)

    pred = unet(
        model_input,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        added_time_ids=added_time_ids,
        return_dict=False,
    )[0]
    target = velocity_target(scheduler, latents, noise, timesteps)
    return F.mse_loss(pred.float(), target.float(), reduction="mean")


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
    (output_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
        variant=args.variant or None,
    )
    pipe.vae.requires_grad_(False)
    pipe.image_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.unet = add_unet_lora(pipe.unet, rank=args.lora_rank, alpha=args.lora_alpha)
    if args.gradient_checkpointing and hasattr(pipe.unet, "enable_gradient_checkpointing"):
        pipe.unet.enable_gradient_checkpointing()

    cross_dim = get_cross_attention_dim(pipe.unet)
    conditioner = AudioROIPhysicsProjector(
        cross_attention_dim=cross_dim,
        num_audio_tokens=args.num_audio_tokens,
        num_roi_tokens=args.num_roi_tokens,
        hidden_dim=args.conditioner_hidden_dim,
    )
    if args.resume_checkpoint:
        load_checkpoint(args.resume_checkpoint, pipe.unet, conditioner, map_location="cpu")

    trainable_params = [p for p in pipe.unet.parameters() if p.requires_grad] + list(conditioner.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.adam_weight_decay,
    )

    train_ds = BoilingSVDManifestDataset(
        args.train_manifest,
        resolution=args.resolution,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        audio_sample_rate=args.audio_sample_rate,
        video_fps=args.video_fps,
        random_start=True,
        load_audio=not bool(args.disable_audio_condition),
        conditioning_image_source=args.conditioning_image_source,
    )
    val_ds = BoilingSVDManifestDataset(
        args.val_manifest,
        resolution=args.resolution,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        audio_sample_rate=args.audio_sample_rate,
        video_fps=args.video_fps,
        random_start=False,
        load_audio=not bool(args.disable_audio_condition),
        conditioning_image_source=args.conditioning_image_source,
    )
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    pipe.vae.to(accelerator.device, dtype=dtype)
    pipe.image_encoder.to(accelerator.device, dtype=dtype)
    pipe.unet, conditioner, optimizer, train_loader, val_loader = accelerator.prepare(
        pipe.unet, conditioner, optimizer, train_loader, val_loader
    )
    scheduler = pipe.scheduler

    global_step = 0
    best_val = math.inf
    progress = tqdm(total=args.max_train_steps, disable=not accelerator.is_local_main_process)
    while global_step < args.max_train_steps:
        for batch in train_loader:
            with accelerator.accumulate(pipe.unet):
                loss = compute_loss(
                    args,
                    pipe,
                    pipe.unet,
                    conditioner,
                    scheduler,
                    batch,
                    accelerator.device,
                    dtype,
                    training=True,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{float(loss.detach().item()):.4f}")

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    save_checkpoint(
                        output_dir / "checkpoints" / f"checkpoint-{global_step:06d}.pt",
                        accelerator.unwrap_model(pipe.unet),
                        accelerator.unwrap_model(conditioner),
                        vars(args),
                        global_step,
                    )

                if global_step % args.validation_steps == 0:
                    pipe.unet.eval()
                    conditioner.eval()
                    val_losses = []
                    with torch.no_grad():
                        for i, val_batch in enumerate(val_loader):
                            if i >= args.num_val_batches:
                                break
                            val_loss = compute_loss(
                                args,
                                pipe,
                                pipe.unet,
                                conditioner,
                                scheduler,
                                val_batch,
                                accelerator.device,
                                dtype,
                                training=False,
                            )
                            val_losses.append(accelerator.gather(val_loss.detach()).mean())
                    mean_val = torch.stack(val_losses).mean().item() if val_losses else float("nan")
                    if accelerator.is_main_process:
                        metrics_path = output_dir / "metrics.jsonl"
                        with metrics_path.open("a", encoding="utf-8", newline="\n") as f:
                            f.write(json.dumps({"step": global_step, "train_loss": float(loss.detach().item()), "val_loss": mean_val}) + "\n")
                        if mean_val < best_val:
                            best_val = mean_val
                            save_checkpoint(
                                output_dir / "best.pt",
                                accelerator.unwrap_model(pipe.unet),
                                accelerator.unwrap_model(conditioner),
                                vars(args),
                                global_step,
                            )
                    pipe.unet.train()
                    conditioner.train()

                if global_step >= args.max_train_steps:
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(
            output_dir / "last.pt",
            accelerator.unwrap_model(pipe.unet),
            accelerator.unwrap_model(conditioner),
            vars(args),
            global_step,
        )
    progress.close()


if __name__ == "__main__":
    main()
