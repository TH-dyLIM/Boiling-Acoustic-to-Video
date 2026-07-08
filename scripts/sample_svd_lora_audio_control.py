from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_svd_lora_audio_control import (
    add_time_ids,
    dtype_from_precision,
    encode_clip_image,
    encode_image_latents,
    get_cross_attention_dim,
)
from svd_audio_control.conditioners import AudioROIPhysicsProjector
from svd_audio_control.dataset import BoilingSVDManifestDataset, collate_fn
from svd_audio_control.lora_utils import add_unet_lora, load_checkpoint
from svd_audio_control.video_io import save_video_gif, save_video_mp4


def video_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    pred01 = ((pred.detach().float().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    gt01 = ((gt.detach().float().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    mse = torch.mean((pred01 - gt01) ** 2).item()
    mae = torch.mean(torch.abs(pred01 - gt01)).item()
    psnr = 99.0 if mse <= 1e-12 else float(-10.0 * torch.log10(torch.tensor(mse)).item())
    return {
        "mse_video": float(mse),
        "mae_video": float(mae),
        "psnr_video": float(psnr),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "manifests" / "val.jsonl"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "svd_lora_audio_control_samples"))
    parser.add_argument("--pretrained_model_name_or_path", default="")
    parser.add_argument("--variant", default="")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--video_fps", type=int, default=100)
    parser.add_argument("--save_fps", type=int, default=10)
    parser.add_argument("--audio_sample_rate", type=int, default=1_000_000)
    parser.add_argument("--conditioning_image_source", choices=["background", "first_frame"], default="background")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--num_audio_tokens", type=int, default=8)
    parser.add_argument("--num_roi_tokens", type=int, default=4)
    parser.add_argument("--conditioner_hidden_dim", type=int, default=256)
    parser.add_argument("--latent_residual_scale", type=float, default=0.0)
    parser.add_argument("--noise_aug_strength", type=float, default=0.02)
    parser.add_argument("--motion_bucket_id", type=int, default=127)
    parser.add_argument("--fps_condition", type=int, default=7)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def apply_checkpoint_args(args: argparse.Namespace, ckpt_args: dict[str, Any]) -> argparse.Namespace:
    keys = [
        "pretrained_model_name_or_path",
        "variant",
        "mixed_precision",
        "resolution",
        "num_frames",
        "frame_stride",
        "video_fps",
        "audio_sample_rate",
        "conditioning_image_source",
        "lora_rank",
        "lora_alpha",
        "num_audio_tokens",
        "num_roi_tokens",
        "conditioner_hidden_dim",
        "latent_residual_scale",
        "noise_aug_strength",
        "motion_bucket_id",
        "fps_condition",
    ]
    for key in keys:
        if key in ckpt_args and (key not in {"pretrained_model_name_or_path", "variant"} or not getattr(args, key)):
            setattr(args, key, ckpt_args[key])
    return args


def decode_latents(vae: Any, latents: torch.Tensor, frames: int) -> torch.Tensor:
    bsz, num_frames, channels, height, width = latents.shape
    flat = latents.reshape(bsz * num_frames, channels, height, width)
    flat = flat / float(vae.config.scaling_factor)
    try:
        decoded = vae.decode(flat, num_frames=num_frames).sample
    except TypeError:
        decoded = vae.decode(flat).sample
    return decoded.reshape(bsz, num_frames, decoded.shape[1], decoded.shape[2], decoded.shape[3])


@torch.no_grad()
def generate(
    args: argparse.Namespace,
    pipe: Any,
    conditioner: AudioROIPhysicsProjector,
    batch: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    cond_images = batch["conditioning_image"].to(device)
    audio = batch["audio"].to(device)
    roi = batch["roi"].to(device)
    physics = batch["physics"].to(device)
    bsz = cond_images.shape[0]
    frames = int(args.num_frames)

    image_embeds = encode_clip_image(pipe, cond_images, device, dtype)
    extra_tokens = conditioner(audio, roi, physics).to(dtype=dtype)
    encoder_hidden_states = torch.cat([image_embeds, extra_tokens], dim=1)
    image_latents = encode_image_latents(pipe.vae, cond_images, frames, args.noise_aug_strength, dtype)
    if float(getattr(args, "latent_residual_scale", 0.0)) != 0.0:
        image_latents = image_latents + float(args.latent_residual_scale) * conditioner.latent_residual(
            audio,
            roi,
            physics,
            frames=frames,
            height=image_latents.shape[-2],
            width=image_latents.shape[-1],
        ).to(dtype=dtype)
    added_time_ids = add_time_ids(args, bsz, device, dtype)

    latent_channels = int(getattr(pipe.vae.config, "latent_channels", 4))
    h = image_latents.shape[-2]
    w = image_latents.shape[-1]
    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    latents = torch.randn((bsz, frames, latent_channels, h, w), generator=generator, device=device, dtype=dtype)
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    latents = latents * pipe.scheduler.init_noise_sigma

    for timestep in tqdm(pipe.scheduler.timesteps, leave=False):
        latent_model_input = pipe.scheduler.scale_model_input(latents, timestep)
        latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
        t = timestep.expand(bsz) if hasattr(timestep, "expand") else torch.tensor([timestep] * bsz, device=device)
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids,
            return_dict=False,
        )[0]
        latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample

    return decode_latents(pipe.vae, latents, frames)


def main() -> None:
    args = parse_args()
    raw_ckpt = torch.load(args.checkpoint, map_location="cpu")
    args = apply_checkpoint_args(args, raw_ckpt.get("args", {}))
    if not args.pretrained_model_name_or_path:
        raise ValueError("pretrained_model_name_or_path is missing. Pass it or use a checkpoint saved by the train script.")

    try:
        from diffusers import StableVideoDiffusionPipeline
    except Exception as exc:
        raise RuntimeError("Install requirements_svd_lora.txt before sampling.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_precision(args.mixed_precision) if device.type == "cuda" else torch.float32
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
        variant=args.variant or None,
    ).to(device)
    pipe.vae.requires_grad_(False)
    pipe.image_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.unet = add_unet_lora(pipe.unet, args.lora_rank, args.lora_alpha)

    conditioner = AudioROIPhysicsProjector(
        cross_attention_dim=get_cross_attention_dim(pipe.unet),
        num_audio_tokens=args.num_audio_tokens,
        num_roi_tokens=args.num_roi_tokens,
        hidden_dim=args.conditioner_hidden_dim,
    ).to(device)
    load_checkpoint(args.checkpoint, pipe.unet, conditioner, map_location=device.type)
    pipe.unet.eval()
    conditioner.eval()

    dataset = BoilingSVDManifestDataset(
        args.manifest,
        resolution=args.resolution,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        audio_sample_rate=args.audio_sample_rate,
        video_fps=args.video_fps,
        random_start=False,
        conditioning_image_source=getattr(args, "conditioning_image_source", "background"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "samples_metadata.jsonl"
    metrics_rows = []

    with meta_path.open("w", encoding="utf-8", newline="\n") as meta_f:
        for i, batch in enumerate(loader):
            if i >= args.max_samples:
                break
            pred = generate(args, pipe, conditioner, batch, device, dtype).cpu()
            gt = batch["pixel_values"].cpu()
            metrics = video_metrics(pred[0], gt[0])
            side_by_side = torch.cat([gt[0], pred[0]], dim=-1)
            stem = batch["stem"][0]
            save_video_gif(side_by_side, out_dir / "gifs" / f"{i:03d}_{stem}_gt_pred.gif", fps=args.save_fps)
            save_video_mp4(pred[0], out_dir / "pred_mp4" / f"{i:03d}_{stem}_pred.mp4", fps=args.save_fps)
            row = {
                "index": i,
                "stem": stem,
                "start_frame": batch["start_frame"][0],
                "physics_raw": batch["physics_raw"][0],
                "physics_norm": batch["physics"][0].tolist(),
                "condition_source": batch["condition_source"][0],
                "gif": str((out_dir / "gifs" / f"{i:03d}_{stem}_gt_pred.gif").resolve()),
                **metrics,
            }
            metrics_rows.append(metrics)
            meta_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False))
    if metrics_rows:
        summary = {
            "num_samples": len(metrics_rows),
            "mse_video_mean": float(sum(row["mse_video"] for row in metrics_rows) / len(metrics_rows)),
            "mae_video_mean": float(sum(row["mae_video"] for row in metrics_rows) / len(metrics_rows)),
            "psnr_video_mean": float(sum(row["psnr_video"] for row in metrics_rows) / len(metrics_rows)),
            "metadata_jsonl": str(meta_path.resolve()),
        }
        (out_dir / "samples_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
