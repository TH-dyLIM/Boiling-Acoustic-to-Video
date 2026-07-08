from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_svd_lora_audio_control import (  # noqa: E402
    compute_loss,
    dtype_from_precision,
    get_cross_attention_dim,
)
from svd_audio_control.conditioners import AudioROIPhysicsProjector  # noqa: E402
from svd_audio_control.dataset import BoilingSVDManifestDataset, collate_fn  # noqa: E402
from svd_audio_control.lora_utils import add_unet_lora, load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "manifests" / "test.jsonl"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "svd_lora_audio_control_test_eval"))
    parser.add_argument("--eval_repeats", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def checkpoint_args_to_namespace(ckpt_args: dict[str, Any]) -> argparse.Namespace:
    defaults = {
        "pretrained_model_name_or_path": "pretrained/stabilityai__stable-video-diffusion-img2vid-xt",
        "variant": "fp16",
        "resolution": 256,
        "num_frames": 8,
        "frame_stride": 1,
        "video_fps": 100,
        "audio_sample_rate": 1_000_000,
        "conditioning_image_source": "background",
        "mixed_precision": "fp16",
        "lora_rank": 32,
        "lora_alpha": 32,
        "num_audio_tokens": 8,
        "num_roi_tokens": 4,
        "conditioner_hidden_dim": 256,
        "condition_dropout": 0.0,
        "latent_residual_scale": 0.0,
        "disable_audio_condition": False,
        "disable_roi_condition": False,
        "disable_physics_condition": True,
        "noise_aug_strength": 0.02,
        "motion_bucket_id": 127,
        "fps_condition": 7,
    }
    defaults.update(ckpt_args)
    return argparse.Namespace(**defaults)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    raw_ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = checkpoint_args_to_namespace(raw_ckpt.get("args", {}))

    try:
        from diffusers import StableVideoDiffusionPipeline
    except Exception as exc:
        raise RuntimeError("Install requirements_svd_lora.txt before evaluation.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_precision(train_args.mixed_precision) if device.type == "cuda" else torch.float32

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        train_args.pretrained_model_name_or_path,
        torch_dtype=dtype,
        variant=train_args.variant or None,
    ).to(device)
    pipe.vae.requires_grad_(False)
    pipe.image_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.unet = add_unet_lora(pipe.unet, train_args.lora_rank, train_args.lora_alpha)

    conditioner = AudioROIPhysicsProjector(
        cross_attention_dim=get_cross_attention_dim(pipe.unet),
        num_audio_tokens=train_args.num_audio_tokens,
        num_roi_tokens=train_args.num_roi_tokens,
        hidden_dim=train_args.conditioner_hidden_dim,
    ).to(device)
    load_checkpoint(args.checkpoint, pipe.unet, conditioner, map_location=device.type)
    pipe.unet.eval()
    conditioner.eval()

    dataset = BoilingSVDManifestDataset(
        args.manifest,
        resolution=train_args.resolution,
        num_frames=train_args.num_frames,
        frame_stride=train_args.frame_stride,
        audio_sample_rate=train_args.audio_sample_rate,
        video_fps=train_args.video_fps,
        random_start=False,
        load_audio=not bool(train_args.disable_audio_condition),
        conditioning_image_source=getattr(train_args, "conditioning_image_source", "background"),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "test_eval_rows.jsonl"
    losses: list[float] = []

    with rows_path.open("w", encoding="utf-8", newline="\n") as f:
        for repeat in range(max(1, int(args.eval_repeats))):
            set_seed(int(args.seed) + repeat)
            progress = tqdm(loader, desc=f"eval repeat {repeat + 1}/{args.eval_repeats}", leave=False)
            for batch_index, batch in enumerate(progress):
                with torch.no_grad():
                    loss = compute_loss(
                        train_args,
                        pipe,
                        pipe.unet,
                        conditioner,
                        pipe.scheduler,
                        batch,
                        device,
                        dtype,
                        training=False,
                    )
                value = float(loss.detach().cpu().item())
                losses.append(value)
                row = {
                    "repeat": repeat,
                    "batch_index": batch_index,
                    "loss": value,
                    "stem": batch["stem"],
                    "start_frame": batch["start_frame"],
                    "condition_source": batch["condition_source"],
                    "physics_raw": batch["physics_raw"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                progress.set_postfix(loss=f"{value:.5f}")

    loss_tensor = torch.tensor(losses, dtype=torch.float32)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(raw_ckpt.get("step", -1)),
        "manifest": str(Path(args.manifest).resolve()),
        "num_manifest_items": len(dataset),
        "eval_repeats": max(1, int(args.eval_repeats)),
        "num_loss_values": len(losses),
        "diffusion_loss_mean": float(loss_tensor.mean().item()) if losses else None,
        "diffusion_loss_std": float(loss_tensor.std(unbiased=False).item()) if losses else None,
        "diffusion_loss_min": float(loss_tensor.min().item()) if losses else None,
        "diffusion_loss_max": float(loss_tensor.max().item()) if losses else None,
        "rows_jsonl": str(rows_path.resolve()),
        "audio_sample_rate": int(train_args.audio_sample_rate),
        "num_frames": int(train_args.num_frames),
        "frame_stride": int(train_args.frame_stride),
        "conditioning_image_source": str(getattr(train_args, "conditioning_image_source", "background")),
        "latent_residual_scale": float(getattr(train_args, "latent_residual_scale", 0.0)),
    }
    summary_path = out_dir / "test_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
