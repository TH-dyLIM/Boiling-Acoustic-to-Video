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

from flow_residual.dataset import ChunkResidualDataset, collate_fn
from flow_residual.metrics import aggregate_metrics, foreground_mask, video_metrics
from residual_video.dataset import audio_segment_from_cached, tensor_u8_to_01
from scripts.train_svd_lora_stochastic_prior import (
    decode_latents,
    decoded_to_video,
    dtype_from_precision,
    generate_batch,
    get_cross_attention_dim,
    make_model,
    parse_int_list,
)
from svd_audio_control.lora_utils import load_checkpoint
from svd_audio_control.video_io import save_video_gif, save_video_mp4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "manifests" / "test.jsonl"))
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--prior_path", default="")
    parser.add_argument("--prior_mode", choices=["global", "per_class", "none"], default="")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "svd_lora_stochastic_prior_samples"))
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--start_frames", default="0,30,60")
    parser.add_argument(
        "--full_video_rollout",
        action="store_true",
        help="Generate each full test video by concatenating sequential latent-diffusion chunks.",
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--save_fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="")
    return parser.parse_args()


def args_from_ckpt(ckpt_args: dict[str, Any], cli: argparse.Namespace) -> argparse.Namespace:
    merged = dict(ckpt_args)
    if cli.mixed_precision:
        merged["mixed_precision"] = cli.mixed_precision
    if cli.prior_path:
        merged["prior_path"] = cli.prior_path
    if cli.prior_mode:
        merged["prior_mode"] = cli.prior_mode
    return argparse.Namespace(**merged)


def make_layout(target01: torch.Tensor, pred01: torch.Tensor, background: torch.Tensor, pred_mask: torch.Tensor) -> torch.Tensor:
    bg = background.unsqueeze(0).expand_as(target01)
    mask_rgb = pred_mask.unsqueeze(1).expand_as(target01)
    return torch.cat([target01, pred01, bg, mask_rgb], dim=-1)


def rollout_item_from_cache(
    dataset: ChunkResidualDataset,
    row_index: int,
    row: dict[str, Any],
    cache: dict[str, Any],
    start: int,
    prev_last_pred: torch.Tensor | None,
) -> dict[str, Any]:
    frames_u8 = cache["frames_u8"]
    total = int(frames_u8.shape[0])
    if total <= 0:
        raise ValueError(f"Empty cached video for row {row_index}: {row.get('stem', '')}")

    idx = (int(start) + torch.arange(dataset.chunk_frames) * dataset.frame_stride).clamp(0, total - 1).long()
    frames_chunk = dataset._resize_video(tensor_u8_to_01(frames_u8[idx]))
    background = dataset._resize_image(tensor_u8_to_01(cache["background_u8"]), is_mask=False)
    roi = dataset._resize_image(tensor_u8_to_01(cache["roi_u8"]), is_mask=True)
    prev_last = prev_last_pred.clone() if prev_last_pred is not None else background.clone()

    cached_audio = cache.get("audio")
    audio_sr = int(cache.get("audio_sr", dataset.audio_sample_rate))
    fps = float(cache.get("fps", dataset.video_fps) or dataset.video_fps)
    if cached_audio is None:
        raise ValueError(f"Full rollout requires cached audio. Missing audio in cache for {row.get('stem', '')}")
    if audio_sr != dataset.audio_sample_rate:
        raise ValueError(
            f"Cached audio sample rate mismatch for {row.get('stem', '')}: "
            f"expected {dataset.audio_sample_rate}, got {audio_sr}"
        )
    audio = audio_segment_from_cached(
        cached_audio,
        start_sec=int(start) / fps,
        duration_sec=(dataset.chunk_frames * dataset.frame_stride) / fps,
        target_sr=dataset.audio_sample_rate,
        target_len=dataset.audio_len,
        normalize=dataset.audio_normalize,
        clamp_value=dataset.audio_clamp_value,
    )
    return dataset._pack(row, int(start), frames_chunk, background, roi, prev_last, audio)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    raw = torch.load(args.checkpoint, map_location="cpu")
    train_args = args_from_ckpt(raw.get("args", {}), args)
    if not getattr(train_args, "pretrained_model_name_or_path", ""):
        raise ValueError("Checkpoint args do not contain pretrained_model_name_or_path.")

    try:
        from diffusers import StableVideoDiffusionPipeline
    except Exception as exc:
        raise RuntimeError("Install requirements_svd_lora.txt before sampling.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_precision(train_args.mixed_precision) if device.type == "cuda" else torch.float32
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        train_args.pretrained_model_name_or_path,
        torch_dtype=dtype,
        variant=getattr(train_args, "variant", "") or None,
    ).to(device)
    pipe.vae.requires_grad_(False)
    pipe.image_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    unet, conditioner = make_model(train_args, pipe)
    conditioner.to(device)
    load_checkpoint(args.checkpoint, unet, conditioner, map_location=device.type)
    unet.to(device)
    unet.eval()
    conditioner.eval()

    dataset = ChunkResidualDataset(
        args.manifest,
        resolution=train_args.resolution,
        chunk_frames=train_args.num_frames,
        frame_stride=train_args.frame_stride,
        audio_sample_rate=train_args.audio_sample_rate,
        video_fps=train_args.video_fps,
        random_start=False,
        cache_dir=args.cache_dir or None,
        prior_path=args.prior_path or getattr(train_args, "prior_path", "") or None,
        prior_mode=args.prior_mode or getattr(train_args, "prior_mode", "global"),
        roi_mode=getattr(train_args, "roi_mode", "normal"),
        fixed_starts_per_clip=([0] if args.full_video_rollout else parse_int_list(args.start_frames)),
        audio_normalize=bool(getattr(train_args, "audio_normalize", True)),
        audio_clamp_value=(float(train_args.audio_clamp_value) if float(getattr(train_args, "audio_clamp_value", 0.0)) > 0.0 else None),
    )
    out_dir = Path(args.output_dir)
    gifs_dir = out_dir / "gifs"
    mp4_dir = out_dir / "pred_mp4"
    gifs_dir.mkdir(parents=True, exist_ok=True)
    mp4_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    meta_lines: list[str] = []

    if args.full_video_rollout:
        if not args.cache_dir:
            raise ValueError("--full_video_rollout requires --cache_dir")
        video_count = 0
        for row_index, row in enumerate(tqdm(dataset.rows)):
            if int(args.max_samples) > 0 and video_count >= int(args.max_samples):
                break
            cache = dataset._load_cache(row_index, row)
            if cache is None:
                raise FileNotFoundError(f"Cache not found for {row.get('stem', '')}; run precompute_residual_tensor_cache first.")

            total_frames = int(cache["frames_u8"].shape[0])
            pred_chunks: list[torch.Tensor] = []
            target_chunks: list[torch.Tensor] = []
            prev_last_pred: torch.Tensor | None = None
            background_cpu: torch.Tensor | None = None
            covered = 0
            chunk_index = 0
            while covered < total_frames:
                item = rollout_item_from_cache(dataset, row_index, row, cache, covered, prev_last_pred)
                batch = collate_fn([item])
                pred = generate_batch(
                    train_args,
                    pipe,
                    unet,
                    conditioner,
                    batch,
                    device,
                    dtype,
                    num_inference_steps=int(args.num_inference_steps),
                    cfg_scale=float(args.cfg_scale),
                    seed=int(args.seed) + row_index * 1000 + chunk_index,
                )
                background_b = batch["background"].to(device).float()
                pred01 = decoded_to_video(train_args, pred, background_b)[0].cpu()
                target01 = batch["pixel_values"][0].float().cpu()
                take = min(int(train_args.num_frames), total_frames - covered)
                pred_take = pred01[:take].detach().cpu()
                target_take = target01[:take].detach().cpu()
                pred_chunks.append(pred_take)
                target_chunks.append(target_take)
                prev_last_pred = pred_take[-1].clone()
                background_cpu = batch["background"][0].float().cpu()
                covered += take
                chunk_index += 1

            if not pred_chunks or background_cpu is None:
                continue
            pred_full = torch.cat(pred_chunks, dim=0)
            target_full = torch.cat(target_chunks, dim=0)
            pred_mask = foreground_mask(
                (pred_full - background_cpu.unsqueeze(0)).unsqueeze(0),
                train_args.foreground_threshold,
            )[0].cpu()
            metrics = video_metrics(
                pred_full.unsqueeze(0).to(device),
                target_full.unsqueeze(0).to(device),
                background=background_cpu.unsqueeze(0).to(device),
                foreground_threshold=train_args.foreground_threshold,
            )
            rows.append(metrics)

            stem = row.get("stem", Path(row.get("video", "")).stem)
            name = f"{video_count:03d}_{stem}_full"
            layout = make_layout(target_full, pred_full, background_cpu, pred_mask)
            gif_path = gifs_dir / f"{name}_gt_pred_bg_mask.gif"
            mp4_path = mp4_dir / f"{name}_pred.mp4"
            save_video_gif(layout, gif_path, fps=args.save_fps)
            save_video_mp4(pred_full, mp4_path, fps=args.save_fps)
            meta = {
                "index": video_count,
                "stem": stem,
                "start_frame": 0,
                "num_frames": int(pred_full.shape[0]),
                "num_chunks": int(chunk_index),
                "full_video_rollout": True,
                "physics_raw": row.get("physics_raw", {}),
                "condition_source": row.get("condition_source", ""),
                "gif": str(gif_path),
                "pred_mp4": str(mp4_path),
                "num_inference_steps": int(args.num_inference_steps),
                "cfg_scale": float(args.cfg_scale),
                "prior_mode": args.prior_mode or getattr(train_args, "prior_mode", "global"),
                "metrics": metrics,
            }
            meta_lines.append(json.dumps(meta, ensure_ascii=False))
            video_count += 1

        (out_dir / "samples_metadata.jsonl").write_text("\n".join(meta_lines) + ("\n" if meta_lines else ""), encoding="utf-8")
        summary = {
            "num_samples": len(meta_lines),
            "full_video_rollout": True,
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": int(raw.get("step", 0)),
            "metrics": aggregate_metrics(rows),
        }
        (out_dir / "samples_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    for index, batch in enumerate(tqdm(loader)):
        if int(args.max_samples) > 0 and index >= int(args.max_samples):
            break
        pred = generate_batch(
            train_args,
            pipe,
            unet,
            conditioner,
            batch,
            device,
            dtype,
            num_inference_steps=int(args.num_inference_steps),
            cfg_scale=float(args.cfg_scale),
            seed=int(args.seed) + index,
        )
        background_b = batch["background"].to(device).float()
        pred01 = decoded_to_video(train_args, pred, background_b)[0].cpu()
        target01 = batch["pixel_values"][0].float().cpu()
        background = batch["background"][0].float().cpu()
        pred_mask = foreground_mask((pred01 - background.unsqueeze(0)).unsqueeze(0), train_args.foreground_threshold)[0].cpu()
        metrics = video_metrics(
            pred01.unsqueeze(0).to(device),
            target01.unsqueeze(0).to(device),
            background=background.unsqueeze(0).to(device),
            foreground_threshold=train_args.foreground_threshold,
        )
        rows.append(metrics)

        stem = batch["stem"][0]
        start_frame = int(batch["start_frame"][0])
        name = f"{index:03d}_{stem}_s{start_frame:04d}"
        layout = make_layout(target01, pred01, background, pred_mask)
        gif_path = gifs_dir / f"{name}_gt_pred_bg_mask.gif"
        mp4_path = mp4_dir / f"{name}_pred.mp4"
        save_video_gif(layout, gif_path, fps=args.save_fps)
        save_video_mp4(pred01, mp4_path, fps=args.save_fps)
        meta = {
            "index": index,
            "stem": stem,
            "start_frame": start_frame,
            "physics_raw": batch["physics_raw"][0],
            "condition_source": batch["condition_source"][0],
            "gif": str(gif_path),
            "pred_mp4": str(mp4_path),
            "num_inference_steps": int(args.num_inference_steps),
            "cfg_scale": float(args.cfg_scale),
            "prior_mode": args.prior_mode or getattr(train_args, "prior_mode", "global"),
            "metrics": metrics,
        }
        meta_lines.append(json.dumps(meta, ensure_ascii=False))

    (out_dir / "samples_metadata.jsonl").write_text("\n".join(meta_lines) + ("\n" if meta_lines else ""), encoding="utf-8")
    summary = {
        "num_samples": len(meta_lines),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(raw.get("step", 0)),
        "metrics": aggregate_metrics(rows),
    }
    (out_dir / "samples_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
