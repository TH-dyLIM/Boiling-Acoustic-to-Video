from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from flow_residual.dataset import ChunkResidualDataset, collate_fn
from flow_residual.metrics import aggregate_metrics, foreground_mask, video_metrics
from residual_video.dataset import audio_segment_from_cached, tensor_u8_to_01
from scripts.train_ldm_residual import (
    decoded_to_video,
    dtype_from_precision,
    load_checkpoint,
    load_vae,
    make_model,
    make_schedulers,
    sample_batch,
)
from svd_audio_control.video_io import save_video_gif, save_video_mp4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "manifests_csv_new" / "test.jsonl"))
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--prior_path", default="")
    parser.add_argument("--prior_mode", choices=["global", "per_class", "none"], default="")
    parser.add_argument("--roi_mode", choices=["normal", "none"], default="")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "ldm_residual_test_samples"))
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--full_video_rollout", action="store_true")
    parser.add_argument("--start_frames", default="0,30,60")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--save_fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="")
    return parser.parse_args()


def args_from_ckpt(ckpt_args: dict[str, Any], cli: argparse.Namespace) -> argparse.Namespace:
    merged = dict(ckpt_args)
    if cli.mixed_precision:
        merged["mixed_precision"] = cli.mixed_precision
    if cli.prior_mode:
        merged["prior_mode"] = cli.prior_mode
    if cli.roi_mode:
        merged["roi_mode"] = cli.roi_mode
    if cli.prior_path:
        merged["prior_path"] = cli.prior_path
    return argparse.Namespace(**merged)


def parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def make_layout(target: torch.Tensor, pred: torch.Tensor, background: torch.Tensor, pred_mask: torch.Tensor) -> torch.Tensor:
    bg = background.unsqueeze(0).expand_as(target)
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.unsqueeze(1)
    mask_rgb = pred_mask.expand_as(target)
    return torch.cat([target, pred, bg, mask_rgb], dim=-1)


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
    idx = (int(start) + torch.arange(dataset.chunk_frames) * dataset.frame_stride).clamp(0, total - 1).long()
    frames_chunk = dataset._resize_video(tensor_u8_to_01(frames_u8[idx]))
    background = dataset._resize_image(tensor_u8_to_01(cache["background_u8"]), is_mask=False)
    roi = dataset._resize_image(tensor_u8_to_01(cache["roi_u8"]), is_mask=True)
    prev_last = prev_last_pred.clone() if prev_last_pred is not None else background.clone()

    cached_audio = cache.get("audio")
    audio_sr = int(cache.get("audio_sr", dataset.audio_sample_rate))
    fps = float(cache.get("fps", dataset.video_fps) or dataset.video_fps)
    if cached_audio is None:
        raise ValueError(f"Missing cached audio for {row.get('stem', '')}")
    if audio_sr != dataset.audio_sample_rate:
        raise ValueError(f"Cached audio sample rate mismatch for {row.get('stem', '')}: expected {dataset.audio_sample_rate}, got {audio_sr}")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_precision(train_args.mixed_precision) if device.type == "cuda" else torch.float32

    vae = load_vae(train_args, device, dtype)
    model = make_model(train_args).to(device)
    load_checkpoint(args.checkpoint, model, optimizer=None)
    model.eval()
    _, sample_scheduler = make_schedulers(train_args)

    audio_kwargs = dict(
        audio_normalize=bool(getattr(train_args, "audio_normalize", False)),
        audio_clamp_value=(float(train_args.audio_clamp_value) if float(getattr(train_args, "audio_clamp_value", 0.0)) > 0.0 else None),
        use_stft=bool(getattr(train_args, "use_stft", False)),
        stft_n_fft=int(getattr(train_args, "stft_n_fft", 2048)),
        stft_hop_length=int(getattr(train_args, "stft_hop_length", 1024)),
        stft_n_freq_bins=int(getattr(train_args, "stft_n_freq_bins", 64)),
        stft_fmin=float(getattr(train_args, "stft_fmin", 100.0)),
        stft_fmax=(float(train_args.stft_fmax) if float(getattr(train_args, "stft_fmax", 0.0)) > 0.0 else None),
    )
    dataset = ChunkResidualDataset(
        args.manifest,
        resolution=train_args.resolution,
        chunk_frames=train_args.chunk_frames,
        frame_stride=train_args.frame_stride,
        audio_sample_rate=train_args.audio_sample_rate,
        video_fps=train_args.video_fps,
        random_start=False,
        cache_dir=args.cache_dir or None,
        prior_path=args.prior_path or getattr(train_args, "prior_path", "") or None,
        prior_mode=args.prior_mode or getattr(train_args, "prior_mode", "none"),
        roi_mode=args.roi_mode or getattr(train_args, "roi_mode", "normal"),
        fixed_starts_per_clip=([0] if args.full_video_rollout else parse_int_list(args.start_frames)),
        **audio_kwargs,
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
                raise FileNotFoundError(f"Cache not found for {row.get('stem', '')}")
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
                pred_video = sample_batch(
                    train_args,
                    vae,
                    model,
                    sample_scheduler,
                    batch,
                    device,
                    dtype,
                    num_steps=int(args.num_inference_steps),
                    seed=int(args.seed) + row_index * 1000 + chunk_index,
                )[0].detach().cpu()
                target_video = batch["pixel_values"][0].float().cpu()
                take = min(int(train_args.chunk_frames), total_frames - covered)
                pred_take = pred_video[:take]
                target_take = target_video[:take]
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
            gif_path = gifs_dir / f"{name}_gt_pred_bg_mask.gif"
            mp4_path = mp4_dir / f"{name}_pred.mp4"
            save_video_gif(make_layout(target_full, pred_full, background_cpu, pred_mask), gif_path, fps=args.save_fps)
            save_video_mp4(pred_full, mp4_path, fps=args.save_fps)
            meta_lines.append(
                json.dumps(
                    {
                        "index": video_count,
                        "stem": stem,
                        "start_frame": 0,
                        "num_frames": int(pred_full.shape[0]),
                        "num_chunks": int(chunk_index),
                        "full_video_rollout": True,
                        "gif": str(gif_path),
                        "pred_mp4": str(mp4_path),
                        "metrics": metrics,
                    },
                    ensure_ascii=False,
                )
            )
            video_count += 1
    else:
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
        for index, batch in enumerate(tqdm(loader)):
            if int(args.max_samples) > 0 and index >= int(args.max_samples):
                break
            pred_video_b = sample_batch(
                train_args,
                vae,
                model,
                sample_scheduler,
                batch,
                device,
                dtype,
                num_steps=int(args.num_inference_steps),
                seed=int(args.seed) + index,
            )
            pred_video = pred_video_b[0].cpu()
            target_video = batch["pixel_values"][0].float().cpu()
            background = batch["background"][0].float().cpu()
            pred_mask = foreground_mask((pred_video - background.unsqueeze(0)).unsqueeze(0), train_args.foreground_threshold)[0].cpu()
            metrics = video_metrics(
                pred_video_b,
                batch["pixel_values"].to(device).float(),
                background=batch["background"].to(device).float(),
                foreground_threshold=train_args.foreground_threshold,
            )
            rows.append(metrics)
            stem = batch["stem"][0]
            start_frame = int(batch["start_frame"][0])
            name = f"{index:03d}_{stem}_s{start_frame:04d}"
            gif_path = gifs_dir / f"{name}_gt_pred_bg_mask.gif"
            mp4_path = mp4_dir / f"{name}_pred.mp4"
            save_video_gif(make_layout(target_video, pred_video, background, pred_mask), gif_path, fps=args.save_fps)
            save_video_mp4(pred_video, mp4_path, fps=args.save_fps)
            meta_lines.append(json.dumps({"index": index, "stem": stem, "start_frame": start_frame, "gif": str(gif_path), "pred_mp4": str(mp4_path), "metrics": metrics}, ensure_ascii=False))

    (out_dir / "samples_metadata.jsonl").write_text("\n".join(meta_lines) + ("\n" if meta_lines else ""), encoding="utf-8")
    summary = {
        "num_samples": len(meta_lines),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(raw.get("step", 0)),
        "full_video_rollout": bool(args.full_video_rollout),
        "metrics": aggregate_metrics(rows),
    }
    (out_dir / "samples_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
