from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.amp import autocast
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from flow_residual.dataset import ChunkResidualDataset
from flow_residual.flow import euler_sample
from sample_flow_residual import (
    args_from_ckpt,
    compose_prediction,
    make_model,
    model_background,
    rollout_item_from_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark no-save inference cost for flow residual acoustic-to-video models on the test set."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prior_path", default="")
    parser.add_argument("--prior_mode", choices=["per_class", "global", "none"], default="none")
    parser.add_argument("--roi_mode", choices=["normal", "none"], default="")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--warmup_chunks", type=int, default=3)
    parser.add_argument("--timing_chunks", type=int, default=14)
    parser.add_argument("--max_full_videos", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def now_after_sync(device: torch.device) -> float:
    sync(device)
    return time.perf_counter()


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    values_sorted = sorted(float(v) for v in values)
    n = len(values_sorted)
    p50 = values_sorted[n // 2] if n % 2 else 0.5 * (values_sorted[n // 2 - 1] + values_sorted[n // 2])
    p90_index = min(n - 1, int(round(0.9 * (n - 1))))
    return {
        "count": float(n),
        "mean": float(statistics.fmean(values_sorted)),
        "std": float(statistics.stdev(values_sorted)) if n > 1 else 0.0,
        "min": float(values_sorted[0]),
        "p50": float(p50),
        "p90": float(values_sorted[p90_index]),
        "max": float(values_sorted[-1]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_dataset(args: argparse.Namespace, train_args: argparse.Namespace, fixed_starts: list[int] | None = None) -> ChunkResidualDataset:
    audio_kwargs = dict(
        audio_normalize=bool(getattr(train_args, "audio_normalize", True)),
        audio_clamp_value=(
            float(train_args.audio_clamp_value)
            if float(getattr(train_args, "audio_clamp_value", 0.0)) > 0.0
            else None
        ),
        use_stft=bool(getattr(train_args, "use_stft", False)),
        stft_n_fft=int(getattr(train_args, "stft_n_fft", 2048)),
        stft_hop_length=int(getattr(train_args, "stft_hop_length", 1024)),
        stft_n_freq_bins=int(getattr(train_args, "stft_n_freq_bins", 64)),
        stft_fmin=float(getattr(train_args, "stft_fmin", 100.0)),
        stft_fmax=(
            float(train_args.stft_fmax)
            if float(getattr(train_args, "stft_fmax", 0.0)) > 0.0
            else None
        ),
    )
    return ChunkResidualDataset(
        args.manifest,
        resolution=train_args.resolution,
        chunk_frames=train_args.chunk_frames,
        frame_stride=train_args.frame_stride,
        audio_sample_rate=train_args.audio_sample_rate,
        video_fps=train_args.video_fps,
        random_start=False,
        cache_dir=args.cache_dir,
        prior_path=args.prior_path or None,
        prior_mode=args.prior_mode,
        roi_mode=args.roi_mode or getattr(train_args, "roi_mode", "normal"),
        fixed_starts_per_clip=fixed_starts,
        **audio_kwargs,
    )


def make_context(
    item: dict[str, Any],
    model: torch.nn.Module,
    train_args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    cfg_scale: float,
) -> tuple[tuple[int, ...], torch.Tensor, Any, Any]:
    background = item["background"].unsqueeze(0).to(device)
    background_cond = model_background(train_args, background)
    roi = item["roi"].unsqueeze(0).to(device)
    prior = item["nucleation_prior"].unsqueeze(0).to(device)
    prev_last = item["prev_last_frame"].unsqueeze(0).to(device)
    audio = item["audio"].unsqueeze(0).to(device)
    scalar_features = item["audio_features"].unsqueeze(0).to(device)
    physics = item["physics"].unsqueeze(0).to(device)
    use_stft = bool(getattr(train_args, "use_stft", False))
    audio_stft = item["audio_stft"].unsqueeze(0).to(device) if use_stft else None
    shape = (1, train_args.chunk_frames, 3, train_args.resolution, train_args.resolution)

    def cond_velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            return model(
                x,
                t,
                background_cond,
                roi,
                prior,
                prev_last if train_args.use_prev_frame else None,
                audio,
                scalar_features,
                physics,
                audio_stft=audio_stft,
            ).float()

    if float(cfg_scale) > 0.0:
        zero_mask = torch.zeros(1, device=device)

        def uncond_velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                return model(
                    x,
                    t,
                    background_cond,
                    roi,
                    prior,
                    prev_last if train_args.use_prev_frame else None,
                    audio,
                    scalar_features,
                    physics,
                    cond_dropout_mask=zero_mask,
                    audio_stft=audio_stft,
                ).float()
    else:
        uncond_velocity = None

    return shape, background, cond_velocity, uncond_velocity


def sample_chunk(
    item: dict[str, Any],
    model: torch.nn.Module,
    train_args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    num_steps: int,
    cfg_scale: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    t0 = now_after_sync(device)
    shape, background, cond_velocity, uncond_velocity = make_context(
        item, model, train_args, device, use_amp, amp_dtype, cfg_scale
    )
    t1 = now_after_sync(device)
    sampled = euler_sample(
        cond_velocity,
        shape,
        device,
        num_steps=int(num_steps),
        cfg_scale=float(cfg_scale),
        uncond_velocity_fn=uncond_velocity,
        seed=int(seed),
    )
    t2 = now_after_sync(device)
    pred_video = compose_prediction(train_args, sampled, background)
    t3 = now_after_sync(device)
    times = {
        "prep_s": t1 - t0,
        "sample_s": t2 - t1,
        "compose_s": t3 - t2,
        "generation_s": t3 - t1,
        "end_to_end_no_save_s": t3 - t0,
    }
    return pred_video, times


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = torch.load(args.checkpoint, map_location="cpu")
    train_args = args_from_ckpt(raw.get("args", {}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.mixed_precision != "no"
    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16

    model = make_model(train_args).to(device)
    missing, unexpected = model.load_state_dict(raw["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    dataset = make_dataset(args, train_args, fixed_starts=[0])
    full_dataset = make_dataset(args, train_args, fixed_starts=None)

    timing_count = len(dataset) if int(args.timing_chunks) <= 0 else min(len(dataset), int(args.timing_chunks))
    chunk_items = [dataset[i] for i in range(timing_count)]
    if not chunk_items:
        raise RuntimeError("No test chunks available for timing.")

    # Warmup is intentionally excluded from timing. It absorbs CUDA kernel setup and allocator effects.
    warmup_count = max(0, int(args.warmup_chunks))
    for i in range(warmup_count):
        item = chunk_items[i % len(chunk_items)]
        sample_chunk(
            item,
            model,
            train_args,
            device,
            use_amp,
            amp_dtype,
            args.num_inference_steps,
            args.cfg_scale,
            args.seed + 100_000 + i,
        )

    chunk_rows: list[dict[str, Any]] = []
    for i, item in enumerate(tqdm(chunk_items, desc="8-frame timing")):
        _, times = sample_chunk(
            item,
            model,
            train_args,
            device,
            use_amp,
            amp_dtype,
            args.num_inference_steps,
            args.cfg_scale,
            args.seed + i,
        )
        chunk_rows.append(
            {
                "index": i,
                "stem": item["stem"],
                "start_frame": int(item["start_frame"]),
                "num_frames": int(train_args.chunk_frames),
                "video_duration_s": float(train_args.chunk_frames) / float(train_args.video_fps),
                "num_inference_steps": int(args.num_inference_steps),
                "cfg_scale": float(args.cfg_scale),
                **times,
                "generated_fps_generation_only": float(train_args.chunk_frames) / max(times["generation_s"], 1e-12),
                "realtime_factor_generation_only": times["generation_s"]
                / (float(train_args.chunk_frames) / float(train_args.video_fps)),
            }
        )

    full_rows: list[dict[str, Any]] = []
    max_full = len(full_dataset.rows) if int(args.max_full_videos) <= 0 else min(len(full_dataset.rows), int(args.max_full_videos))
    for row_index, row in enumerate(tqdm(full_dataset.rows[:max_full], desc="full-rollout timing")):
        cache = full_dataset._load_cache(row_index, row)
        if cache is None:
            raise FileNotFoundError(f"Missing cache for test row {row_index}: {row.get('stem', '')}")

        total_frames = int(cache["frames_u8"].shape[0])
        covered = 0
        chunk_index = 0
        prev_last_pred: torch.Tensor | None = None
        prep_s = 0.0
        sample_s = 0.0
        compose_s = 0.0
        generation_s = 0.0
        end_to_end_s = 0.0

        while covered < total_frames:
            item_start = now_after_sync(device)
            item = rollout_item_from_cache(full_dataset, row_index, row, cache, covered, prev_last_pred)
            item_mid = now_after_sync(device)
            pred_video, times = sample_chunk(
                item,
                model,
                train_args,
                device,
                use_amp,
                amp_dtype,
                args.num_inference_steps,
                args.cfg_scale,
                args.seed + row_index * 1000 + chunk_index,
            )
            item_end = now_after_sync(device)
            take = min(int(train_args.chunk_frames), total_frames - covered)
            pred_take = pred_video[0, :take].detach().cpu()
            prev_last_pred = pred_take[-1].clone()
            prep_s += (item_mid - item_start) + times["prep_s"]
            sample_s += times["sample_s"]
            compose_s += times["compose_s"]
            generation_s += times["generation_s"]
            end_to_end_s += item_end - item_start
            covered += take
            chunk_index += 1

        duration_s = total_frames / float(train_args.video_fps)
        stem = row.get("stem", Path(row.get("video", "")).stem)
        full_rows.append(
            {
                "index": row_index,
                "stem": stem,
                "num_frames": total_frames,
                "num_chunks": chunk_index,
                "video_duration_s": duration_s,
                "num_inference_steps": int(args.num_inference_steps),
                "cfg_scale": float(args.cfg_scale),
                "prep_s": prep_s,
                "sample_s": sample_s,
                "compose_s": compose_s,
                "generation_s": generation_s,
                "end_to_end_no_save_s": end_to_end_s,
                "generated_fps_generation_only": total_frames / max(generation_s, 1e-12),
                "generated_fps_end_to_end_no_save": total_frames / max(end_to_end_s, 1e-12),
                "realtime_factor_generation_only": generation_s / max(duration_s, 1e-12),
                "realtime_factor_end_to_end_no_save": end_to_end_s / max(duration_s, 1e-12),
            }
        )

    write_csv(out_dir / "chunk_8frame_timing.csv", chunk_rows)
    write_csv(out_dir / "full_rollout_timing.csv", full_rows)

    chunk_summary = {
        key: stats([float(row[key]) for row in chunk_rows])
        for key in [
            "prep_s",
            "sample_s",
            "compose_s",
            "generation_s",
            "end_to_end_no_save_s",
            "generated_fps_generation_only",
            "realtime_factor_generation_only",
        ]
    }
    full_summary = {
        key: stats([float(row[key]) for row in full_rows])
        for key in [
            "num_frames",
            "num_chunks",
            "video_duration_s",
            "prep_s",
            "sample_s",
            "compose_s",
            "generation_s",
            "end_to_end_no_save_s",
            "generated_fps_generation_only",
            "generated_fps_end_to_end_no_save",
            "realtime_factor_generation_only",
            "realtime_factor_end_to_end_no_save",
        ]
    }
    hardware = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "mixed_precision": args.mixed_precision,
        "peak_cuda_memory_allocated_gb": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
        ),
        "peak_cuda_memory_reserved_gb": (
            torch.cuda.max_memory_reserved(device) / (1024 ** 3) if device.type == "cuda" else 0.0
        ),
    }
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(raw.get("step", 0)),
        "manifest": str(args.manifest),
        "cache_dir": str(args.cache_dir),
        "model": {
            "model_type": getattr(train_args, "model_type", "unet3d"),
            "resolution": int(train_args.resolution),
            "chunk_frames": int(train_args.chunk_frames),
            "video_fps": int(train_args.video_fps),
            "base_channels": int(getattr(train_args, "base_channels", 0)),
            "audio_dim": int(getattr(train_args, "audio_dim", 0)),
            "audio_tokens": int(getattr(train_args, "audio_tokens", 0)),
            "cond_dim": int(getattr(train_args, "cond_dim", 0)),
            "time_dim": int(getattr(train_args, "time_dim", 0)),
            "audio_normalize": bool(getattr(train_args, "audio_normalize", True)),
            "audio_clamp_value": float(getattr(train_args, "audio_clamp_value", 0.0)),
            "use_stft": bool(getattr(train_args, "use_stft", False)),
            "prediction_mode": getattr(train_args, "prediction_mode", "residual"),
            "prior_mode": args.prior_mode,
        },
        "sampling": {
            "num_inference_steps": int(args.num_inference_steps),
            "cfg_scale": float(args.cfg_scale),
            "effective_unet_forward_passes_per_chunk": int(args.num_inference_steps)
            * (2 if float(args.cfg_scale) > 0.0 else 1),
            "warmup_chunks": int(args.warmup_chunks),
            "timed_chunks": len(chunk_rows),
            "timed_full_videos": len(full_rows),
        },
        "hardware": hardware,
        "chunk_8frame_summary": chunk_summary,
        "full_rollout_summary": full_summary,
    }
    (out_dir / "inference_cost_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    chunk_mean = chunk_summary["generation_s"]["mean"]
    chunk_std = chunk_summary["generation_s"]["std"]
    full_mean = full_summary["generation_s"]["mean"]
    full_std = full_summary["generation_s"]["std"]
    full_rt = full_summary["realtime_factor_generation_only"]["mean"]
    full_fps = full_summary["generated_fps_generation_only"]["mean"]
    md = f"""# Inference Cost Benchmark

Model: `{Path(args.checkpoint).as_posix()}`

Dataset: test split only (`{Path(args.manifest).as_posix()}`)

## Hardware and Sampling

- Device: {hardware["gpu_name"]}
- Torch: {hardware["torch_version"]}
- CUDA: {hardware["cuda_version"]}
- Precision: {args.mixed_precision}
- Euler sampling steps: {int(args.num_inference_steps)}
- CFG scale: {float(args.cfg_scale)}
- Effective model forward passes per 8-frame chunk: {summary["sampling"]["effective_unet_forward_passes_per_chunk"]}
- Chunk size: {int(train_args.chunk_frames)} frames at {int(train_args.video_fps)} fps ({int(train_args.chunk_frames) / float(train_args.video_fps):.3f} s)
- Peak CUDA memory allocated: {hardware["peak_cuda_memory_allocated_gb"]:.3f} GB
- Peak CUDA memory reserved: {hardware["peak_cuda_memory_reserved_gb"]:.3f} GB

## Timing Results

- 8-frame generation time, sampling + composition only: {chunk_mean:.4f} ± {chunk_std:.4f} s
- Full-rollout generation time, sampling + composition only: {full_mean:.4f} ± {full_std:.4f} s
- Full-rollout generated FPS, generation only: {full_fps:.2f} fps
- Full-rollout real-time factor, generation only: {full_rt:.2f}× video duration

`generation_s` excludes disk I/O, GIF/MP4 writing, and metric computation. `end_to_end_no_save_s` includes cache tensor extraction, audio chunking, GPU transfer, sampling, and residual composition, but still excludes video saving and metric computation.

## Paper-ready Sentence Template

Using {hardware["gpu_name"]}, the proposed no-prior residual conditional flow-matching model generated an 8-frame, 80-ms boiling-video chunk in {chunk_mean:.3f} ± {chunk_std:.3f} s with {int(args.num_inference_steps)} Euler steps and CFG scale {float(args.cfg_scale):.1f}. Full test-video rollout required {full_mean:.3f} ± {full_std:.3f} s per sequence on average, corresponding to {full_fps:.2f} generated frames per second and a real-time factor of {full_rt:.2f}× relative to 100-fps video.
"""
    (out_dir / "inference_cost_paper_summary.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
