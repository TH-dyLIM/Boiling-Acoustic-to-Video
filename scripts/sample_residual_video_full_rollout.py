from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from flow_residual.metrics import (  # noqa: E402
    aggregate_metrics as aggregate_flow_metrics,
    video_metrics as distribution_video_metrics,
)
from residual_video.dataset import (  # noqa: E402
    _image_01,
    _read_video_rgb,
    _rgb_frame_to_01,
    audio_scalar_features,
    audio_segment_from_cached,
    read_full_audio_file,
    read_jsonl,
)
from residual_video.metrics import (  # noqa: E402
    compose_video,
    effective_residual,
    foreground_mask as residual_foreground_mask,
    video_metrics as residual_video_metrics,
)
from residual_video.model import AudioResidualUNet  # noqa: E402
from svd_audio_control.video_io import save_video_gif, save_video_mp4  # noqa: E402


METRIC_COLUMNS = [
    "mse_video_mean",
    "mae_video_mean",
    "psnr_video_mean",
    "edge_mae_video_mean",
    "fg_mae_video_mean",
    "fg_coverage_mean",
    "pred_fg_coverage_mean",
    "bubble_mask_iou_mean",
    "void_fraction_mae_mean",
    "pred_void_fraction_mean_mean",
    "target_void_fraction_mean_mean",
    "void_fraction_ks_mean",
    "nucleation_count_mae_mean",
    "pred_nucleation_count_mean_mean",
    "target_nucleation_count_mean_mean",
    "nucleation_count_ks_mean",
    "departure_freq_mae_mean",
    "pred_departure_freq_mean_mean",
    "target_departure_freq_mean_mean",
    "spatial_density_l1_mean",
    "distribution_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "manifests_csv_new" / "test.jsonl"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_fps", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def args_from_ckpt(ckpt_args: dict[str, Any]) -> argparse.Namespace:
    defaults = {
        "resolution": 256,
        "num_frames": 8,
        "frame_stride": 1,
        "video_fps": 100,
        "audio_sample_rate": 1_000_000,
        "audio_channels": 64,
        "base_channels": 64,
        "residual_scale": 2.0,
        "residual_target_scale": 2.0,
        "use_mask_head": False,
        "mask_bias_init": 0.0,
        "foreground_threshold": 0.035,
        "dropout": 0.0,
        "physics_dim": 3,
        "physics_channels": 8,
        "audio_feature_dim": 6,
        "audio_feature_channels": 8,
        "use_previous_frame_condition": False,
        "prediction_base": "background",
        "audio_context_frames": 1,
        "audio_context_future_frames": 0,
    }
    defaults.update(ckpt_args)
    return argparse.Namespace(**defaults)


def make_model(args: argparse.Namespace) -> AudioResidualUNet:
    return AudioResidualUNet(
        num_frames=args.num_frames,
        audio_channels=args.audio_channels,
        base_channels=args.base_channels,
        residual_scale=args.residual_scale,
        dropout=args.dropout,
        use_mask_head=getattr(args, "use_mask_head", False),
        mask_bias_init=getattr(args, "mask_bias_init", 0.0),
        physics_dim=getattr(args, "physics_dim", 3),
        physics_channels=getattr(args, "physics_channels", 8),
        audio_feature_dim=getattr(args, "audio_feature_dim", 6),
        audio_feature_channels=getattr(args, "audio_feature_channels", 8),
        use_previous_frame_condition=getattr(args, "use_previous_frame_condition", False),
    )


def split_model_output(output: torch.Tensor | dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(output, dict):
        return output["residual"], output.get("mask")
    return output, None


def residual_vis(residual: torch.Tensor) -> torch.Tensor:
    vis = residual.detach().float().abs().mean(dim=1, keepdim=True).clamp(0.0, 0.25) * 4.0
    return vis.repeat(1, 3, 1, 1)


def distribution_score(metrics: dict[str, Any]) -> float:
    void = float(metrics.get("void_fraction_mae_mean", 0.0))
    spatial = float(metrics.get("spatial_density_l1_mean", 0.0))
    nuc_mae = float(metrics.get("nucleation_count_mae_mean", 0.0))
    nuc_target = max(float(metrics.get("target_nucleation_count_mean_mean", 1.0)), 1e-6)
    dep_mae = float(metrics.get("departure_freq_mae_mean", 0.0))
    dep_target = max(float(metrics.get("target_departure_freq_mean_mean", 1.0)), 1e-6)
    return void + 0.8 * spatial + 0.4 * (nuc_mae / nuc_target) + 0.02 * (dep_mae / dep_target)


def write_summary_csv(path: Path, row: dict[str, Any]) -> None:
    base_cols = ["experiment_id", "checkpoint", "checkpoint_step", "num_samples", "eval_mode", "save_fps"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_cols + METRIC_COLUMNS)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in base_cols + METRIC_COLUMNS})


def write_per_video_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = ["index", "stem", "num_frames", "num_chunks", "gif", "pred_mp4"]
    metric_keys = sorted({k for row in rows for k in row if k not in keys})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys + metric_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys + metric_keys})


def load_video_tensors(row: dict[str, Any], resolution: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    frames_rgb, fps = _read_video_rgb(row["video"])
    frames = torch.stack([_rgb_frame_to_01(frame_rgb, resolution) for frame_rgb in frames_rgb], dim=0)
    bg_path = row.get("background", "")
    if bg_path and Path(bg_path).exists():
        background = _image_01(bg_path, resolution, "RGB")
    else:
        background = frames[0].clone()
    roi_path = row.get("roi", "")
    if roi_path and Path(roi_path).exists():
        roi = _image_01(roi_path, resolution, "L")
    else:
        roi = torch.zeros(1, resolution, resolution, dtype=torch.float32)
    return frames, background, roi, float(fps)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    raw = torch.load(args.checkpoint, map_location="cpu")
    train_args = args_from_ckpt(raw.get("args", {}))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = make_model(train_args).to(device)
    missing, unexpected = model.load_state_dict(raw["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys in {args.checkpoint}: {unexpected}")
    model.eval()

    rows = read_jsonl(args.manifest)
    out_dir = Path(args.output_dir)
    gifs_dir = out_dir / "gifs"
    mp4_dir = out_dir / "pred_mp4"
    gifs_dir.mkdir(parents=True, exist_ok=True)
    mp4_dir.mkdir(parents=True, exist_ok=True)

    metadata_lines: list[str] = []
    metric_rows: list[dict[str, float]] = []
    per_video_rows: list[dict[str, Any]] = []
    max_samples = int(args.max_samples)

    chunk_frames = int(train_args.num_frames)
    frame_stride = int(train_args.frame_stride)
    audio_len = int(round((chunk_frames * frame_stride / float(train_args.video_fps)) * int(train_args.audio_sample_rate)))

    for row_index, row in enumerate(tqdm(rows, desc="videos")):
        if max_samples > 0 and row_index >= max_samples:
            break
        stem = row.get("stem", Path(row.get("video", "")).stem)
        target_full, background, roi, fps = load_video_tensors(row, int(train_args.resolution))
        full_audio, _ = read_full_audio_file(row["audio"], int(train_args.audio_sample_rate))
        total_frames = int(target_full.shape[0])

        pred_chunks: list[torch.Tensor] = []
        target_chunks: list[torch.Tensor] = []
        prev_last_pred: torch.Tensor | None = None
        covered = 0
        chunk_index = 0
        while covered < total_frames:
            idx = covered + torch.arange(chunk_frames, dtype=torch.long) * frame_stride
            idx = idx.clamp(0, total_frames - 1)
            target_chunk = target_full[idx]

            audio = audio_segment_from_cached(
                full_audio,
                start_sec=covered / float(fps),
                duration_sec=(chunk_frames * frame_stride) / float(fps),
                target_sr=int(train_args.audio_sample_rate),
                target_len=audio_len,
            )
            previous_frame = prev_last_pred.clone() if prev_last_pred is not None else background.clone()
            physics = torch.tensor(row.get("physics_norm", [0.0, 0.0, 0.0]), dtype=torch.float32)
            audio_features = audio_scalar_features(audio)

            background_b = background.unsqueeze(0).to(device)
            roi_b = roi.unsqueeze(0).to(device)
            audio_b = audio.unsqueeze(0).to(device)
            previous_b = previous_frame.unsqueeze(0).to(device)
            physics_b = physics.unsqueeze(0).to(device)
            audio_features_b = audio_features.unsqueeze(0).to(device)
            pred_residual, pred_mask = split_model_output(
                model(
                    background_b,
                    roi_b,
                    audio_b,
                    physics=physics_b,
                    audio_features=audio_features_b,
                    previous_frame=previous_b,
                )
            )
            pred_chunk = compose_video(
                background_b,
                pred_residual,
                clamp=True,
                mask=pred_mask,
                residual_target_scale=float(train_args.residual_target_scale),
                base_frame=previous_b if train_args.prediction_base == "previous_frame" else None,
            )[0].detach().cpu()
            take = min(chunk_frames, total_frames - covered)
            pred_take = pred_chunk[:take]
            target_take = target_chunk[:take]
            pred_chunks.append(pred_take)
            target_chunks.append(target_take)
            prev_last_pred = pred_take[-1].clone()
            covered += take
            chunk_index += 1

        pred_full = torch.cat(pred_chunks, dim=0).clamp(0.0, 1.0)
        target_full_eval = torch.cat(target_chunks, dim=0).clamp(0.0, 1.0)
        background_b = background.unsqueeze(0)
        pred_b = pred_full.unsqueeze(0)
        target_b = target_full_eval.unsqueeze(0)

        dist_metrics = distribution_video_metrics(
            pred_b,
            target_b,
            background=background_b,
            foreground_threshold=float(train_args.foreground_threshold),
        )
        pixel_metrics = residual_video_metrics(
            pred_b,
            target_b,
            background=background_b,
            foreground_threshold=float(train_args.foreground_threshold),
        )
        metrics = {**dist_metrics, **pixel_metrics}
        metric_rows.append(metrics)

        target_residual_panel = residual_vis(target_full_eval - background.unsqueeze(0))
        pred_residual_panel = residual_vis(pred_full - background.unsqueeze(0))
        bg_panel = background.unsqueeze(0).expand_as(pred_full)
        pred_mask_panel = residual_foreground_mask(
            pred_b - background_b.unsqueeze(1),
            float(train_args.foreground_threshold),
        )[0].repeat(1, 3, 1, 1)
        layout = torch.cat([target_full_eval, pred_full, bg_panel, target_residual_panel, pred_residual_panel, pred_mask_panel], dim=-1)

        safe_stem = f"{len(per_video_rows):03d}_{stem}_full"
        gif_path = gifs_dir / f"{safe_stem}_gt_pred_bg_resmask.gif"
        mp4_path = mp4_dir / f"{safe_stem}_pred.mp4"
        save_video_gif(layout, gif_path, fps=int(args.save_fps))
        save_video_mp4(pred_full, mp4_path, fps=int(args.save_fps))

        meta = {
            "index": len(per_video_rows),
            "stem": stem,
            "num_frames": int(pred_full.shape[0]),
            "num_chunks": int(chunk_index),
            "gif_layout": "GT | Pred | BG | GT residual abs | Pred residual abs | Pred mask",
            "prediction_source": "model(background, roi, audio chunk) -> residual -> full-video chunk concat",
            "save_fps": int(args.save_fps),
            "checkpoint": str(Path(args.checkpoint)),
            "checkpoint_step": int(raw.get("step", -1)),
            "physics_raw": row.get("physics_raw", {}),
            "condition_source": row.get("condition_source", ""),
            "gif": str(gif_path.resolve()),
            "pred_mp4": str(mp4_path.resolve()),
            "metrics": metrics,
        }
        metadata_lines.append(json.dumps(meta, ensure_ascii=False))
        per_video_rows.append(
            {
                "index": meta["index"],
                "stem": stem,
                "num_frames": int(pred_full.shape[0]),
                "num_chunks": int(chunk_index),
                "gif": str(gif_path.resolve()),
                "pred_mp4": str(mp4_path.resolve()),
                **metrics,
            }
        )

    aggregate = aggregate_flow_metrics(metric_rows)
    aggregate["distribution_score"] = distribution_score(aggregate)
    summary = {
        "num_samples": len(per_video_rows),
        "full_video_rollout": True,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(raw.get("step", -1)),
        "save_fps": int(args.save_fps),
        "eval_mode": "deterministic_residual_full_video_rollout",
        "metrics": aggregate,
        "metadata_jsonl": str((out_dir / "samples_metadata.jsonl").resolve()),
        "per_video_csv": str((out_dir / "per_video_metrics.csv").resolve()),
        "metrics_summary_csv": str((out_dir / "metrics_summary.csv").resolve()),
    }
    (out_dir / "samples_metadata.jsonl").write_text(
        "\n".join(metadata_lines) + ("\n" if metadata_lines else ""),
        encoding="utf-8",
    )
    (out_dir / "samples_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_per_video_csv(out_dir / "per_video_metrics.csv", per_video_rows)
    experiment_id = out_dir.name
    summary_row = {
        "experiment_id": experiment_id,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(raw.get("step", -1)),
        "num_samples": len(per_video_rows),
        "eval_mode": "deterministic_residual_full_video_rollout",
        "save_fps": int(args.save_fps),
        **aggregate,
    }
    write_summary_csv(out_dir / "metrics_summary.csv", summary_row)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
