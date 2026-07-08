from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from residual_video.dataset import (
    _image_01,
    _read_video_rgb,
    _rgb_frame_to_01,
    audio_scalar_features,
    read_audio_segment,
    read_jsonl,
)
from residual_video.metrics import compose_video, video_metrics
from residual_video.model import AudioResidualUNet
from svd_audio_control.video_io import save_video_gif, save_video_mp4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "manifests" / "test.jsonl"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "residual_frame_ar_samples"))
    parser.add_argument("--max_samples", type=int, default=13)
    parser.add_argument("--rollout_frames", type=int, default=10)
    parser.add_argument("--save_fps", type=int, default=10)
    return parser.parse_args()


def args_from_ckpt(ckpt_args: dict[str, Any]) -> argparse.Namespace:
    defaults = {
        "resolution": 256,
        "num_frames": 1,
        "frame_stride": 1,
        "video_fps": 100,
        "audio_sample_rate": 1_000_000,
        "audio_channels": 32,
        "base_channels": 32,
        "residual_scale": 1.0,
        "residual_target_scale": 1.0,
        "use_mask_head": False,
        "mask_bias_init": 0.0,
        "foreground_threshold": 0.035,
        "dropout": 0.0,
        "physics_dim": 3,
        "physics_channels": 8,
        "audio_feature_dim": 6,
        "audio_feature_channels": 8,
        "use_previous_frame_condition": True,
        "prediction_base": "background",
        "audio_context_frames": 1,
        "audio_context_future_frames": 0,
    }
    defaults.update(ckpt_args)
    return argparse.Namespace(**defaults)


def make_model(args: argparse.Namespace) -> AudioResidualUNet:
    return AudioResidualUNet(
        num_frames=1,
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
        use_previous_frame_condition=True,
    )


def split_model_output(output: torch.Tensor | dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(output, dict):
        return output["residual"], output.get("mask")
    return output, None


@torch.no_grad()
def main() -> None:
    args = parse_args()
    raw = torch.load(args.checkpoint, map_location="cpu")
    train_args = args_from_ckpt(raw.get("args", {}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(train_args).to(device)
    missing, unexpected = model.load_state_dict(raw["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys in {args.checkpoint}: {unexpected}")
    model.eval()

    rows = read_jsonl(args.manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "samples_metadata.jsonl"
    metric_rows = []
    with meta_path.open("w", encoding="utf-8", newline="\n") as meta_f:
        for index, row in enumerate(rows[: int(args.max_samples)]):
            frames_rgb, fps = _read_video_rgb(row["video"])
            bg_path = row.get("background", "")
            if bg_path and Path(bg_path).exists():
                background = _image_01(bg_path, train_args.resolution, "RGB")
            else:
                background = _rgb_frame_to_01(frames_rgb[0], train_args.resolution)
            roi_path = row.get("roi", "")
            if roi_path and Path(roi_path).exists():
                roi = _image_01(roi_path, train_args.resolution, "L")
            else:
                roi = torch.zeros(1, train_args.resolution, train_args.resolution, dtype=torch.float32)

            background_b = background.unsqueeze(0).to(device)
            roi_b = roi.unsqueeze(0).to(device)
            physics = torch.tensor(row.get("physics_norm", [0.0, 0.0, 0.0]), dtype=torch.float32).unsqueeze(0).to(device)
            previous_frame = background.unsqueeze(0).to(device)
            context_frames = max(1, int(getattr(train_args, "audio_context_frames", 1)))
            future_frames = max(0, min(int(getattr(train_args, "audio_context_future_frames", 0)), context_frames - 1))
            past_frames = context_frames - future_frames - 1
            duration_sec = context_frames * float(train_args.frame_stride) / float(fps)
            target_len = int(round(duration_sec * int(train_args.audio_sample_rate)))
            pred_frames = []
            target_frames = []
            mask_frames = []

            for step in range(int(args.rollout_frames)):
                frame_index = min(step * int(train_args.frame_stride), frames_rgb.shape[0] - 1)
                target_frame = _rgb_frame_to_01(frames_rgb[frame_index], train_args.resolution)
                audio_start_index = frame_index - past_frames * int(train_args.frame_stride)
                start_sec = audio_start_index / float(fps)
                audio = read_audio_segment(
                    row["audio"],
                    start_sec=start_sec,
                    duration_sec=duration_sec,
                    target_sr=int(train_args.audio_sample_rate),
                    target_len=target_len,
                )
                audio_features = audio_scalar_features(audio)
                pred_residual, pred_mask = split_model_output(
                    model(
                        background_b,
                        roi_b,
                        audio.unsqueeze(0).to(device),
                        physics=physics,
                        audio_features=audio_features.unsqueeze(0).to(device),
                        previous_frame=previous_frame,
                    )
                )
                pred_video = compose_video(
                    background_b,
                    pred_residual,
                    clamp=True,
                    mask=pred_mask,
                    residual_target_scale=train_args.residual_target_scale,
                    base_frame=previous_frame if train_args.prediction_base == "previous_frame" else None,
                )
                pred_frame = pred_video[:, 0]
                pred_frames.append(pred_frame[0].detach().cpu())
                target_frames.append(target_frame)
                if pred_mask is not None:
                    mask_frames.append(pred_mask[0, 0].detach().cpu().repeat(3, 1, 1))
                previous_frame = pred_frame.detach()

            pred_seq = torch.stack(pred_frames, dim=0)
            target_seq = torch.stack(target_frames, dim=0)
            pred_b = pred_seq.unsqueeze(0).to(device)
            target_b = target_seq.unsqueeze(0).to(device)
            metrics = video_metrics(
                pred_b,
                target_b,
                background=background_b,
                foreground_threshold=train_args.foreground_threshold,
            )
            metric_rows.append(metrics)
            bg_seq = background.unsqueeze(0).repeat(pred_seq.shape[0], 1, 1, 1)
            if mask_frames:
                mask_seq = torch.stack(mask_frames, dim=0)
            else:
                mask_seq = torch.zeros_like(pred_seq)
            side_by_side = torch.cat([target_seq, pred_seq, bg_seq, mask_seq], dim=-1)
            stem = row.get("stem", Path(row["video"]).stem)
            gif_path = out_dir / "gifs" / f"{index:03d}_{stem}_gt_pred_bg_mask_ar.gif"
            mp4_path = out_dir / "pred_mp4" / f"{index:03d}_{stem}_pred_ar.mp4"
            save_video_gif(side_by_side, gif_path, fps=args.save_fps)
            save_video_mp4(pred_seq, mp4_path, fps=args.save_fps)
            meta = {
                "index": index,
                "stem": stem,
                "rollout_frames": int(args.rollout_frames),
                "audio_window_sec": duration_sec,
                "audio_context_frames": context_frames,
                "audio_context_future_frames": future_frames,
                "prediction_source": "autoregressive: previous predicted frame + next 10 ms audio -> next frame",
                "prediction_base": train_args.prediction_base,
                "gif_layout": "GT | Pred autoregressive | BG | Pred mask",
                "gif": str(gif_path.resolve()),
                "pred_mp4": str(mp4_path.resolve()),
                "physics_raw": row.get("physics_raw", {}),
                "condition_source": row.get("condition_source", ""),
                **metrics,
            }
            meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            print(json.dumps(meta, ensure_ascii=False))

    if metric_rows:
        keys = sorted({key for metric in metric_rows for key in metric})
        summary = {
            "num_samples": len(metric_rows),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_step": int(raw.get("step", -1)),
            "metadata_jsonl": str(meta_path.resolve()),
            "metrics": {
                f"{key}_mean": float(sum(row[key] for row in metric_rows if key in row) / sum(1 for row in metric_rows if key in row))
                for key in keys
            },
        }
        (out_dir / "samples_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
