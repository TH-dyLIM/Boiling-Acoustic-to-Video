from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from residual_video.dataset import ResidualVideoManifestDataset, collate_fn
from residual_video.metrics import compose_video, effective_residual, video_metrics
from residual_video.model import AudioResidualUNet
from svd_audio_control.video_io import save_video_gif, save_video_mp4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "manifests" / "test.jsonl"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "residual_video_samples"))
    parser.add_argument("--max_samples", type=int, default=13)
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

    dataset = ResidualVideoManifestDataset(
        args.manifest,
        resolution=train_args.resolution,
        num_frames=train_args.num_frames,
        frame_stride=train_args.frame_stride,
        audio_sample_rate=train_args.audio_sample_rate,
        video_fps=train_args.video_fps,
        random_start=False,
        load_audio=True,
        audio_context_frames=train_args.audio_context_frames,
        audio_context_future_frames=train_args.audio_context_future_frames,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "samples_metadata.jsonl"
    metric_rows = []
    summary_rows = []

    with meta_path.open("w", encoding="utf-8", newline="\n") as meta_f:
        for i, batch in enumerate(loader):
            if i >= int(args.max_samples):
                break
            background = batch["background"].to(device)
            roi = batch["roi"].to(device)
            audio = batch["audio"].to(device)
            previous_frame = batch["previous_frame"].to(device)
            physics = batch["physics"].to(device)
            audio_features = batch["audio_features"].to(device)
            target = batch["pixel_values"].to(device)
            pred_residual, pred_mask = split_model_output(
                model(
                    background,
                    roi,
                    audio,
                    physics=physics,
                    audio_features=audio_features,
                    previous_frame=previous_frame,
                )
            )
            pred = compose_video(
                background,
                pred_residual,
                clamp=True,
                mask=pred_mask,
                residual_target_scale=train_args.residual_target_scale,
                base_frame=previous_frame if train_args.prediction_base == "previous_frame" else None,
            )

            metrics = video_metrics(
                pred,
                target,
                background=background,
                previous_frame=previous_frame,
                pred_mask=pred_mask,
                foreground_threshold=train_args.foreground_threshold,
            )
            metric_rows.append(metrics)
            regime = batch["physics_raw"][0].get("regime", "unknown") if isinstance(batch["physics_raw"][0], dict) else "unknown"
            summary_rows.append({"regime": str(regime), **metrics})
            gt = target[0].cpu()
            pred_cpu = pred[0].cpu()
            bg = background[0].unsqueeze(0).repeat(gt.shape[0], 1, 1, 1).cpu()
            pred_effective_residual = effective_residual(
                pred_residual,
                pred_mask,
                train_args.residual_target_scale,
            )[0].cpu()
            target_residual = (target - background.unsqueeze(1))[0].cpu()
            pred_residual_panel = residual_vis(pred_effective_residual)
            target_residual_panel = residual_vis(target_residual)
            if pred_mask is not None:
                mask_panel = pred_mask[0].detach().cpu().repeat(1, 3, 1, 1)
            else:
                mask_panel = torch.zeros_like(pred_residual_panel)
            max_abs_gt_pred = float((gt - pred_cpu).abs().max().item())
            pred_gt_exact_equal = bool(torch.equal(gt, pred_cpu))
            pred_bg_mse = float(torch.mean((pred_cpu - bg) ** 2).item())
            side_by_side = torch.cat([gt, pred_cpu, bg, target_residual_panel, pred_residual_panel, mask_panel], dim=-1)
            stem = batch["stem"][0]
            gif_path = out_dir / "gifs" / f"{i:03d}_{stem}_gt_pred_bg_resmask.gif"
            mp4_path = out_dir / "pred_mp4" / f"{i:03d}_{stem}_pred.mp4"
            save_video_gif(side_by_side, gif_path, fps=args.save_fps)
            save_video_mp4(pred_cpu, mp4_path, fps=args.save_fps)
            row = {
                "index": i,
                "stem": stem,
                "start_frame": batch["start_frame"][0],
                "physics_raw": batch["physics_raw"][0],
                "condition_source": batch["condition_source"][0],
                "gif_layout": "GT | Pred | BG | GT residual abs | Pred residual abs | Pred mask",
                "prediction_source": "model(background, roi, audio) -> scaled residual/mask -> background + mask * residual / residual_target_scale",
                "residual_target_scale": train_args.residual_target_scale,
                "pred_gt_exact_equal": pred_gt_exact_equal,
                "pred_gt_max_abs": max_abs_gt_pred,
                "pred_bg_mse": pred_bg_mse,
                "gif": str(gif_path.resolve()),
                "pred_mp4": str(mp4_path.resolve()),
                **metrics,
            }
            meta_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False))

    if metric_rows:
        metric_keys = sorted({key for row in metric_rows for key in row})
        metric_means = {
            f"{key}_mean": float(sum(row[key] for row in metric_rows if key in row) / sum(1 for row in metric_rows if key in row))
            for key in metric_keys
        }
        by_regime = {}
        for regime in sorted({row["regime"] for row in summary_rows}):
            rows = [row for row in summary_rows if row["regime"] == regime]
            by_regime[regime] = {
                f"{key}_mean": float(sum(row[key] for row in rows if key in row) / sum(1 for row in rows if key in row))
                for key in metric_keys
                if any(key in row for row in rows)
            }
            by_regime[regime]["num_samples"] = len(rows)
        summary = {
            "num_samples": len(metric_rows),
            "mse_video_mean": metric_means.get("mse_video_mean", 0.0),
            "mae_video_mean": metric_means.get("mae_video_mean", 0.0),
            "psnr_video_mean": metric_means.get("psnr_video_mean", 0.0),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_step": int(raw.get("step", -1)),
            "metadata_jsonl": str(meta_path.resolve()),
            "metrics": metric_means,
            "metrics_by_regime": by_regime,
        }
        (out_dir / "samples_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
