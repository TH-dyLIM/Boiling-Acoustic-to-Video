from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from flow_residual.dataset import ChunkResidualDataset, collate_fn
from flow_residual.flow import euler_sample
from flow_residual.metrics import (
    aggregate_metrics,
    compose_video,
    foreground_mask,
    video_metrics,
)
from flow_residual.model import FlowResidualUNet
from flow_residual.token_model import DiTVideoTokenFlowTransformer, VideoTokenFlowTransformer
from residual_video.dataset import audio_segment_from_cached, tensor_u8_to_01
from svd_audio_control.video_io import save_video_gif, save_video_mp4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "manifests" / "test.jsonl"))
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--prior_path", default="")
    parser.add_argument("--prior_mode", choices=["per_class", "global", "none"], default="")
    parser.add_argument("--roi_mode", choices=["normal", "none"], default="")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "flow_residual_chunk_test_samples"))
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--num_chunks_per_clip", type=int, default=1)
    parser.add_argument(
        "--full_video_rollout",
        action="store_true",
        help="Generate each full test video sequentially. The last predicted frame of a chunk is used as the next chunk condition.",
    )
    parser.add_argument("--save_fps", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    return parser.parse_args()


def args_from_ckpt(ckpt_args: dict[str, Any]) -> argparse.Namespace:
    defaults = dict(
        resolution=128,
        chunk_frames=12,
        frame_stride=1,
        video_fps=100,
        audio_sample_rate=1_000_000,
        model_type="unet3d",
        base_channels=64,
        audio_dim=64,
        audio_tokens=16,
        scalar_feature_dim=6,
        physics_dim=3,
        time_dim=128,
        cond_dim=256,
        patch_size=16,
        transformer_dim=384,
        transformer_depth=8,
        transformer_heads=8,
        transformer_mlp_ratio=4.0,
        transformer_checkpoint=False,
        dropout=0.0,
        residual_scale=2.0,
        background_mode="normal",
        prediction_mode="residual",
        use_prev_frame=True,
        foreground_threshold=0.04,
        audio_normalize=True,
        audio_clamp_value=0.0,
        use_stft=False,
        stft_n_fft=2048,
        stft_hop_length=1024,
        stft_n_freq_bins=64,
        stft_fmin=100.0,
        stft_fmax=0.0,
    )
    defaults.update(ckpt_args)
    return argparse.Namespace(**defaults)


def make_model(train_args: argparse.Namespace) -> FlowResidualUNet:
    if getattr(train_args, "model_type", "unet3d") == "dit_token_transformer":
        return DiTVideoTokenFlowTransformer(
            chunk_frames=train_args.chunk_frames,
            resolution=train_args.resolution,
            patch_size=train_args.patch_size,
            hidden_dim=train_args.transformer_dim,
            depth=train_args.transformer_depth,
            num_heads=train_args.transformer_heads,
            mlp_ratio=train_args.transformer_mlp_ratio,
            audio_dim=train_args.audio_dim,
            audio_tokens=train_args.audio_tokens,
            scalar_feature_dim=train_args.scalar_feature_dim,
            physics_dim=train_args.physics_dim,
            time_dim=train_args.time_dim,
            cond_dim=train_args.cond_dim,
            dropout=train_args.dropout,
            residual_scale=train_args.residual_scale,
            use_prev_frame=train_args.use_prev_frame,
            use_checkpoint=False,
            use_refine_head=bool(getattr(train_args, "use_refine_head", False)),
        )
    if getattr(train_args, "model_type", "unet3d") == "token_transformer":
        return VideoTokenFlowTransformer(
            chunk_frames=train_args.chunk_frames,
            resolution=train_args.resolution,
            patch_size=train_args.patch_size,
            hidden_dim=train_args.transformer_dim,
            depth=train_args.transformer_depth,
            num_heads=train_args.transformer_heads,
            mlp_ratio=train_args.transformer_mlp_ratio,
            audio_dim=train_args.audio_dim,
            audio_tokens=train_args.audio_tokens,
            scalar_feature_dim=train_args.scalar_feature_dim,
            physics_dim=train_args.physics_dim,
            time_dim=train_args.time_dim,
            cond_dim=train_args.cond_dim,
            dropout=train_args.dropout,
            residual_scale=train_args.residual_scale,
            use_prev_frame=train_args.use_prev_frame,
            use_checkpoint=False,
        )
    return FlowResidualUNet(
        chunk_frames=train_args.chunk_frames,
        base_channels=train_args.base_channels,
        audio_dim=train_args.audio_dim,
        audio_tokens=train_args.audio_tokens,
        scalar_feature_dim=train_args.scalar_feature_dim,
        physics_dim=train_args.physics_dim,
        time_dim=train_args.time_dim,
        cond_dim=train_args.cond_dim,
        dropout=train_args.dropout,
        residual_scale=train_args.residual_scale,
        use_prev_frame=train_args.use_prev_frame,
        use_stft=bool(getattr(train_args, "use_stft", False)),
        stft_freq_bins=int(getattr(train_args, "stft_n_freq_bins", 64)),
    )


def model_background(train_args: argparse.Namespace, background: torch.Tensor) -> torch.Tensor:
    if getattr(train_args, "background_mode", "normal") == "none":
        return torch.zeros_like(background)
    return background


def compose_prediction(train_args: argparse.Namespace, sampled: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    scale = max(float(train_args.residual_scale), 1e-6)
    if getattr(train_args, "prediction_mode", "residual") == "video":
        return (sampled / scale).clamp(0.0, 1.0)
    return compose_video(background, sampled / scale, clamp=True)


def make_layout_video(
    target: torch.Tensor,
    pred: torch.Tensor,
    background: torch.Tensor,
    pred_mask: torch.Tensor,
) -> torch.Tensor:
    # target/pred are single-sample clips: T,3,H,W. background is 3,H,W,
    # pred_mask is T,1,H,W. Build panels with the same T,3,H,W shape.
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
    train_args = args_from_ckpt(raw.get("args", {}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.mixed_precision != "no"
    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16

    model = make_model(train_args).to(device)
    missing, unexpected = model.load_state_dict(raw["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    model.eval()

    fixed_starts = sorted({int(c * 30) for c in range(int(args.num_chunks_per_clip))}) or [0]
    cache_dir = args.cache_dir or None
    prior_path = args.prior_path or None
    prior_mode = args.prior_mode or getattr(train_args, "prior_mode", "per_class")
    roi_mode = args.roi_mode or getattr(train_args, "roi_mode", "normal")

    audio_kwargs = dict(
        audio_normalize=bool(getattr(train_args, "audio_normalize", True)),
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
        cache_dir=cache_dir,
        prior_path=prior_path,
        prior_mode=prior_mode,
        roi_mode=roi_mode,
        fixed_starts_per_clip=fixed_starts,
        **audio_kwargs,
    )

    if args.full_video_rollout:
        if cache_dir is None:
            raise ValueError("--full_video_rollout requires --cache_dir so full frame/audio tensors can be read safely.")

        out_dir = Path(args.output_dir)
        gifs_dir = out_dir / "gifs"
        mp4_dir = out_dir / "pred_mp4"
        gifs_dir.mkdir(parents=True, exist_ok=True)
        mp4_dir.mkdir(parents=True, exist_ok=True)

        metadata_lines: list[str] = []
        rows: list[dict[str, float]] = []
        video_count = 0
        for row_index, row in enumerate(tqdm(dataset.rows)):
            if int(args.max_samples) > 0 and video_count >= int(args.max_samples):
                break
            cache = dataset._load_cache(row_index, row)
            if cache is None:
                raise FileNotFoundError(
                    f"Full rollout cache not found for row {row_index} ({row.get('stem', '')}). "
                    f"Run precompute_residual_tensor_cache first or remove --full_video_rollout."
                )

            total_frames = int(cache["frames_u8"].shape[0])
            pred_chunks: list[torch.Tensor] = []
            target_chunks: list[torch.Tensor] = []
            prev_last_pred: torch.Tensor | None = None
            background_cpu: torch.Tensor | None = None
            covered = 0
            chunk_index = 0

            use_stft = bool(getattr(train_args, "use_stft", False))
            while covered < total_frames:
                item = rollout_item_from_cache(dataset, row_index, row, cache, covered, prev_last_pred)
                background = item["background"].unsqueeze(0).to(device)
                background_cond = model_background(train_args, background)
                roi = item["roi"].unsqueeze(0).to(device)
                prior = item["nucleation_prior"].unsqueeze(0).to(device)
                prev_last = item["prev_last_frame"].unsqueeze(0).to(device)
                audio = item["audio"].unsqueeze(0).to(device)
                scalar_features = item["audio_features"].unsqueeze(0).to(device)
                physics = item["physics"].unsqueeze(0).to(device)
                target_video = item["pixel_values"].unsqueeze(0).to(device)
                audio_stft = item["audio_stft"].unsqueeze(0).to(device) if use_stft else None
                shape = (1, train_args.chunk_frames, 3, train_args.resolution, train_args.resolution)

                def cond_velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                        return model(
                            x, t, background_cond, roi, prior,
                            prev_last if train_args.use_prev_frame else None,
                            audio, scalar_features, physics,
                            audio_stft=audio_stft,
                        ).float()

                if float(args.cfg_scale) > 0.0:
                    zero_mask = torch.zeros(1, device=device)

                    def uncond_velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                            return model(
                                x, t, background_cond, roi, prior,
                                prev_last if train_args.use_prev_frame else None,
                                audio, scalar_features, physics,
                                cond_dropout_mask=zero_mask,
                                audio_stft=audio_stft,
                            ).float()
                else:
                    uncond_velocity = None

                sampled = euler_sample(
                    cond_velocity,
                    shape,
                    device,
                    num_steps=int(args.num_inference_steps),
                    cfg_scale=float(args.cfg_scale),
                    uncond_velocity_fn=uncond_velocity,
                    seed=int(args.seed) + row_index * 1000 + chunk_index,
                )
                pred_video = compose_prediction(train_args, sampled, background)

                take = min(int(train_args.chunk_frames), total_frames - covered)
                pred_take = pred_video[0, :take].detach().cpu()
                target_take = target_video[0, :take].detach().cpu()
                pred_chunks.append(pred_take)
                target_chunks.append(target_take)
                prev_last_pred = pred_take[-1].clone()
                background_cpu = item["background"].detach().cpu()
                covered += take
                chunk_index += 1

            if not pred_chunks or background_cpu is None:
                continue
            pred_full = torch.cat(pred_chunks, dim=0)
            target_full = torch.cat(target_chunks, dim=0)
            background_batch = background_cpu.unsqueeze(0)
            pred_batch = pred_full.unsqueeze(0)
            target_batch = target_full.unsqueeze(0)
            pred_mask = foreground_mask(
                pred_batch - background_batch.unsqueeze(1),
                train_args.foreground_threshold,
            )[0]
            metrics = video_metrics(
                pred_batch,
                target_batch,
                background=background_batch,
                foreground_threshold=train_args.foreground_threshold,
            )
            rows.append(metrics)

            stem = row.get("stem", Path(row.get("video", "")).stem)
            safe_stem = f"{video_count:03d}_{stem}_full"
            layout = make_layout_video(target_full, pred_full, background_cpu, pred_mask).cpu()
            gif_path = gifs_dir / f"{safe_stem}_gt_pred_bg_mask.gif"
            mp4_path = mp4_dir / f"{safe_stem}_pred.mp4"
            save_video_gif(layout.unsqueeze(0), gif_path, fps=int(args.save_fps))
            save_video_mp4(pred_full.cpu(), mp4_path, fps=int(args.save_fps))

            meta = {
                "index": video_count,
                "stem": stem,
                "start_frame": 0,
                "num_frames": int(pred_full.shape[0]),
                "num_chunks": int(chunk_index),
                "autoregressive_prev_frame": True,
                "physics_raw": row.get("physics_raw", {}),
                "condition_source": row.get("condition_source", ""),
                "gif": str(gif_path),
                "pred_mp4": str(mp4_path),
                "num_inference_steps": int(args.num_inference_steps),
                "cfg_scale": float(args.cfg_scale),
                "background_mode": getattr(train_args, "background_mode", "normal"),
                "prediction_mode": getattr(train_args, "prediction_mode", "residual"),
                "metrics": metrics,
            }
            metadata_lines.append(json.dumps(meta, ensure_ascii=False))
            video_count += 1

        (out_dir / "samples_metadata.jsonl").write_text(
            "\n".join(metadata_lines) + ("\n" if metadata_lines else ""),
            encoding="utf-8",
        )
        summary = {
            "num_samples": len(metadata_lines),
            "full_video_rollout": True,
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": int(raw.get("step", 0)),
            "num_inference_steps": int(args.num_inference_steps),
            "cfg_scale": float(args.cfg_scale),
            "prior_mode": prior_mode,
            "roi_mode": roi_mode,
            "background_mode": getattr(train_args, "background_mode", "normal"),
            "prediction_mode": getattr(train_args, "prediction_mode", "residual"),
            "metrics": aggregate_metrics(rows),
        }
        (out_dir / "samples_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    out_dir = Path(args.output_dir)
    gifs_dir = out_dir / "gifs"
    mp4_dir = out_dir / "pred_mp4"
    gifs_dir.mkdir(parents=True, exist_ok=True)
    mp4_dir.mkdir(parents=True, exist_ok=True)

    metadata_lines: list[str] = []
    rows: list[dict[str, float]] = []
    use_stft = bool(getattr(train_args, "use_stft", False))
    for index, batch in enumerate(tqdm(loader)):
        if int(args.max_samples) > 0 and index >= int(args.max_samples):
            break
        background = batch["background"].to(device)
        background_cond = model_background(train_args, background)
        roi = batch["roi"].to(device)
        prior = batch["nucleation_prior"].to(device)
        prev_last = batch["prev_last_frame"].to(device)
        audio = batch["audio"].to(device)
        scalar_features = batch["audio_features"].to(device)
        physics = batch["physics"].to(device)
        target_video = batch["pixel_values"].to(device)
        audio_stft = batch["audio_stft"].to(device) if use_stft else None
        bsz = background.shape[0]
        shape = (bsz, train_args.chunk_frames, 3, train_args.resolution, train_args.resolution)

        def cond_velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                return model(
                    x, t, background_cond, roi, prior,
                    prev_last if train_args.use_prev_frame else None,
                    audio, scalar_features, physics,
                    audio_stft=audio_stft,
                ).float()

        if float(args.cfg_scale) > 0.0:
            zero_mask = torch.zeros(bsz, device=device)

            def uncond_velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    return model(
                        x, t, background_cond, roi, prior,
                        prev_last if train_args.use_prev_frame else None,
                        audio, scalar_features, physics,
                        cond_dropout_mask=zero_mask,
                        audio_stft=audio_stft,
                    ).float()
        else:
            uncond_velocity = None

        sampled = euler_sample(
            cond_velocity,
            shape,
            device,
            num_steps=int(args.num_inference_steps),
            cfg_scale=float(args.cfg_scale),
            uncond_velocity_fn=uncond_velocity,
            seed=int(args.seed) + index,
        )
        pred_video = compose_prediction(train_args, sampled, background)
        pred_mask = foreground_mask(pred_video - background.unsqueeze(1), train_args.foreground_threshold)

        metrics = video_metrics(
            pred_video,
            target_video,
            background=background,
            foreground_threshold=train_args.foreground_threshold,
        )
        rows.append(metrics)

        stem = batch["stem"][0]
        start_frame = batch["start_frame"][0]
        safe_stem = f"{index:03d}_{stem}_s{start_frame:04d}"
        layout = make_layout_video(target_video[0], pred_video[0], background[0], pred_mask[0]).cpu()
        gif_path = gifs_dir / f"{safe_stem}_gt_pred_bg_mask.gif"
        mp4_path = mp4_dir / f"{safe_stem}_pred.mp4"
        save_video_gif(layout.unsqueeze(0), gif_path, fps=int(args.save_fps))
        save_video_mp4(pred_video[0].cpu(), mp4_path, fps=int(args.save_fps))

        meta = {
            "index": index,
            "stem": stem,
            "start_frame": int(start_frame),
            "physics_raw": batch["physics_raw"][0],
            "condition_source": batch["condition_source"][0],
            "gif": str(gif_path),
            "pred_mp4": str(mp4_path),
            "num_inference_steps": int(args.num_inference_steps),
            "cfg_scale": float(args.cfg_scale),
            "prior_mode": prior_mode,
            "roi_mode": roi_mode,
            "background_mode": getattr(train_args, "background_mode", "normal"),
            "prediction_mode": getattr(train_args, "prediction_mode", "residual"),
            "metrics": metrics,
        }
        metadata_lines.append(json.dumps(meta, ensure_ascii=False))

    (out_dir / "samples_metadata.jsonl").write_text("\n".join(metadata_lines) + ("\n" if metadata_lines else ""), encoding="utf-8")
    summary = {
        "num_samples": len(metadata_lines),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(raw.get("step", 0)),
        "num_inference_steps": int(args.num_inference_steps),
        "cfg_scale": float(args.cfg_scale),
        "prior_mode": prior_mode,
        "roi_mode": roi_mode,
        "background_mode": getattr(train_args, "background_mode", "normal"),
        "prediction_mode": getattr(train_args, "prediction_mode", "residual"),
        "metrics": aggregate_metrics(rows),
    }
    (out_dir / "samples_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
