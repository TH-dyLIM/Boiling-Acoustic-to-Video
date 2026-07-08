from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from flow_residual.dataset import read_jsonl, safe_cache_stem  # noqa: E402
from flow_residual.flow import euler_sample  # noqa: E402
from flow_residual.metrics import compose_video  # noqa: E402
from flow_residual.model import FlowResidualUNet  # noqa: E402
from residual_video.dataset import audio_scalar_features, audio_segment_from_cached  # noqa: E402
from svd_audio_control.video_io import save_video_gif, save_video_mp4, tensor_to_pil  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run flow_noprior_c128_rawamp inference on test audio with "
            "virtual background/ROI conditions."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "outputs" / "flow_residual_csv8_noprior_c128_rawamp" / "best.pt"),
    )
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "manifests_csv_new" / "test.jsonl"))
    parser.add_argument("--cache_dir", default=str(PROJECT_ROOT / "cache" / "svd_csv8_residual_new" / "test"))
    parser.add_argument("--virtual_dir", default=str(PROJECT_ROOT / "0virtualtest"))
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "outputs" / "virtualtest_flow_noprior_c128_rawamp"),
    )
    parser.add_argument("--num_inference_steps", type=int, default=24)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--save_fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--max_videos", type=int, default=0, help="0 means all test videos.")
    parser.add_argument("--max_cases", type=int, default=0, help="0 means all virtual cases.")
    parser.add_argument("--case_ids", default="", help="Comma-separated case ids, e.g. test1,test3.")
    parser.add_argument("--stems", default="", help="Comma-separated test stems to infer, e.g. 40_6,200_3.")
    parser.add_argument("--preview_frames", type=int, default=6)
    parser.add_argument("--save_gif", action="store_true", help="Also save GIF previews.")
    return parser.parse_args()


def args_from_ckpt(ckpt_args: dict[str, Any]) -> argparse.Namespace:
    defaults = dict(
        resolution=128,
        chunk_frames=8,
        frame_stride=1,
        video_fps=100,
        audio_sample_rate=1_000_000,
        base_channels=128,
        audio_dim=96,
        audio_tokens=24,
        scalar_feature_dim=6,
        physics_dim=3,
        time_dim=192,
        cond_dim=384,
        dropout=0.05,
        residual_scale=2.0,
        use_prev_frame=True,
        foreground_threshold=0.04,
        audio_normalize=False,
        audio_clamp_value=10.0,
        use_stft=False,
        stft_n_freq_bins=64,
    )
    defaults.update(ckpt_args)
    return argparse.Namespace(**defaults)


def make_model(train_args: argparse.Namespace) -> FlowResidualUNet:
    model_type = getattr(train_args, "model_type", "unet3d")
    if model_type != "unet3d":
        raise ValueError(f"This virtual inference script expects a U-Net flow model, got {model_type!r}")
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


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def load_cache(cache_dir: Path, row_index: int, row: dict[str, Any]) -> dict[str, Any]:
    stem = row.get("stem", Path(row.get("video", "")).stem)
    path = cache_dir / f"{safe_cache_stem(row_index, stem)}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing cache for {stem}: {path}")
    return torch.load(path, map_location="cpu")


def pil_to_01(path: Path, resolution: int, mode: str) -> torch.Tensor:
    img = Image.open(path).convert(mode)
    if mode == "L":
        img = img.resize((resolution, resolution), Image.Resampling.NEAREST)
        arr = torch.from_numpy(np.asarray(img, dtype="float32"))[None] / 255.0
        return (arr > 0.5).float()
    img = img.resize((resolution, resolution), Image.Resampling.BICUBIC)
    arr = torch.from_numpy(np.asarray(img, dtype="float32")).permute(2, 0, 1) / 255.0
    return arr.clamp(0.0, 1.0)


def discover_virtual_cases(virtual_dir: Path) -> list[dict[str, Any]]:
    cases: dict[str, dict[str, Path]] = {}
    for path in sorted(virtual_dir.iterdir()):
        if not path.is_file():
            continue
        m = re.match(r"(.+?)_(bg|roi)\.(jpg|jpeg|png|bmp|tif|tiff)$", path.name, flags=re.IGNORECASE)
        if not m:
            continue
        case_id, kind = m.group(1), m.group(2).lower()
        cases.setdefault(case_id, {})[kind] = path
    out = []
    for case_id, pair in sorted(cases.items()):
        if "bg" not in pair or "roi" not in pair:
            raise FileNotFoundError(f"Incomplete virtual case {case_id}: {pair}")
        out.append({"case_id": case_id, "background_path": pair["bg"], "roi_path": pair["roi"]})
    if not out:
        raise FileNotFoundError(f"No *_bg + *_roi pairs found in {virtual_dir}")
    return out


def compose_prediction(train_args: argparse.Namespace, sampled: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    scale = max(float(train_args.residual_scale), 1e-6)
    return compose_video(background, sampled / scale, clamp=True)


def residual_abs_video(pred: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    return (pred - background.unsqueeze(1)).abs().amax(dim=2, keepdim=True).expand(-1, -1, 3, -1, -1).clamp(0, 1)


def save_preview_frames(pred_full: torch.Tensor, out_dir: Path, count: int) -> None:
    if count <= 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    total = int(pred_full.shape[0])
    if total <= 0:
        return
    if count >= total:
        indices = list(range(total))
    else:
        indices = torch.linspace(0, total - 1, count).round().long().tolist()
    for idx in indices:
        img = tensor_to_pil(pred_full[int(idx)])
        img.save(out_dir / f"frame_{int(idx):06d}.png")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    manifest = Path(args.manifest)
    cache_dir = Path(args.cache_dir)
    virtual_dir = Path(args.virtual_dir)
    output_dir = Path(args.output_dir)

    raw = torch.load(checkpoint, map_location="cpu")
    train_args = args_from_ckpt(raw.get("args", {}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.mixed_precision != "no"
    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16

    model = make_model(train_args).to(device)
    model.load_state_dict(raw["model"], strict=True)
    model.eval()

    rows = read_jsonl_rows(manifest)
    indexed_rows = list(enumerate(rows))
    if args.stems.strip():
        requested_stems = [x.strip() for x in args.stems.split(",") if x.strip()]
        row_by_stem = {
            str(row.get("stem", Path(row.get("video", "")).stem)): (row_index, row)
            for row_index, row in indexed_rows
        }
        missing_stems = [stem for stem in requested_stems if stem not in row_by_stem]
        if missing_stems:
            raise ValueError(f"Requested stems not found in manifest: {missing_stems}")
        indexed_rows = [row_by_stem[stem] for stem in requested_stems]
    if int(args.max_videos) > 0:
        indexed_rows = indexed_rows[: int(args.max_videos)]

    cases = discover_virtual_cases(virtual_dir)
    if args.case_ids.strip():
        keep = {x.strip() for x in args.case_ids.split(",") if x.strip()}
        cases = [case for case in cases if case["case_id"] in keep]
    if int(args.max_cases) > 0:
        cases = cases[: int(args.max_cases)]
    if not cases:
        raise ValueError("No virtual cases selected.")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(raw.get("step", 0)),
        "manifest": str(manifest),
        "cache_dir": str(cache_dir),
        "virtual_dir": str(virtual_dir),
        "output_dir": str(output_dir),
        "num_cases": len(cases),
        "num_test_videos": len(indexed_rows),
        "selected_stems": [
            str(row.get("stem", Path(row.get("video", "")).stem)) for _, row in indexed_rows
        ],
        "num_inference_steps": int(args.num_inference_steps),
        "cfg_scale": float(args.cfg_scale),
        "seed": int(args.seed),
        "model": {
            "type": "FlowResidualUNet",
            "base_channels": int(train_args.base_channels),
            "audio_dim": int(train_args.audio_dim),
            "audio_tokens": int(train_args.audio_tokens),
            "chunk_frames": int(train_args.chunk_frames),
            "resolution": int(train_args.resolution),
            "residual_scale": float(train_args.residual_scale),
            "audio_normalize": bool(getattr(train_args, "audio_normalize", False)),
            "audio_clamp_value": float(getattr(train_args, "audio_clamp_value", 0.0)),
            "prior_mode": "none",
        },
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    total_jobs = len(cases) * len(indexed_rows)
    pbar = tqdm(total=total_jobs, desc="virtual inference")
    for case_index, case in enumerate(cases):
        case_id = case["case_id"]
        case_out = output_dir / case_id
        pred_dir = case_out / "pred_mp4"
        residual_dir = case_out / "residual_abs_mp4"
        preview_dir = case_out / "preview_frames"
        gif_dir = case_out / "gifs"
        pred_dir.mkdir(parents=True, exist_ok=True)
        residual_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)
        if args.save_gif:
            gif_dir.mkdir(parents=True, exist_ok=True)

        virtual_bg_cpu = pil_to_01(Path(case["background_path"]), int(train_args.resolution), "RGB")
        virtual_roi_cpu = pil_to_01(Path(case["roi_path"]), int(train_args.resolution), "L")
        zero_prior_cpu = torch.zeros(1, int(train_args.resolution), int(train_args.resolution), dtype=torch.float32)
        physics_cpu = torch.zeros(int(train_args.physics_dim), dtype=torch.float32)

        tensor_to_pil(virtual_bg_cpu).save(case_out / f"{case_id}_background_128.png")
        tensor_to_pil(virtual_roi_cpu.expand(3, -1, -1)).save(case_out / f"{case_id}_roi_128.png")

        metadata_lines: list[str] = []
        for selected_index, (row_index, row) in enumerate(indexed_rows):
            cache = load_cache(cache_dir, row_index, row)
            stem = row.get("stem", Path(row.get("video", "")).stem)
            total_frames = int(cache["frames_u8"].shape[0])
            fps = float(cache.get("fps", train_args.video_fps) or train_args.video_fps)
            cached_audio = cache.get("audio", None)
            audio_sr = int(cache.get("audio_sr", train_args.audio_sample_rate))
            if cached_audio is None:
                raise ValueError(f"Cache has no full audio tensor for {stem}")
            if audio_sr != int(train_args.audio_sample_rate):
                raise ValueError(f"Audio SR mismatch for {stem}: expected {train_args.audio_sample_rate}, got {audio_sr}")

            pred_chunks: list[torch.Tensor] = []
            prev_last_cpu = virtual_bg_cpu.clone()
            covered = 0
            chunk_index = 0
            while covered < total_frames:
                duration_sec = (int(train_args.chunk_frames) * int(train_args.frame_stride)) / fps
                audio = audio_segment_from_cached(
                    cached_audio,
                    start_sec=covered / fps,
                    duration_sec=duration_sec,
                    target_sr=int(train_args.audio_sample_rate),
                    target_len=int(round(duration_sec * int(train_args.audio_sample_rate))),
                    normalize=bool(getattr(train_args, "audio_normalize", False)),
                    clamp_value=(
                        float(train_args.audio_clamp_value)
                        if float(getattr(train_args, "audio_clamp_value", 0.0)) > 0.0
                        else None
                    ),
                )
                scalar_features = audio_scalar_features(audio)

                background = virtual_bg_cpu.unsqueeze(0).to(device)
                roi = virtual_roi_cpu.unsqueeze(0).to(device)
                prior = zero_prior_cpu.unsqueeze(0).to(device)
                prev_last = prev_last_cpu.unsqueeze(0).to(device)
                audio = audio.unsqueeze(0).to(device)
                scalar_features = scalar_features.unsqueeze(0).to(device)
                physics = physics_cpu.unsqueeze(0).to(device)
                shape = (
                    1,
                    int(train_args.chunk_frames),
                    3,
                    int(train_args.resolution),
                    int(train_args.resolution),
                )

                def cond_velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                        return model(
                            x,
                            t,
                            background,
                            roi,
                            prior,
                            prev_last if bool(train_args.use_prev_frame) else None,
                            audio,
                            scalar_features,
                            physics,
                        ).float()

                if float(args.cfg_scale) > 0.0:
                    zero_mask = torch.zeros(1, device=device)

                    def uncond_velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                            return model(
                                x,
                                t,
                                background,
                                roi,
                                prior,
                                prev_last if bool(train_args.use_prev_frame) else None,
                                audio,
                                scalar_features,
                                physics,
                                cond_dropout_mask=zero_mask,
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
                seed=int(args.seed) + case_index * 100000 + row_index * 1000 + chunk_index,
                )
                pred_video = compose_prediction(train_args, sampled, background)[0].detach().cpu()
                take = min(int(train_args.chunk_frames), total_frames - covered)
                pred_take = pred_video[:take]
                pred_chunks.append(pred_take)
                prev_last_cpu = pred_take[-1].clone()
                covered += take
                chunk_index += 1

            pred_full = torch.cat(pred_chunks, dim=0)
            background_batch = virtual_bg_cpu.unsqueeze(0)
            residual_abs = residual_abs_video(pred_full.unsqueeze(0), background_batch)[0]

            safe_name = f"{selected_index:03d}_{stem}"
            pred_mp4 = pred_dir / f"{safe_name}_virtual_{case_id}_pred.mp4"
            residual_mp4 = residual_dir / f"{safe_name}_virtual_{case_id}_abs_residual.mp4"
            save_video_mp4(pred_full, pred_mp4, fps=int(args.save_fps))
            save_video_mp4(residual_abs, residual_mp4, fps=int(args.save_fps))
            save_preview_frames(pred_full, preview_dir / safe_name, int(args.preview_frames))
            if args.save_gif:
                layout = torch.cat(
                    [
                        virtual_bg_cpu.unsqueeze(0).expand_as(pred_full),
                        virtual_roi_cpu.expand(3, -1, -1).unsqueeze(0).expand_as(pred_full),
                        pred_full,
                        residual_abs,
                    ],
                    dim=-1,
                )
                save_video_gif(layout, gif_dir / f"{safe_name}_bg_roi_pred_residual.gif", fps=int(args.save_fps))

            meta = {
                "case_id": case_id,
                "stem": stem,
                "row_index": row_index,
                "selected_index": selected_index,
                "num_frames": int(pred_full.shape[0]),
                "num_chunks": int(chunk_index),
                "fps_source": fps,
                "save_fps": int(args.save_fps),
                "background_path": str(case["background_path"]),
                "roi_path": str(case["roi_path"]),
                "pred_mp4": str(pred_mp4),
                "residual_abs_mp4": str(residual_mp4),
                "preview_frames_dir": str(preview_dir / safe_name),
                "uses_test_audio": True,
                "uses_virtual_background_roi": True,
                "uses_nucleation_prior": False,
                "autoregressive_prev_frame": bool(train_args.use_prev_frame),
            }
            metadata_lines.append(json.dumps(meta, ensure_ascii=False))
            pbar.update(1)

        (case_out / "samples_metadata.jsonl").write_text(
            "\n".join(metadata_lines) + ("\n" if metadata_lines else ""),
            encoding="utf-8",
        )
    pbar.close()
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
