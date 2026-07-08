from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from residual_video.metrics import video_metrics as residual_video_metrics  # noqa: E402
from residual_video.dataset import tensor_u8_to_01  # noqa: E402


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


MODEL_DIRS = [
    "flow_residual_csv8_noprior_c128",
    "flow_residual_csv8_noprior_c64_stft",
    "flow_residual_csv8_moviegen_token_xl_noprior_d768_p8",
    "flow_residual_csv8_moviegen_token_noprior_d512",
    "flow_residual_csv8_noprior_c64",
    "flow_residual_csv8_noprior_c32",
    "flow_residual_csv8_large_main_noprior",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def distribution_score(metrics: dict[str, Any]) -> float | str:
    required = ["void_fraction_mae_mean", "spatial_density_l1_mean", "nucleation_count_mae_mean"]
    if not all(k in metrics for k in required):
        return ""
    void = float(metrics.get("void_fraction_mae_mean", 0.0))
    spatial = float(metrics.get("spatial_density_l1_mean", 0.0))
    nuc_mae = float(metrics.get("nucleation_count_mae_mean", 0.0))
    nuc_target = max(float(metrics.get("target_nucleation_count_mean_mean", 1.0)), 1e-6)
    dep_mae = float(metrics.get("departure_freq_mae_mean", 0.0))
    dep_target = max(float(metrics.get("target_departure_freq_mean_mean", 1.0)), 1e-6)
    return void + 0.8 * spatial + 0.4 * (nuc_mae / nuc_target) + 0.02 * (dep_mae / dep_target)


def read_mp4_rgb(path: Path) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    arr = np.stack(frames, axis=0)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0


def mean(values: list[float]) -> float | str:
    return float(sum(values) / len(values)) if values else ""


def recompute_edge_fg(
    sample_dir: Path,
    cache_path: Path,
    foreground_threshold: float,
) -> dict[str, float | str]:
    metadata = read_jsonl(sample_dir / "samples_metadata.jsonl")
    if not metadata:
        return {}
    cache = torch.load(cache_path, map_location="cpu")
    target = tensor_u8_to_01(cache["frames_u8"])
    background = tensor_u8_to_01(cache["background_u8"])

    rows: list[dict[str, float]] = []
    for meta in metadata:
        pred_path = Path(meta["pred_mp4"])
        if not pred_path.is_absolute():
            pred_path = PROJECT_ROOT / pred_path
        pred = read_mp4_rgb(pred_path)
        n = min(int(pred.shape[0]), int(target.shape[0]))
        rows.append(
            residual_video_metrics(
                pred[:n].unsqueeze(0),
                target[:n].unsqueeze(0),
                background=background.unsqueeze(0),
                foreground_threshold=foreground_threshold,
            )
        )
    keys = ["edge_mae_video", "fg_mae_video", "fg_coverage", "pred_fg_coverage", "bubble_mask_iou"]
    return {f"{key}_mean": mean([float(row[key]) for row in rows if key in row]) for key in keys}


def write_one_row_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_cols = [
        "experiment_id",
        "stem",
        "checkpoint",
        "checkpoint_step",
        "num_samples",
        "eval_mode",
        "prior_mode",
        "roi_mode",
        "sample_dir",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_cols + METRIC_COLUMNS)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in base_cols + METRIC_COLUMNS})


def write_markdown(path: Path, row: dict[str, Any]) -> None:
    show_cols = [
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

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = [
        f"# 895chf_3 Best Prediction Metrics: {row.get('experiment_id', '')}",
        "",
        f"- checkpoint: `{row.get('checkpoint', '')}`",
        f"- sample_dir: `{row.get('sample_dir', '')}`",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for col in show_cols:
        lines.append(f"| {col} | {fmt(row.get(col, ''))} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest_and_cache() -> tuple[Path, Path]:
    dataset_root = PROJECT_ROOT.parent / "dataset_split_audio_csv_new_seed42"
    stem = "895chf_3"
    row = {
        "split": "test",
        "stem": stem,
        "video": str(dataset_root / "test" / "video_100fps" / f"{stem}.mp4"),
        "audio": str(dataset_root / "test" / "audio_csv" / f"{stem}.csv"),
        "audio_format": "csv",
        "background": str(dataset_root / "test" / "background" / f"{stem}.jpg"),
        "roi": str(dataset_root / "test" / "Heating_ROI" / f"{stem}.png"),
        "physics_norm": [0.0, 0.0, 0.0],
        "physics_raw": {},
        "condition_source": "",
    }
    missing = [value for key, value in row.items() if key in {"video", "audio", "background", "roi"} and not Path(value).exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files for {stem}: {missing}")

    manifest = PROJECT_ROOT / "manifests_csv_new" / "test_895chf_3.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    cache_root = PROJECT_ROOT / "cache" / "svd_csv8_residual_new_895chf_3"
    cache_dir = cache_root / "test"
    cache_path = cache_dir / "00000_895chf_3.pt"
    if not cache_path.exists():
        cmd = [
            sys.executable,
            "scripts/precompute_residual_tensor_cache.py",
            "--manifest",
            f"test={manifest}",
            "--cache_root",
            str(cache_root),
            "--resolution",
            "128",
            "--audio_sample_rate",
            "1000000",
            "--overwrite",
        ]
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    return manifest, cache_path


def run_sample(model_dir_name: str, manifest: Path) -> dict[str, Any]:
    model_dir = PROJECT_ROOT / "outputs" / model_dir_name
    checkpoint = model_dir / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    sample_dir = model_dir / "test_895chf_3_best"
    cache_dir = PROJECT_ROOT / "cache" / "svd_csv8_residual_new_895chf_3" / "test"
    cmd = [
        sys.executable,
        "scripts/sample_flow_residual.py",
        "--checkpoint",
        str(checkpoint),
        "--manifest",
        str(manifest),
        "--cache_dir",
        str(cache_dir),
        "--prior_mode",
        "none",
        "--output_dir",
        str(sample_dir),
        "--full_video_rollout",
        "--num_inference_steps",
        "30",
        "--cfg_scale",
        "1.0",
        "--save_fps",
        "10",
        "--mixed_precision",
        "fp16",
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    return {"model_dir": model_dir, "checkpoint": checkpoint, "sample_dir": sample_dir}


def main() -> None:
    manifest, cache_path = build_manifest_and_cache()
    all_rows: list[dict[str, Any]] = []
    for model_dir_name in MODEL_DIRS:
        info = run_sample(model_dir_name, manifest)
        model_dir = info["model_dir"]
        sample_dir = info["sample_dir"]
        summary_path = sample_dir / "samples_summary.json"
        summary = load_json(summary_path)
        metrics = dict(summary.get("metrics", {}))
        train_args = load_json(model_dir / "train_args.json")
        edge_fg = recompute_edge_fg(
            sample_dir,
            cache_path=cache_path,
            foreground_threshold=float(train_args.get("foreground_threshold", 0.04)),
        )
        metrics.update(edge_fg)
        metrics["distribution_score"] = distribution_score(metrics)

        row = {
            "experiment_id": model_dir_name,
            "stem": "895chf_3",
            "checkpoint": str(info["checkpoint"]),
            "checkpoint_step": summary.get("checkpoint_step", ""),
            "num_samples": summary.get("num_samples", ""),
            "eval_mode": "full_video_rollout_single_895chf_3",
            "prior_mode": summary.get("prior_mode", "none"),
            "roi_mode": summary.get("roi_mode", "normal"),
            "sample_dir": str(sample_dir),
        }
        for col in METRIC_COLUMNS:
            row[col] = metrics.get(col, "")

        write_json(model_dir / "895chf_3_best_metrics.json", row)
        write_one_row_csv(model_dir / "895chf_3_best_metrics.csv", row)
        write_markdown(model_dir / "895chf_3_best_metrics.md", row)
        all_rows.append(row)

    combined_dir = PROJECT_ROOT / "outputs" / "benchmark_metrics_csv_audio_new_noprior"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = combined_dir / "895chf_3_best_all_models_metrics.csv"
    base_cols = [
        "experiment_id",
        "stem",
        "checkpoint",
        "checkpoint_step",
        "num_samples",
        "eval_mode",
        "prior_mode",
        "roi_mode",
        "sample_dir",
    ]
    with combined_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_cols + METRIC_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key, "") for key in base_cols + METRIC_COLUMNS})
    write_json(combined_dir / "895chf_3_best_all_models_metrics.json", all_rows)
    print(json.dumps({"saved_models": len(all_rows), "combined_csv": str(combined_csv)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
