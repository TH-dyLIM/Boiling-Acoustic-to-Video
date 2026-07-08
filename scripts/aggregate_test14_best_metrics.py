from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from residual_video.dataset import tensor_u8_to_01  # noqa: E402
from residual_video.metrics import video_metrics as residual_video_metrics  # noqa: E402


MODEL_DIRS = [
    "flow_residual_csv8_noprior_c128",
    "flow_residual_csv8_noprior_c64_stft",
    "flow_residual_csv8_moviegen_token_xl_noprior_d768_p8",
    "flow_residual_csv8_moviegen_token_noprior_d512",
    "flow_residual_csv8_noprior_c64",
    "flow_residual_csv8_noprior_c32",
    "flow_residual_csv8_large_main_noprior",
]

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


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


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


def cache_path_for(meta: dict[str, Any], single_895: bool) -> Path:
    index = int(meta.get("index", 0))
    stem = str(meta.get("stem", ""))
    if single_895:
        cache_dir = PROJECT_ROOT / "cache" / "svd_csv8_residual_new_895chf_3" / "test"
    else:
        cache_dir = PROJECT_ROOT / "cache" / "svd_csv8_residual_new" / "test"
    exact = cache_dir / f"{index:05d}_{stem}.pt"
    if exact.exists():
        return exact
    candidates = sorted(cache_dir.glob(f"{index:05d}_*.pt"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Cache not found for index={index}, stem={stem}, dir={cache_dir}")


def mean(values: list[float]) -> float | str:
    return float(sum(values) / len(values)) if values else ""


def aggregate_key(rows: list[dict[str, float]], key: str) -> float | str:
    values = [float(row[key]) for row in rows if key in row and row[key] is not None]
    return mean(values)


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


def metric_row_from_sample_dir(
    sample_dir: Path,
    foreground_threshold: float,
    single_895: bool,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    metadata_path = sample_dir / "samples_metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    flow_rows: list[dict[str, float]] = []
    edge_rows: list[dict[str, float]] = []
    for meta in read_jsonl(metadata_path):
        metrics = meta.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError(f"Missing metrics in {metadata_path}: {meta}")
        flow_rows.append({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})

        pred_path = resolve_path(meta["pred_mp4"])
        pred = read_mp4_rgb(pred_path)
        cache = torch.load(cache_path_for(meta, single_895=single_895), map_location="cpu")
        target = tensor_u8_to_01(cache["frames_u8"])
        background = tensor_u8_to_01(cache["background_u8"])
        n = min(int(pred.shape[0]), int(target.shape[0]))
        edge_rows.append(
            residual_video_metrics(
                pred[:n].unsqueeze(0),
                target[:n].unsqueeze(0),
                background=background.unsqueeze(0),
                foreground_threshold=foreground_threshold,
            )
        )
    return flow_rows, edge_rows


def aggregate_model(model_dir_name: str) -> dict[str, Any]:
    model_dir = PROJECT_ROOT / "outputs" / model_dir_name
    train_args = load_json(model_dir / "train_args.json")
    foreground_threshold = float(train_args.get("foreground_threshold", 0.04))

    sample_dirs = [
        (model_dir / "test_full_rollout_best", False),
        (model_dir / "test_895chf_3_best", True),
    ]
    flow_rows: list[dict[str, float]] = []
    edge_rows: list[dict[str, float]] = []
    for sample_dir, single_895 in sample_dirs:
        rows_a, rows_b = metric_row_from_sample_dir(sample_dir, foreground_threshold, single_895)
        flow_rows.extend(rows_a)
        edge_rows.extend(rows_b)

    if len(flow_rows) != 14 or len(edge_rows) != 14:
        raise ValueError(f"{model_dir_name}: expected 14 rows, got flow={len(flow_rows)}, edge={len(edge_rows)}")

    metrics: dict[str, Any] = {}
    flow_metric_names = [
        "mse_video",
        "mae_video",
        "psnr_video",
        "void_fraction_mae",
        "pred_void_fraction_mean",
        "target_void_fraction_mean",
        "void_fraction_ks",
        "nucleation_count_mae",
        "pred_nucleation_count_mean",
        "target_nucleation_count_mean",
        "nucleation_count_ks",
        "departure_freq_mae",
        "pred_departure_freq_mean",
        "target_departure_freq_mean",
        "spatial_density_l1",
    ]
    for name in flow_metric_names:
        metrics[f"{name}_mean"] = aggregate_key(flow_rows, name)

    edge_metric_names = [
        "edge_mae_video",
        "fg_mae_video",
        "fg_coverage",
        "pred_fg_coverage",
        "bubble_mask_iou",
    ]
    for name in edge_metric_names:
        metrics[f"{name}_mean"] = aggregate_key(edge_rows, name)
    metrics["distribution_score"] = distribution_score(metrics)

    summary_13 = load_json(model_dir / "test_full_rollout_best" / "samples_summary.json")
    summary_895 = load_json(model_dir / "test_895chf_3_best" / "samples_summary.json")
    row = {
        "experiment_id": model_dir_name,
        "stem_set": "test_full_rollout_13_plus_895chf_3",
        "num_samples": 14,
        "eval_mode": "full_video_rollout_test14",
        "checkpoint": str(model_dir / "best.pt"),
        "checkpoint_step": summary_895.get("checkpoint_step", summary_13.get("checkpoint_step", "")),
        "prior_mode": "none",
        "roi_mode": summary_895.get("roi_mode", summary_13.get("roi_mode", "normal")),
        "source_dirs": [
            str(model_dir / "test_full_rollout_best"),
            str(model_dir / "test_895chf_3_best"),
        ],
        "metric_note": "Aggregated from 13 original test full-rollout predictions plus the newly added 895chf_3 prediction.",
    }
    for col in METRIC_COLUMNS:
        row[col] = metrics.get(col, "")
    return row


def write_one_row_csv(path: Path, row: dict[str, Any]) -> None:
    base_cols = [
        "experiment_id",
        "stem_set",
        "num_samples",
        "eval_mode",
        "checkpoint",
        "checkpoint_step",
        "prior_mode",
        "roi_mode",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_cols + METRIC_COLUMNS)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in base_cols + METRIC_COLUMNS})


def write_markdown(path: Path, row: dict[str, Any]) -> None:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = [
        f"# Test14 Best Metrics: {row['experiment_id']}",
        "",
        f"- num_samples: `{row['num_samples']}`",
        f"- eval_mode: `{row['eval_mode']}`",
        f"- checkpoint: `{row['checkpoint']}`",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for col in METRIC_COLUMNS:
        lines.append(f"| {col} | {fmt(row.get(col, ''))} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_combined_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        "experiment_id",
        "psnr_video_mean",
        "edge_mae_video_mean",
        "fg_mae_video_mean",
        "bubble_mask_iou_mean",
        "void_fraction_mae_mean",
        "nucleation_count_mae_mean",
        "departure_freq_mae_mean",
        "spatial_density_l1_mean",
        "distribution_score",
    ]

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = [
        "# Test14 Best Metrics",
        "",
        "Aggregated from the original 13 test full-rollout predictions plus `895chf_3`.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows: list[dict[str, Any]] = []
    for model_dir_name in MODEL_DIRS:
        row = aggregate_model(model_dir_name)
        model_dir = PROJECT_ROOT / "outputs" / model_dir_name
        write_json(model_dir / "test14_best_metrics.json", row)
        write_one_row_csv(model_dir / "test14_best_metrics.csv", row)
        write_markdown(model_dir / "test14_best_metrics.md", row)
        rows.append(row)

    out_dir = PROJECT_ROOT / "outputs" / "benchmark_metrics_csv_audio_new_noprior"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cols = [
        "experiment_id",
        "stem_set",
        "num_samples",
        "eval_mode",
        "checkpoint",
        "checkpoint_step",
        "prior_mode",
        "roi_mode",
    ]
    combined_csv = out_dir / "test14_best_all_models_metrics.csv"
    with combined_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_cols + METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in base_cols + METRIC_COLUMNS})
    write_json(out_dir / "test14_best_all_models_metrics.json", rows)
    write_combined_markdown(out_dir / "test14_best_all_models_metrics.md", rows)
    print(json.dumps({"saved_models": len(rows), "combined_csv": str(combined_csv)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
