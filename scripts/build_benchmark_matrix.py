from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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

BASE_COLUMNS = [
    "experiment_id",
    "family",
    "method",
    "audio_input",
    "visual_condition",
    "target",
    "objective",
    "prior",
    "roi",
    "base_channels",
    "eval_mode",
    "status",
    "num_samples",
    "checkpoint_step",
    "summary_path",
    "notes",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def is_prior_free(entry: dict[str, Any]) -> bool:
    prior = str(entry.get("prior", "")).strip().lower()
    return prior in {"none", "no", "false", "0", ""}


def resolve_path(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def candidate_summary_paths(entry: dict[str, Any]) -> list[Path]:
    out: list[Path] = []
    for raw in entry.get("summary_candidates", []):
        p = resolve_path(raw)
        if p is None:
            continue
        out.append(p / "samples_summary.json" if p.suffix == "" else p)

    explicit = resolve_path(entry.get("summary_path", ""))
    if explicit is not None:
        out.append(explicit / "samples_summary.json" if explicit.suffix == "" else explicit)

    root = resolve_path(entry.get("output_dir", ""))
    if root is not None:
        preferred_subdirs = [
            "test_full_rollout_best",
            "test_samples_best",
            "test_full_rollout_last",
            "test_samples_last",
            "samples_best",
            "samples_last",
        ]
        out.append(root / "samples_summary.json")
        out.extend(root / sub / "samples_summary.json" for sub in preferred_subdirs)
    # Keep order while removing duplicates.
    unique: list[Path] = []
    seen: set[str] = set()
    for p in out:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def find_summary(entry: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    for path in candidate_summary_paths(entry):
        if path.exists():
            return path, load_json(path)
    return None, None


def has_any_output(entry: dict[str, Any]) -> bool:
    root = resolve_path(entry.get("output_dir", ""))
    if root is None or not root.exists():
        return False
    patterns = ["*.pt", "*.ckpt", "*.mp4", "*.gif", "progress.csv", "metrics.jsonl", "train.log"]
    for pattern in patterns:
        if any(root.rglob(pattern)):
            return True
    return False


def metric_value(metrics: dict[str, Any], key: str) -> Any:
    value = metrics.get(key, "")
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return value


def compute_distribution_score(metrics: dict[str, Any]) -> str:
    required = ["void_fraction_mae_mean", "spatial_density_l1_mean", "nucleation_count_mae_mean"]
    if not all(k in metrics for k in required):
        return ""
    void = float(metrics.get("void_fraction_mae_mean", 0.0))
    spatial = float(metrics.get("spatial_density_l1_mean", 0.0))
    nuc_mae = float(metrics.get("nucleation_count_mae_mean", 0.0))
    nuc_target = max(float(metrics.get("target_nucleation_count_mean_mean", 1.0)), 1e-6)
    dep_mae = float(metrics.get("departure_freq_mae_mean", 0.0))
    dep_target = max(float(metrics.get("target_departure_freq_mean_mean", 1.0)), 1e-6)
    score = void + 0.8 * spatial + 0.4 * (nuc_mae / nuc_target) + 0.02 * (dep_mae / dep_target)
    return f"{score:.6g}"


def build_row(entry: dict[str, Any]) -> dict[str, Any]:
    summary_path, summary = find_summary(entry)
    row = {col: entry.get(col, "") for col in BASE_COLUMNS}
    row["experiment_id"] = entry.get("id", entry.get("experiment_id", ""))
    row["summary_path"] = str(summary_path.relative_to(PROJECT_ROOT)) if summary_path and summary_path.is_relative_to(PROJECT_ROOT) else (str(summary_path) if summary_path else "")

    if summary is None:
        row["status"] = "trained_no_standard_test" if has_any_output(entry) else entry.get("status", "pending")
        for col in METRIC_COLUMNS:
            row[col] = ""
        return row

    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    merged_metrics = {**summary, **metrics}
    row["status"] = entry.get("status", "complete")
    row["num_samples"] = summary.get("num_samples", "")
    row["checkpoint_step"] = summary.get("checkpoint_step", "")
    if summary.get("full_video_rollout"):
        row["eval_mode"] = "full_video_rollout"
    elif not row.get("eval_mode"):
        row["eval_mode"] = "chunk_or_sample"

    for col in METRIC_COLUMNS:
        row[col] = metric_value(merged_metrics, col)
    row["distribution_score"] = compute_distribution_score(merged_metrics)
    return row


def fmt_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.5g}"
    text = str(value)
    if text.replace(".", "", 1).replace("-", "", 1).isdigit():
        try:
            return f"{float(text):.5g}"
        except ValueError:
            return text
    return text.replace("|", "\\|")


def write_markdown(rows: list[dict[str, Any]], out_path: Path) -> None:
    show_cols = [
        "experiment_id",
        "family",
        "audio_input",
        "prior",
        "roi",
        "base_channels",
        "eval_mode",
        "status",
        "psnr_video_mean",
        "mae_video_mean",
        "void_fraction_mae_mean",
        "nucleation_count_mae_mean",
        "departure_freq_mae_mean",
        "spatial_density_l1_mean",
        "bubble_mask_iou_mean",
        "distribution_score",
    ]
    lines = []
    lines.append("| " + " | ".join(show_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(show_cols)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt_cell(row.get(col, "")) for col in show_cols) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_schema(out_path: Path) -> None:
    text = """# Benchmark Metric Schema

This table is the common test board for the CSV-audio boiling-video dataset.

Default benchmark policy:

- Prior-conditioned experiments are excluded by default because nucleation priors require video-derived spatial statistics that are not available in real deployment.
- Use `--include_prior` only for archival/internal comparison.

Primary metrics for stochastic video generation:

- `void_fraction_mae_mean`: frame-wise active boiling area error. Lower is better.
- `nucleation_count_mae_mean`: connected foreground component count error after 4x downsampling. Lower is better.
- `departure_freq_mae_mean`: temporal 0-to-1 foreground transition count error. Lower is better.
- `spatial_density_l1_mean`: L1 distance between predicted and target foreground density maps. Lower is better.
- `distribution_score`: composite score used only for quick ranking: `void + 0.8*spatial + 0.4*nucleation_rel + 0.02*departure_rel`. Lower is better.

Secondary sanity metrics:

- `psnr_video_mean`, `mae_video_mean`, `mse_video_mean`: pixel-level sanity checks. They can reward blurry averages, so do not use them alone.
- `edge_mae_video_mean`, `fg_mae_video_mean`, `bubble_mask_iou_mean`: useful for deterministic residual/PatchGAN baselines.

Fair-comparison rule:

- Final paper tables should prefer `full_video_rollout` for models that can roll out full 100-frame test clips.
- Older deterministic/PatchGAN/SVD sample folders may be chunk-level only; keep `eval_mode` visible so those numbers are not over-interpreted.
"""
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "configs" / "benchmark_matrix_csv_audio_new.json"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "benchmark_metrics_csv_audio_new_noprior"))
    parser.add_argument(
        "--include_prior",
        action="store_true",
        help="Include prior-conditioned experiments. Default benchmark is prior-free only.",
    )
    args = parser.parse_args()

    registry_path = resolve_path(args.registry)
    if registry_path is None or not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {args.registry}")
    registry = load_json(registry_path)
    all_entries = registry.get("experiments", [])
    entries = all_entries if args.include_prior else [entry for entry in all_entries if is_prior_free(entry)]
    if not entries:
        raise ValueError(f"No experiments in registry: {registry_path}")

    rows = [build_row(entry) for entry in entries]
    out_dir = resolve_path(args.output_dir)
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "benchmark_matrix.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_COLUMNS + METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    write_markdown(rows, out_dir / "benchmark_matrix.md")
    write_schema(out_dir / "metric_schema.md")

    missing = [row for row in rows if row.get("status") != "complete"]
    with (out_dir / "missing_or_pending_experiments.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_COLUMNS + METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(missing)

    summary = {
        "registry": str(registry_path),
        "num_experiments": len(rows),
        "include_prior": bool(args.include_prior),
        "filtered_prior_experiments": int(len(all_entries) - len(entries)),
        "complete": sum(1 for row in rows if row.get("status") == "complete"),
        "pending_or_missing": len(missing),
        "csv": str(csv_path),
        "markdown": str(out_dir / "benchmark_matrix.md"),
        "schema": str(out_dir / "metric_schema.md"),
    }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
