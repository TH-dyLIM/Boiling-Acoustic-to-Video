from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from train_transformer_physics_stft import (
    WavWindowDataset,
    build_heat_maps,
    build_regime_maps,
    compute_scalers,
    expand_rows,
    load_labels,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = PROJECT_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute and save 100 ms STFT window cache for physics transformers.")
    parser.add_argument(
        "--labels-xlsx",
        default=str((WORK_ROOT / "Results_Pool_100%_Spectrogram" / "audio_wav_physics_labels_8011.xlsx").resolve()),
    )
    parser.add_argument(
        "--output-dir",
        default=str((WORK_ROOT / "Results_Pool_100%_Spectrogram" / "transformer_wav100ms_separate").resolve()),
    )
    parser.add_argument("--target-sr", type=int, default=1_000_000)
    parser.add_argument("--window-sec", type=float, default=0.10)
    parser.add_argument("--train-windows-per-file", type=int, default=12)
    parser.add_argument("--eval-windows-per-file", type=int, default=12)
    parser.add_argument("--window-selection", default="energy_topk", choices=["uniform", "energy_topk"])
    parser.add_argument("--nfft", type=int, default=1000)
    parser.add_argument("--win", type=int, default=50)
    parser.add_argument("--hop", type=int, default=50)
    parser.add_argument("--frmax", type=float, default=300000.0)
    parser.add_argument("--dbmin", type=float, default=30.0)
    parser.add_argument("--dbmax", type=float, default=90.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated subset of train,val,test.")
    parser.add_argument("--force", type=int, default=0, help="Delete and regenerate matching cached tensors.")
    return parser.parse_args()


def make_records(labels_df: pd.DataFrame, args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    for split in ["train", "val", "test"]:
        split_df = labels_df[labels_df["split"] == split].reset_index(drop=True)
        if split == "train":
            windows_per_file = args.train_windows_per_file
        else:
            windows_per_file = args.eval_windows_per_file
        records[split] = expand_rows(split_df, args.target_sr, args.window_sec, windows_per_file, args.window_selection)
    return records


def precompute_split(
    split: str,
    records: list[dict[str, Any]],
    scalers: dict[str, float],
    regime_to_index: dict[int, int],
    heat_to_index: dict[float, int],
    cache_root: Path,
    args: argparse.Namespace,
) -> dict[str, int]:
    dataset = WavWindowDataset(
        records,
        scalers,
        regime_to_index,
        heat_to_index,
        target_sr=args.target_sr,
        nfft=args.nfft,
        win=args.win,
        hop=args.hop,
        frmax=args.frmax,
        dbmin=args.dbmin,
        dbmax=args.dbmax,
        image_size=args.image_size,
        cache_dir=cache_root / split,
    )
    generated = 0
    existing = 0
    forced = 0
    for index, row in enumerate(tqdm(dataset.records, desc=f"Precompute {split}", leave=False)):
        cache_path = dataset._cache_path(row)
        if cache_path.exists() and int(args.force):
            cache_path.unlink()
            forced += 1
        if cache_path.exists():
            existing += 1
            continue
        _ = dataset[index]
        generated += 1
    return {
        "total": len(records),
        "existing": existing,
        "generated": generated,
        "forced_deleted": forced,
    }


def main() -> None:
    args = parse_args()
    labels_df = load_labels(Path(args.labels_xlsx))
    train_df = labels_df[labels_df["split"] == "train"].reset_index(drop=True)
    if train_df.empty:
        raise ValueError("Missing train split in label workbook.")
    scalers = compute_scalers(train_df)
    regime_to_index, _ = build_regime_maps(labels_df)
    heat_to_index, _ = build_heat_maps(labels_df)

    selected_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    invalid = selected_splits.difference({"train", "val", "test"})
    if invalid:
        raise ValueError(f"Invalid split names: {sorted(invalid)}")

    records = make_records(labels_df, args)
    output_dir = Path(args.output_dir)
    cache_root = output_dir / "stft_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "labels_xlsx": str(Path(args.labels_xlsx).resolve()),
        "output_dir": str(output_dir.resolve()),
        "cache_root": str(cache_root.resolve()),
        "window_sec": float(args.window_sec),
        "window_selection": str(args.window_selection),
        "splits": {},
    }
    for split in ["train", "val", "test"]:
        if split not in selected_splits:
            continue
        summary["splits"][split] = precompute_split(
            split,
            records[split],
            scalers,
            regime_to_index,
            heat_to_index,
            cache_root,
            args,
        )

    summary_path = output_dir / "stft_cache_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
