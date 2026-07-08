from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = PROJECT_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build wav-level physics labels from prefix metadata.")
    parser.add_argument(
        "--split-root",
        default=str((WORK_ROOT / "dataset_split_8011_seed42").resolve()),
        help="Root folder containing train/val/test/audio_wav.",
    )
    parser.add_argument(
        "--source-labels",
        default=str((WORK_ROOT / "Results_Pool_100%_Spectrogram" / "audio_poolboiling_labels.xlsx").resolve()),
        help="Prefix label workbook.",
    )
    parser.add_argument(
        "--output-xlsx",
        default=str((WORK_ROOT / "Results_Pool_100%_Spectrogram" / "audio_wav_physics_labels_8011.xlsx").resolve()),
        help="Output wav-level Excel workbook.",
    )
    parser.add_argument(
        "--extra-csv-audio-dir",
        default=str((WORK_ROOT / "dataset_preprocessing" / "audio_211105").resolve()),
        help="Optional folder of extra CSV audio files. The first number in the filename is used as heat flux.",
    )
    parser.add_argument(
        "--extra-csv-split",
        default="train",
        choices=["train", "val", "test"],
        help="Split assigned to extra CSV audio files.",
    )
    return parser.parse_args()


def load_prefix_labels(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = ["Prefix Label Name", "HeatFlux", "HTC", "BoilingRegime"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    df = df.copy()
    df["Prefix Label Name"] = df["Prefix Label Name"].astype(str)
    return df


def match_prefix(stem: str, prefixes: list[str]) -> str | None:
    for prefix in prefixes:
        if stem == prefix or stem.startswith(prefix + "_"):
            return prefix
    return None


def csv_audio_info(path: Path) -> tuple[int, int, float]:
    with path.open("rb") as f:
        line_count = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1024 * 1024), b""))
    num_samples = max(0, int(line_count) - 1)
    preview = pd.read_csv(path, nrows=4)
    if "Time" in preview.columns and preview.shape[0] >= 2:
        dt = float(preview["Time"].iloc[1]) - float(preview["Time"].iloc[0])
        sample_rate = int(round(1.0 / dt)) if dt > 0 else 1_000_000
    else:
        sample_rate = 1_000_000
    duration = float(num_samples / sample_rate) if sample_rate > 0 else 0.0
    return sample_rate, num_samples, duration


def first_number_as_heat_flux(stem: str) -> float | None:
    match = re.match(r"^(\d+(?:\.\d+)?)", stem)
    return float(match.group(1)) if match else None


def nearest_physics_for_heat_flux(source_df: pd.DataFrame, heat_flux: float) -> dict[str, object]:
    grouped = (
        source_df.groupby("HeatFlux")
        .agg(
            htc=("HTC", "median"),
            boiling_regime=("BoilingRegime", lambda s: int(s.mode().iloc[0]) if not s.mode().empty else int(round(float(s.median())))),
        )
        .reset_index()
    )
    idx = (grouped["HeatFlux"].astype(float) - float(heat_flux)).abs().idxmin()
    row = grouped.loc[idx]
    return {
        "matched_prefix": f"nearest_heat_flux:{float(row['HeatFlux']):g}",
        "htc": float(row["htc"]),
        "boiling_regime": int(row["boiling_regime"]),
        "experiment_date": "audio_211105_nearest_heat_flux",
    }


def collect_rows(
    split_root: Path,
    source_df: pd.DataFrame,
    extra_csv_audio_dir: Path | None = None,
    extra_csv_split: str = "train",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_map = source_df.set_index("Prefix Label Name").to_dict(orient="index")
    prefixes = sorted(source_map.keys(), key=len, reverse=True)

    rows: list[dict[str, object]] = []
    unmatched: list[dict[str, object]] = []

    for split in ["train", "val", "test"]:
        audio_dir = split_root / split / "audio_wav"
        if not audio_dir.exists():
            raise FileNotFoundError(f"Missing audio folder: {audio_dir}")
        for wav_path in sorted(audio_dir.glob("*.wav")):
            stem = wav_path.stem
            prefix = match_prefix(stem, prefixes)
            info = sf.info(str(wav_path))
            base = {
                "split": split,
                "wav_name": wav_path.name,
                "stem": stem,
                "wav_path": str(wav_path.resolve()),
                "sample_rate": int(info.samplerate),
                "num_samples": int(info.frames),
                "duration_sec": float(info.frames / info.samplerate),
                "matched_prefix": prefix or "",
                "data_source": "dataset_split_8011_seed42",
                "audio_format": "wav",
            }
            if prefix is None:
                unmatched.append(base)
                continue

            src = source_map[prefix]
            rows.append(
                {
                    **base,
                    "heat_flux": float(src["HeatFlux"]),
                    "htc": float(src["HTC"]),
                    "boiling_regime": int(src["BoilingRegime"]),
                    "experiment_date": "" if pd.isna(src.get("Experiment Date")) else str(src.get("Experiment Date")),
                }
            )

    if extra_csv_audio_dir is not None and extra_csv_audio_dir.exists():
        for csv_path in sorted(extra_csv_audio_dir.glob("*.csv")):
            stem = csv_path.stem
            heat_flux = first_number_as_heat_flux(stem)
            sample_rate, num_samples, duration_sec = csv_audio_info(csv_path)
            base = {
                "split": extra_csv_split,
                "wav_name": csv_path.name,
                "stem": stem,
                "wav_path": str(csv_path.resolve()),
                "sample_rate": int(sample_rate),
                "num_samples": int(num_samples),
                "duration_sec": float(duration_sec),
                "data_source": "audio_211105_csv",
                "audio_format": "csv",
            }
            if heat_flux is None:
                unmatched.append({**base, "matched_prefix": ""})
                continue
            physics = nearest_physics_for_heat_flux(source_df, heat_flux)
            rows.append(
                {
                    **base,
                    "matched_prefix": physics["matched_prefix"],
                    "heat_flux": float(heat_flux),
                    "htc": float(physics["htc"]),
                    "boiling_regime": int(physics["boiling_regime"]),
                    "experiment_date": physics["experiment_date"],
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(unmatched)


def build_summary(labels_df: pd.DataFrame, unmatched_df: pd.DataFrame) -> pd.DataFrame:
    by_split = (
        labels_df.groupby("split")
        .agg(
            wav_count=("wav_name", "count"),
            heat_flux_min=("heat_flux", "min"),
            heat_flux_max=("heat_flux", "max"),
            htc_min=("htc", "min"),
            htc_max=("htc", "max"),
        )
        .reset_index()
    )
    if unmatched_df.shape[0] > 0 and "split" in unmatched_df.columns:
        unmatched_counts = unmatched_df.groupby("split").size()
    else:
        unmatched_counts = pd.Series(dtype="int64")
    by_split["unmatched_count"] = by_split["split"].map(unmatched_counts).fillna(0).astype(int)
    return by_split


def main() -> None:
    args = parse_args()
    split_root = Path(args.split_root)
    source_labels = Path(args.source_labels)
    output_xlsx = Path(args.output_xlsx)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    source_df = load_prefix_labels(source_labels)
    extra_csv_audio_dir = Path(args.extra_csv_audio_dir) if args.extra_csv_audio_dir else None
    labels_df, unmatched_df = collect_rows(split_root, source_df, extra_csv_audio_dir, args.extra_csv_split)
    if unmatched_df.shape[0] > 0:
        raise RuntimeError(f"Found unmatched wav files: {unmatched_df['wav_name'].tolist()}")

    summary_df = build_summary(labels_df, unmatched_df)
    labels_df = labels_df.sort_values(["split", "stem"]).reset_index(drop=True)
    source_df = source_df.sort_values("Prefix Label Name").reset_index(drop=True)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        labels_df.to_excel(writer, index=False, sheet_name="wav_labels")
        source_df.to_excel(writer, index=False, sheet_name="source_prefix_labels")
        summary_df.to_excel(writer, index=False, sheet_name="summary")
        unmatched_df.to_excel(writer, index=False, sheet_name="unmatched")

    print(f"Saved wav label workbook: {output_xlsx}")
    print(f"Matched wavs: {len(labels_df)}")
    print(f"Unmatched wavs: {len(unmatched_df)}")


if __name__ == "__main__":
    main()
