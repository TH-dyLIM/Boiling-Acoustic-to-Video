from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from svd_audio_control.video_io import IMG_EXTS, VIDEO_EXTS, find_with_ext


def load_physics_map(path: str | Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            split = row.get("split", "")
            stem = row.get("stem", "")
            if not split or not stem:
                continue
            out[(split, stem)] = {
                "physics_norm": [
                    float(row.get("cond_heat_flux_norm", 0.0) or 0.0),
                    float(row.get("cond_htc_norm", 0.0) or 0.0),
                    float(row.get("cond_regime_norm", 0.0) or 0.0),
                ],
                "physics_raw": {
                    "heat_flux": float(row.get("pred_heat_flux", 0.0) or 0.0),
                    "htc": float(row.get("pred_htc", 0.0) or 0.0),
                    "regime": float(row.get("pred_regime", 0.0) or 0.0),
                },
                "condition_source": row.get("condition_source", ""),
            }
    return out


def build_split_rows(
    split_root: Path,
    split: str,
    physics: dict[tuple[str, str], dict[str, Any]],
    audio_subdir: str = "audio_wav",
    audio_ext: str = ".wav",
) -> list[dict[str, Any]]:
    root = split_root / split
    video_dir = root / "video_100fps"
    audio_dir = root / audio_subdir
    bg_dir = root / "background"
    roi_dir = root / "Heating_ROI"
    if not video_dir.exists():
        raise FileNotFoundError(f"Missing video folder: {video_dir}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Missing audio folder: {audio_dir}")

    rows = []
    videos = []
    audio_ext = audio_ext if str(audio_ext).startswith(".") else f".{audio_ext}"
    for ext in VIDEO_EXTS:
        videos.extend(sorted(video_dir.glob(f"*{ext}")))
    for video in sorted(videos):
        stem = video.stem
        audio = audio_dir / f"{stem}{audio_ext}"
        if not audio.exists():
            hits = list(audio_dir.rglob(f"{stem}{audio_ext}"))
            if not hits:
                raise FileNotFoundError(f"Missing audio {stem}{audio_ext} under {audio_dir}")
            audio = hits[0]
        bg = find_with_ext(bg_dir, stem, IMG_EXTS)
        roi = find_with_ext(roi_dir, stem, IMG_EXTS)
        phys = physics.get((split, stem), {})
        rows.append(
            {
                "split": split,
                "stem": stem,
                "video": str(video.resolve()),
                "audio": str(audio.resolve()),
                "audio_format": audio_ext.lstrip(".").lower(),
                "background": str(bg.resolve()) if bg else "",
                "roi": str(roi.resolve()) if roi else "",
                "physics_norm": phys.get("physics_norm", [0.0, 0.0, 0.0]),
                "physics_raw": phys.get("physics_raw", {}),
                "condition_source": phys.get("condition_source", ""),
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_root",
        default=str((PROJECT_ROOT.parent / "dataset_split_8011_seed42").resolve()),
    )
    parser.add_argument(
        "--physics_csv",
        default=str((PROJECT_ROOT.parent / "MM-Diffusion_customized_260421_wav_260313_split8011_physicscond" / "physics_condition_map.csv").resolve()),
    )
    parser.add_argument("--out_dir", default=str((PROJECT_ROOT / "manifests").resolve()))
    parser.add_argument("--audio_subdir", default="audio_wav")
    parser.add_argument("--audio_ext", default=".wav")
    args = parser.parse_args()

    split_root = Path(args.split_root)
    physics = load_physics_map(args.physics_csv)
    out_dir = Path(args.out_dir)
    summary = {}
    for split in ["train", "val", "test"]:
        rows = build_split_rows(
            split_root,
            split,
            physics,
            audio_subdir=args.audio_subdir,
            audio_ext=args.audio_ext,
        )
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        summary[split] = len(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote manifests to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
