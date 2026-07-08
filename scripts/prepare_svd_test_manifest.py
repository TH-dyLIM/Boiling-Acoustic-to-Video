from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from svd_audio_control.video_io import IMG_EXTS, VIDEO_EXTS, find_with_ext


def find_required(root: Path, stem: str, exts: tuple[str, ...] | list[str], kind: str) -> Path:
    hit = find_with_ext(root, stem, exts)
    if hit is None:
        raise FileNotFoundError(f"Missing {kind} for stem '{stem}' under {root}")
    return hit


def build_rows(
    split_root: Path,
    split: str,
    video_subdir: str,
    audio_subdir: str,
    background_subdir: str,
    roi_subdir: str,
    audio_ext: str,
) -> list[dict[str, Any]]:
    root = split_root / split
    video_dir = root / video_subdir
    audio_dir = root / audio_subdir
    bg_dir = root / background_subdir
    roi_dir = root / roi_subdir
    if not video_dir.exists():
        raise FileNotFoundError(f"Missing video folder: {video_dir}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Missing audio folder: {audio_dir}")
    if not bg_dir.exists():
        raise FileNotFoundError(f"Missing background folder: {bg_dir}")
    if not roi_dir.exists():
        raise FileNotFoundError(f"Missing Heating_ROI folder: {roi_dir}")

    videos: list[Path] = []
    for ext in VIDEO_EXTS:
        videos.extend(sorted(video_dir.glob(f"*{ext}")))
    if not videos:
        raise FileNotFoundError(f"No videos found under {video_dir}")

    audio_ext = audio_ext if audio_ext.startswith(".") else f".{audio_ext}"
    rows: list[dict[str, Any]] = []
    for video in sorted(videos):
        stem = video.stem
        audio = audio_dir / f"{stem}{audio_ext}"
        if not audio.exists():
            hits = sorted(audio_dir.rglob(f"{stem}{audio_ext}"))
            if not hits:
                raise FileNotFoundError(f"Missing audio {stem}{audio_ext} under {audio_dir}")
            audio = hits[0]
        bg = find_required(bg_dir, stem, IMG_EXTS, "background")
        roi = find_required(roi_dir, stem, IMG_EXTS, "ROI")
        rows.append(
            {
                "split": split,
                "stem": stem,
                "video": str(video.resolve()),
                "audio": str(audio.resolve()),
                "audio_format": audio_ext.lstrip(".").lower(),
                "background": str(bg.resolve()),
                "roi": str(roi.resolve()),
                "physics_norm": [0.0, 0.0, 0.0],
                "physics_raw": {},
                "condition_source": "dataset_test_external_bg_roi",
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
    parser.add_argument("--split_root", default=str((PROJECT_ROOT.parent / "dataset_test").resolve()))
    parser.add_argument("--split", default="test")
    parser.add_argument("--out_manifest", default=str((PROJECT_ROOT / "manifests_dataset_test" / "test.jsonl").resolve()))
    parser.add_argument("--video_subdir", default="video_100fps")
    parser.add_argument("--audio_subdir", default="audio_csv")
    parser.add_argument("--background_subdir", default="background")
    parser.add_argument("--roi_subdir", default="Heating_ROI")
    parser.add_argument("--audio_ext", default=".csv")
    args = parser.parse_args()

    rows = build_rows(
        split_root=Path(args.split_root),
        split=str(args.split),
        video_subdir=str(args.video_subdir),
        audio_subdir=str(args.audio_subdir),
        background_subdir=str(args.background_subdir),
        roi_subdir=str(args.roi_subdir),
        audio_ext=str(args.audio_ext),
    )
    out_manifest = Path(args.out_manifest)
    write_jsonl(out_manifest, rows)
    summary = {
        "split_root": str(Path(args.split_root).resolve()),
        "split": str(args.split),
        "out_manifest": str(out_manifest.resolve()),
        "num_rows": len(rows),
        "stems": [row["stem"] for row in rows],
    }
    (out_manifest.parent / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
