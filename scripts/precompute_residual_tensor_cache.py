from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from residual_video.dataset import (
    _image_01,
    _read_video_rgb,
    _rgb_frame_to_01,
    read_full_audio_file,
    read_jsonl,
    safe_cache_stem,
    tensor_01_to_u8,
)


def load_json_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def build_one_cache(row: dict[str, Any], resolution: int, audio_sample_rate: int) -> dict[str, Any]:
    frames_rgb, fps = _read_video_rgb(row["video"])
    frames_u8 = torch.stack([tensor_01_to_u8(_rgb_frame_to_01(frame, resolution)) for frame in frames_rgb], dim=0)

    bg_path = row.get("background", "")
    if bg_path and Path(bg_path).exists():
        background_u8 = tensor_01_to_u8(_image_01(bg_path, resolution, "RGB"))
    else:
        background_u8 = frames_u8[0].clone()

    roi_path = row.get("roi", "")
    if roi_path and Path(roi_path).exists():
        roi_u8 = tensor_01_to_u8(_image_01(roi_path, resolution, "L"))
    else:
        roi_u8 = torch.zeros(1, resolution, resolution, dtype=torch.uint8)

    audio, audio_sr = read_full_audio_file(row["audio"], audio_sample_rate)
    return {
        "frames_u8": frames_u8,
        "background_u8": background_u8,
        "roi_u8": roi_u8,
        "audio": audio,
        "audio_sr": audio_sr,
        "fps": float(fps),
        "resolution": int(resolution),
        "stem": row.get("stem", Path(row["video"]).stem),
        "video": row.get("video", ""),
        "audio_path": row.get("audio", ""),
        "background": row.get("background", ""),
        "roi": row.get("roi", ""),
    }


def parse_manifest_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), resolve_path(path.strip())
    p = resolve_path(value)
    return p.stem, p


def default_manifests(config: dict[str, Any]) -> list[tuple[str, Path]]:
    manifests: list[tuple[str, Path]] = []
    for name, key in (("train", "train_manifest"), ("val", "val_manifest")):
        if config.get(key):
            manifests.append((name, resolve_path(config[key])))
    test_path = PROJECT_ROOT / "manifests" / "test.jsonl"
    if test_path.exists():
        manifests.append(("test", test_path))
    return manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "residual_video_sharp.json"))
    parser.add_argument("--cache_root", default="")
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument("--resolution", type=int, default=0)
    parser.add_argument("--audio_sample_rate", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json_config(args.config)
    resolution = int(args.resolution or config.get("resolution", 256))
    audio_sample_rate = int(args.audio_sample_rate or config.get("audio_sample_rate", 1_000_000))
    cache_root = resolve_path(args.cache_root or config.get("cache_root", "cache/residual_frame_ar_delta_nophysics_sharp"))
    manifests = [parse_manifest_arg(item) for item in args.manifest] if args.manifest else default_manifests(config)
    if not manifests:
        raise ValueError("No manifests were provided and config has no train/val manifest.")

    total_saved = 0
    total_skipped = 0
    for split_name, manifest_path in manifests:
        rows = read_jsonl(manifest_path)
        if args.limit > 0:
            rows = rows[: args.limit]
        out_dir = cache_root / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Precomputing {split_name}: {manifest_path} -> {out_dir} ({len(rows)} rows)")
        for index, row in enumerate(tqdm(rows, desc=split_name)):
            out_path = out_dir / f"{safe_cache_stem(index, row.get('stem', Path(row.get('video', '')).stem))}.pt"
            if out_path.exists() and not args.overwrite:
                total_skipped += 1
                continue
            cache = build_one_cache(row, resolution=resolution, audio_sample_rate=audio_sample_rate)
            tmp_path = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
            try:
                torch.save(cache, tmp_path)
                os.replace(tmp_path, out_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
            total_saved += 1
    print(json.dumps({"cache_root": str(cache_root), "saved": total_saved, "skipped": total_skipped}, indent=2))


if __name__ == "__main__":
    main()
