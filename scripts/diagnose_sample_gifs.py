from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from svd_audio_control.video_io import frames_to_uint8, pil_to_tensor


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def mad_next(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(x, axis=0))))


def temporal_std(x: np.ndarray) -> float:
    return float(np.mean(np.std(x, axis=0)))


def bg_uint8(path: str | Path, resolution: int) -> np.ndarray:
    tensor = pil_to_tensor(Image.open(path), resolution, resolution, "RGB")
    return frames_to_uint8(tensor.unsqueeze(0))[0].astype(np.float32) / 255.0


def parse_stem(gif_path: Path) -> str:
    name = gif_path.name
    if name.endswith("_gt_pred.gif"):
        name = name[: -len("_gt_pred.gif")]
    parts = name.split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument("--manifest", default="manifests/test.jsonl")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--out_name", default="gif_diagnostics.jsonl")
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    gif_dir = sample_dir / "gifs"
    manifest_rows = read_jsonl(args.manifest)
    manifest_by_stem = {row.get("stem", Path(row["video"]).stem): row for row in manifest_rows}

    rows = []
    contact_items = []
    for gif_path in sorted(gif_dir.glob("*.gif")):
        frames = np.stack(imageio.mimread(gif_path), axis=0).astype(np.float32) / 255.0
        half = frames.shape[2] // 2
        gt = frames[:, :, :half, :3]
        pred = frames[:, :, half:, :3]
        stem = parse_stem(gif_path)
        row = manifest_by_stem.get(stem)
        out = {
            "gif": gif_path.name,
            "stem": stem,
            "gt_next_mae": mad_next(gt),
            "pred_next_mae": mad_next(pred),
            "pred_motion_ratio": mad_next(pred) / max(mad_next(gt), 1e-9),
            "gt_temporal_std": temporal_std(gt),
            "pred_temporal_std": temporal_std(pred),
            "pred_temporal_std_ratio": temporal_std(pred) / max(temporal_std(gt), 1e-9),
            "gt_pred_mse": mse(gt, pred),
        }
        if row and row.get("background"):
            bg = bg_uint8(row["background"], args.resolution)
            out["gt_bg_mse"] = mse(gt, bg[None])
            out["pred_bg_mse"] = mse(pred, bg[None])
            out["pred_vs_gt_bg_ratio"] = out["pred_bg_mse"] / max(out["gt_bg_mse"], 1e-9)
        rows.append(out)
        if len(contact_items) < 6:
            contact_items.append((gif_path.name, gt, pred))

    out_path = sample_dir / args.out_name
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if contact_items:
        cell_w = cell_h = 128
        label_h = 30
        cols = 6
        sheet = Image.new("RGB", (cols * cell_w, len(contact_items) * (cell_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for row_i, (name, gt, pred) in enumerate(contact_items):
            y0 = row_i * (cell_h + label_h)
            draw.text((4, y0 + 2), name.replace("_gt_pred.gif", ""), fill=(0, 0, 0))
            for frame_group, idx in enumerate([0, gt.shape[0] // 2, gt.shape[0] - 1]):
                for kind, arr in enumerate([gt[idx], pred[idx]]):
                    img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8)).resize((cell_w, cell_h), Image.BICUBIC)
                    x0 = (frame_group * 2 + kind) * cell_w
                    sheet.paste(img, (x0, y0 + label_h))
                    draw.text((x0 + 4, y0 + label_h + 4), ("GT" if kind == 0 else "Pred") + f" f{idx}", fill=(255, 255, 255))
        sheet.save(sample_dir / "gif_contactsheet_gt_pred.png")

    summary = {}
    for key in [
        "gt_next_mae",
        "pred_next_mae",
        "pred_motion_ratio",
        "gt_temporal_std",
        "pred_temporal_std",
        "pred_temporal_std_ratio",
        "gt_bg_mse",
        "pred_bg_mse",
        "pred_vs_gt_bg_ratio",
        "gt_pred_mse",
    ]:
        values = [row[key] for row in rows if key in row]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_min"] = float(np.min(values))
            summary[f"{key}_max"] = float(np.max(values))
    summary["num_gifs"] = len(rows)
    summary["diagnostics_jsonl"] = str(out_path.resolve())
    summary_path = sample_dir / "gif_diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
