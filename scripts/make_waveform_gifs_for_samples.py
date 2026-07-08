from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO_DIR = PROJECT_ROOT.parent / "dataset_split_audio_csv_new_seed42" / "test" / "audio_csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_dir",
        required=True,
        help="Sample output directory containing samples_metadata.jsonl and gifs/.",
    )
    parser.add_argument("--audio_dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--comparison_dir", default="")
    parser.add_argument("--video_fps", type=float, default=100.0)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--write_comparison", action="store_true")
    parser.add_argument(
        "--window_mode",
        choices=["sample", "full_audio"],
        default="sample",
        help="sample: show only the time span covered by the prediction GIF. full_audio: show the full CSV waveform.",
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv_waveform(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        data = np.loadtxt(str(path), delimiter=",", skiprows=1, dtype=np.float64)
    except Exception:
        data = np.genfromtxt(str(path), delimiter=",", skip_header=1, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.size == 0:
        raise ValueError(f"Empty CSV waveform: {path}")

    if data.shape[1] >= 2:
        t = data[:, 0].astype(np.float64, copy=False)
        y = data[:, 1].astype(np.float32, copy=False)
        finite = np.isfinite(t) & np.isfinite(y)
        t = t[finite]
        y = y[finite]
        if t.size < 2:
            raise ValueError(f"Not enough finite waveform samples: {path}")
        t = t - float(t[0])
        dt = np.diff(t[: min(t.size, 20000)])
        dt = dt[np.isfinite(dt) & (dt > 0)]
        sample_rate = float(round(1.0 / float(np.median(dt)))) if dt.size else 1_000_000.0
        return t, y, sample_rate

    y = data[:, 0].astype(np.float32, copy=False)
    y = y[np.isfinite(y)]
    if y.size < 2:
        raise ValueError(f"Not enough finite waveform samples: {path}")
    sample_rate = 1_000_000.0
    t = np.arange(y.size, dtype=np.float64) / sample_rate
    return t, y, sample_rate


def gif_frames_and_durations(path: Path) -> tuple[list[Image.Image], list[int]]:
    with Image.open(path) as im:
        frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(im)]
        durations = [int(frame.info.get("duration", im.info.get("duration", 100)) or 100) for frame in ImageSequence.Iterator(im)]
    if not frames:
        raise ValueError(f"No frames in GIF: {path}")
    if len(durations) != len(frames):
        durations = [100] * len(frames)
    return frames, durations


def find_prediction_gif(sample_dir: Path, meta: dict[str, Any]) -> Path:
    raw = str(meta.get("gif", ""))
    if raw:
        p = resolve(raw)
        if p.exists():
            return p
    gifs_dir = sample_dir / "gifs"
    index = int(meta.get("index", 0))
    stem = str(meta.get("stem", ""))
    patterns = [
        f"{index:03d}_{stem}*.gif",
        f"{index:03d}_*.gif",
        f"*{stem}*.gif",
    ]
    for pattern in patterns:
        candidates = sorted(gifs_dir.glob(pattern))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"Could not find prediction GIF for index={index}, stem={stem} in {gifs_dir}")


def robust_ylim(y: np.ndarray) -> float:
    if y.size == 0:
        return 1.0
    q = float(np.percentile(np.abs(y), 99.5))
    m = float(np.max(np.abs(y)))
    limit = max(q, min(m, q * 3.0), 1e-8)
    return limit * 1.12


def envelope_points(t: np.ndarray, y: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    if t.size <= bins * 2:
        return t, y
    edges = np.linspace(float(t[0]), float(t[-1]), bins + 1)
    xs: list[float] = []
    ys: list[float] = []
    indices = np.searchsorted(t, edges)
    for i in range(bins):
        lo = int(indices[i])
        hi = int(indices[i + 1])
        if hi <= lo:
            continue
        chunk = y[lo:hi]
        if chunk.size == 0:
            continue
        mid = 0.5 * (edges[i] + edges[i + 1])
        ymin = float(np.min(chunk))
        ymax = float(np.max(chunk))
        xs.extend([mid, mid])
        ys.extend([ymin, ymax])
    if not xs:
        return t, y
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float32)


def draw_waveform_frame(
    t: np.ndarray,
    y: np.ndarray,
    cursor_time: float,
    start_time: float,
    end_time: float,
    sample_rate: float,
    stem: str,
    frame_index: int,
    total_frames: int,
    size: tuple[int, int],
) -> Image.Image:
    width, height = size
    img = Image.new("RGB", (width, height), (252, 252, 250))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    margin_l, margin_r, margin_t, margin_b = 64, 24, 30, 42
    x0, x1 = margin_l, width - margin_r
    y0, y1 = margin_t, height - margin_b
    mid_y = (y0 + y1) // 2

    draw.rectangle((0, 0, width - 1, height - 1), outline=(210, 210, 205))
    draw.line((x0, y0, x0, y1), fill=(65, 65, 65), width=1)
    draw.line((x0, mid_y, x1, mid_y), fill=(205, 205, 205), width=1)
    draw.line((x0, y1, x1, y1), fill=(65, 65, 65), width=1)

    if end_time <= start_time:
        end_time = start_time + 1e-6
    mask = (t >= start_time) & (t <= end_time)
    tt = t[mask]
    yy = y[mask]
    if tt.size < 2:
        tt = np.array([start_time, end_time], dtype=np.float64)
        yy = np.zeros_like(tt, dtype=np.float32)

    t_draw, y_draw = envelope_points(tt, yy, max(64, x1 - x0))
    ylim = robust_ylim(yy)

    def tx(value: float) -> int:
        frac = (float(value) - start_time) / (end_time - start_time)
        return int(round(x0 + max(0.0, min(1.0, frac)) * (x1 - x0)))

    def ty(value: float) -> int:
        frac = float(value) / ylim
        return int(round(mid_y - max(-1.0, min(1.0, frac)) * ((y1 - y0) * 0.47)))

    cursor_x = tx(cursor_time)
    if cursor_x > x0 + 1:
        draw.rectangle((x0 + 1, y0 + 1, cursor_x, y1 - 1), fill=(236, 243, 255))

    points = [(tx(float(a)), ty(float(b))) for a, b in zip(t_draw, y_draw)]
    if len(points) >= 2:
        draw.line(points, fill=(26, 92, 179), width=1)

    draw.line((cursor_x, y0, cursor_x, y1), fill=(220, 45, 45), width=2)
    draw.ellipse((cursor_x - 4, mid_y - 4, cursor_x + 4, mid_y + 4), fill=(220, 45, 45))

    cursor_ms = (cursor_time - start_time) * 1000.0
    span_ms = (end_time - start_time) * 1000.0
    rms = math.sqrt(float(np.mean(yy.astype(np.float64) ** 2))) if yy.size else 0.0
    peak = float(np.max(np.abs(yy))) if yy.size else 0.0
    title = f"{stem} | waveform | frame {frame_index + 1}/{total_frames}"
    info = f"window {start_time:.3f}-{end_time:.3f}s ({span_ms:.0f} ms), cursor {cursor_ms:.1f} ms, fs~{sample_rate/1000:.1f} kHz"
    amp = f"RMS {rms:.4g}, peak {peak:.4g}"
    draw.text((12, 9), title, fill=(20, 20, 20), font=font)
    draw.text((x0, height - 30), info, fill=(55, 55, 55), font=font)
    draw.text((width - 180, 9), amp, fill=(55, 55, 55), font=font)

    draw.text((10, y0 - 4), f"+/-{ylim:.3g}", fill=(80, 80, 80), font=font)
    draw.text((x0, y1 + 6), f"{start_time:.3f}s", fill=(80, 80, 80), font=font)
    draw.text((x1 - 48, y1 + 6), f"{end_time:.3f}s", fill=(80, 80, 80), font=font)
    return img


def save_gif(frames: list[Image.Image], durations: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    durations = durations[: len(frames)]
    if len(durations) < len(frames):
        durations.extend([durations[-1] if durations else 100] * (len(frames) - len(durations)))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )


def combine_prediction_and_waveform(
    prediction_frames: list[Image.Image],
    waveform_frames: list[Image.Image],
) -> list[Image.Image]:
    count = min(len(prediction_frames), len(waveform_frames))
    out: list[Image.Image] = []
    for i in range(count):
        pred = prediction_frames[i].convert("RGB")
        wave = waveform_frames[i].convert("RGB")
        if wave.width != pred.width:
            new_h = max(160, int(round(wave.height * pred.width / max(1, wave.width))))
            wave = wave.resize((pred.width, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (pred.width, pred.height + wave.height), (255, 255, 255))
        canvas.paste(pred, (0, 0))
        canvas.paste(wave, (0, pred.height))
        out.append(canvas)
    return out


def main() -> None:
    args = parse_args()
    sample_dir = resolve(args.sample_dir)
    audio_dir = resolve(args.audio_dir)
    out_dir = resolve(args.out_dir) if args.out_dir else sample_dir / "waveform_gifs"
    comparison_dir = resolve(args.comparison_dir) if args.comparison_dir else sample_dir / "waveform_prediction_gifs"
    metadata_path = sample_dir / "samples_metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing samples_metadata.jsonl: {metadata_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Missing audio_dir: {audio_dir}")

    rows = read_jsonl(metadata_path)
    if args.max_samples > 0:
        rows = rows[: int(args.max_samples)]

    summary: list[dict[str, Any]] = []
    for meta in rows:
        index = int(meta.get("index", len(summary)))
        stem = str(meta.get("stem", ""))
        audio_path = audio_dir / f"{stem}.csv"
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing CSV audio for {stem}: {audio_path}")
        pred_gif = find_prediction_gif(sample_dir, meta)
        pred_frames, durations = gif_frames_and_durations(pred_gif)
        frame_count = len(pred_frames)

        start_frame = int(meta.get("start_frame", 0))
        start_time = start_frame / float(args.video_fps)
        if args.window_mode == "sample":
            end_time = start_time + frame_count / float(args.video_fps)
        else:
            end_time = float("inf")

        t, y, sample_rate = read_csv_waveform(audio_path)
        if args.window_mode == "full_audio":
            start_time = 0.0
            end_time = float(t[-1])
        else:
            end_time = min(float(end_time), float(t[-1]))

        waveform_frames = []
        for frame_idx in range(frame_count):
            cursor_time = start_time + frame_idx / float(args.video_fps)
            cursor_time = min(cursor_time, end_time)
            waveform_frames.append(
                draw_waveform_frame(
                    t=t,
                    y=y,
                    cursor_time=cursor_time,
                    start_time=start_time,
                    end_time=end_time,
                    sample_rate=sample_rate,
                    stem=stem,
                    frame_index=frame_idx,
                    total_frames=frame_count,
                    size=(int(args.width), int(args.height)),
                )
            )

        out_name = f"{index:03d}_{stem}_waveform.gif"
        out_path = out_dir / out_name
        save_gif(waveform_frames, durations, out_path)

        comparison_path = ""
        if args.write_comparison:
            combined = combine_prediction_and_waveform(pred_frames, waveform_frames)
            comparison_path_obj = comparison_dir / f"{index:03d}_{stem}_prediction_waveform.gif"
            save_gif(combined, durations[: len(combined)], comparison_path_obj)
            comparison_path = str(comparison_path_obj)

        summary.append(
            {
                "index": index,
                "stem": stem,
                "prediction_gif": str(pred_gif),
                "audio_csv": str(audio_path),
                "waveform_gif": str(out_path),
                "comparison_gif": comparison_path,
                "frames": frame_count,
                "gif_duration_ms_total": int(sum(durations)),
                "video_fps_timebase": float(args.video_fps),
                "waveform_window_sec": [float(start_time), float(end_time)],
                "sample_rate_estimated": float(sample_rate),
            }
        )
        print(f"[waveform] {stem}: {frame_count} frames -> {out_path}")

    summary_path = out_dir / "waveform_gifs_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"saved": len(summary), "out_dir": str(out_dir), "summary": str(summary_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
