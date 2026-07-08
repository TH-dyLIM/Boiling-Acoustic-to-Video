"""Render scrolling waveform GIFs that align frame-by-frame with the
prediction GIFs under ``test_full_rollout_best/gifs/``.

For each prediction GIF (``{idx:03d}_{stem}_full_gt_pred_bg_mask.gif``), this
script reads the matching ``audio_csv/{stem}.csv``, then renders one waveform
panel per video frame. At video frame ``k`` (corresponding to real time
``t = k / video_fps``) the panel shows the last ``window_sec`` seconds of
audio with the *current* sample fixed at the right edge — so historical
samples scroll right-to-left as the GIF advances, exactly matching the
playback timing (default 10 fps, 100 ms/frame).

Two visual styles are supported (default ``matlab``):

- ``matlab``        white background, MATLAB-blue line, mV/ms axes
                    (matches the user's reference figure)
- ``oscilloscope``  black background, cyan line, V/s axes (original)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after .use)
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATLAB_BLUE = "#0072BD"


def read_csv_waveform(csv_path: Path, default_sr: int = 1_000_000) -> tuple[np.ndarray, int]:
    """Read an oscilloscope-style CSV (Time,Voltage). Returns (samples, sample_rate)."""
    try:
        data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1, dtype=np.float64)
    except Exception:
        data = np.genfromtxt(str(csv_path), delimiter=",", skip_header=1, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.size == 0:
        return np.zeros((0,), dtype=np.float32), int(default_sr)
    if data.shape[1] >= 2:
        t = data[:, 0]
        x = data[:, 1].astype(np.float32)
        finite_t = t[np.isfinite(t)]
        dt = np.diff(finite_t[: min(20_000, finite_t.size)])
        dt = dt[np.isfinite(dt) & (dt > 0)]
        sr = int(round(1.0 / float(np.median(dt)))) if dt.size else int(default_sr)
    else:
        x = data[:, 0].astype(np.float32)
        sr = int(default_sr)
    x = x[np.isfinite(x)].astype(np.float32, copy=False)
    return x, int(sr)


# ============================================================
# Oscilloscope (black-bg) renderer
# ============================================================
def render_waveform_frame_oscilloscope(
    window: np.ndarray,
    ylim: float,
    width: int,
    height: int,
    window_sec: float,
    current_time: float,
    stem: str,
    is_padded_left: bool,
) -> Image.Image:
    img_np = np.full((height, width, 3), (10, 10, 18), dtype=np.uint8)
    mid_y = height // 2
    img_np[mid_y, :, :] = (60, 60, 80)
    img_np[:, width // 2, :] = (40, 40, 60)
    n = window.size
    if n > 0 and ylim > 1e-9:
        bin_w = max(1, int(np.ceil(n / float(width))))
        target_len = bin_w * width
        if n < target_len:
            window_p = np.concatenate([window, np.zeros(target_len - n, dtype=window.dtype)])
        else:
            window_p = window[:target_len]
        reshaped = window_p.reshape(width, bin_w)
        vmin = reshaped.min(axis=1)
        vmax = reshaped.max(axis=1)
        ymid = (height - 1) * 0.5
        scale = (height - 1) * 0.5 / float(ylim)
        y_top = np.clip((ymid - vmax * scale).round().astype(np.int32), 0, height - 1)
        y_bot = np.clip((ymid - vmin * scale).round().astype(np.int32), 0, height - 1)
        for x in range(width):
            y0 = int(y_top[x])
            y1 = int(y_bot[x])
            if y0 > y1:
                y0, y1 = y1, y0
            img_np[y0 : y1 + 1, x, :] = (0, 230, 255)
    img_np[:, -1, :] = (255, 255, 200)
    img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img)
    suffix = "  (padded)" if is_padded_left else ""
    draw.text((4, 2), stem, fill=(220, 220, 220))
    draw.text(
        (4, height - 14),
        f"t={current_time:6.3f}s  win={window_sec:.2f}s  y=[-{ylim:.3g},+{ylim:.3g}]{suffix}",
        fill=(180, 180, 200),
    )
    return img


# ============================================================
# MATLAB-style renderer (white bg, blue line, mV/ms)
# ============================================================
def _downsample_for_line(arr: np.ndarray, target_n: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Subsample to ``target_n`` points (uniform indexing) while preserving the
    visual envelope well enough for line plotting at typical figure widths."""
    n = arr.size
    if n == 0:
        return np.array([]), np.array([])
    if n <= target_n:
        x = np.linspace(0.0, 1.0, n)
        return x, arr.astype(np.float64)
    idx = np.linspace(0.0, n - 1, target_n).round().astype(np.int64)
    x = np.linspace(0.0, 1.0, target_n)
    return x, arr[idx].astype(np.float64)


def make_matlab_figure(
    width_px: int,
    height_px: int,
    window_ms: float,
    ymax_mv: float,
    dpi: int = 100,
):
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(0.0, window_ms)
    ax.set_ylim(-ymax_mv, ymax_mv)
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Voltage (mV)", fontsize=12)
    ax.tick_params(direction="out", length=5, width=1.0, labelsize=11)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="out", length=3, width=0.7)
    for s in ax.spines.values():
        s.set_color("black")
        s.set_linewidth(1.0)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    (line,) = ax.plot([], [], color=MATLAB_BLUE, linewidth=0.5)
    # Lightweight annotation in top-right showing the absolute current time
    # (kept small/subtle so the plot still reads like the MATLAB reference).
    time_text = ax.text(
        0.985, 0.965, "", transform=ax.transAxes, ha="right", va="top",
        fontsize=10, color="#555555",
    )
    fig.tight_layout(pad=0.8)
    return fig, ax, line, time_text


def render_waveform_frame_matlab(
    fig,
    ax,
    line,
    time_text,
    window: np.ndarray,
    window_ms: float,
    current_time: float,
    units_scale: float = 1000.0,  # V -> mV
    target_n: int = 4096,
) -> Image.Image:
    x_norm, v = _downsample_for_line(window, target_n=target_n)
    if x_norm.size == 0:
        line.set_data([], [])
    else:
        line.set_data(x_norm * window_ms, v * units_scale)
    time_text.set_text(f"t = {current_time:.3f} s")
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba[:, :, :3].copy())


PRED_SUFFIX = "_gt_pred_bg_mask"
PRED_TRAIL = "_full" + PRED_SUFFIX


def parse_stem_from_pred_gif(filename: str) -> str:
    name = Path(filename).stem
    if name.endswith(PRED_TRAIL):
        name = name[: -len(PRED_TRAIL)]
    elif name.endswith(PRED_SUFFIX):
        name = name[: -len(PRED_SUFFIX)]
    m = re.match(r"^\d+_(.+)$", name)
    return m.group(1) if m else name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_gifs_dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "flow_residual_csv8_large_main_noprior"
        / "test_full_rollout_best"
        / "gifs",
    )
    parser.add_argument(
        "--audio_csv_dir",
        type=Path,
        default=PROJECT_ROOT.parent / "dataset_split_audio_csv_new_seed42" / "test" / "audio_csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "flow_residual_csv8_large_main_noprior"
        / "test_full_rollout_best"
        / "gifs_waveform",
    )
    parser.add_argument(
        "--style",
        choices=["matlab", "oscilloscope"],
        default="matlab",
        help="Visual style. 'matlab' = white bg / blue line / mV-ms axes; "
        "'oscilloscope' = black bg / cyan line / V-s.",
    )
    parser.add_argument("--video_fps", type=float, default=100.0)
    parser.add_argument(
        "--gif_fps",
        type=float,
        default=10.0,
        help="GIF playback rate (matches prediction GIFs saved at fps=10).",
    )
    parser.add_argument(
        "--window_sec",
        type=float,
        default=0.5,
        help="Time-window length shown at any moment (seconds).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Output GIF width in pixels. 0 = style default (matlab:640, osc:512).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Output GIF height in pixels. 0 = style default (matlab:400, osc:128).",
    )
    parser.add_argument(
        "--ymax_mv",
        type=float,
        default=0.0,
        help="Fixed y-axis half-range in mV (matlab style). 0 = auto from clip.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=0.0,
        help="Fixed y-axis half-range in volts (oscilloscope style). 0 = auto.",
    )
    parser.add_argument(
        "--ymax_quantile",
        type=float,
        default=0.999,
        help="Quantile of |signal| used for auto y-axis (scaled by 1.05).",
    )
    parser.add_argument("--default_sr", type=int, default=1_000_000)
    parser.add_argument("--max_videos", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.width <= 0:
        args.width = 640 if args.style == "matlab" else 512
    if args.height <= 0:
        args.height = 400 if args.style == "matlab" else 128

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ref_gifs = sorted(args.pred_gifs_dir.glob("*.gif"))
    if not ref_gifs:
        print(f"No reference GIFs in {args.pred_gifs_dir}", file=sys.stderr)
        sys.exit(1)
    if int(args.max_videos) > 0:
        ref_gifs = ref_gifs[: int(args.max_videos)]

    duration_ms = int(round(1000.0 / max(args.gif_fps, 1e-3)))
    written = 0
    skipped = 0

    for ref in tqdm(ref_gifs, desc=f"waveform GIFs ({args.style})"):
        stem = parse_stem_from_pred_gif(ref.name)
        out_name = ref.stem.replace(PRED_SUFFIX, "_waveform") + ".gif"
        out_path = args.output_dir / out_name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        csv_path = args.audio_csv_dir / f"{stem}.csv"
        if not csv_path.exists():
            print(f"[skip] missing CSV for stem={stem!r}: {csv_path}")
            continue

        try:
            with Image.open(ref) as im:
                n_frames = int(getattr(im, "n_frames", 1))
        except Exception as e:
            print(f"[skip] failed to read {ref.name}: {e}")
            continue
        if n_frames <= 0:
            print(f"[skip] {ref.name}: zero frames")
            continue

        wave, sr = read_csv_waveform(csv_path, default_sr=args.default_sr)
        if wave.size == 0:
            print(f"[skip] empty CSV: {csv_path}")
            continue

        # y-axis
        abs_wave = np.abs(wave)
        if args.style == "matlab":
            if args.ymax_mv > 0:
                ylim_mv = float(args.ymax_mv)
            else:
                q = float(np.quantile(abs_wave, args.ymax_quantile)) if abs_wave.size else 1e-3
                ylim_v = q * 1.05 if q > 1e-9 else 1e-3
                ylim_mv = ylim_v * 1000.0
        else:
            if args.ymax > 0:
                ylim_v = float(args.ymax)
            else:
                q = float(np.quantile(abs_wave, args.ymax_quantile)) if abs_wave.size else 1.0
                ylim_v = q * 1.05 if q > 1e-9 else 1.0

        window_samples = max(1, int(round(args.window_sec * sr)))
        window_ms = float(args.window_sec * 1000.0)

        if args.style == "matlab":
            fig, ax, line, time_text = make_matlab_figure(
                args.width, args.height, window_ms, ylim_mv
            )

        frames: list[Image.Image] = []
        try:
            for k in range(n_frames):
                t_cur = k / float(args.video_fps)
                i_end = int(round(t_cur * sr))
                i_start = i_end - window_samples
                if i_start < 0:
                    left_pad = -i_start
                    w_end = max(0, min(i_end, wave.size))
                    buf = np.zeros(window_samples, dtype=np.float32)
                    buf[left_pad : left_pad + w_end] = wave[:w_end]
                    is_padded_left = True
                elif i_start >= wave.size:
                    buf = np.zeros(window_samples, dtype=np.float32)
                    is_padded_left = False
                else:
                    w_end = min(i_end, wave.size)
                    actual = wave[i_start:w_end]
                    if actual.size < window_samples:
                        buf = np.zeros(window_samples, dtype=np.float32)
                        buf[: actual.size] = actual
                    else:
                        buf = actual
                    is_padded_left = False

                if args.style == "matlab":
                    frames.append(
                        render_waveform_frame_matlab(
                            fig, ax, line, time_text,
                            buf, window_ms, t_cur,
                        )
                    )
                else:
                    frames.append(
                        render_waveform_frame_oscilloscope(
                            buf, ylim_v, args.width, args.height,
                            window_sec=args.window_sec, current_time=t_cur,
                            stem=stem, is_padded_left=is_padded_left,
                        )
                    )
        finally:
            if args.style == "matlab":
                plt.close(fig)

        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
        written += 1

    print(
        f"Done. style={args.style} wrote={written} skipped_existing={skipped} "
        f"total_reference={len(ref_gifs)} -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
