from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from scipy.stats import ks_2samp, wasserstein_distance
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from residual_video.dataset import read_full_audio_file  # noqa: E402


FAST_METRIC_COLUMNS = [
    "mse",
    "mae",
    "psnr",
    "ssim",
    "roi_residual_mae",
    "roi_edge_mae",
    "temporal_gradient_mae",
    "residual_energy_wasserstein",
    "residual_energy_ks",
    "frame_change_rate_error",
    "temporal_residual_autocorr_error",
    "spatial_power_spectrum_error",
    "audio_motion_corr_gt",
    "audio_motion_corr_pred",
    "audio_motion_corr_error",
    "audio_motion_lag_gt_frames",
    "audio_motion_lag_pred_frames",
    "audio_motion_lag_error_frames",
]

DEEP_METRIC_COLUMNS = [
    "fid_inception_v3",
    "fvd_r3d18",
    "precision_inception_v3",
    "recall_inception_v3",
    "precision_r3d18",
    "recall_r3d18",
    "lpips_alex",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated boiling prediction MP4s against test-set GT videos "
            "using segmentation-free image, residual, temporal, spectral, and audio-motion metrics."
        )
    )
    parser.add_argument(
        "--dataset_root",
        default=str(WORKSPACE_ROOT / "dataset_split_audio_csv_new_seed42" / "test"),
        help="Test split directory containing video_100fps, audio_csv, background, and Heating_ROI.",
    )
    parser.add_argument(
        "--model_dir",
        action="append",
        required=True,
        help="Result directory. The script auto-detects pred_mp4 or test_full_rollout_best/pred_mp4.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "outputs" / "segmentation_free_video_metrics"),
        help="Central output directory for combined CSV/XLSX summaries.",
    )
    parser.add_argument("--audio_sample_rate", type=int, default=1_000_000)
    parser.add_argument("--default_fps", type=float, default=100.0)
    parser.add_argument("--max_videos", type=int, default=0)
    parser.add_argument("--residual_sample_limit", type=int, default=250_000)
    parser.add_argument("--spectrum_bins", type=int, default=32)
    parser.add_argument("--autocorr_lags", type=int, default=20)
    parser.add_argument("--audio_lag_frames", type=int, default=10)
    parser.add_argument("--deep_metrics", action="store_true", help="Also compute FID, R3D-18 FVD proxy, and LPIPS.")
    parser.add_argument("--feature_frame_stride", type=int, default=4)
    parser.add_argument("--feature_batch_size", type=int, default=16)
    parser.add_argument("--precision_recall_k", type=int, default=3)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def safe_model_id(path: Path) -> str:
    text = path.name
    if text in {"test_full_rollout_best", "test_samples_best", "test_rollout_ema060000", "pred_mp4"} and path.parent.name:
        text = f"{path.parent.name}__{text}"
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("._") or "model"
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:8]
    if len(text) > 72:
        text = text[:72].rstrip("._-")
    return f"{text}__{digest}"


def read_video_rgb_01(path: Path, size: tuple[int, int] | None = None) -> tuple[torch.Tensor, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if size is not None:
            frame_rgb = cv2.resize(frame_rgb, size, interpolation=cv2.INTER_AREA)
        frames.append(frame_rgb)
    cap.release()
    if not frames:
        raise RuntimeError(f"Failed to decode frames: {path}")
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous(), fps if fps > 0 else 0.0


def read_image_01(path: Path, size: tuple[int, int], mode: str) -> torch.Tensor:
    image = Image.open(path).convert(mode).resize(size, Image.BILINEAR if mode == "RGB" else Image.NEAREST)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if mode == "L":
        return torch.from_numpy(arr).unsqueeze(0)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def build_test_index(dataset_root: Path) -> dict[str, dict[str, Path]]:
    video_dir = dataset_root / "video_100fps"
    audio_dir = dataset_root / "audio_csv"
    bg_dir = dataset_root / "background"
    roi_dir = dataset_root / "Heating_ROI"
    index: dict[str, dict[str, Path]] = {}
    for video_path in sorted(video_dir.glob("*.mp4")):
        stem = video_path.stem
        index[stem] = {
            "video": video_path,
            "audio": audio_dir / f"{stem}.csv",
            "background": first_existing([bg_dir / f"{stem}.jpg", bg_dir / f"{stem}.png", bg_dir / f"{stem}.jpeg"]),
            "roi": first_existing([roi_dir / f"{stem}.png", roi_dir / f"{stem}.jpg", roi_dir / f"{stem}.jpeg"]),
        }
    if not index:
        raise RuntimeError(f"No GT videos found under {video_dir}")
    return index


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def find_prediction_dir(model_dir: Path) -> Path:
    candidates = [
        model_dir / "pred_mp4",
        model_dir / "test_full_rollout_best" / "pred_mp4",
        model_dir / "test_samples_best" / "pred_mp4",
        model_dir / "test_rollout_ema060000" / "pred_mp4",
    ]
    for path in candidates:
        if path.exists() and any(path.glob("*.mp4")):
            return path
    pred_dirs = [p for p in model_dir.rglob("pred_mp4") if p.is_dir() and any(p.glob("*.mp4"))]
    if not pred_dirs:
        raise RuntimeError(f"No pred_mp4 directory with MP4 files found under {model_dir}")
    pred_dirs.sort(key=prediction_dir_priority)
    return pred_dirs[0]


def prediction_dir_priority(path: Path) -> tuple[int, int, str]:
    text = str(path).lower()
    if "best" in text:
        group = 0
    elif "ema" in text:
        group = 1
    elif "last" in text:
        group = 3
    else:
        group = 2
    return (group, len(path.parts), str(path))


def match_pred_to_stem(pred_path: Path, stems: list[str]) -> str | None:
    name = pred_path.stem
    for suffix in ("_full_pred", "_pred", "_prediction", "_generated"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    for stem in sorted(stems, key=len, reverse=True):
        if name == stem or name.endswith(f"_{stem}"):
            return stem
    return None


def sobel_edges(video: torch.Tensor) -> torch.Tensor:
    # video: T,C,H,W
    t, c, h, w = video.shape
    x = video.reshape(t * c, 1, h, w)
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=video.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=video.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8).reshape(t, c, h, w)


def gaussian_window(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).view(1, 1, window_size, window_size)
    return window


def ssim_video(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_y = rgb_to_gray(pred).unsqueeze(1)
    target_y = rgb_to_gray(target).unsqueeze(1)
    window = gaussian_window().to(pred_y)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.conv2d(pred_y, window, padding=5)
    mu_y = F.conv2d(target_y, window, padding=5)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(pred_y * pred_y, window, padding=5) - mu_x2
    sigma_y2 = F.conv2d(target_y * target_y, window, padding=5) - mu_y2
    sigma_xy = F.conv2d(pred_y * target_y, window, padding=5) - mu_xy
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-8)
    return float(ssim_map.mean().item())


def rgb_to_gray(video: torch.Tensor) -> torch.Tensor:
    return 0.299 * video[:, 0] + 0.587 * video[:, 1] + 0.114 * video[:, 2]


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(1)
    if mask.shape[1] == 1 and value.shape[1] != 1:
        mask = mask.expand(-1, value.shape[1], -1, -1)
    mask = mask.expand_as(value)
    denom = mask.sum().clamp_min(1.0)
    return float((value * mask).sum().item() / denom.item())


def deterministic_sample(values: torch.Tensor, limit: int) -> np.ndarray:
    arr = values.detach().cpu().float().reshape(-1).numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size > int(limit) > 0:
        idx = np.linspace(0, arr.size - 1, int(limit)).astype(np.int64)
        arr = arr[idx]
    return arr


def autocorr_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros((0,), dtype=np.float64)
    x = x - float(np.mean(x))
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return np.zeros((min(max_lag, max(0, x.size - 1)),), dtype=np.float64)
    out = []
    for lag in range(1, min(max_lag, x.size - 1) + 1):
        out.append(float(np.dot(x[:-lag], x[lag:]) / denom))
    return np.asarray(out, dtype=np.float64)


def radial_power_spectrum(video_gray: torch.Tensor, roi: torch.Tensor, bins: int) -> np.ndarray:
    # video_gray: T,H,W, roi: 1,H,W
    roi2 = roi.squeeze(0).detach().cpu().numpy().astype(np.float32)
    h, w = roi2.shape
    yy, xx = np.indices((h, w))
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = radius / max(float(radius.max()), 1e-6)
    bin_idx = np.clip((radius * int(bins)).astype(np.int32), 0, int(bins) - 1)
    accum = np.zeros((int(bins),), dtype=np.float64)
    count = np.zeros((int(bins),), dtype=np.float64)
    frames = video_gray.detach().cpu().numpy().astype(np.float32)
    for frame in frames:
        x = frame * roi2
        if roi2.sum() > 1:
            x = x - float(x[roi2 > 0.5].mean())
        power = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
        for b in range(int(bins)):
            mask = bin_idx == b
            accum[b] += float(power[mask].mean()) if np.any(mask) else 0.0
            count[b] += 1.0
    spec = accum / np.maximum(count, 1.0)
    spec = spec / max(float(spec.sum()), 1e-12)
    return spec.astype(np.float64)


def frame_signal_from_video(video: torch.Tensor, background: torch.Tensor, roi: torch.Tensor) -> np.ndarray:
    residual_energy = (video - background.unsqueeze(0)).abs().mean(dim=1)
    mask = roi.squeeze(0)
    denom = mask.sum().clamp_min(1.0)
    return ((residual_energy * mask).sum(dim=(1, 2)) / denom).detach().cpu().numpy().astype(np.float64)


def frame_change_signal(video: torch.Tensor, roi: torch.Tensor) -> np.ndarray:
    if video.shape[0] < 2:
        return np.zeros((video.shape[0],), dtype=np.float64)
    diff = (video[1:] - video[:-1]).abs().mean(dim=1)
    mask = roi.squeeze(0)
    denom = mask.sum().clamp_min(1.0)
    sig = ((diff * mask).sum(dim=(1, 2)) / denom).detach().cpu().numpy().astype(np.float64)
    return np.concatenate([[0.0], sig], axis=0)


def audio_frame_rms(audio_path: Path, n_frames: int, fps: float, target_sr: int) -> np.ndarray:
    if not audio_path.exists():
        return np.zeros((n_frames,), dtype=np.float64)
    audio, sr = read_full_audio_file(audio_path, int(target_sr))
    y = audio.detach().float().view(-1).cpu().numpy().astype(np.float64)
    if y.size == 0:
        return np.zeros((n_frames,), dtype=np.float64)
    y = y - float(np.mean(y))
    out = np.zeros((n_frames,), dtype=np.float64)
    for i in range(n_frames):
        start = int(round((i / float(fps)) * sr))
        end = int(round(((i + 1) / float(fps)) * sr))
        chunk = y[max(0, start) : min(y.size, max(start + 1, end))]
        out[i] = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
    return out


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    std = float(np.std(x))
    if std <= 1e-12:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / std


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n < 2:
        return 0.0
    aa = standardize(a[:n])
    bb = standardize(b[:n])
    if float(np.std(aa)) <= 1e-12 or float(np.std(bb)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def best_lag_frames(audio: np.ndarray, motion: np.ndarray, max_lag: int) -> int:
    n = min(audio.size, motion.size)
    if n < 3:
        return 0
    a = standardize(audio[:n])
    m = standardize(motion[:n])
    best_lag = 0
    best_score = -float("inf")
    for lag in range(-int(max_lag), int(max_lag) + 1):
        if lag < 0:
            aa, mm = a[-lag:], m[: n + lag]
        elif lag > 0:
            aa, mm = a[: n - lag], m[lag:]
        else:
            aa, mm = a, m
        if aa.size < 3:
            continue
        score = float(np.mean(aa * mm))
        if score > best_score:
            best_score = score
            best_lag = lag
    return int(best_lag)


def compute_fast_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    background: torch.Tensor,
    roi: torch.Tensor,
    audio_path: Path,
    fps: float,
    args: argparse.Namespace,
) -> dict[str, float]:
    pred = pred.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)
    background = background.float().clamp(0.0, 1.0)
    roi = (roi.float() > 0.5).float()
    if float(roi.sum().item()) < 1.0:
        roi = torch.ones_like(roi)

    diff = pred - target
    mse = float((diff * diff).mean().item())
    mae = float(diff.abs().mean().item())
    psnr = 99.0 if mse <= 1e-12 else float(-10.0 * math.log10(mse))
    ssim = ssim_video(pred, target)

    pred_res = pred - background.unsqueeze(0)
    target_res = target - background.unsqueeze(0)
    roi_res_mae = masked_mean((pred_res - target_res).abs(), roi)
    roi_edge_mae = masked_mean((sobel_edges(pred_res) - sobel_edges(target_res)).abs(), roi)
    if pred.shape[0] > 1:
        pred_grad = pred[1:] - pred[:-1]
        target_grad = target[1:] - target[:-1]
        temporal_gradient_mae = masked_mean((pred_grad - target_grad).abs(), roi)
    else:
        temporal_gradient_mae = 0.0

    pred_energy = pred_res.abs().mean(dim=1)
    target_energy = target_res.abs().mean(dim=1)
    roi2 = roi.squeeze(0)
    pred_dist = deterministic_sample(pred_energy[:, roi2 > 0.5], args.residual_sample_limit)
    target_dist = deterministic_sample(target_energy[:, roi2 > 0.5], args.residual_sample_limit)
    residual_w = float(wasserstein_distance(pred_dist, target_dist)) if pred_dist.size and target_dist.size else float("nan")
    residual_ks = float(ks_2samp(pred_dist, target_dist).statistic) if pred_dist.size and target_dist.size else float("nan")

    pred_change = frame_change_signal(pred, roi)
    target_change = frame_change_signal(target, roi)
    frame_change_rate_error = float(np.mean(np.abs(pred_change[: target_change.size] - target_change[: pred_change.size])))

    pred_energy_ts = frame_signal_from_video(pred, background, roi)
    target_energy_ts = frame_signal_from_video(target, background, roi)
    pred_ac = autocorr_1d(pred_energy_ts, args.autocorr_lags)
    target_ac = autocorr_1d(target_energy_ts, args.autocorr_lags)
    if pred_ac.size and target_ac.size:
        n = min(pred_ac.size, target_ac.size)
        autocorr_error = float(np.mean(np.abs(pred_ac[:n] - target_ac[:n])))
    else:
        autocorr_error = 0.0

    pred_spec = radial_power_spectrum(rgb_to_gray(pred_res), roi, args.spectrum_bins)
    target_spec = radial_power_spectrum(rgb_to_gray(target_res), roi, args.spectrum_bins)
    spectrum_error = float(np.mean(np.abs(pred_spec - target_spec)))

    fps = fps if fps > 0 else float(args.default_fps)
    audio_env = audio_frame_rms(audio_path, pred.shape[0], fps, args.audio_sample_rate)
    gt_motion = target_change
    pred_motion = pred_change
    corr_gt = corrcoef_safe(audio_env, gt_motion)
    corr_pred = corrcoef_safe(audio_env, pred_motion)
    lag_gt = best_lag_frames(audio_env, gt_motion, args.audio_lag_frames)
    lag_pred = best_lag_frames(audio_env, pred_motion, args.audio_lag_frames)

    return {
        "mse": mse,
        "mae": mae,
        "psnr": psnr,
        "ssim": ssim,
        "roi_residual_mae": roi_res_mae,
        "roi_edge_mae": roi_edge_mae,
        "temporal_gradient_mae": temporal_gradient_mae,
        "residual_energy_wasserstein": residual_w,
        "residual_energy_ks": residual_ks,
        "frame_change_rate_error": frame_change_rate_error,
        "temporal_residual_autocorr_error": autocorr_error,
        "spatial_power_spectrum_error": spectrum_error,
        "audio_motion_corr_gt": corr_gt,
        "audio_motion_corr_pred": corr_pred,
        "audio_motion_corr_error": abs(corr_pred - corr_gt),
        "audio_motion_lag_gt_frames": float(lag_gt),
        "audio_motion_lag_pred_frames": float(lag_pred),
        "audio_motion_lag_error_frames": float(abs(lag_pred - lag_gt)),
    }


class DeepMetricExtractor:
    def __init__(self, enabled: bool, device: torch.device, batch_size: int, frame_stride: int) -> None:
        self.enabled = bool(enabled)
        self.device = device
        self.batch_size = int(batch_size)
        self.frame_stride = max(1, int(frame_stride))
        self.inception = None
        self.video_model = None
        self.lpips_model = None
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.feature_error = ""
        if self.enabled:
            self._init_models()

    def _init_models(self) -> None:
        try:
            from torchvision.models import Inception_V3_Weights, inception_v3
            from torchvision.models.video import R3D_18_Weights, r3d_18

            self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
            self.inception.fc = torch.nn.Identity()
            self.inception.eval().to(self.device)
            for p in self.inception.parameters():
                p.requires_grad_(False)

            self.video_model = r3d_18(weights=R3D_18_Weights.DEFAULT)
            self.video_model.fc = torch.nn.Identity()
            self.video_model.eval().to(self.device)
            for p in self.video_model.parameters():
                p.requires_grad_(False)
        except Exception as exc:
            self.feature_error = f"FID/FVD feature models unavailable: {exc}"
            warnings.warn(self.feature_error)
            self.inception = None
            self.video_model = None

        try:
            import lpips  # type: ignore

            self.lpips_model = lpips.LPIPS(net="alex").eval().to(self.device)
            for p in self.lpips_model.parameters():
                p.requires_grad_(False)
        except Exception as exc:
            warnings.warn(f"LPIPS unavailable; install lpips to enable it. Reason: {exc}")
            self.lpips_model = None

    @torch.no_grad()
    def image_features(self, video: torch.Tensor) -> np.ndarray:
        if not self.enabled or self.inception is None:
            return np.zeros((0, 2048), dtype=np.float64)
        frames = video[:: self.frame_stride].float().clamp(0.0, 1.0)
        feats: list[np.ndarray] = []
        for start in range(0, frames.shape[0], self.batch_size):
            batch = frames[start : start + self.batch_size].to(self.device)
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
            batch = (batch - self.imagenet_mean) / self.imagenet_std
            out = self.inception(batch)
            if isinstance(out, tuple):
                out = out[0]
            feats.append(out.detach().cpu().double().numpy())
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, 2048), dtype=np.float64)

    @torch.no_grad()
    def video_features(self, video: torch.Tensor, frames: int = 16) -> np.ndarray:
        if not self.enabled or self.video_model is None:
            return np.zeros((0, 512), dtype=np.float64)
        x = video.float().clamp(0.0, 1.0)
        if x.shape[0] != frames:
            idx = torch.linspace(0, x.shape[0] - 1, frames).round().long()
            x = x[idx]
        x = x.unsqueeze(0).permute(0, 2, 1, 3, 4).to(self.device)
        b, c, t, h, w = x.shape
        x2 = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x2 = F.interpolate(x2, size=(112, 112), mode="bilinear", align_corners=False)
        x2 = (x2 - self.imagenet_mean) / self.imagenet_std
        x = x2.reshape(b, t, c, 112, 112).permute(0, 2, 1, 3, 4)
        out = self.video_model(x)
        return out.detach().cpu().double().numpy()

    @torch.no_grad()
    def lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if not self.enabled or self.lpips_model is None:
            return float("nan")
        n = min(pred.shape[0], target.shape[0])
        pred = pred[:n:self.frame_stride].float().clamp(0.0, 1.0)
        target = target[:n:self.frame_stride].float().clamp(0.0, 1.0)
        vals: list[float] = []
        for start in range(0, pred.shape[0], self.batch_size):
            p = pred[start : start + self.batch_size].to(self.device) * 2.0 - 1.0
            t = target[start : start + self.batch_size].to(self.device) * 2.0 - 1.0
            vals.append(float(self.lpips_model(p, t).mean().detach().cpu().item()))
        return float(np.mean(vals)) if vals else float("nan")


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] < 2 or b.shape[0] < 2:
        return float("nan")
    mu_a = np.mean(a, axis=0)
    mu_b = np.mean(b, axis=0)
    sigma_a = np.cov(a, rowvar=False)
    sigma_b = np.cov(b, rowvar=False)
    eps = 1e-6
    sigma_a = np.atleast_2d(sigma_a) + np.eye(sigma_a.shape[0]) * eps
    sigma_b = np.atleast_2d(sigma_b) + np.eye(sigma_b.shape[0]) * eps
    covmean = linalg.sqrtm(sigma_a @ sigma_b)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_a - mu_b
    return float(diff.dot(diff) + np.trace(sigma_a + sigma_b - 2.0 * covmean))


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    dist2 = np.maximum(aa + bb - 2.0 * (a @ b.T), 0.0)
    return np.sqrt(dist2)


def _manifold_radii(features: np.ndarray, k: int) -> np.ndarray | None:
    features = np.asarray(features, dtype=np.float64)
    n = int(features.shape[0])
    if n < 2:
        return None
    k_eff = min(max(1, int(k)), n - 1)
    dist = _pairwise_distances(features, features)
    np.fill_diagonal(dist, np.inf)
    return np.partition(dist, k_eff - 1, axis=1)[:, k_eff - 1]


def _fraction_inside_manifold(query: np.ndarray, support: np.ndarray, support_radii: np.ndarray) -> float:
    if query.size == 0 or support.size == 0 or support_radii.size == 0:
        return float("nan")
    dist = _pairwise_distances(query, support)
    inside = np.any(dist <= support_radii.reshape(1, -1), axis=1)
    return float(np.mean(inside))


def manifold_precision_recall(real_features: np.ndarray, pred_features: np.ndarray, k: int = 3) -> tuple[float, float]:
    """Improved precision/recall-style manifold coverage in feature space.

    Precision: fraction of predicted samples inside the GT feature manifold.
    Recall: fraction of GT samples inside the predicted feature manifold.
    """
    real_features = np.asarray(real_features, dtype=np.float64)
    pred_features = np.asarray(pred_features, dtype=np.float64)
    if real_features.shape[0] < 2 or pred_features.shape[0] < 2:
        return float("nan"), float("nan")
    real_radii = _manifold_radii(real_features, k)
    pred_radii = _manifold_radii(pred_features, k)
    if real_radii is None or pred_radii is None:
        return float("nan"), float("nan")
    precision = _fraction_inside_manifold(pred_features, real_features, real_radii)
    recall = _fraction_inside_manifold(real_features, pred_features, pred_radii)
    return precision, recall


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    keys = sorted({key for row in rows for key in row if isinstance(row.get(key), (int, float, np.floating))})
    for key in keys:
        values = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float, np.floating)) and np.isfinite(float(value)):
                values.append(float(value))
        if values:
            out[f"{key}_mean"] = float(np.mean(values))
            out[f"{key}_std"] = float(np.std(values))
    return out


def evaluate_model(
    model_dir: Path,
    test_index: dict[str, dict[str, Path]],
    out_root: Path,
    deep: DeepMetricExtractor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    pred_dir = find_prediction_dir(model_dir)
    stems = sorted(test_index)
    pred_paths = sorted(pred_dir.glob("*.mp4"))
    matched: list[tuple[str, Path]] = []
    unmatched = []
    used_stems = set()
    for pred_path in pred_paths:
        stem = match_pred_to_stem(pred_path, stems)
        if stem is None:
            unmatched.append(str(pred_path))
            continue
        if stem in used_stems:
            continue
        used_stems.add(stem)
        matched.append((stem, pred_path))
    if args.max_videos > 0:
        matched = matched[: int(args.max_videos)]

    model_id = safe_model_id(model_dir)
    model_out = out_root / model_id
    model_out.mkdir(parents=True, exist_ok=True)

    per_video_rows: list[dict[str, Any]] = []
    pred_img_features: list[np.ndarray] = []
    gt_img_features: list[np.ndarray] = []
    pred_vid_features: list[np.ndarray] = []
    gt_vid_features: list[np.ndarray] = []

    for stem, pred_path in tqdm(matched, desc=model_dir.name, leave=False):
        item = test_index[stem]
        pred, pred_fps = read_video_rgb_01(pred_path)
        _, h, w = pred.shape[1:]
        size = (w, h)
        target, gt_fps = read_video_rgb_01(item["video"], size=size)
        n = min(pred.shape[0], target.shape[0])
        pred = pred[:n]
        target = target[:n]
        background = read_image_01(item["background"], size=size, mode="RGB") if item["background"].exists() else target[0].clone()
        roi = read_image_01(item["roi"], size=size, mode="L") if item["roi"].exists() else torch.ones(1, h, w)
        fps = gt_fps or pred_fps or float(args.default_fps)

        metrics = compute_fast_metrics(pred, target, background, roi, item["audio"], fps, args)
        metrics["lpips_alex"] = deep.lpips(pred, target) if args.deep_metrics else float("nan")
        if args.deep_metrics:
            pred_img_features.append(deep.image_features(pred))
            gt_img_features.append(deep.image_features(target))
            pred_vid_features.append(deep.video_features(pred))
            gt_vid_features.append(deep.video_features(target))
        row: dict[str, Any] = {
            "model_id": model_id,
            "model_dir": str(model_dir),
            "pred_dir": str(pred_dir),
            "stem": stem,
            "pred_mp4": str(pred_path),
            "gt_mp4": str(item["video"]),
            "audio_csv": str(item["audio"]),
            "background": str(item["background"]),
            "roi": str(item["roi"]),
            "frames_used": int(n),
            "pred_frames": int(pred.shape[0]),
            "gt_frames": int(target.shape[0]),
            "fps": float(fps),
        }
        row.update(metrics)
        per_video_rows.append(row)

    summary = aggregate(per_video_rows)
    if args.deep_metrics:
        pred_img = np.concatenate([x for x in pred_img_features if x.size], axis=0) if any(x.size for x in pred_img_features) else np.zeros((0, 2048))
        gt_img = np.concatenate([x for x in gt_img_features if x.size], axis=0) if any(x.size for x in gt_img_features) else np.zeros((0, 2048))
        pred_vid = np.concatenate([x for x in pred_vid_features if x.size], axis=0) if any(x.size for x in pred_vid_features) else np.zeros((0, 512))
        gt_vid = np.concatenate([x for x in gt_vid_features if x.size], axis=0) if any(x.size for x in gt_vid_features) else np.zeros((0, 512))
        summary["fid_inception_v3"] = frechet_distance(pred_img, gt_img)
        summary["fvd_r3d18"] = frechet_distance(pred_vid, gt_vid)
        precision, recall = manifold_precision_recall(gt_img, pred_img, k=int(args.precision_recall_k))
        summary["precision_inception_v3"] = precision
        summary["recall_inception_v3"] = recall
        video_precision, video_recall = manifold_precision_recall(gt_vid, pred_vid, k=int(args.precision_recall_k))
        summary["precision_r3d18"] = video_precision
        summary["recall_r3d18"] = video_recall
    else:
        summary["fid_inception_v3"] = float("nan")
        summary["fvd_r3d18"] = float("nan")
        summary["precision_inception_v3"] = float("nan")
        summary["recall_inception_v3"] = float("nan")
        summary["precision_r3d18"] = float("nan")
        summary["recall_r3d18"] = float("nan")
    lpips_values = [
        float(row.get("lpips_alex", float("nan")))
        for row in per_video_rows
        if np.isfinite(float(row.get("lpips_alex", float("nan"))))
    ]
    summary["lpips_alex"] = float(np.mean(lpips_values)) if lpips_values else float("nan")

    summary_row: dict[str, Any] = {
        "model_id": model_id,
        "model_dir": str(model_dir),
        "pred_dir": str(pred_dir),
        "num_matched_videos": len(per_video_rows),
        "num_test_videos": len(test_index),
        "missing_stems": ";".join([stem for stem in stems if stem not in used_stems]),
        "unmatched_pred_count": len(unmatched),
    }
    summary_row.update(summary)

    per_video_csv = model_out / "segfree_per_video_metrics.csv"
    summary_json = model_out / "segfree_summary_metrics.json"
    pd.DataFrame(per_video_rows).to_csv(per_video_csv, index=False, encoding="utf-8-sig")
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_row, f, ensure_ascii=False, indent=2)

    # Also place a compact copy inside the model directory for quick lookup.
    pd.DataFrame(per_video_rows).to_csv(model_dir / "segfree_per_video_metrics.csv", index=False, encoding="utf-8-sig")
    with (model_dir / "segfree_summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary_row, f, ensure_ascii=False, indent=2)
    return summary_row


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    test_index = build_test_index(dataset_root)
    deep = DeepMetricExtractor(
        enabled=bool(args.deep_metrics),
        device=device,
        batch_size=args.feature_batch_size,
        frame_stride=args.feature_frame_stride,
    )

    summaries: list[dict[str, Any]] = []
    for model_dir_text in args.model_dir:
        model_dir = Path(model_dir_text)
        try:
            summaries.append(evaluate_model(model_dir, test_index, out_root, deep, args))
        except Exception as exc:
            row = {
                "model_id": safe_model_id(model_dir),
                "model_dir": str(model_dir),
                "error": str(exc),
            }
            summaries.append(row)
            warnings.warn(f"Failed to evaluate {model_dir}: {exc}")

    summary_df = pd.DataFrame(summaries)
    summary_csv = out_root / "segfree_summary_all_models.csv"
    summary_xlsx = out_root / "segfree_summary_all_models.xlsx"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    try:
        summary_df.to_excel(summary_xlsx, index=False)
    except Exception as exc:
        warnings.warn(f"Failed to write xlsx summary: {exc}")

    with (out_root / "segfree_eval_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_root": str(dataset_root),
                "output_dir": str(out_root),
                "model_dirs": args.model_dir,
                "deep_metrics": bool(args.deep_metrics),
                "deep_metric_note": "FID uses torchvision Inception-v3 frame features. FVD uses torchvision R3D-18 Kinetics feature proxy, not official I3D FVD.",
                "precision_recall_note": "Precision/recall are k-NN manifold metrics computed in Inception-v3 frame-feature space and R3D-18 video-feature space.",
                "precision_recall_k": int(args.precision_recall_k),
                "columns_fast": FAST_METRIC_COLUMNS,
                "columns_deep": DEEP_METRIC_COLUMNS,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps({"summary_csv": str(summary_csv), "summary_xlsx": str(summary_xlsx)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
