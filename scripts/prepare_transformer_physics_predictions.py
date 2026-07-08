from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from PIL import Image
from scipy.signal import resample_poly
from tqdm.auto import tqdm
from torchvision.models.vision_transformer import VisionTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_FEATURE_NAMES = [
    "log_rms",
    "log_peak",
    "log_mean_abs",
    "log_std",
    "zero_crossing_rate",
    "crest_factor",
    "spectral_centroid",
    "start_fraction",
]


class TransformerModel(nn.Module):
    """Architecture used by the wav-window STFT transformer training."""

    def __init__(
        self,
        num_classes: int = 3,
        num_heat_classes: int = 0,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 8,
        audio_feature_dim: int = 0,
    ):
        super().__init__()
        self.vit = VisionTransformer(
            image_size=224,
            patch_size=16,
            num_layers=num_layers,
            num_heads=nhead,
            hidden_dim=d_model,
            mlp_dim=d_model * 4,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=1000,
        )
        self.vit.heads = nn.Identity()
        self.audio_feature_dim = int(audio_feature_dim)
        if self.audio_feature_dim > 0:
            self.audio_encoder = nn.Sequential(
                nn.Linear(self.audio_feature_dim, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
            )
            head_dim = d_model * 2
        else:
            self.audio_encoder = None
            head_dim = d_model
        self.fc_heat_flux = nn.Linear(head_dim, 1)
        self.fc_heat_class = nn.Linear(head_dim, num_heat_classes) if num_heat_classes > 0 else None
        self.fc_htc = nn.Linear(head_dim, 1)
        self.fc_regime = nn.Linear(head_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        audio_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        features = self.vit(x)
        if self.audio_encoder is not None:
            if audio_features is None:
                audio_features = torch.zeros(
                    (features.shape[0], self.audio_feature_dim),
                    dtype=features.dtype,
                    device=features.device,
                )
            features = torch.cat([features, self.audio_encoder(audio_features)], dim=1)
        heat_flux = self.fc_heat_flux(features).squeeze(-1)
        heat_class = self.fc_heat_class(features) if self.fc_heat_class is not None else None
        htc = self.fc_htc(features).squeeze(-1)
        regime = self.fc_regime(features)
        return heat_flux, htc, regime, heat_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_root", default=str((PROJECT_ROOT.parent / "dataset_split_8011_seed42").resolve()))
    parser.add_argument("--results_root", default=str((PROJECT_ROOT.parent / "Results_Pool_100%_Spectrogram").resolve()))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--scaler_labels", default="")
    parser.add_argument(
        "--output_csv",
        default=str((PROJECT_ROOT / "outputs" / "physics_transformer_predictions" / "physics_condition_map_transformer.csv").resolve()),
    )
    parser.add_argument("--window_sec", type=float, default=0.10)
    parser.add_argument("--windows_per_file", type=int, default=16, help="0 means all non-overlapping windows.")
    parser.add_argument("--target_sr", type=int, default=1_000_000)
    parser.add_argument("--nfft", type=int, default=1000)
    parser.add_argument("--win", type=int, default=50)
    parser.add_argument("--hop", type=int, default=50)
    parser.add_argument("--frmax", type=float, default=300000.0)
    parser.add_argument("--dbmin", type=float, default=30.0)
    parser.add_argument("--dbmax", type=float, default=90.0)
    parser.add_argument("--predict_batch_size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--aggregation", default="median", choices=["median", "mean"])
    parser.add_argument("--save_window_jsonl", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--use_audio_features", type=int, default=1)
    return parser.parse_args()


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    results_root = Path(args.results_root)
    if not args.checkpoint:
        candidate = results_root / "transformer_wav100ms_vit" / "transformer_best_model.pth"
        if candidate.exists():
            args.checkpoint = str(candidate.resolve())
        else:
            args.checkpoint = str((results_root / "transformer_best_model.pth").resolve())
    if not args.scaler_labels:
        labels = results_root / "transformer_wav100ms_vit" / "target_scalers.json"
        if not labels.exists():
            labels = results_root / "audio_wav_physics_labels_8011.xlsx"
        if not labels.exists():
            labels = results_root / "dataset_stft_pool_test" / "labels.xlsx"
        if not labels.exists():
            labels = results_root / "dataset_signal_pool_test_10%" / "labels.xlsx"
        args.scaler_labels = str(labels.resolve())
    return args


def load_scaler_range(labels_path: str | Path) -> dict[str, float]:
    path = Path(labels_path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        heat_from_index = {int(k): float(v) for k, v in data.get("heat_from_index", {}).items()}
        regime_from_index = {int(k): int(v) for k, v in data.get("regime_from_index", {}).items()}
        return {
            "heat_min": float(data["heat_min"]),
            "heat_max": float(data["heat_max"]),
            "htc_min": float(data["htc_min"]),
            "htc_max": float(data["htc_max"]),
            "regime_min": int(min(regime_from_index.values())) if regime_from_index else 0,
            "regime_max": int(max(regime_from_index.values())) if regime_from_index else 0,
            "num_classes": int(len(regime_from_index)) if regime_from_index else 1,
            "regime_values": [int(regime_from_index[idx]) for idx in sorted(regime_from_index)],
            "heat_values": [float(heat_from_index[idx]) for idx in sorted(heat_from_index)],
        }
    df = pd.read_excel(path)
    if {"heat_flux", "htc", "boiling_regime"}.issubset(df.columns):
        heat = df["heat_flux"].astype(float)
        htc = df["htc"].astype(float)
        regime = df["boiling_regime"].astype(int)
    else:
        heat = df.iloc[:, 2].astype(float)
        htc = df.iloc[:, 3].astype(float)
        regime = df.iloc[:, 4].astype(int)
    return {
        "heat_min": float(heat.min()),
        "heat_max": float(heat.max()),
        "htc_min": float(htc.min()),
        "htc_max": float(htc.max()),
        "regime_min": int(regime.min()),
        "regime_max": int(regime.max()),
        "num_classes": int(regime.max() - regime.min() + 1),
        "regime_values": sorted(int(v) for v in regime.unique()),
        "heat_values": sorted(float(v) for v in heat.unique()),
    }


def inverse_minmax(x: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    return x * (max_value - min_value) + min_value


def normalize_minmax(x: float, min_value: float, max_value: float) -> float:
    denom = max(max_value - min_value, 1e-12)
    return float(np.clip((x - min_value) / denom, 0.0, 1.0))


def load_audio_mono(path: str | Path, target_sr: int) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = audio[:, 0].astype(np.float32)
    if int(sr) != int(target_sr):
        g = math.gcd(int(sr), int(target_sr))
        x = resample_poly(x, int(target_sr) // g, int(sr) // g).astype(np.float32)
        sr = int(target_sr)
    return x, int(sr)


def make_window_segment(x: np.ndarray, start: int, window_samples: int) -> np.ndarray:
    segment = np.zeros((window_samples,), dtype=np.float32)
    available = max(0, min(window_samples, x.shape[0] - start))
    if available > 0:
        segment[:available] = x[start : start + available]
    return segment


def compute_audio_features(segment: np.ndarray, start: int, total_samples: int, sr: int) -> torch.Tensor:
    eps = 1e-12
    abs_segment = np.abs(segment)
    rms = float(np.sqrt(np.mean(np.square(segment.astype(np.float64))) + eps))
    peak = float(np.max(abs_segment) if abs_segment.size else 0.0)
    mean_abs = float(np.mean(abs_segment) if abs_segment.size else 0.0)
    std = float(np.std(segment) if segment.size else 0.0)
    signs = np.signbit(segment)
    zcr = float(np.mean(signs[1:] != signs[:-1])) if segment.size > 1 else 0.0
    crest = float(min(peak / max(rms, eps), 20.0) / 20.0)
    spectrum = np.abs(np.fft.rfft(segment.astype(np.float32)))
    freqs = np.fft.rfftfreq(segment.size, d=1.0 / float(sr)) if segment.size else np.array([0.0])
    spectral_sum = float(np.sum(spectrum) + eps)
    centroid = float(np.sum(freqs * spectrum) / spectral_sum) if spectrum.size else 0.0
    centroid_norm = float(np.clip(centroid / max(sr * 0.5, eps), 0.0, 1.0))
    start_fraction = float(np.clip(start / max(total_samples - segment.size, 1), 0.0, 1.0))

    def log_amp(value: float) -> float:
        return float(np.clip((math.log10(max(value, eps)) + 8.0) / 8.0, 0.0, 1.0))

    return torch.tensor(
        [
            log_amp(rms),
            log_amp(peak),
            log_amp(mean_abs),
            log_amp(std),
            float(np.clip(zcr, 0.0, 1.0)),
            crest,
            centroid_norm,
            start_fraction,
        ],
        dtype=torch.float32,
    )


def select_window_starts(num_samples: int, window_samples: int, windows_per_file: int) -> list[int]:
    if num_samples <= 0:
        return [0]
    if num_samples <= window_samples:
        return [0]
    max_start = num_samples - window_samples
    if windows_per_file <= 0:
        starts = list(range(0, max_start + 1, window_samples))
        if starts[-1] != max_start:
            starts.append(max_start)
        return starts
    starts = np.linspace(0, max_start, num=min(windows_per_file, max_start + 1))
    return sorted({int(round(v)) for v in starts})


def power_to_db_like_librosa(mag: np.ndarray, ref: float = 1e-6, amin: float = 1e-20) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(mag, amin) / ref)


def jet_colormap(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def window_to_stft_tensor(
    x: np.ndarray,
    start: int,
    window_samples: int,
    sr: int,
    nfft: int,
    win: int,
    hop: int,
    frmax: float,
    dbmin: float,
    dbmax: float,
) -> torch.Tensor:
    segment = make_window_segment(x, start, window_samples)

    wav = torch.from_numpy(segment)
    stft = torch.stft(
        wav,
        n_fft=nfft,
        hop_length=hop,
        win_length=win,
        window=torch.hann_window(win),
        center=True,
        return_complex=True,
    )
    db = power_to_db_like_librosa(stft.abs().numpy(), ref=1e-6)
    max_bin = int(min(db.shape[0] - 1, math.floor(frmax / (sr / float(nfft)))))
    db = db[: max_bin + 1]
    img = np.clip((db - dbmin) / max(dbmax - dbmin, 1e-12), 0.0, 1.0)
    img = Image.fromarray((np.flipud(img) * 255.0).round().astype(np.uint8))
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    img = np.asarray(img, dtype=np.float32) / 255.0
    rgb = jet_colormap(img)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def list_audio_rows(split_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        audio_dir = split_root / split / "audio_wav"
        if not audio_dir.exists():
            raise FileNotFoundError(f"Missing audio folder: {audio_dir}")
        for wav in sorted(audio_dir.glob("*.wav")):
            rows.append({"split": split, "stem": wav.stem, "audio": wav})
    return rows


def parse_heat_flux_from_stem(stem: str) -> int | None:
    parts = stem.split("_")
    if len(parts) >= 3 and re.fullmatch(r"\d{6}", parts[0]) and parts[1].isdigit():
        return int(parts[1])
    if parts and parts[0].isdigit():
        return int(parts[0])
    match = re.search(r"(?<!\d)(\d{1,3})(?!\d)", stem)
    return int(match.group(1)) if match else None


@torch.no_grad()
def predict_file(
    model: TransformerModel,
    wav_path: Path,
    args: argparse.Namespace,
    scaler: dict[str, float],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x, sr = load_audio_mono(wav_path, args.target_sr)
    window_samples = int(round(args.window_sec * sr))
    starts = select_window_starts(x.shape[0], window_samples, args.windows_per_file)

    heat_norms: list[np.ndarray] = []
    htc_norms: list[np.ndarray] = []
    regime_probs: list[np.ndarray] = []
    heat_probs: list[np.ndarray] = []
    window_rows: list[dict[str, Any]] = []

    tensors = []
    feature_tensors = []
    tensor_starts = []
    for start in starts:
        segment = make_window_segment(x, start, window_samples)
        tensors.append(
            window_to_stft_tensor(
                x,
                start=start,
                window_samples=window_samples,
                sr=sr,
                nfft=args.nfft,
                win=args.win,
                hop=args.hop,
                frmax=args.frmax,
                dbmin=args.dbmin,
                dbmax=args.dbmax,
            )
        )
        feature_tensors.append(compute_audio_features(segment, start, int(x.shape[0]), sr))
        tensor_starts.append(start)
        if len(tensors) >= args.predict_batch_size:
            _predict_batch(
                model,
                tensors,
                feature_tensors,
                tensor_starts,
                device,
                heat_norms,
                htc_norms,
                regime_probs,
                heat_probs,
                window_rows,
                sr,
                bool(args.use_audio_features),
            )
            tensors, feature_tensors, tensor_starts = [], [], []
    if tensors:
        _predict_batch(
            model,
            tensors,
            feature_tensors,
            tensor_starts,
            device,
            heat_norms,
            htc_norms,
            regime_probs,
            heat_probs,
            window_rows,
            sr,
            bool(args.use_audio_features),
        )

    heat_norm_arr = np.concatenate(heat_norms)
    htc_norm_arr = np.concatenate(htc_norms)
    prob_arr = np.concatenate(regime_probs, axis=0)
    heat_prob_arr = np.concatenate(heat_probs, axis=0) if heat_probs else np.empty((0, 0), dtype=np.float32)

    if args.aggregation == "mean":
        heat_norm = float(np.mean(heat_norm_arr))
        htc_norm = float(np.mean(htc_norm_arr))
    else:
        heat_norm = float(np.median(heat_norm_arr))
        htc_norm = float(np.median(htc_norm_arr))
    mean_prob = np.mean(prob_arr, axis=0)
    regime_values = scaler.get("regime_values", list(range(prob_arr.shape[1])))
    pred_regime = int(regime_values[int(np.argmax(mean_prob))])
    heat_regression_raw = float(inverse_minmax(np.array([heat_norm]), scaler["heat_min"], scaler["heat_max"])[0])
    if heat_prob_arr.size and scaler.get("heat_values"):
        mean_heat_prob = np.mean(heat_prob_arr, axis=0)
        heat_values = np.array(scaler["heat_values"], dtype=np.float64)
        heat_raw = float(heat_values[int(np.argmax(mean_heat_prob))])
        heat_expected = float(mean_heat_prob @ heat_values)
        heat_confidence = float(np.max(mean_heat_prob))
    else:
        heat_raw = heat_regression_raw
        heat_expected = heat_regression_raw
        heat_confidence = 0.0
    htc_raw = float(inverse_minmax(np.array([htc_norm]), scaler["htc_min"], scaler["htc_max"])[0])

    summary = {
        "pred_heat_flux": heat_raw,
        "pred_heat_flux_expected": heat_expected,
        "pred_heat_flux_regression": heat_regression_raw,
        "pred_htc": htc_raw,
        "pred_regime": pred_regime,
        "cond_heat_flux_norm": normalize_minmax(heat_raw, scaler["heat_min"], scaler["heat_max"]),
        "cond_htc_norm": normalize_minmax(htc_raw, scaler["htc_min"], scaler["htc_max"]),
        "cond_regime_norm": normalize_minmax(pred_regime, scaler["regime_min"], scaler["regime_max"]),
        "heat_flux_window_std": float(np.std(inverse_minmax(heat_norm_arr, scaler["heat_min"], scaler["heat_max"]))),
        "htc_window_std": float(np.std(inverse_minmax(htc_norm_arr, scaler["htc_min"], scaler["htc_max"]))),
        "regime_confidence": float(np.max(mean_prob)),
        "heat_flux_confidence": heat_confidence,
        "windows_used": int(len(starts)),
        "window_sec": float(args.window_sec),
        "condition_source": "frozen_transformer_stft_window_ensemble",
    }
    return summary, window_rows


def _predict_batch(
    model: TransformerModel,
    tensors: list[torch.Tensor],
    feature_tensors: list[torch.Tensor],
    starts: list[int],
    device: torch.device,
    heat_norms: list[np.ndarray],
    htc_norms: list[np.ndarray],
    regime_probs: list[np.ndarray],
    heat_probs: list[np.ndarray],
    window_rows: list[dict[str, Any]],
    sr: int,
    use_audio_features: bool,
) -> None:
    batch = torch.stack(tensors, dim=0).to(device)
    features = torch.stack(feature_tensors, dim=0).to(device) if use_audio_features else None
    heat, htc, logits, heat_logits = model(batch, features)
    probs = torch.softmax(logits, dim=-1)
    heat_prob_tensor = torch.softmax(heat_logits, dim=-1) if heat_logits is not None else None
    heat_np = heat.detach().cpu().numpy().reshape(-1)
    htc_np = htc.detach().cpu().numpy().reshape(-1)
    probs_np = probs.detach().cpu().numpy()
    heat_probs_np = heat_prob_tensor.detach().cpu().numpy() if heat_prob_tensor is not None else None
    heat_norms.append(heat_np)
    htc_norms.append(htc_np)
    regime_probs.append(probs_np)
    if heat_probs_np is not None:
        heat_probs.append(heat_probs_np)
    for i, start in enumerate(starts):
        window_rows.append(
            {
                "start_sample": int(start),
                "start_sec": float(start / sr),
                "heat_flux_norm": float(heat_np[i]),
                "htc_norm": float(htc_np[i]),
                "regime_pred": int(np.argmax(probs_np[i])),
                "regime_confidence": float(np.max(probs_np[i])),
                "heat_class_pred": int(np.argmax(heat_probs_np[i])) if heat_probs_np is not None else "",
                "heat_class_confidence": float(np.max(heat_probs_np[i])) if heat_probs_np is not None else "",
            }
        )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = resolve_defaults(parse_args())
    split_root = Path(args.split_root)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scaler = load_scaler_range(args.scaler_labels)

    device = resolve_device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu")
    num_classes = int(state["fc_regime.weight"].shape[0]) if "fc_regime.weight" in state else int(scaler["num_classes"])
    num_heat_classes = int(state["fc_heat_class.weight"].shape[0]) if "fc_heat_class.weight" in state else 0
    audio_feature_dim = int(state["audio_encoder.0.weight"].shape[1]) if "audio_encoder.0.weight" in state else 0
    args.use_audio_features = int(bool(audio_feature_dim) and bool(args.use_audio_features))
    model = TransformerModel(
        num_classes=num_classes,
        num_heat_classes=num_heat_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        audio_feature_dim=audio_feature_dim,
    ).to(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture mismatch. Use a checkpoint trained by train_transformer_physics_stft.py."
        ) from exc
    model.eval()

    audio_rows = list_audio_rows(split_root)
    out_rows: list[dict[str, Any]] = []
    window_jsonl = output_csv.with_name(output_csv.stem + "_windows.jsonl")
    window_f = window_jsonl.open("w", encoding="utf-8", newline="\n") if args.save_window_jsonl else None
    try:
        for row in tqdm(audio_rows, desc="Predicting frozen transformer physics"):
            summary, window_rows = predict_file(model, Path(row["audio"]), args, scaler, device)
            parsed_heat_flux = parse_heat_flux_from_stem(row["stem"])
            out = {
                "split": row["split"],
                "stem": row["stem"],
                "parsed_heat_flux": "" if parsed_heat_flux is None else parsed_heat_flux,
                **summary,
                "scaler_labels": str(Path(args.scaler_labels).resolve()),
                "checkpoint": str(Path(args.checkpoint).resolve()),
            }
            out_rows.append(out)
            if window_f is not None:
                for w in window_rows:
                    window_f.write(json.dumps({"split": row["split"], "stem": row["stem"], **w}, ensure_ascii=False) + "\n")
    finally:
        if window_f is not None:
            window_f.close()

    fieldnames = [
        "split",
        "stem",
        "parsed_heat_flux",
        "pred_heat_flux",
        "pred_heat_flux_expected",
        "pred_heat_flux_regression",
        "pred_htc",
        "pred_regime",
        "cond_heat_flux_norm",
        "cond_htc_norm",
        "cond_regime_norm",
        "condition_source",
        "windows_used",
        "window_sec",
        "heat_flux_window_std",
        "htc_window_std",
        "heat_flux_confidence",
        "regime_confidence",
        "scaler_labels",
        "checkpoint",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    counts = {}
    for row in out_rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    summary = {
        "output_csv": str(output_csv.resolve()),
        "window_jsonl": str(window_jsonl.resolve()) if args.save_window_jsonl else "",
        "counts": counts,
        "scaler": scaler,
        "args": vars(args),
    }
    comparable = [
        (float(row["parsed_heat_flux"]), float(row["pred_heat_flux"]))
        for row in out_rows
        if row["parsed_heat_flux"] != ""
    ]
    if comparable:
        err = np.array([abs(a - b) for a, b in comparable], dtype=np.float64)
        summary["parsed_heat_flux_diagnostic"] = {
            "count": int(err.size),
            "mae": float(err.mean()),
            "median_abs_error": float(np.median(err)),
        }
    summary_path = output_csv.with_name(output_csv.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
