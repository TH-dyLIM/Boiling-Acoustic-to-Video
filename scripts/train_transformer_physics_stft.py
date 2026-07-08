from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from torchvision.models.vision_transformer import VisionTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = PROJECT_ROOT.parent
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ViT-based physics predictor from 100 ms wav STFT windows.")
    parser.add_argument(
        "--labels-xlsx",
        default=str((WORK_ROOT / "Results_Pool_100%_Spectrogram" / "audio_wav_physics_labels_8011.xlsx").resolve()),
    )
    parser.add_argument(
        "--output-dir",
        default=str((WORK_ROOT / "Results_Pool_100%_Spectrogram" / "transformer_wav100ms_vit").resolve()),
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--regime-loss-weight", type=float, default=1.0)
    parser.add_argument("--heat-loss-weight", type=float, default=1.0)
    parser.add_argument("--heat-class-loss-weight", type=float, default=1.0)
    parser.add_argument("--htc-loss-weight", type=float, default=1.0)
    parser.add_argument("--use-audio-features", type=int, default=1)
    parser.add_argument("--cache-stft", type=int, default=1)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TransformerModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_heat_classes: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        image_size: int,
        audio_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        self.vit = VisionTransformer(
            image_size=image_size,
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
        self.fc_heat_class = nn.Linear(head_dim, num_heat_classes)
        self.fc_htc = nn.Linear(head_dim, 1)
        self.fc_regime = nn.Linear(head_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        audio_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        heat_class = self.fc_heat_class(features)
        htc = self.fc_htc(features).squeeze(-1)
        regime = self.fc_regime(features)
        return heat_flux, htc, regime, heat_class


def he_init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="wav_labels")
    required = ["split", "wav_name", "stem", "wav_path", "heat_flux", "htc", "boiling_regime"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required label columns in {path}: {missing}")
    return df.copy()


def compute_scalers(train_df: pd.DataFrame) -> dict[str, float]:
    return {
        "heat_min": float(train_df["heat_flux"].min()),
        "heat_max": float(train_df["heat_flux"].max()),
        "htc_min": float(train_df["htc"].min()),
        "htc_max": float(train_df["htc"].max()),
    }


def normalize_minmax(x: float, min_value: float, max_value: float) -> float:
    denom = max(max_value - min_value, 1e-12)
    return float(np.clip((x - min_value) / denom, 0.0, 1.0))


def inverse_minmax(x: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    return x * (max_value - min_value) + min_value


def normalize_minmax_array(x: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    denom = max(max_value - min_value, 1e-12)
    return np.clip((x - min_value) / denom, 0.0, 1.0)


def build_regime_maps(df: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    values = sorted(int(v) for v in df["boiling_regime"].astype(int).unique())
    to_index = {value: idx for idx, value in enumerate(values)}
    from_index = {idx: value for value, idx in to_index.items()}
    return to_index, from_index


def build_heat_maps(df: pd.DataFrame) -> tuple[dict[float, int], dict[int, float]]:
    values = sorted(float(v) for v in df["heat_flux"].astype(float).unique())
    to_index = {value: idx for idx, value in enumerate(values)}
    from_index = {idx: value for value, idx in to_index.items()}
    return to_index, from_index


def load_csv_audio(path: str | Path) -> tuple[np.ndarray, int]:
    df = pd.read_csv(path)
    if "Voltage" in df.columns:
        y = df["Voltage"].to_numpy(dtype=np.float32)
    else:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError(f"No numeric audio column found in CSV: {path}")
        y = numeric.iloc[:, -1].to_numpy(dtype=np.float32)
    if "Time" in df.columns and df.shape[0] >= 2:
        dt = float(df["Time"].iloc[1]) - float(df["Time"].iloc[0])
        sr = int(round(1.0 / dt)) if dt > 0 else 1_000_000
    else:
        sr = 1_000_000
    return y.astype(np.float32), int(sr)


@lru_cache(maxsize=64)
def load_audio_mono_cached(path_str: str, target_sr: int) -> tuple[np.ndarray, int]:
    path = Path(path_str)
    if path.suffix.lower() == ".csv":
        x, sr = load_csv_audio(path)
    else:
        audio, sr = sf.read(path_str, dtype="float32", always_2d=True)
        x = audio[:, 0].astype(np.float32)
    if int(sr) != int(target_sr):
        g = math.gcd(int(sr), int(target_sr))
        x = resample_poly(x, int(target_sr) // g, int(sr) // g).astype(np.float32)
        sr = int(target_sr)
    return x, int(sr)


@lru_cache(maxsize=16)
def hann_window_cached(win: int) -> torch.Tensor:
    return torch.hann_window(win)


def select_window_starts(num_samples: int, window_samples: int, windows_per_file: int) -> list[int]:
    if num_samples <= 0 or num_samples <= window_samples:
        return [0]
    max_start = num_samples - window_samples
    if windows_per_file <= 0:
        starts = list(range(0, max_start + 1, window_samples))
        if starts[-1] != max_start:
            starts.append(max_start)
        return starts
    starts = np.linspace(0, max_start, num=min(windows_per_file, max_start + 1))
    return sorted({int(round(v)) for v in starts})


def select_energy_window_starts(x: np.ndarray, window_samples: int, windows_per_file: int) -> list[int]:
    num_samples = int(x.shape[0])
    if num_samples <= 0 or num_samples <= window_samples:
        return [0]
    if windows_per_file <= 0:
        return select_window_starts(num_samples, window_samples, windows_per_file)

    max_start = num_samples - window_samples
    candidate_count = max(windows_per_file * 6, windows_per_file)
    candidates = np.linspace(0, max_start, num=min(candidate_count, max_start + 1))
    scored: list[tuple[float, int]] = []
    for value in candidates:
        start = int(round(float(value)))
        segment = x[start : start + window_samples]
        if segment.size == 0:
            score = 0.0
        else:
            score = float(np.sqrt(np.mean(np.square(segment.astype(np.float64)))))
        scored.append((score, start))
    best = sorted(scored, key=lambda item: (-item[0], item[1]))[:windows_per_file]
    return sorted({int(start) for _, start in best})


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

    values = [
        log_amp(rms),
        log_amp(peak),
        log_amp(mean_abs),
        log_amp(std),
        float(np.clip(zcr, 0.0, 1.0)),
        crest,
        centroid_norm,
        start_fraction,
    ]
    return torch.tensor(values, dtype=torch.float32)


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
    image_size: int,
) -> torch.Tensor:
    segment = make_window_segment(x, start, window_samples)

    wav = torch.from_numpy(segment)
    stft = torch.stft(
        wav,
        n_fft=nfft,
        hop_length=hop,
        win_length=win,
        window=hann_window_cached(win),
        center=True,
        return_complex=True,
    )
    db = power_to_db_like_librosa(stft.abs().numpy(), ref=1e-6)
    max_bin = int(min(db.shape[0] - 1, math.floor(frmax / (sr / float(nfft)))))
    db = db[: max_bin + 1]
    img = np.clip((db - dbmin) / max(dbmax - dbmin, 1e-12), 0.0, 1.0)
    img = Image.fromarray((np.flipud(img) * 255.0).round().astype(np.uint8))
    img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    img = np.asarray(img, dtype=np.float32) / 255.0
    rgb = jet_colormap(img)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def expand_rows(
    df: pd.DataFrame,
    target_sr: int,
    window_sec: float,
    windows_per_file: int,
    window_selection: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    window_samples = int(round(window_sec * target_sr))
    for record in df.to_dict(orient="records"):
        num_samples = int(record["num_samples"])
        sample_rate = int(record["sample_rate"])
        if sample_rate != target_sr:
            duration_sec = float(record.get("duration_sec", 0.0) or 0.0)
            if duration_sec <= 0.0 and str(record["wav_path"]).lower().endswith(".wav"):
                audio_info = sf.info(record["wav_path"])
                duration_sec = float(audio_info.frames / audio_info.samplerate)
            num_samples = int(round(duration_sec * target_sr))
        if window_selection == "energy_topk":
            x, _ = load_audio_mono_cached(str(record["wav_path"]), target_sr)
            starts = select_energy_window_starts(x, window_samples, windows_per_file)
        else:
            starts = select_window_starts(num_samples, window_samples, windows_per_file)
        for window_index, start in enumerate(starts):
            rows.append(
                {
                    **record,
                    "window_index": int(window_index),
                    "start_sample": int(start),
                    "window_samples": int(window_samples),
                }
            )
    return rows


class WavWindowDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        scalers: dict[str, float],
        regime_to_index: dict[int, int],
        heat_to_index: dict[float, int],
        target_sr: int,
        nfft: int,
        win: int,
        hop: int,
        frmax: float,
        dbmin: float,
        dbmax: float,
        image_size: int,
        cache_dir: Path | None = None,
    ) -> None:
        self.records = list(records)
        self.scalers = scalers
        self.regime_to_index = regime_to_index
        self.heat_to_index = heat_to_index
        self.target_sr = int(target_sr)
        self.nfft = int(nfft)
        self.win = int(win)
        self.hop = int(hop)
        self.frmax = float(frmax)
        self.dbmin = float(dbmin)
        self.dbmax = float(dbmax)
        self.image_size = int(image_size)
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.records)

    def _cache_path(self, row: dict[str, Any]) -> Path:
        safe_stem = str(row["stem"]).replace(" ", "_")
        name = (
            f"{safe_stem}__{int(row['start_sample']):07d}"
            f"__w{int(row['window_samples'])}"
            f"__n{self.nfft}_win{self.win}_hop{self.hop}"
            f"__f{int(self.frmax)}_db{int(self.dbmin)}_{int(self.dbmax)}.pt"
        )
        return self.cache_dir / name

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        image: torch.Tensor
        x, sr = load_audio_mono_cached(str(row["wav_path"]), self.target_sr)
        segment = make_window_segment(x, int(row["start_sample"]), int(row["window_samples"]))
        audio_features = compute_audio_features(
            segment,
            start=int(row["start_sample"]),
            total_samples=int(x.shape[0]),
            sr=sr,
        )
        cache_path = self._cache_path(row) if self.cache_dir is not None else None
        if cache_path is not None and cache_path.exists():
            image = torch.load(cache_path, map_location="cpu")
        else:
            image = window_to_stft_tensor(
                x,
                start=int(row["start_sample"]),
                window_samples=int(row["window_samples"]),
                sr=sr,
                nfft=self.nfft,
                win=self.win,
                hop=self.hop,
                frmax=self.frmax,
                dbmin=self.dbmin,
                dbmax=self.dbmax,
                image_size=self.image_size,
            )
            if cache_path is not None:
                torch.save(image, cache_path)

        return {
            "image": image,
            "heat_norm": torch.tensor(
                normalize_minmax(float(row["heat_flux"]), self.scalers["heat_min"], self.scalers["heat_max"]),
                dtype=torch.float32,
            ),
            "htc_norm": torch.tensor(
                normalize_minmax(float(row["htc"]), self.scalers["htc_min"], self.scalers["htc_max"]),
                dtype=torch.float32,
            ),
            "regime_index": torch.tensor(self.regime_to_index[int(row["boiling_regime"])], dtype=torch.long),
            "heat_class_index": torch.tensor(self.heat_to_index[float(row["heat_flux"])], dtype=torch.long),
            "audio_features": audio_features,
            "heat_flux": torch.tensor(float(row["heat_flux"]), dtype=torch.float32),
            "htc": torch.tensor(float(row["htc"]), dtype=torch.float32),
            "boiling_regime": torch.tensor(int(row["boiling_regime"]), dtype=torch.long),
            "stem": str(row["stem"]),
            "wav_name": str(row["wav_name"]),
            "split": str(row["split"]),
            "start_sample": torch.tensor(int(row["start_sample"]), dtype=torch.long),
            "window_index": torch.tensor(int(row["window_index"]), dtype=torch.long),
        }


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_regime_class_weights(train_df: pd.DataFrame, regime_to_index: dict[int, int]) -> torch.Tensor:
    counts = Counter(regime_to_index[int(v)] for v in train_df["boiling_regime"].astype(int).tolist())
    total = sum(counts.values())
    weights = []
    for class_index in range(len(regime_to_index)):
        count = max(counts.get(class_index, 0), 1)
        weights.append(total / (len(regime_to_index) * count))
    return torch.tensor(weights, dtype=torch.float32)


def compute_heat_class_weights(train_df: pd.DataFrame, heat_to_index: dict[float, int]) -> torch.Tensor:
    counts = Counter(heat_to_index[float(v)] for v in train_df["heat_flux"].astype(float).tolist())
    total = sum(counts.values())
    weights = []
    for class_index in range(len(heat_to_index)):
        count = max(counts.get(class_index, 0), 1)
        weights.append(total / (len(heat_to_index) * count))
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion_regression: nn.Module,
    criterion_regime: nn.Module,
    criterion_heat_class: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    running = {"loss": 0.0, "heat": 0.0, "heat_class": 0.0, "htc": 0.0, "regime": 0.0}
    total = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        audio_features = batch["audio_features"].to(device) if int(args.use_audio_features) else None
        heat_norm = batch["heat_norm"].to(device)
        htc_norm = batch["htc_norm"].to(device)
        regime_index = batch["regime_index"].to(device)
        heat_class_index = batch["heat_class_index"].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred_heat, pred_htc, pred_regime, pred_heat_class = model(images, audio_features)

        heat_loss = criterion_regression(pred_heat, heat_norm)
        heat_class_loss = criterion_heat_class(pred_heat_class, heat_class_index)
        htc_loss = criterion_regression(pred_htc, htc_norm)
        regime_loss = criterion_regime(pred_regime, regime_index)
        loss = (
            args.heat_loss_weight * heat_loss
            + args.heat_class_loss_weight * heat_class_loss
            + args.htc_loss_weight * htc_loss
            + args.regime_loss_weight * regime_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = images.shape[0]
        total += batch_size
        running["loss"] += float(loss.item()) * batch_size
        running["heat"] += float(heat_loss.item()) * batch_size
        running["heat_class"] += float(heat_class_loss.item()) * batch_size
        running["htc"] += float(htc_loss.item()) * batch_size
        running["regime"] += float(regime_loss.item()) * batch_size

    return {
        "train_total_loss": running["loss"] / max(total, 1),
        "train_heat_loss": running["heat"] / max(total, 1),
        "train_heat_class_loss": running["heat_class"] / max(total, 1),
        "train_htc_loss": running["htc"] / max(total, 1),
        "train_regime_loss": running["regime"] / max(total, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion_regression: nn.Module,
    criterion_regime: nn.Module,
    criterion_heat_class: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    scalers: dict[str, float],
    regime_from_index: dict[int, int],
    heat_from_index: dict[int, float],
    split_name: str,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    model.eval()
    running = {"loss": 0.0, "heat": 0.0, "heat_class": 0.0, "htc": 0.0, "regime": 0.0}
    total = 0
    rows: list[dict[str, Any]] = []
    num_classes = len(regime_from_index)
    num_heat_classes = len(heat_from_index)

    for batch in tqdm(loader, desc=f"Eval {split_name}", leave=False):
        images = batch["image"].to(device)
        audio_features = batch["audio_features"].to(device) if int(args.use_audio_features) else None
        heat_norm = batch["heat_norm"].to(device)
        htc_norm = batch["htc_norm"].to(device)
        regime_index = batch["regime_index"].to(device)
        heat_class_index = batch["heat_class_index"].to(device)

        pred_heat, pred_htc, pred_regime, pred_heat_class = model(images, audio_features)

        heat_loss = criterion_regression(pred_heat, heat_norm)
        heat_class_loss = criterion_heat_class(pred_heat_class, heat_class_index)
        htc_loss = criterion_regression(pred_htc, htc_norm)
        regime_loss = criterion_regime(pred_regime, regime_index)
        loss = (
            args.heat_loss_weight * heat_loss
            + args.heat_class_loss_weight * heat_class_loss
            + args.htc_loss_weight * htc_loss
            + args.regime_loss_weight * regime_loss
        )

        batch_size = images.shape[0]
        total += batch_size
        running["loss"] += float(loss.item()) * batch_size
        running["heat"] += float(heat_loss.item()) * batch_size
        running["heat_class"] += float(heat_class_loss.item()) * batch_size
        running["htc"] += float(htc_loss.item()) * batch_size
        running["regime"] += float(regime_loss.item()) * batch_size

        probs = torch.softmax(pred_regime, dim=-1).cpu().numpy()
        heat_probs = torch.softmax(pred_heat_class, dim=-1).cpu().numpy()
        pred_regime_index = np.argmax(probs, axis=1)
        pred_regime_raw = np.array([regime_from_index[int(v)] for v in pred_regime_index], dtype=np.int64)
        pred_heat_class_index = np.argmax(heat_probs, axis=1)
        pred_heat_class_raw = np.array([heat_from_index[int(v)] for v in pred_heat_class_index], dtype=np.float64)
        pred_heat_norm = pred_heat.detach().cpu().numpy().reshape(-1)
        pred_htc_norm = pred_htc.detach().cpu().numpy().reshape(-1)
        pred_heat_raw = inverse_minmax(pred_heat_norm, scalers["heat_min"], scalers["heat_max"])
        pred_htc_raw = inverse_minmax(pred_htc_norm, scalers["htc_min"], scalers["htc_max"])

        for i in range(batch_size):
            row = {
                "split": str(batch["split"][i]),
                "stem": str(batch["stem"][i]),
                "wav_name": str(batch["wav_name"][i]),
                "start_sample": int(batch["start_sample"][i].item()),
                "window_index": int(batch["window_index"][i].item()),
                "target_heat_flux": float(batch["heat_flux"][i].item()),
                "pred_heat_flux": float(pred_heat_class_raw[i]),
                "pred_heat_flux_regression": float(pred_heat_raw[i]),
                "target_htc": float(batch["htc"][i].item()),
                "pred_htc": float(pred_htc_raw[i]),
                "target_regime": int(batch["boiling_regime"][i].item()),
                "pred_regime": int(pred_regime_raw[i]),
                "target_heat_class_index": int(batch["heat_class_index"][i].item()),
                "pred_heat_class_index": int(pred_heat_class_index[i]),
                "target_heat_norm": float(batch["heat_norm"][i].item()),
                "pred_heat_norm": float(pred_heat_norm[i]),
                "target_htc_norm": float(batch["htc_norm"][i].item()),
                "pred_htc_norm": float(pred_htc_norm[i]),
                "audio_rms_feature": float(batch["audio_features"][i, 0].item()),
                "audio_peak_feature": float(batch["audio_features"][i, 1].item()),
            }
            for class_index in range(num_classes):
                row[f"regime_prob_{regime_from_index[class_index]}"] = float(probs[i, class_index])
            for class_index in range(num_heat_classes):
                row[f"heat_prob_{heat_from_index[class_index]:g}"] = float(heat_probs[i, class_index])
            rows.append(row)

    window_df = pd.DataFrame(rows)
    prob_cols = [f"regime_prob_{regime_from_index[idx]}" for idx in range(num_classes)]
    heat_prob_cols = [f"heat_prob_{heat_from_index[idx]:g}" for idx in range(num_heat_classes)]
    agg_spec: dict[str, Any] = {
        "split": "first",
        "wav_name": "first",
        "target_heat_flux": "first",
        "target_htc": "first",
        "target_regime": "first",
        "target_heat_norm": "first",
        "target_htc_norm": "first",
        "pred_heat_norm": "median",
        "pred_htc_norm": "median",
        "pred_heat_flux_regression": "median",
    }
    for col in prob_cols:
        agg_spec[col] = "mean"
    for col in heat_prob_cols:
        agg_spec[col] = "mean"
    file_df = window_df.groupby("stem", as_index=False).agg(agg_spec)

    pred_prob = file_df[prob_cols].to_numpy()
    pred_class_indices = np.argmax(pred_prob, axis=1)
    pred_regime_values = np.array([regime_from_index[int(idx)] for idx in pred_class_indices], dtype=np.int64)
    file_df["pred_regime"] = pred_regime_values
    pred_heat_prob = file_df[heat_prob_cols].to_numpy()
    pred_heat_class_indices = np.argmax(pred_heat_prob, axis=1)
    heat_class_values = np.array([heat_from_index[idx] for idx in range(num_heat_classes)], dtype=np.float64)
    file_df["pred_heat_flux"] = heat_class_values[pred_heat_class_indices]
    file_df["pred_heat_flux_expected"] = pred_heat_prob @ heat_class_values
    file_df["pred_htc"] = inverse_minmax(file_df["pred_htc_norm"].to_numpy(), scalers["htc_min"], scalers["htc_max"])

    metrics = {
        f"{split_name}_total_loss": running["loss"] / max(total, 1),
        f"{split_name}_heat_loss": running["heat"] / max(total, 1),
        f"{split_name}_heat_class_loss": running["heat_class"] / max(total, 1),
        f"{split_name}_htc_loss": running["htc"] / max(total, 1),
        f"{split_name}_regime_loss": running["regime"] / max(total, 1),
    }
    metrics.update(
        {
            f"{split_name}_window_heat_mae": float(np.mean(np.abs(window_df["pred_heat_flux"] - window_df["target_heat_flux"]))),
            f"{split_name}_window_heat_regression_mae": float(np.mean(np.abs(window_df["pred_heat_flux_regression"] - window_df["target_heat_flux"]))),
            f"{split_name}_window_heat_class_acc": float(np.mean(window_df["pred_heat_class_index"] == window_df["target_heat_class_index"])),
            f"{split_name}_window_htc_mae": float(np.mean(np.abs(window_df["pred_htc"] - window_df["target_htc"]))),
            f"{split_name}_window_regime_acc": float(np.mean(window_df["pred_regime"] == window_df["target_regime"])),
            f"{split_name}_file_heat_mae": float(np.mean(np.abs(file_df["pred_heat_flux"] - file_df["target_heat_flux"]))),
            f"{split_name}_file_heat_expected_mae": float(np.mean(np.abs(file_df["pred_heat_flux_expected"] - file_df["target_heat_flux"]))),
            f"{split_name}_file_heat_regression_mae": float(np.mean(np.abs(file_df["pred_heat_flux_regression"] - file_df["target_heat_flux"]))),
            f"{split_name}_file_heat_class_acc": float(np.mean(file_df["pred_heat_flux"] == file_df["target_heat_flux"])),
            f"{split_name}_file_htc_mae": float(np.mean(np.abs(file_df["pred_htc"] - file_df["target_htc"]))),
            f"{split_name}_file_regime_acc": float(np.mean(file_df["pred_regime"] == file_df["target_regime"])),
            f"{split_name}_file_heat_mae_norm": float(
                np.mean(
                    np.abs(
                        normalize_minmax_array(file_df["pred_heat_flux"].to_numpy(), scalers["heat_min"], scalers["heat_max"])
                        - file_df["target_heat_norm"].to_numpy()
                    )
                )
            ),
            f"{split_name}_file_htc_mae_norm": float(np.mean(np.abs(file_df["pred_htc_norm"] - file_df["target_htc_norm"]))),
        }
    )
    metrics[f"{split_name}_selection_score"] = (
        metrics[f"{split_name}_file_heat_mae_norm"]
        + metrics[f"{split_name}_file_htc_mae_norm"]
        + (1.0 - metrics[f"{split_name}_file_regime_acc"])
    )
    return metrics, window_df, file_df


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    labels_path = Path(args.labels_xlsx)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "stft_cache"
    if not int(args.cache_stft):
        cache_dir = None

    device = resolve_device(args.device)
    save_json(output_dir / "train_args.json", vars(args))

    labels_df = load_labels(labels_path)
    train_df = labels_df[labels_df["split"] == "train"].reset_index(drop=True)
    val_df = labels_df[labels_df["split"] == "val"].reset_index(drop=True)
    test_df = labels_df[labels_df["split"] == "test"].reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Train/val/test splits must all be present in the label workbook.")

    scalers = compute_scalers(train_df)
    regime_to_index, regime_from_index = build_regime_maps(labels_df)
    heat_to_index, heat_from_index = build_heat_maps(labels_df)
    save_json(
        output_dir / "target_scalers.json",
        {
            **scalers,
            "regime_to_index": {str(k): int(v) for k, v in regime_to_index.items()},
            "regime_from_index": {str(k): int(v) for k, v in regime_from_index.items()},
            "heat_to_index": {str(k): int(v) for k, v in heat_to_index.items()},
            "heat_from_index": {str(k): float(v) for k, v in heat_from_index.items()},
            "audio_feature_names": AUDIO_FEATURE_NAMES,
        },
    )

    train_records = expand_rows(train_df, args.target_sr, args.window_sec, args.train_windows_per_file, args.window_selection)
    val_records = expand_rows(val_df, args.target_sr, args.window_sec, args.eval_windows_per_file, args.window_selection)
    test_records = expand_rows(test_df, args.target_sr, args.window_sec, args.eval_windows_per_file, args.window_selection)

    datasets = {
        "train": WavWindowDataset(
            train_records,
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
            cache_dir=(cache_dir / "train") if cache_dir is not None else None,
        ),
        "val": WavWindowDataset(
            val_records,
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
            cache_dir=(cache_dir / "val") if cache_dir is not None else None,
        ),
        "test": WavWindowDataset(
            test_records,
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
            cache_dir=(cache_dir / "test") if cache_dir is not None else None,
        ),
    }
    loaders = {
        "train": make_loader(datasets["train"], args.batch_size, shuffle=True, num_workers=args.num_workers),
        "val": make_loader(datasets["val"], args.batch_size, shuffle=False, num_workers=args.num_workers),
        "test": make_loader(datasets["test"], args.batch_size, shuffle=False, num_workers=args.num_workers),
    }

    model = TransformerModel(
        num_classes=len(regime_to_index),
        num_heat_classes=len(heat_to_index),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        image_size=args.image_size,
        audio_feature_dim=len(AUDIO_FEATURE_NAMES) if int(args.use_audio_features) else 0,
    ).to(device)
    model.apply(he_init_weights)
    class_weights = compute_regime_class_weights(train_df, regime_to_index).to(device)
    heat_class_weights = compute_heat_class_weights(train_df, heat_to_index).to(device)
    criterion_regression = nn.MSELoss()
    criterion_regime = nn.CrossEntropyLoss(weight=class_weights)
    criterion_heat_class = nn.CrossEntropyLoss(weight=heat_class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history: list[dict[str, float]] = []
    best_score = float("inf")
    best_epoch = -1
    stale_epochs = 0
    best_path = output_dir / "transformer_best_model.pth"
    best_val_file_df = pd.DataFrame()
    best_val_window_df = pd.DataFrame()

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion_regression,
            criterion_regime,
            criterion_heat_class,
            device,
            args,
        )
        val_metrics, val_window_df, val_file_df = evaluate(
            model,
            loaders["val"],
            criterion_regression,
            criterion_regime,
            criterion_heat_class,
            device,
            args,
            scalers,
            regime_from_index,
            heat_from_index,
            "val",
        )

        metrics = {"epoch": float(epoch), **train_metrics, **val_metrics}
        history.append(metrics)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        scheduler.step(val_metrics["val_selection_score"])

        print(
            " ".join(
                [
                    f"train_loss={train_metrics['train_total_loss']:.4f}",
                    f"val_score={val_metrics['val_selection_score']:.4f}",
                    f"val_heat_mae={val_metrics['val_file_heat_mae']:.3f}",
                    f"val_htc_mae={val_metrics['val_file_htc_mae']:.3f}",
                    f"val_reg_acc={val_metrics['val_file_regime_acc']:.3f}",
                ]
            )
        )

        if val_metrics["val_selection_score"] < best_score:
            best_score = val_metrics["val_selection_score"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), best_path)
            best_val_file_df = val_file_df.copy()
            best_val_window_df = val_window_df.copy()
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if not best_path.exists():
        raise RuntimeError("Best model checkpoint was not saved.")

    model.load_state_dict(torch.load(best_path, map_location=device))
    val_metrics, val_window_df, val_file_df = evaluate(
        model,
        loaders["val"],
        criterion_regression,
        criterion_regime,
        criterion_heat_class,
        device,
        args,
        scalers,
        regime_from_index,
        heat_from_index,
        "val",
    )
    test_metrics, test_window_df, test_file_df = evaluate(
        model,
        loaders["test"],
        criterion_regression,
        criterion_regime,
        criterion_heat_class,
        device,
        args,
        scalers,
        regime_from_index,
        heat_from_index,
        "test",
    )

    best_val_window_df.to_csv(output_dir / "val_window_predictions_best.csv", index=False)
    best_val_file_df.to_excel(output_dir / "val_file_predictions_best.xlsx", index=False)
    val_window_df.to_csv(output_dir / "val_window_predictions_final.csv", index=False)
    val_file_df.to_excel(output_dir / "val_file_predictions_final.xlsx", index=False)
    test_window_df.to_csv(output_dir / "test_window_predictions.csv", index=False)
    test_file_df.to_excel(output_dir / "test_file_predictions.xlsx", index=False)

    summary = {
        "best_epoch": int(best_epoch),
        "best_val_selection_score": float(best_score),
        "train_wavs": int(train_df.shape[0]),
        "val_wavs": int(val_df.shape[0]),
        "test_wavs": int(test_df.shape[0]),
        "train_windows": int(len(train_records)),
        "val_windows": int(len(val_records)),
        "test_windows": int(len(test_records)),
        "scalers": scalers,
        "regime_values": [int(regime_from_index[idx]) for idx in range(len(regime_from_index))],
        "heat_flux_values": [float(heat_from_index[idx]) for idx in range(len(heat_from_index))],
        "audio_feature_names": AUDIO_FEATURE_NAMES,
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    save_json(output_dir / "training_summary.json", summary)
    (output_dir / "transformer_hyperparameters.txt").write_text(
        "\n".join(
            [
                f"Best epoch: {best_epoch}",
                f"Best val selection score: {best_score:.6f}",
                f"window_sec: {args.window_sec}",
                f"window_selection: {args.window_selection}",
                f"target_sr: {args.target_sr}",
                f"nfft: {args.nfft}",
                f"win: {args.win}",
                f"hop: {args.hop}",
                f"d_model: {args.d_model}",
                f"nhead: {args.nhead}",
                f"num_layers: {args.num_layers}",
                f"batch_size: {args.batch_size}",
                f"learning_rate: {args.learning_rate}",
                f"heat_class_loss_weight: {args.heat_class_loss_weight}",
                f"use_audio_features: {args.use_audio_features}",
            ]
        ),
        encoding="utf-8",
    )

    print("\nTraining finished.")
    print(f"Best checkpoint: {best_path}")
    print(f"Summary: {output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
