from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torchvision.models.vision_transformer import VisionTransformer

from train_transformer_physics_stft import (
    AUDIO_FEATURE_NAMES,
    WavWindowDataset,
    build_heat_maps,
    build_regime_maps,
    compute_heat_class_weights,
    compute_regime_class_weights,
    compute_scalers,
    expand_rows,
    he_init_weights,
    inverse_minmax,
    load_labels,
    make_loader,
    normalize_minmax_array,
    resolve_device,
    save_json,
    seed_everything,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = PROJECT_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train three independent ViT physics predictors from 100 ms wav STFT windows.")
    parser.add_argument(
        "--labels-xlsx",
        default=str((WORK_ROOT / "Results_Pool_100%_Spectrogram" / "audio_wav_physics_labels_8011.xlsx").resolve()),
    )
    parser.add_argument(
        "--output-dir",
        default=str((WORK_ROOT / "Results_Pool_100%_Spectrogram" / "transformer_wav100ms_separate").resolve()),
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
    parser.add_argument("--cache-stft", type=int, default=1)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--use-audio-features", type=int, default=1)
    return parser.parse_args()


class SingleTaskTransformer(nn.Module):
    def __init__(
        self,
        output_dim: int,
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
        self.head = nn.Linear(head_dim, output_dim)

    def forward(self, image: torch.Tensor, audio_features: torch.Tensor | None = None) -> torch.Tensor:
        features = self.vit(image)
        if self.audio_encoder is not None:
            if audio_features is None:
                audio_features = torch.zeros(
                    (features.shape[0], self.audio_feature_dim),
                    dtype=features.dtype,
                    device=features.device,
                )
            features = torch.cat([features, self.audio_encoder(audio_features)], dim=1)
        return self.head(features)


def task_target(batch: dict[str, Any], task: str, device: torch.device) -> torch.Tensor:
    if task == "heat_flux":
        return batch["heat_class_index"].to(device)
    if task == "htc":
        return batch["htc_norm"].to(device)
    if task == "boiling_regime":
        return batch["regime_index"].to(device)
    raise ValueError(f"Unknown task: {task}")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    task: str,
    device: torch.device,
    use_audio_features: bool,
) -> dict[str, float]:
    model.train()
    running = 0.0
    total = 0
    for batch in tqdm(loader, desc=f"Train {task}", leave=False):
        images = batch["image"].to(device)
        features = batch["audio_features"].to(device) if use_audio_features else None
        target = task_target(batch, task, device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(images, features)
        if task == "htc":
            loss = criterion(pred.squeeze(-1), target)
        else:
            loss = criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = images.shape[0]
        running += float(loss.item()) * batch_size
        total += batch_size
    return {f"train_{task}_loss": running / max(total, 1)}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    task: str,
    device: torch.device,
    use_audio_features: bool,
    scalers: dict[str, float],
    heat_from_index: dict[int, float],
    regime_from_index: dict[int, int],
    split_name: str,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    model.eval()
    rows: list[dict[str, Any]] = []
    running = 0.0
    total = 0
    heat_values = np.array([heat_from_index[idx] for idx in range(len(heat_from_index))], dtype=np.float64)

    for batch in tqdm(loader, desc=f"Eval {split_name} {task}", leave=False):
        images = batch["image"].to(device)
        features = batch["audio_features"].to(device) if use_audio_features else None
        target = task_target(batch, task, device)
        pred = model(images, features)
        if task == "htc":
            loss = criterion(pred.squeeze(-1), target)
        else:
            loss = criterion(pred, target)

        batch_size = images.shape[0]
        running += float(loss.item()) * batch_size
        total += batch_size

        base = {
            "split": [str(v) for v in batch["split"]],
            "stem": [str(v) for v in batch["stem"]],
            "wav_name": [str(v) for v in batch["wav_name"]],
            "start_sample": [int(v.item()) for v in batch["start_sample"]],
            "window_index": [int(v.item()) for v in batch["window_index"]],
            "target_heat_flux": [float(v.item()) for v in batch["heat_flux"]],
            "target_htc": [float(v.item()) for v in batch["htc"]],
            "target_regime": [int(v.item()) for v in batch["boiling_regime"]],
        }

        if task == "heat_flux":
            probs = torch.softmax(pred, dim=-1).detach().cpu().numpy()
            pred_idx = np.argmax(probs, axis=1)
            pred_heat = heat_values[pred_idx]
            expected_heat = probs @ heat_values
            for i in range(batch_size):
                row = {key: values[i] for key, values in base.items()}
                row.update(
                    {
                        "pred_heat_flux": float(pred_heat[i]),
                        "pred_heat_flux_expected": float(expected_heat[i]),
                        "target_heat_class_index": int(batch["heat_class_index"][i].item()),
                        "pred_heat_class_index": int(pred_idx[i]),
                    }
                )
                for class_index, value in heat_from_index.items():
                    row[f"heat_prob_{value:g}"] = float(probs[i, class_index])
                rows.append(row)
        elif task == "htc":
            pred_norm = pred.squeeze(-1).detach().cpu().numpy()
            pred_htc = inverse_minmax(pred_norm, scalers["htc_min"], scalers["htc_max"])
            for i in range(batch_size):
                row = {key: values[i] for key, values in base.items()}
                row.update(
                    {
                        "pred_htc": float(pred_htc[i]),
                        "pred_htc_norm": float(pred_norm[i]),
                        "target_htc_norm": float(batch["htc_norm"][i].item()),
                    }
                )
                rows.append(row)
        elif task == "boiling_regime":
            probs = torch.softmax(pred, dim=-1).detach().cpu().numpy()
            pred_idx = np.argmax(probs, axis=1)
            pred_regime = np.array([regime_from_index[int(idx)] for idx in pred_idx], dtype=np.int64)
            for i in range(batch_size):
                row = {key: values[i] for key, values in base.items()}
                row.update(
                    {
                        "pred_regime": int(pred_regime[i]),
                        "target_regime_index": int(batch["regime_index"][i].item()),
                        "pred_regime_index": int(pred_idx[i]),
                    }
                )
                for class_index, value in regime_from_index.items():
                    row[f"regime_prob_{value}"] = float(probs[i, class_index])
                rows.append(row)

    window_df = pd.DataFrame(rows)
    metrics: dict[str, float] = {f"{split_name}_{task}_loss": running / max(total, 1)}

    if task == "heat_flux":
        prob_cols = [f"heat_prob_{heat_from_index[idx]:g}" for idx in range(len(heat_from_index))]
        agg_spec: dict[str, Any] = {
            "split": "first",
            "wav_name": "first",
            "target_heat_flux": "first",
            "target_htc": "first",
            "target_regime": "first",
        }
        for col in prob_cols:
            agg_spec[col] = "mean"
        file_df = window_df.groupby("stem", as_index=False).agg(agg_spec)
        mean_probs = file_df[prob_cols].to_numpy()
        pred_idx = np.argmax(mean_probs, axis=1)
        file_df["pred_heat_flux"] = heat_values[pred_idx]
        file_df["pred_heat_flux_expected"] = mean_probs @ heat_values
        metrics.update(
            {
                f"{split_name}_file_heat_mae": float(np.mean(np.abs(file_df["pred_heat_flux"] - file_df["target_heat_flux"]))),
                f"{split_name}_file_heat_expected_mae": float(
                    np.mean(np.abs(file_df["pred_heat_flux_expected"] - file_df["target_heat_flux"]))
                ),
                f"{split_name}_file_heat_acc": float(np.mean(file_df["pred_heat_flux"] == file_df["target_heat_flux"])),
                f"{split_name}_file_heat_mae_norm": float(
                    np.mean(
                        np.abs(
                            normalize_minmax_array(file_df["pred_heat_flux"].to_numpy(), scalers["heat_min"], scalers["heat_max"])
                            - normalize_minmax_array(
                                file_df["target_heat_flux"].to_numpy(), scalers["heat_min"], scalers["heat_max"]
                            )
                        )
                    )
                ),
            }
        )
        metrics[f"{split_name}_selection_score"] = metrics[f"{split_name}_file_heat_mae_norm"]
    elif task == "htc":
        file_df = (
            window_df.groupby("stem", as_index=False)
            .agg(
                {
                    "split": "first",
                    "wav_name": "first",
                    "target_heat_flux": "first",
                    "target_htc": "first",
                    "target_regime": "first",
                    "pred_htc": "median",
                    "pred_htc_norm": "median",
                    "target_htc_norm": "first",
                }
            )
            .copy()
        )
        metrics.update(
            {
                f"{split_name}_file_htc_mae": float(np.mean(np.abs(file_df["pred_htc"] - file_df["target_htc"]))),
                f"{split_name}_file_htc_mae_norm": float(np.mean(np.abs(file_df["pred_htc_norm"] - file_df["target_htc_norm"]))),
            }
        )
        metrics[f"{split_name}_selection_score"] = metrics[f"{split_name}_file_htc_mae_norm"]
    else:
        prob_cols = [f"regime_prob_{regime_from_index[idx]}" for idx in range(len(regime_from_index))]
        agg_spec = {
            "split": "first",
            "wav_name": "first",
            "target_heat_flux": "first",
            "target_htc": "first",
            "target_regime": "first",
        }
        for col in prob_cols:
            agg_spec[col] = "mean"
        file_df = window_df.groupby("stem", as_index=False).agg(agg_spec)
        pred_probs = file_df[prob_cols].to_numpy()
        pred_idx = np.argmax(pred_probs, axis=1)
        file_df["pred_regime"] = np.array([regime_from_index[int(idx)] for idx in pred_idx], dtype=np.int64)
        metrics[f"{split_name}_file_regime_acc"] = float(np.mean(file_df["pred_regime"] == file_df["target_regime"]))
        metrics[f"{split_name}_selection_score"] = 1.0 - metrics[f"{split_name}_file_regime_acc"]

    return metrics, window_df, file_df


def train_task(
    task: str,
    model_index: int,
    args: argparse.Namespace,
    loaders: dict[str, torch.utils.data.DataLoader],
    train_df: pd.DataFrame,
    scalers: dict[str, float],
    heat_to_index: dict[float, int],
    heat_from_index: dict[int, float],
    regime_to_index: dict[int, int],
    regime_from_index: dict[int, int],
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_dir = Path(args.output_dir) / f"transformer_{model_index}_{task}"
    task_dir.mkdir(parents=True, exist_ok=True)

    if task == "heat_flux":
        output_dim = len(heat_to_index)
        criterion = nn.CrossEntropyLoss(weight=compute_heat_class_weights(train_df, heat_to_index).to(device))
    elif task == "htc":
        output_dim = 1
        criterion = nn.MSELoss()
    elif task == "boiling_regime":
        output_dim = len(regime_to_index)
        criterion = nn.CrossEntropyLoss(weight=compute_regime_class_weights(train_df, regime_to_index).to(device))
    else:
        raise ValueError(f"Unknown task: {task}")

    model = SingleTaskTransformer(
        output_dim=output_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        image_size=args.image_size,
        audio_feature_dim=len(AUDIO_FEATURE_NAMES) if int(args.use_audio_features) else 0,
    ).to(device)
    model.apply(he_init_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history: list[dict[str, float]] = []
    best_score = float("inf")
    best_epoch = -1
    stale_epochs = 0
    best_path = task_dir / "best_model.pth"
    best_val_file_df = pd.DataFrame()
    best_val_window_df = pd.DataFrame()
    use_audio_features = bool(int(args.use_audio_features))

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n[{task}] Epoch {epoch}/{args.num_epochs}")
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, criterion, task, device, use_audio_features)
        val_metrics, val_window_df, val_file_df = evaluate(
            model,
            loaders["val"],
            criterion,
            task,
            device,
            use_audio_features,
            scalers,
            heat_from_index,
            regime_from_index,
            "val",
        )
        metrics = {"epoch": float(epoch), **train_metrics, **val_metrics}
        history.append(metrics)
        pd.DataFrame(history).to_csv(task_dir / "training_history.csv", index=False)
        scheduler.step(val_metrics["val_selection_score"])
        print(format_epoch_log(task, train_metrics, val_metrics))

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
                print(f"[{task}] Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    train_metrics, train_window_df, train_file_df = evaluate(
        model,
        loaders["train"],
        criterion,
        task,
        device,
        use_audio_features,
        scalers,
        heat_from_index,
        regime_from_index,
        "train_eval",
    )
    val_metrics, val_window_df, val_file_df = evaluate(
        model,
        loaders["val"],
        criterion,
        task,
        device,
        use_audio_features,
        scalers,
        heat_from_index,
        regime_from_index,
        "val",
    )
    test_metrics, test_window_df, test_file_df = evaluate(
        model,
        loaders["test"],
        criterion,
        task,
        device,
        use_audio_features,
        scalers,
        heat_from_index,
        regime_from_index,
        "test",
    )

    best_val_window_df.to_csv(task_dir / "val_window_predictions_best.csv", index=False)
    best_val_file_df.to_excel(task_dir / "val_file_predictions_best.xlsx", index=False)
    train_window_df.to_csv(task_dir / "train_window_predictions.csv", index=False)
    train_file_df.to_excel(task_dir / "train_file_predictions.xlsx", index=False)
    val_window_df.to_csv(task_dir / "val_window_predictions_final.csv", index=False)
    val_file_df.to_excel(task_dir / "val_file_predictions_final.xlsx", index=False)
    test_window_df.to_csv(task_dir / "test_window_predictions.csv", index=False)
    test_file_df.to_excel(task_dir / "test_file_predictions.xlsx", index=False)

    summary = {
        "task": task,
        "model_index": int(model_index),
        "best_epoch": int(best_epoch),
        "best_monitor_name": monitor_name_for_task(task),
        "best_monitor_value": float(best_score),
        "checkpoint": str(best_path.resolve()),
        "train_eval_metrics": {k: float(v) for k, v in train_metrics.items()},
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    save_json(task_dir / "training_summary.json", summary)
    return summary, train_file_df, val_file_df, test_file_df


def merge_task_predictions(predictions: dict[str, pd.DataFrame], scalers: dict[str, float], regime_from_index: dict[int, int]) -> pd.DataFrame:
    merged = predictions["heat_flux"][
        ["stem", "split", "wav_name", "target_heat_flux", "target_htc", "target_regime", "pred_heat_flux", "pred_heat_flux_expected"]
    ].copy()
    merged = merged.merge(predictions["htc"][["stem", "pred_htc"]], on="stem", how="left")
    merged = merged.merge(predictions["boiling_regime"][["stem", "pred_regime"]], on="stem", how="left")
    regime_values = np.array([regime_from_index[idx] for idx in range(len(regime_from_index))], dtype=np.float64)
    regime_min = float(regime_values.min()) if regime_values.size else 0.0
    regime_max = float(regime_values.max()) if regime_values.size else 1.0
    merged["cond_heat_flux_norm"] = normalize_minmax_array(
        merged["pred_heat_flux"].to_numpy(dtype=np.float64), scalers["heat_min"], scalers["heat_max"]
    )
    merged["cond_htc_norm"] = normalize_minmax_array(merged["pred_htc"].to_numpy(dtype=np.float64), scalers["htc_min"], scalers["htc_max"])
    merged["cond_regime_norm"] = normalize_minmax_array(merged["pred_regime"].to_numpy(dtype=np.float64), regime_min, regime_max)
    merged["condition_source"] = "separate_transformer_stft_window_ensemble"
    return merged


def format_epoch_log(task: str, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> str:
    train_loss = list(train_metrics.values())[0]
    if task == "heat_flux":
        return " ".join(
            [
                f"train_ce_loss={train_loss:.4f}",
                f"val_heat_mae_label_units={val_metrics['val_file_heat_mae']:.3f}",
                f"val_heat_expected_mae_label_units={val_metrics['val_file_heat_expected_mae']:.3f}",
                f"val_heat_acc={val_metrics['val_file_heat_acc']:.3f}",
                f"best_monitor_heat_mae_norm={val_metrics['val_selection_score']:.4f}",
            ]
        )
    if task == "htc":
        return " ".join(
            [
                f"train_mse_loss={train_loss:.4f}",
                f"val_htc_mae={val_metrics['val_file_htc_mae']:.4f}",
                f"best_monitor_htc_mae_norm={val_metrics['val_selection_score']:.4f}",
            ]
        )
    if task == "boiling_regime":
        return " ".join(
            [
                f"train_ce_loss={train_loss:.4f}",
                f"val_regime_acc={val_metrics['val_file_regime_acc']:.3f}",
                f"best_monitor_1_minus_acc={val_metrics['val_selection_score']:.4f}",
            ]
        )
    return f"train_loss={train_loss:.4f} val_monitor={val_metrics['val_selection_score']:.4f}"


def monitor_name_for_task(task: str) -> str:
    if task == "heat_flux":
        return "val_file_heat_mae_norm"
    if task == "htc":
        return "val_file_htc_mae_norm"
    if task == "boiling_regime":
        return "1_minus_val_file_regime_acc"
    return "val_selection_score"


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "train_args.json", vars(args))

    labels_df = load_labels(Path(args.labels_xlsx))
    train_df = labels_df[labels_df["split"] == "train"].reset_index(drop=True)
    val_df = labels_df[labels_df["split"] == "val"].reset_index(drop=True)
    test_df = labels_df[labels_df["split"] == "test"].reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Train/val/test splits must all be present in the label workbook.")

    scalers = compute_scalers(train_df)
    heat_to_index, heat_from_index = build_heat_maps(labels_df)
    regime_to_index, regime_from_index = build_regime_maps(labels_df)
    save_json(
        output_dir / "target_scalers.json",
        {
            **scalers,
            "heat_to_index": {str(k): int(v) for k, v in heat_to_index.items()},
            "heat_from_index": {str(k): float(v) for k, v in heat_from_index.items()},
            "regime_to_index": {str(k): int(v) for k, v in regime_to_index.items()},
            "regime_from_index": {str(k): int(v) for k, v in regime_from_index.items()},
            "audio_feature_names": AUDIO_FEATURE_NAMES,
        },
    )

    train_records = expand_rows(train_df, args.target_sr, args.window_sec, args.train_windows_per_file, args.window_selection)
    val_records = expand_rows(val_df, args.target_sr, args.window_sec, args.eval_windows_per_file, args.window_selection)
    test_records = expand_rows(test_df, args.target_sr, args.window_sec, args.eval_windows_per_file, args.window_selection)

    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "stft_cache"
    if not int(args.cache_stft):
        cache_dir = None
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

    device = resolve_device(args.device)
    task_order = [("heat_flux", 1), ("htc", 2), ("boiling_regime", 3)]
    summaries: dict[str, Any] = {}
    train_predictions: dict[str, pd.DataFrame] = {}
    val_predictions: dict[str, pd.DataFrame] = {}
    test_predictions: dict[str, pd.DataFrame] = {}
    for task, model_index in task_order:
        summary, train_file_df, val_file_df, test_file_df = train_task(
            task,
            model_index,
            args,
            loaders,
            train_df,
            scalers,
            heat_to_index,
            heat_from_index,
            regime_to_index,
            regime_from_index,
            device,
        )
        summaries[task] = summary
        train_predictions[task] = train_file_df
        val_predictions[task] = val_file_df
        test_predictions[task] = test_file_df

    train_combined = merge_task_predictions(train_predictions, scalers, regime_from_index)
    val_combined = merge_task_predictions(val_predictions, scalers, regime_from_index)
    test_combined = merge_task_predictions(test_predictions, scalers, regime_from_index)
    combined = pd.concat([train_combined, val_combined, test_combined], ignore_index=True)
    train_combined.to_excel(output_dir / "train_file_predictions_combined.xlsx", index=False)
    val_combined.to_excel(output_dir / "val_file_predictions_combined.xlsx", index=False)
    test_combined.to_excel(output_dir / "test_file_predictions_combined.xlsx", index=False)
    combined.to_excel(output_dir / "physics_condition_map_separate.xlsx", index=False)
    combined.to_csv(output_dir / "physics_condition_map_separate.csv", index=False, encoding="utf-8-sig")
    combined_summary = {
        "train_wavs": int(train_df.shape[0]),
        "val_wavs": int(val_df.shape[0]),
        "test_wavs": int(test_df.shape[0]),
        "train_windows": int(len(train_records)),
        "val_windows": int(len(val_records)),
        "test_windows": int(len(test_records)),
        "tasks": summaries,
    }
    save_json(output_dir / "training_summary.json", combined_summary)
    print("\nSeparate transformer training finished.")
    print(f"Summary: {output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
