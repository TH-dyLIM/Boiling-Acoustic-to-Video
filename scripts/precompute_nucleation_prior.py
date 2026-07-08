from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from flow_residual.dataset import write_prior_blob
from flow_residual.nucleation_prior import compute_priors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "manifests" / "train.jsonl"))
    parser.add_argument("--cache_dir", default=str(PROJECT_ROOT / "cache" / "residual_frame_ar_delta_nophysics_sharp" / "train"))
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--foreground_threshold", type=float, default=0.04)
    parser.add_argument("--smoothing_sigma", type=float, default=2.0)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "cache" / "flow_residual" / "nucleation_prior.pt"))
    args = parser.parse_args()
    if args.config:
        with Path(args.config).open("r", encoding="utf-8-sig") as f:
            cfg = json.load(f)
        for key, value in cfg.items():
            if hasattr(args, key) and getattr(args, key, None) in (None, "", 0):
                setattr(args, key, value)
        for key in ("prior_manifest", "prior_cache_dir", "prior_resolution", "prior_threshold", "prior_sigma", "prior_out"):
            base_key = key.replace("prior_", "")
            mapping = {
                "manifest": "manifest",
                "cache_dir": "cache_dir",
                "resolution": "resolution",
                "threshold": "foreground_threshold",
                "sigma": "smoothing_sigma",
                "out": "out",
            }
            target = mapping.get(base_key)
            if target and key in cfg:
                setattr(args, target, cfg[key])
    return args


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir if args.cache_dir and Path(args.cache_dir).exists() else None
    per_class, global_prior, counts = compute_priors(
        manifest_path=args.manifest,
        cache_dir=cache_dir,
        resolution=int(args.resolution),
        foreground_threshold=float(args.foreground_threshold),
        smoothing_sigma=float(args.smoothing_sigma),
    )
    write_prior_blob(
        args.out,
        per_class=per_class,
        global_prior=global_prior,
        meta={
            "manifest": str(args.manifest),
            "cache_dir": str(cache_dir) if cache_dir else None,
            "resolution": int(args.resolution),
            "foreground_threshold": float(args.foreground_threshold),
            "smoothing_sigma": float(args.smoothing_sigma),
            "counts": counts,
        },
    )
    print(json.dumps({"out": str(args.out), "classes": list(per_class.keys()), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
