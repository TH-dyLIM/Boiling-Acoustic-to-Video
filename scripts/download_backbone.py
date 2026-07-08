from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="stabilityai/stable-video-diffusion-img2vid-xt")
    parser.add_argument("--local_dir", default="")
    parser.add_argument("--allow_patterns", nargs="*", default=None)
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("Install requirements_svd_lora.txt before downloading models.") from exc

    local_dir = args.local_dir or str((Path(__file__).resolve().parents[1] / "pretrained" / args.model_id.replace("/", "__")).resolve())
    path = snapshot_download(
        repo_id=args.model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=args.allow_patterns,
    )
    print(path)


if __name__ == "__main__":
    main()

