from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_manifest", default="manifests/test.jsonl")
    parser.add_argument("--stem", default="400_4")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--out_manifest", default="manifests/debug_oneclip_400_4.jsonl")
    args = parser.parse_args()

    rows = read_jsonl(args.source_manifest)
    matches = [row for row in rows if row.get("stem") == args.stem or Path(row.get("video", "")).stem == args.stem]
    if not matches:
        raise ValueError(f"Stem {args.stem!r} not found in {args.source_manifest}")
    row = dict(matches[0])
    row["start_frame"] = int(args.start_frame)
    row["debug_note"] = "single fixed clip for overfit/capacity diagnosis"
    write_jsonl(args.out_manifest, [row])
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
