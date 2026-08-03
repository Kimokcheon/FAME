#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def _bytes(value):
    if isinstance(value, dict):
        return value.get("bytes") or b""
    if hasattr(value, "as_py"):
        return _bytes(value.as_py())
    return b""


def write_blob(root: Path, rel_path: str, blob: bytes, overwrite: bool) -> bool:
    out = root / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not overwrite:
        return False
    out.write_bytes(blob)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract FAME image/mask pixels from Parquet shards.")
    parser.add_argument("--root", type=Path, required=True, help="Downloaded FAME dataset repository root.")
    parser.add_argument("--out-root", type=Path, default=None, help="Output root. Defaults to --root.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    out_root = (args.out_root or root).resolve()
    shards = sorted((root / "data_shards").glob("fame_image_mask_pairs-*.parquet"))
    if not shards:
        raise SystemExit(f"No Parquet shards found under {root / 'data_shards'}")
    rows = written = skipped = 0
    for shard in shards:
        table = pq.read_table(shard, columns=["image", "mask", "image_path", "mask_path"])
        for row in table.to_pylist():
            rows += 1
            if write_blob(out_root, row["image_path"], _bytes(row["image"]), args.overwrite):
                written += 1
            else:
                skipped += 1
            if write_blob(out_root, row["mask_path"], _bytes(row["mask"]), args.overwrite):
                written += 1
            else:
                skipped += 1
    print(f"rows={rows} written_files={written} skipped_existing={skipped} out_root={out_root}")


if __name__ == "__main__":
    main()

