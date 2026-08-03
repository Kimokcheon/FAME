#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the released FAME benchmark from Hugging Face.")
    parser.add_argument("--repo-id", default="Kimokcheon/FAME_benchmark")
    parser.add_argument("--local-dir", type=Path, default=Path("data/FAME_benchmark"))
    parser.add_argument("--endpoint", default=None, help="Optional Hugging Face endpoint or mirror.")
    parser.add_argument("--protocol-only", action="store_true", help="Download README, metadata, tools, and protocol archive only.")
    args = parser.parse_args()
    allow_patterns = None
    if args.protocol_only:
        allow_patterns = [
            "README.md",
            "LICENSE",
            "metadata/*",
            "protocols/FAME_benchmark_protocols.tar.gz",
            "tools/*",
            "viewer/*",
            "assets/*",
        ]
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(args.local_dir),
        endpoint=args.endpoint,
        allow_patterns=allow_patterns,
    )
    print(path)


if __name__ == "__main__":
    main()

