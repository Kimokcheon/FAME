#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fame_eval.collect import collect_summary_files, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect FAME summary_all.csv files.")
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()
    rows = collect_summary_files(args.eval_root)
    write_csv(args.out_csv, rows)
    print(f"rows={len(rows)} out={args.out_csv}")


if __name__ == "__main__":
    main()
