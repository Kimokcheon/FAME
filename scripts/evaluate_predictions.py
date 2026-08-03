#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fame_eval.evaluate import evaluate_prediction_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one FAME prediction directory.")
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--release-root", default=None)
    parser.add_argument("--support-k", type=int, default=0)
    parser.add_argument("--min-pred-area-frac", type=float, default=0.0)
    args = parser.parse_args()
    summary = evaluate_prediction_dir(
        test_jsonl=args.test_jsonl,
        pred_dir=args.pred_dir,
        out_dir=args.out_dir,
        method=args.method,
        task=args.task,
        release_root=args.release_root,
        support_k=args.support_k,
        min_pred_area_frac=args.min_pred_area_frac,
    )
    print(summary)


if __name__ == "__main__":
    main()
