#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fame_eval.evaluate import evaluate_prediction_dir
from fame_eval.protocol import ensure_protocols_extracted, list_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all IID tasks for one method.")
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--setting", default="k10_seed0", help="Use zero_shot or k10_seed0.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--tasks", default="", help="Optional comma-separated task list.")
    parser.add_argument("--min-pred-area-frac", type=float, default=0.0)
    args = parser.parse_args()
    protocol_root = ensure_protocols_extracted(args.release_root)
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()] if args.tasks else list_tasks(args.release_root)
    for task in tasks:
        if args.setting == "zero_shot":
            test_jsonl = protocol_root / "episodes" / "zero_shot" / task / "test.jsonl"
            support_k = 0
        else:
            test_jsonl = protocol_root / "episodes" / args.setting / task / "test.jsonl"
            support_k = 10
        pred_dir = args.prediction_root / task / "pred_masks"
        if not pred_dir.exists():
            pred_dir = args.prediction_root / task
        out_dir = args.output_root / task
        summary = evaluate_prediction_dir(
            test_jsonl=test_jsonl,
            pred_dir=pred_dir,
            out_dir=out_dir,
            method=args.method,
            task=task,
            release_root=args.release_root,
            support_k=support_k,
            min_pred_area_frac=args.min_pred_area_frac,
        )
        print(task, summary.get("positive_dice_mean"), summary.get("specificity"))


if __name__ == "__main__":
    main()
