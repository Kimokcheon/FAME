#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fame_eval.evaluate import evaluate_prediction_dir
from fame_eval.protocol import ensure_protocols_extracted, list_ood_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all FAME OOD pairs for one method.")
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--setting", default="k10_seed0")
    parser.add_argument("--method", required=True)
    parser.add_argument("--pairs", default="", help="Optional comma-separated OOD pair IDs.")
    parser.add_argument("--min-pred-area-frac", type=float, default=0.0)
    args = parser.parse_args()
    protocol_root = ensure_protocols_extracted(args.release_root)
    pairs = [x.strip() for x in args.pairs.split(",") if x.strip()] if args.pairs else list_ood_pairs(args.release_root, args.setting)
    for pair_id in pairs:
        test_jsonl = protocol_root / "ood_episodes" / args.setting / pair_id / "test.jsonl"
        pred_dir = args.prediction_root / pair_id / "pred_masks"
        if not pred_dir.exists():
            pred_dir = args.prediction_root / pair_id
        out_dir = args.output_root / pair_id
        summary = evaluate_prediction_dir(
            test_jsonl=test_jsonl,
            pred_dir=pred_dir,
            out_dir=out_dir,
            method=args.method,
            task=pair_id,
            release_root=args.release_root,
            support_k=10,
            min_pred_area_frac=args.min_pred_area_frac,
        )
        print(pair_id, summary.get("positive_dice_mean"), summary.get("specificity"))


if __name__ == "__main__":
    main()
