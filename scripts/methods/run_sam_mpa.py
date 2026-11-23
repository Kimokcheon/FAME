"""SAM-MPA / MedSAM-MPA over FAME episodes.

The support mask is propagated to the query with a registration, a spatial prompt
is derived from the coarse mask, and the SAM decoder refines it. Among the three
multimask outputs we keep the one that agrees most with the propagated mask.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _common import (  # noqa: E402
    add_common_args,
    pick_device,
    pick_mask,
    run_tasks,
    support_to_query_coarse,
)


def build_predictor(repo: Path, checkpoint: Path, device):
    sys.path.insert(0, str(repo))
    from segment_anything import SamPredictor, sam_model_registry

    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    return SamPredictor(model.to(device).eval())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SAM-MPA or MedSAM-MPA on FAME episodes."
    )
    parser.add_argument("--method", default="SAM-MPA", help="Name used in logs and output paths.")
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="Official SAM or MedSAM repository that provides segment_anything.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    add_common_args(parser)
    args = parser.parse_args()

    device = pick_device(args.device)
    predictor = build_predictor(args.repo, args.checkpoint, device)

    def predict_one(support_row, query_row):
        query, coarse, box, _, _ = support_to_query_coarse(support_row, query_row)
        if box is None:
            return np.zeros(query.shape[:2], dtype=bool)
        predictor.set_image(query)
        masks, scores, _ = predictor.predict(
            box=np.asarray(box, dtype=np.float32)[None],
            multimask_output=True,
        )
        return np.asarray(masks[pick_mask(masks, scores, coarse)] > 0)

    run_tasks(args, args.method, predict_one)


if __name__ == "__main__":
    main()
