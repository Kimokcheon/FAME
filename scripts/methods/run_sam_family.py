"""Box-prompted SAM-family baselines over FAME episodes.

The support mask is warped to the query with a registration, a box is derived
from the warped mask, and the SAM-family predictor segments the query inside
that box. No query labels are used.

Supported methods:
    sam        SAM ViT-B
    medsam     MedSAM ViT-B
    mobilesam  MobileSAM (TinyViT)
    sam2       SAM 2.1 Hiera-T
    medsam2    MedSAM2 Hiera-T
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


def build_predictor(method: str, args: argparse.Namespace, device):
    if method in ("sam", "medsam"):
        if args.sam_repo is None:
            raise SystemExit(f"--sam-repo is required for {method}")
        sys.path.insert(0, str(args.sam_repo))
        from segment_anything import SamPredictor, sam_model_registry

        model = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint))
        return SamPredictor(model.to(device).eval())
    if method == "mobilesam":
        if args.mobilesam_root is None:
            raise SystemExit("--mobilesam-root is required for mobilesam")
        sys.path.insert(0, str(args.mobilesam_root))
        from mobile_sam import SamPredictor, sam_model_registry

        model = sam_model_registry["vit_t"](checkpoint=str(args.checkpoint))
        return SamPredictor(model.to(device).eval())
    if args.sam2_repo is None:
        raise SystemExit(f"--sam2-repo is required for {method}")
    sys.path.insert(0, str(args.sam2_repo))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(
        args.sam2_config,
        str(args.checkpoint),
        device=str(device),
        mode="eval",
    )
    return SAM2ImagePredictor(model)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a box-prompted SAM-family baseline on FAME episodes."
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["sam", "medsam", "mobilesam", "sam2", "medsam2"],
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sam-repo", type=Path, help="Official SAM or MedSAM repository.")
    parser.add_argument("--sam2-repo", type=Path, help="Official SAM 2 repository.")
    parser.add_argument("--mobilesam-root", type=Path, help="Official MobileSAM repository.")
    parser.add_argument(
        "--sam2-config",
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
        help="SAM 2 model config passed to build_sam2.",
    )
    parser.add_argument("--device", default="auto")
    add_common_args(parser)
    args = parser.parse_args()

    device = pick_device(args.device)
    predictor = build_predictor(args.method, args, device)

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
