#!/usr/bin/env python3
"""Run MedPro on FAME episodes with the official SAM2 pipeline.

MedPro propagates the support mask with SAM2, repeats the propagation on the
support-masked ROI (optionally after histogram matching), and lets a DINOv2
selector choose the better of the two masks. The benchmark run is one-shot.

Install the official code and weights separately:

    /path/to/MedPro/
      notebooks/          # SAM2, dino_select.py, build_fsmedsam2.py
      SAM2/, sam2-main/
      weights/sam2_hiera_small.pt
    /path/to/dinov2_weights/

Example:

    python scripts/methods/run_medpro.py \
      --release-root data/FAME_benchmark \
      --out-root outputs/MedPro/predictions \
      --medpro-root /path/to/MedPro \
      --dino-root /path/to/dinov2_weights
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from _common import add_common_args, pick_device, run_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MedPro runner for FAME episodes.")
    add_common_args(parser)
    parser.add_argument(
        "--medpro-root",
        type=Path,
        required=True,
        help="Path to the official MedPro repository.",
    )
    parser.add_argument(
        "--dino-root",
        type=Path,
        required=True,
        help="Directory containing the DINOv2 weights used by MedPro.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--padding", type=int, default=50)
    parser.add_argument("--no-histogram-match", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu.")
    return parser.parse_args()


def roi_bbox(mask: np.ndarray, padding: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    height, width = mask.shape
    if len(ys) == 0:
        return 0, height, 0, width
    return (
        max(0, int(ys.min()) - padding),
        min(height, int(ys.max()) + padding + 1),
        max(0, int(xs.min()) - padding),
        min(width, int(xs.max()) + padding + 1),
    )


def propagate(predictor, support: np.ndarray, query: np.ndarray, mask: np.ndarray) -> np.ndarray:
    state = predictor.init_state_by_np_data(images_np=np.stack((support, query)))
    predictor.reset_state(state)
    predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=mask)
    logits = None
    for frame_idx, _, out_mask_logits in predictor.propagate_in_video(
        state, start_frame_idx=1
    ):
        if frame_idx == 1:
            logits = out_mask_logits[0, 0].float().cpu().numpy()
    if logits is None:
        raise RuntimeError("SAM2 did not return the query frame")
    return logits > 0.0


def predict_medpro(
    predictor,
    selector,
    support: np.ndarray,
    support_mask: np.ndarray,
    query: np.ndarray,
    padding: int,
    histogram_match: bool,
) -> np.ndarray:
    from skimage.exposure import match_histograms

    candidates = [propagate(predictor, support, query, support_mask)]
    y1, y2, x1, x2 = roi_bbox(support_mask, padding)
    roi_support = support[:, y1:y2, x1:x2]
    roi_query = query[:, y1:y2, x1:x2]
    if histogram_match:
        matched = match_histograms(
            roi_support.transpose(1, 2, 0),
            roi_query.transpose(1, 2, 0),
            channel_axis=-1,
        )
        roi_support = matched.transpose(2, 0, 1).astype(np.float32)
    roi_pred = propagate(
        predictor, roi_support, roi_query, support_mask[y1:y2, x1:x2]
    )
    local_full = np.zeros_like(support_mask, dtype=bool)
    local_full[y1:y2, x1:x2] = roi_pred
    candidates.append(local_full)

    import torch

    query_tensor = torch.from_numpy(query).to(selector.device)[None, None]
    return selector.select_best_mask(query_tensor, candidates).astype(bool)


def main() -> None:
    args = parse_args()
    root = args.medpro_root.resolve()
    notebooks = root / "notebooks"
    if not (notebooks / "dino_select.py").is_file():
        raise SystemExit(f"MedPro code not found under {notebooks}; expected dino_select.py.")

    sys.path.insert(0, str(root))
    sys.path.insert(0, str(notebooks))
    old_cwd = os.getcwd()
    os.chdir(notebooks)
    try:
        import torch
        import SAM2

        sys.modules.setdefault("sam2", SAM2)
        from SAM2.build_fsmedsam2 import build_fsmedsam2_video_predictor
        from dino_select import DinoSelector

        device = pick_device(args.device)
        if device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        predictor = build_fsmedsam2_video_predictor(
            "sam2_hiera_s.yaml",
            str(root / "weights" / "sam2_hiera_small.pt"),
            device=device,
        )
        selector = DinoSelector(model_path=str(args.dino_root), device=device)
        print(f"[MedPro] SAM2 + DINO selector on {device}", flush=True)

        size = args.image_size

        def image_array(path: str | Path) -> np.ndarray:
            image = Image.open(path).convert("RGB").resize(
                (size, size), Image.Resampling.BILINEAR
            )
            return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0

        def mask_array(path: str | Path) -> np.ndarray:
            mask = Image.open(path).convert("L").resize(
                (size, size), Image.Resampling.NEAREST
            )
            return (np.asarray(mask) > 0).astype(np.uint8)

        reference: dict = {"key": None}

        def prepare_reference(support: dict) -> None:
            key = support["image"]
            if reference["key"] == key:
                return
            support_image = image_array(support["image"])
            support_mask = mask_array(support["mask"])
            selector.set_reference(
                torch.from_numpy(support_image).to(device)[None, None],
                torch.from_numpy(support_mask).to(device)[None, None],
            )
            reference.update(
                key=key, image=support_image, mask=support_mask
            )

        def predict_one(support: dict, query: dict) -> np.ndarray:
            prepare_reference(support)
            query_image = image_array(query["image"])
            return predict_medpro(
                predictor,
                selector,
                reference["image"],
                reference["mask"],
                query_image,
                args.padding,
                not args.no_histogram_match,
            )

        run_tasks(
            args,
            "MedPro",
            predict_one,
            reset=lambda: reference.__setitem__("key", None),
        )
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
