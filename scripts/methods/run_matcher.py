"""Official Matcher (one-shot, SAM-H + DINOv2-L) over FAME episodes.

Protocol follows FAME: support image + GT support mask -> query mask. The support
mask is used as-is; we do not re-predict it with SAM (that is demo-only behaviour).
Settings mirror the FAME benchmark run: 518 input, points_per_side 16,
sample_range (1, 1), one sampling iteration, one merged mask.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _common import (  # noqa: E402
    add_common_args,
    image_shape,
    pick_device,
    read_mask,
    read_rgb,
    run_tasks,
)

# Matcher's code uses deprecated numpy aliases (np.int / np.float / np.bool).
for _alias, _type in (("int", int), ("float", float), ("bool", bool), ("object", object)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _type)


def build_models(
    matcher_root: Path,
    sam_checkpoint: Path,
    dinov2_checkpoint: Path,
    points_per_side: int,
    device,
):
    sys.path.insert(0, str(matcher_root))
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    from dinov2.models import vision_transformer as vits
    import dinov2.utils.utils as dinov2_utils
    from gradio_demo.Matcher import Matcher

    sam = sam_model_registry["default"](checkpoint=str(sam_checkpoint))
    sam.to(device=device)
    sam.eval()

    dinov2 = vits.__dict__["vit_large"](
        img_size=518,
        patch_size=14,
        init_values=1e-5,
        ffn_layer="mlp",
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )
    dinov2_utils.load_pretrained_weights(dinov2, str(dinov2_checkpoint), "teacher")
    dinov2.eval()
    dinov2.to(device=device)

    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        points_per_batch=64,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        stability_score_offset=1.0,
        sel_stability_score_thresh=0.90,
        sel_pred_iou_thresh=0.85,
        box_nms_thresh=0.65,
        sel_output_layer=3,
        output_layer=0,
        dense_pred=True,
        multimask_output=False,
        sel_multimask_output=True,
    )

    score_filter_cfg = {
        "emd": 0.0,
        "purity": 0.02,
        "coverage": 0.0,
        "score_filter": True,
        "score": 0.33,
        "score_norm": 0.1,
        "topk_scores_threshold": 0.0,
    }
    matcher = Matcher(
        encoder=dinov2,
        generator=generator,
        num_centers=8,
        use_box=False,
        use_points_or_centers=True,
        sample_range=(1, 1),
        max_sample_iterations=1,
        alpha=1.0,
        beta=0.0,
        exp=0.0,
        score_filter_cfg=score_filter_cfg,
        num_merging_mask=1,
        device=device,
    )
    return matcher


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official Matcher on FAME episodes.")
    parser.add_argument(
        "--matcher-root",
        type=Path,
        required=True,
        help="Official Matcher repository.",
    )
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--dinov2-checkpoint", type=Path, required=True)
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    add_common_args(parser)
    args = parser.parse_args()

    device = pick_device(args.device)
    matcher = build_models(
        args.matcher_root,
        args.sam_checkpoint,
        args.dinov2_checkpoint,
        args.points_per_side,
        device,
    )
    transform = transforms.ToTensor()

    def predict_one(support_row, query_row):
        query_shape = image_shape(query_row["image"])
        support = read_rgb(support_row["image"])
        query = read_rgb(query_row["image"])

        support_t = transform(
            Image.fromarray(support).resize((args.img_size, args.img_size), Image.BILINEAR)
        )
        query_t = transform(
            Image.fromarray(query).resize((args.img_size, args.img_size), Image.BILINEAR)
        )
        support_mask = read_mask(support_row["mask"])
        support_mask = (
            np.asarray(
                Image.fromarray(support_mask.astype(np.uint8) * 255).resize(
                    (args.img_size, args.img_size), Image.NEAREST
                )
            )
            > 127
        )

        with torch.no_grad():
            matcher.set_reference(
                support_t[None, None].to(device),
                torch.from_numpy(support_mask).float()[None, None].to(device),
            )
            matcher.set_target(query_t[None].to(device), query_shape)
            prediction, _ = matcher.predict()
            matcher.clear()

        prediction = (
            prediction.float()
            if torch.is_tensor(prediction)
            else torch.from_numpy(np.asarray(prediction)).float()
        )
        if prediction.dim() == 2:
            prediction = prediction[None, None]
        elif prediction.dim() == 3:
            prediction = prediction[None]
        prediction = F.interpolate(
            prediction, query_shape, mode="nearest"
        )[0, 0].cpu().numpy()
        return prediction > 0

    run_tasks(args, "Matcher", predict_one)


if __name__ == "__main__":
    main()
