#!/usr/bin/env python3
"""Run SegGPT on FAME episodes with the official Painter implementation.

SegGPT is installed separately:

    git clone https://github.com/baaivision/Painter
    cd Painter/SegGPT/SegGPT_inference
    wget https://huggingface.co/BAAI/SegGPT/resolve/main/seggpt_vit_large.pth

The support image-mask pair is used as the in-context prompt and the predicted
query half is thresholded into a binary mask.

Example:

    python scripts/methods/run_seggpt.py \
      --release-root data/FAME_benchmark \
      --out-root outputs/SegGPT/predictions \
      --painter-root /path/to/Painter/SegGPT/SegGPT_inference \
      --checkpoint /path/to/seggpt_vit_large.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from _common import add_common_args, pick_device, read_mask, read_rgb, run_tasks

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SegGPT runner for FAME episodes.")
    add_common_args(parser)
    parser.add_argument(
        "--painter-root",
        type=Path,
        required=True,
        help="Path to Painter/SegGPT/SegGPT_inference.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to seggpt_vit_large.pth.",
    )
    parser.add_argument(
        "--arch",
        default="seggpt_vit_large_patch16_input896x448",
        help="SegGPT architecture name in models_seggpt.",
    )
    parser.add_argument(
        "--seg-type",
        choices=["instance", "semantic"],
        default="instance",
        help="SegGPT segmentation embedding type.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=448,
        help="Square input size used by the official inference script.",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.painter_root.resolve()
    if not (root / "models_seggpt.py").is_file():
        raise SystemExit(f"SegGPT code not found under {root}; expected models_seggpt.py.")
    sys.path.insert(0, str(root))

    import torch
    import models_seggpt
    from seggpt_engine import run_one_image

    device = pick_device(args.device)
    model = getattr(models_seggpt, args.arch)()
    model.seg_type = args.seg_type
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device).eval()
    print(f"[SegGPT] loaded {args.arch} on {device}", flush=True)

    size = args.image_size

    def predict_one(support: dict, query: dict) -> np.ndarray:
        prompt_rgb = read_rgb(support["image"])
        prompt_mask = read_mask(support["mask"])
        query_rgb = read_rgb(query["image"])

        prompt_image = Image.fromarray(prompt_rgb).resize((size, size))
        query_image = Image.fromarray(query_rgb).resize((size, size))
        prompt_target = Image.fromarray(
            (prompt_mask.astype(np.uint8) * 255)
        ).convert("RGB").resize((size, size), Image.NEAREST)

        prompt_pixels = np.asarray(prompt_image, dtype=np.float32) / 255.0
        query_pixels = np.asarray(query_image, dtype=np.float32) / 255.0
        target_pixels = np.asarray(prompt_target, dtype=np.float32) / 255.0

        stacked_image = np.concatenate([prompt_pixels, query_pixels], axis=0)
        stacked_target = np.concatenate([target_pixels, target_pixels], axis=0)
        stacked_image = (stacked_image - IMAGENET_MEAN) / IMAGENET_STD
        stacked_target = (stacked_target - IMAGENET_MEAN) / IMAGENET_STD

        torch.manual_seed(2)
        output = run_one_image(
            stacked_image[None], stacked_target[None], model, device
        )

        query_half = output[0, size:, :, :]
        mask = (query_half.mean(dim=-1) > 128).cpu().numpy().astype(np.uint8) * 255
        resized = Image.fromarray(mask).resize(
            (query_rgb.shape[1], query_rgb.shape[0]), Image.NEAREST
        )
        return np.asarray(resized) > 127

    run_tasks(args, "SegGPT", predict_one)


if __name__ == "__main__":
    main()
