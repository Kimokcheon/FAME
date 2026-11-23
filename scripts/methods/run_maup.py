#!/usr/bin/env python3
"""Run MAUP on FAME episodes with the official few-shot implementation.

MAUP is training-free: a DINOv2/SAM pipeline (``models.fewshot.FewShotSeg``)
takes the support pair and predicts the query at 256x256. The benchmark run is
restricted to the one-shot setting.

Install the official code and weights separately:

    git clone https://github.com/YazhouZhu19/MAUP
    # checkpoints/sam_vit_h_4b8939.pth and dinov2_weights/ must exist

Example:

    python scripts/methods/run_maup.py \
      --release-root data/FAME_benchmark \
      --out-root outputs/MAUP/predictions \
      --maup-root /path/to/MAUP
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
    parser = argparse.ArgumentParser(description="MAUP runner for FAME episodes.")
    add_common_args(parser)
    parser.add_argument(
        "--maup-root",
        type=Path,
        required=True,
        help="Path to the official MAUP repository.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.maup_root.resolve()
    if not (root / "models" / "fewshot.py").is_file():
        raise SystemExit(f"MAUP code not found under {root}; expected models/fewshot.py.")

    sys.path.insert(0, str(root / "segment-anything"))
    sys.path.insert(0, str(root))
    old_cwd = os.getcwd()
    os.chdir(root)
    try:
        import torch
        from models.fewshot import FewShotSeg

        device = pick_device(args.device)
        model = FewShotSeg().to(device).eval()
        print(f"[MAUP] loaded FewShotSeg on {device}", flush=True)
        size = args.image_size

        def image_tensor(path: str | Path) -> "torch.Tensor":
            image = Image.open(path).convert("RGB").resize(
                (size, size), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            return ((tensor - mean) / std).contiguous()

        def mask_tensor(path: str | Path) -> "torch.Tensor":
            mask = Image.open(path).convert("L").resize(
                (size, size), Image.Resampling.NEAREST
            )
            return torch.from_numpy(
                (np.asarray(mask) > 0).astype(np.float32)
            ).contiguous()

        def predict_one(support: dict, query: dict) -> np.ndarray:
            support_image = image_tensor(support["image"]).unsqueeze(0).to(device)
            support_mask = mask_tensor(support["mask"]).unsqueeze(0).to(device)
            query_image = image_tensor(query["image"]).unsqueeze(0).to(device)
            with torch.inference_mode():
                logits = model(
                    [[support_image]], [[support_mask]], [query_image], None, train=False
                )
            return logits.argmax(dim=1)[0].detach().cpu().numpy().astype(bool)

        run_tasks(args, "MAUP", predict_one)
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
