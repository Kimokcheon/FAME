#!/usr/bin/env python3
"""Run SegGPT on FAME episodes with the Hugging Face implementation.

This mirrors the benchmark run: ``BAAI/seggpt-vit-large`` is used through
``SegGptImageProcessor`` / ``SegGptForImageSegmentation``, the support pair is
the in-context prompt, and the query prediction is thresholded at label 1.

Example:

    python scripts/methods/run_seggpt.py \
      --release-root data/FAME_benchmark \
      --out-root outputs/SegGPT/predictions
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image

from _common import add_common_args, pick_device, run_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SegGPT runner for FAME episodes.")
    add_common_args(parser)
    parser.add_argument(
        "--model-id",
        default="BAAI/seggpt-vit-large",
        help="Hugging Face model id or local directory.",
    )
    parser.add_argument("--amp", action="store_true", help="Use FP16 autocast on CUDA.")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import SegGptForImageSegmentation, SegGptImageProcessor

    device = pick_device(args.device)
    processor = SegGptImageProcessor.from_pretrained(args.model_id)
    model = (
        SegGptForImageSegmentation.from_pretrained(args.model_id)
        .to(device)
        .eval()
    )
    print(f"[SegGPT] loaded {args.model_id} on {device}", flush=True)

    def predict_one(support: dict, query: dict) -> np.ndarray:
        query_image = Image.open(query["image"]).convert("RGB")
        support_image = Image.open(support["image"]).convert("RGB")
        support_mask = np.asarray(
            Image.open(support["mask"]).convert("L"), dtype=np.uint8
        )
        support_mask = Image.fromarray((support_mask > 0).astype(np.uint8), mode="L")

        inputs = processor(
            images=[query_image],
            prompt_images=[support_image],
            prompt_masks=[support_mask],
            return_tensors="pt",
            num_labels=1,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            if args.amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

        labels = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[query_image.size[::-1]],
            num_labels=1,
        )
        return (labels[0].detach().cpu().numpy() == 1).astype(bool)

    run_tasks(args, "SegGPT", predict_one)


if __name__ == "__main__":
    main()
