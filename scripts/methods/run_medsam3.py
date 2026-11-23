#!/usr/bin/env python3
"""Run MedSAM3 on FAME episodes with the official LoRA checkpoints.

MedSAM3 is a text-prompted SAM3 model. The per-task lesion concept is taken
from ``--concept-map`` when provided, otherwise it is derived from the FAME task
name (for example ``ISIC2018__ISIC2018_skin_lesion`` becomes ``skin lesion``).
Use ``--text-prompt`` to force one concept for every task.

Install the official code and LoRA weights separately:

    git clone https://github.com/Joey-S-Liu/MedSAM3
    # base SAM3 weights and the MedSAM3_v1 LoRA weights

Example:

    python scripts/methods/run_medsam3.py \
      --release-root data/FAME_benchmark \
      --out-root outputs/MedSAM3/predictions \
      --medsam3-root /path/to/MedSAM3 \
      --base-model /path/to/sam3 \
      --lora-weights /path/to/MedSAM3_v1 \
      --lora-config /path/to/MedSAM3/configs/full_lora_config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from _common import add_common_args, pick_device, run_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MedSAM3 runner for FAME episodes.")
    add_common_args(parser)
    parser.add_argument(
        "--medsam3-root",
        type=Path,
        required=True,
        help="Path to the official MedSAM3 repository.",
    )
    parser.add_argument(
        "--base-model",
        default="facebook/sam3",
        help="Base SAM3 model name or local directory.",
    )
    parser.add_argument(
        "--lora-weights",
        default="",
        help="MedSAM3 LoRA weights (directory or file); empty runs the base SAM3 model.",
    )
    parser.add_argument(
        "--lora-config",
        default="",
        help="Optional LoRA YAML config used when applying the adapter.",
    )
    parser.add_argument(
        "--text-prompt",
        default="",
        help="Force one text concept for every task.",
    )
    parser.add_argument(
        "--concept-map",
        type=Path,
        default=None,
        help="Optional JSON mapping task name to text concept.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu.")
    return parser.parse_args()


def task_concept(task: str) -> str:
    dataset, _, name = task.partition("__")
    if not name:
        return task.replace("_", " ")
    prefix = f"{dataset}_"
    if name.startswith(prefix):
        name = name[len(prefix):]
    return name.replace("_", " ")


def main() -> None:
    args = parse_args()
    root = args.medsam3_root.resolve()
    if not (root / "lora_layers.py").is_file():
        raise SystemExit(f"MedSAM3 code not found under {root}; expected lora_layers.py.")
    sys.path.insert(0, str(root))

    import torch
    import torch.nn.functional as F
    from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
    from transformers import Sam3Model, Sam3Processor

    device = pick_device(args.device)
    model = Sam3Model.from_pretrained(args.base_model)
    processor = Sam3Processor.from_pretrained(args.base_model)

    if args.lora_weights:
        config = LoRAConfig()
        if args.lora_config:
            import yaml

            with open(args.lora_config) as handle:
                config = LoRAConfig(**yaml.safe_load(handle)["lora"])
        model = apply_lora_to_model(model, config)
        load_lora_weights(model, args.lora_weights)
    model.to(device).eval()
    print(f"[MedSAM3] base={args.base_model} device={device}", flush=True)

    concept_map = {}
    if args.concept_map:
        concept_map = json.loads(args.concept_map.read_text())

    def concept_for(task: str) -> str:
        if args.text_prompt:
            return args.text_prompt
        return concept_map.get(task, task_concept(task))

    def field(outputs, name):
        if isinstance(outputs, dict):
            return outputs.get(name)
        return getattr(outputs, name, None)

    def predict_one(task: str, support: dict, query: dict) -> np.ndarray:
        image = Image.open(query["image"]).convert("RGB")
        height, width = image.size[1], image.size[0]
        inputs = processor(
            images=image, text=concept_for(task), return_tensors="pt"
        )
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        with torch.no_grad():
            outputs = model(**inputs)

        pred_masks = field(outputs, "pred_masks")
        if pred_masks is None or pred_masks.shape[1] == 0:
            return np.zeros((height, width), dtype=bool)
        pred_masks = pred_masks[0]

        pred_logits = field(outputs, "pred_logits")
        if pred_logits is not None:
            scores = pred_logits[0].sigmoid().max(dim=-1).values
            pred_masks = pred_masks[scores > args.threshold]
        if pred_masks.shape[0] == 0:
            return np.zeros((height, width), dtype=bool)

        mask = (pred_masks.sigmoid() > args.mask_threshold).any(dim=0).float()
        mask = F.interpolate(
            mask[None, None], size=(height, width), mode="bilinear", align_corners=False
        )[0, 0]
        return (mask > 0.5).cpu().numpy()

    run_tasks(args, "MedSAM3", predict_one, task_aware=True)


if __name__ == "__main__":
    main()
