#!/usr/bin/env python3
"""Run MedSAM3 on FAME episodes with the official LoRA weights.

The benchmark run uses the ``medsam3_text`` variant: SAM3 with MedSAM3-v1 LoRA,
prompted by the per-task target text from ``task_info.json``. LoRA training is
performed by the official MedSAM3 procedure before this runner. The paper used
one LoRA per task, produced under
``<root>/<task>/output/lora/last_lora_weights.pt``; pass that root with
``--medsam3-lora-root`` (or a single ``--medsam3-lora`` file for one task).

Install the official code and weights separately:

    git clone https://github.com/Joey-S-Liu/MedSAM3          # LoRA layers
    git clone https://github.com/Kakarot1103/CG-ICS          # SAM3 code / checkpoint

Example:

    python scripts/methods/run_medsam3.py \
      --release-root data/FAME_benchmark \
      --out-root outputs/MedSAM3/predictions \
      --cgics-root /path/to/CG-ICS \
      --sam3-checkpoint /path/to/CG-ICS/Pretrained_models/sam3.pt \
      --medsam3-root /path/to/MedSAM3 \
      --medsam3-lora-root /path/to/oneshot_tasks
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from _common import add_common_args, pick_device, run_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MedSAM3 runner for FAME episodes.")
    add_common_args(parser)
    parser.add_argument(
        "--cgics-root",
        type=Path,
        required=True,
        help="Path to the CG-ICS repository (SAM3 code and checkpoint).",
    )
    parser.add_argument(
        "--sam3-checkpoint",
        type=Path,
        required=True,
        help="Path to the SAM3 checkpoint (sam3.pt).",
    )
    parser.add_argument(
        "--medsam3-root",
        type=Path,
        required=True,
        help="Path to the MedSAM3 repository containing lora_layers.py.",
    )
    parser.add_argument(
        "--medsam3-lora",
        type=Path,
        default=None,
        help="Single LoRA weight file applied to every task.",
    )
    parser.add_argument(
        "--medsam3-lora-root",
        type=Path,
        default=None,
        help=(
            "Per-task LoRA root produced by the official 1-shot training queue: "
            "<root>/<task>/output/lora/last_lora_weights.pt."
        ),
    )
    parser.add_argument(
        "--lora-config",
        type=Path,
        default=None,
        help="Defaults to <medsam3-root>/configs/full_lora_config.yaml.",
    )
    parser.add_argument(
        "--prompt-field",
        default="target_display",
        help="Field read from task_info.json for the target text.",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cgics_root = args.cgics_root.resolve()
    medsam3_root = args.medsam3_root.resolve()
    lora_config = args.lora_config or (medsam3_root / "configs" / "full_lora_config.yaml")
    if not (cgics_root / "model" / "sam3.py").is_file():
        raise SystemExit(f"CG-ICS/SAM3 code not found under {cgics_root}.")
    if not lora_config.is_file():
        raise SystemExit(f"LoRA config not found: {lora_config}")
    if args.medsam3_lora is None and args.medsam3_lora_root is None:
        raise SystemExit("pass --medsam3-lora or --medsam3-lora-root.")

    old_cwd = os.getcwd()
    os.chdir(cgics_root)
    sys.path.insert(0, str(cgics_root / "sam3"))
    sys.path.insert(0, str(cgics_root))
    try:
        import torch
        import yaml
        from model.sam3 import Sam3Segmenter

        # Import the CG-ICS ``sam3`` package before adding the MedSAM3 tree: the
        # LoRA layers come from MedSAM3, but the SAM3 code must stay CG-ICS's.
        sys.path.insert(0, str(medsam3_root))
        from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
        from sam3.model_builder import build_sam3_image_model

        device = pick_device(args.device)
        config = yaml.safe_load(lora_config.read_text(encoding="utf-8"))["lora"]
        base_model = build_sam3_image_model(
            checkpoint_path=str(args.sam3_checkpoint),
            device=str(device),
            eval_mode=True,
            bpe_path=str(cgics_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"),
        )
        lora = LoRAConfig(
            rank=config["rank"],
            alpha=config["alpha"],
            dropout=0.0,
            target_modules=config["target_modules"],
            apply_to_vision_encoder=config["apply_to_vision_encoder"],
            apply_to_text_encoder=config["apply_to_text_encoder"],
            apply_to_geometry_encoder=config["apply_to_geometry_encoder"],
            apply_to_detr_encoder=config["apply_to_detr_encoder"],
            apply_to_detr_decoder=config["apply_to_detr_decoder"],
            apply_to_mask_decoder=config["apply_to_mask_decoder"],
        )
        base_model = apply_lora_to_model(base_model, lora)
        if args.medsam3_lora is not None:
            load_lora_weights(base_model, str(args.medsam3_lora))
        segmenter = Sam3Segmenter(
            checkpoint_path=str(args.sam3_checkpoint),
            device=str(device),
            model=base_model,
        ).eval()
        print(
            f"[MedSAM3] lora={args.medsam3_lora or args.medsam3_lora_root} device={device}",
            flush=True,
        )
        active_lora = {"task": None}

        def apply_task_lora(task: str) -> None:
            if args.medsam3_lora_root is None or active_lora["task"] == task:
                return
            weights = (
                args.medsam3_lora_root / task / "output" / "lora" / "last_lora_weights.pt"
            )
            if not weights.is_file():
                raise SystemExit(f"missing per-task LoRA weights: {weights}")
            load_lora_weights(base_model, str(weights))
            active_lora["task"] = task

        def task_prompt(task: str) -> str:
            info_path = (
                args.release_root / "FAME_benchmark" / "tasks" / task / "task_info.json"
            )
            info = json.loads(info_path.read_text(encoding="utf-8"))
            prompt = info.get(args.prompt_field) or info.get("target")
            if not prompt:
                dataset, _, name = task.partition("__")
                prompt = (name or task).replace("_", " ")
            return prompt

        def predict_one(task: str, support: dict, query: dict) -> np.ndarray:
            del support
            apply_task_lora(task)
            query_pil = (
                Image.open(query["image"])
                .convert("RGB")
                .resize((args.image_size, args.image_size), Image.Resampling.BILINEAR)
            )
            segmenter.ref_states = None
            segmenter.query_state = segmenter.processor.set_image(query_pil)
            segmenter.cat_states = []
            try:
                result = segmenter.infer(
                    segmenter.query_state, prompt=task_prompt(task)
                )
                return (
                    result["semantic_mask"].detach().float().cpu().numpy() > 0.5
                )
            finally:
                segmenter.clear()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        run_tasks(args, "MedSAM3", predict_one, task_aware=True)
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
