#!/usr/bin/env python3
"""Run CG-ICS on FAME episodes with the official implementation.

CG-ICS couples SAM3 with a local Qwen3-VL concept model. The benchmark run used
the fast prompt-cache path (``--bon 1 --num-loops 0 --max-restarts 0``) and the
plain SAM3 backend; both the full tree search and the MedSAM3 backend are
supported.

Install the official code and weights separately:

    git clone https://github.com/Kakarot1103/CG-ICS
    # CG-ICS/Pretrained_models/sam3.pt
    # CG-ICS/Pretrained_models/qwen3-vl-4B-ins

Example:

    python scripts/methods/run_cgics.py \
      --release-root data/FAME_benchmark \
      --out-root outputs/CG-ICS/predictions \
      --cgics-root /path/to/CG-ICS \
      --sam3-checkpoint /path/to/CG-ICS/Pretrained_models/sam3.pt \
      --qwen-model /path/to/CG-ICS/Pretrained_models/qwen3-vl-4B-ins \
      --bon 1 --num-loops 0 --max-restarts 0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from _common import add_common_args, pick_device, run_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CG-ICS runner for FAME episodes.")
    add_common_args(parser)
    parser.add_argument(
        "--cgics-root",
        type=Path,
        required=True,
        help="Path to the official CG-ICS repository.",
    )
    parser.add_argument(
        "--sam3-checkpoint",
        type=Path,
        required=True,
        help="Path to the SAM3 checkpoint (sam3.pt).",
    )
    parser.add_argument(
        "--qwen-model",
        type=Path,
        required=True,
        help="Path to the local Qwen3-VL model directory.",
    )
    parser.add_argument(
        "--qwen-dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--sam3-backend",
        choices=["sam3", "medsam3"],
        default="sam3",
    )
    parser.add_argument(
        "--medsam3-root",
        type=Path,
        default=None,
        help="Path to MedSAM3_src when --sam3-backend medsam3 is used.",
    )
    parser.add_argument(
        "--medsam3-lora",
        type=Path,
        default=None,
        help="MedSAM3 LoRA weights when --sam3-backend medsam3 is used.",
    )
    parser.add_argument("--lora-config", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--bon", type=int, default=5)
    parser.add_argument("--num-loops", type=int, default=3)
    parser.add_argument("--num-expand-per-node", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--expansion-threshold", type=float, default=0.5)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--early-stop-threshold", type=float, default=0.8)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu.")
    return parser.parse_args()


def tensor_iou(pred, target) -> float:
    pred_binary = (pred > 0.5).float().cpu()
    target_binary = target.float()
    while target_binary.ndim > 2:
        target_binary = target_binary.squeeze(0)
    target_binary = (target_binary > 0.5).float().cpu()
    if target_binary.sum() == 0:
        return 1.0 if pred_binary.sum() == 0 else 0.0
    intersection = pred_binary * target_binary
    union = pred_binary + target_binary - intersection
    return float(intersection.sum() / union.sum())


def build_segmenter(args: argparse.Namespace, device):
    from model.sam3 import Sam3Segmenter

    root = args.cgics_root.resolve()
    if args.sam3_backend == "sam3":
        return Sam3Segmenter(
            checkpoint_path=str(args.sam3_checkpoint), device=str(device)
        ).eval()

    if args.medsam3_root is None or args.medsam3_lora is None:
        raise SystemExit("--medsam3-root and --medsam3-lora are required for medsam3.")
    lora_config = args.lora_config or (args.medsam3_root / "configs" / "full_lora_config.yaml")
    sys.path.insert(0, str(args.medsam3_root.resolve()))
    import yaml
    from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
    from sam3.model_builder import build_sam3_image_model

    config = yaml.safe_load(lora_config.read_text(encoding="utf-8"))["lora"]
    base_model = build_sam3_image_model(
        checkpoint_path=str(args.sam3_checkpoint),
        device=str(device),
        eval_mode=True,
        bpe_path=str(root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"),
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
    load_lora_weights(base_model, str(args.medsam3_lora))
    return Sam3Segmenter(
        checkpoint_path=str(args.sam3_checkpoint),
        device=str(device),
        model=base_model,
    ).eval()


def main() -> None:
    args = parse_args()
    root = args.cgics_root.resolve()
    if not (root / "model" / "sam3.py").is_file():
        raise SystemExit(f"CG-ICS code not found under {root}; expected model/sam3.py.")

    old_cwd = os.getcwd()
    os.chdir(root)
    sys.path.insert(0, str(root / "sam3"))
    sys.path.insert(0, str(root))
    try:
        import torch
        import torchvision
        from model.mllm import MLLM
        from model.tree_search import TreeSearcher
        from utils.connected import split_connected_components

        device = pick_device(args.device)
        segmenter = build_segmenter(args, device)
        mllm = MLLM(
            model_name="qwen3-vl-4b",
            seg_type="object",
            local_model_path=str(args.qwen_model),
            device=str(device),
            dtype=args.qwen_dtype,
        )
        searcher = TreeSearcher(
            segmenter=segmenter,
            mllm=mllm,
            args=SimpleNamespace(
                BON=args.bon,
                num_loops=args.num_loops,
                num_expand_per_node=args.num_expand_per_node,
                alpha=args.alpha,
                beta=args.beta,
                expansion_threshold=args.expansion_threshold,
                max_restarts=args.max_restarts,
                early_stop_threshold=args.early_stop_threshold,
            ),
            verbose=False,
        )
        print(
            f"[CG-ICS] backend={args.sam3_backend} device={device} "
            f"fast_cache={args.bon == 1 and args.num_loops == 0 and args.max_restarts == 0}",
            flush=True,
        )

        size = args.image_size
        fast_cache = (
            args.bon == 1
            and args.num_loops == 0
            and args.max_restarts == 0
            and not (args.alpha == 0 and args.beta == 0)
        )

        def image_tensor(path: str | Path) -> "torch.Tensor":
            image = Image.open(path).convert("RGB").resize(
                (size, size), Image.Resampling.BILINEAR
            )
            return torchvision.transforms.functional.to_tensor(image)

        def mask_tensor(path: str | Path) -> "torch.Tensor":
            mask = Image.open(path).convert("L").resize(
                (size, size), Image.Resampling.NEAREST
            )
            return torch.from_numpy((np.asarray(mask) > 0).astype(np.float32))

        def build_prompt_cache(support_tensor, support_mask, support_pil):
            from utils.ref_image_edit import make_ref_image_for_llm

            ref_instances = split_connected_components(support_mask)
            ref_masked_pil, _ = make_ref_image_for_llm(
                support_tensor, ref_instances, crop=True, alpha=0.3
            )
            candidates = mllm.generate(
                mode="initial",
                ref_image=support_pil,
                ref_image_masked=ref_masked_pil,
                num_candidates=args.bon,
            )
            prompt = candidates[0] if candidates else "object"

            segmenter.ref_states = [segmenter.processor.set_image(support_pil)]
            try:
                result = segmenter.infer(segmenter.ref_states[0], prompt=prompt)
                ref_iou = tensor_iou(result["semantic_mask"], support_mask)
            finally:
                segmenter.clear()
            return {"prompt": prompt, "ref_score": ref_iou ** args.alpha}

        def select_cached_prompt(cache):
            result = segmenter.infer(segmenter.query_state, prompt=cache["prompt"])
            query_score = float(result["presence_score"])
            score = cache["ref_score"] * (query_score ** args.beta)
            return cache["prompt"] if score >= 1e-6 else "visual"

        prompt_caches: dict[str, dict] = {}

        def predict_one(support: dict, query: dict) -> np.ndarray:
            support_tensor = image_tensor(support["image"])
            support_mask = mask_tensor(support["mask"])
            support_pil = torchvision.transforms.functional.to_pil_image(support_tensor)
            query_tensor = image_tensor(query["image"])
            query_pil = torchvision.transforms.functional.to_pil_image(query_tensor)
            cat_pil = Image.fromarray(
                np.concatenate(
                    [np.asarray(support_pil), np.asarray(query_pil)], axis=1
                )
            )
            cat_mask = split_connected_components(
                torch.cat([support_mask, torch.zeros_like(support_mask)], dim=1)
            )

            try:
                if fast_cache:
                    key = support["image"]
                    if key not in prompt_caches:
                        prompt_caches[key] = build_prompt_cache(
                            support_tensor, support_mask, support_pil
                        )
                    segmenter.ref_states = None
                    segmenter.query_state = segmenter.processor.set_image(query_pil)
                    segmenter.cat_states = [segmenter.processor.set_image(cat_pil)]
                    prompt = select_cached_prompt(prompt_caches[key])
                else:
                    segmenter.build_state([support_pil], query_pil, [cat_pil])
                    prompt = searcher.select(
                        [support_tensor], [support_mask], query_tensor
                    )

                cat_result = segmenter.infer(segmenter.cat_states[0], mask=cat_mask)
                instances = cat_result["instances"]
                width = instances.shape[-1]
                visual_mask = instances[:, :, :, width // 2:].squeeze(1).to(device)
                if visual_mask.shape[0] > 0:
                    result = segmenter.infer(
                        segmenter.query_state, prompt=prompt, mask=visual_mask
                    )
                else:
                    result = segmenter.infer(segmenter.query_state, prompt=prompt)
                return (
                    result["semantic_mask"].detach().float().cpu().numpy() > 0.5
                )
            finally:
                segmenter.clear()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        run_tasks(args, "CG-ICS", predict_one, reset=prompt_caches.clear)
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
