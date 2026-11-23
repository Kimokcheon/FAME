#!/usr/bin/env python3
"""Run CG-ICS on FAME episodes with the official implementation.

CG-ICS combines SAM3 with a Qwen3-VL concept server, so the vLLM-compatible
OpenAI endpoint must be running before this script starts:

    bash scripts/llm_server.sh          # inside the CG-ICS repository

The official code and the SAM3 checkpoint are installed separately:

    git clone https://github.com/Kakarot1103/CG-ICS
    # place sam3.pt under CG-ICS/Pretrained_models/sam3.pt

Example:

    python scripts/methods/run_cgics.py \
      --release-root data/FAME_benchmark \
      --out-root outputs/CG-ICS/predictions \
      --cgics-root /path/to/CG-ICS \
      --sam3-checkpoint /path/to/CG-ICS/Pretrained_models/sam3.pt
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from _common import add_common_args, pick_device, read_mask, read_rgb, run_tasks


@dataclass
class SearchArgs:
    BON: int
    num_loops: int
    num_expand_per_node: int
    alpha: float
    beta: float
    expansion_threshold: float
    max_restarts: int
    early_stop_threshold: float


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
        "--llm-base-url",
        default="http://localhost:22002/v1",
        help="OpenAI-compatible Qwen3-VL endpoint.",
    )
    parser.add_argument("--llm-model", default="qwen3-vl-4b")
    parser.add_argument("--llm-api-key", default="EMPTY")
    parser.add_argument(
        "--seg-type",
        choices=["object", "part"],
        default="object",
        help="Concept prompt type used by the MLLM.",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--bon", type=int, default=5, help="Initial concept candidates.")
    parser.add_argument("--num-loops", type=int, default=3, help="Maximum tree depth.")
    parser.add_argument("--num-expand-per-node", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--expansion-threshold", type=float, default=0.5)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--early-stop-threshold", type=float, default=0.8)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.cgics_root.resolve()
    if not (root / "model" / "sam3.py").is_file():
        raise SystemExit(f"CG-ICS code not found under {root}; expected model/sam3.py.")
    sys.path.insert(0, str(root))

    import torch
    import torch.nn.functional as F
    import torchvision
    from model.mllm import MLLM
    from model.sam3 import Sam3Segmenter
    from model.tree_search import TreeSearcher
    from utils.connected import split_connected_components

    device = pick_device(args.device)
    segmenter = Sam3Segmenter(
        checkpoint_path=str(args.sam3_checkpoint), device=str(device)
    )
    segmenter.eval()
    mllm = MLLM(
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
        model_name=args.llm_model,
        seg_type=args.seg_type,
    )
    search_args = SearchArgs(
        BON=args.bon,
        num_loops=args.num_loops,
        num_expand_per_node=args.num_expand_per_node,
        alpha=args.alpha,
        beta=args.beta,
        expansion_threshold=args.expansion_threshold,
        max_restarts=args.max_restarts,
        early_stop_threshold=args.early_stop_threshold,
    )
    searcher = TreeSearcher(segmenter=segmenter, mllm=mllm, args=search_args)
    print(f"[CG-ICS] SAM3 on {device}, concept server {args.llm_base_url}", flush=True)

    size = args.image_size

    def prepare(rgb: np.ndarray, mask: np.ndarray | None = None):
        image = Image.fromarray(rgb).convert("RGB").resize((size, size), Image.BILINEAR)
        tensor = torchvision.transforms.functional.to_tensor(image)
        if mask is None:
            return image, tensor, None
        mask_tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
        mask_tensor = F.interpolate(mask_tensor, size=(size, size), mode="nearest")[0, 0]
        return image, tensor, mask_tensor

    def predict_one(support: dict, query: dict) -> np.ndarray:
        query_rgb = read_rgb(query["image"])
        ref_pil, ref_tensor, ref_mask = prepare(
            read_rgb(support["image"]), read_mask(support["mask"])
        )
        query_pil, query_tensor, _ = prepare(query_rgb)

        cat_pil = Image.fromarray(
            np.concatenate([np.asarray(ref_pil), np.asarray(query_pil)], axis=1)
        )
        segmenter.build_state([ref_pil], query_pil, [cat_pil])

        prompt = searcher.select([ref_tensor], [ref_mask], query_tensor)

        cat_mask = torch.cat([ref_mask, torch.zeros_like(ref_mask)], dim=1)
        cat_instances = split_connected_components(cat_mask)
        query_instances = None
        if cat_instances.numel() > 0:
            cat_result = segmenter.infer(
                segmenter.cat_states[0], mask=cat_instances
            )
            instances = cat_result["instances"]
            width = instances.shape[-1]
            query_instances = instances[:, :, :, width // 2:].squeeze(1)

        result = segmenter.infer(
            segmenter.query_state,
            prompt=prompt,
            mask=query_instances
            if query_instances is not None and query_instances.numel() > 0
            else None,
        )
        prediction = result["semantic_mask"].detach().float().cpu().numpy()
        segmenter.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mask = (prediction > 0.5).astype(np.uint8) * 255
        resized = Image.fromarray(mask).resize(
            (query_rgb.shape[1], query_rgb.shape[0]), Image.NEAREST
        )
        return np.asarray(resized) > 127

    run_tasks(args, "CG-ICS", predict_one)


if __name__ == "__main__":
    main()
