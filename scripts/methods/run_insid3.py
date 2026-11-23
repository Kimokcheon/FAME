"""Official INSID3 (training-free DINOv3 correspondence) over FAME episodes."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _common import add_common_args, pick_device, run_tasks  # noqa: E402

WEIGHTS = {
    "small": "pretrain/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "base": "pretrain/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "large": "pretrain/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official INSID3 on FAME episodes.")
    parser.add_argument("--insid3-root", type=Path, required=True)
    parser.add_argument(
        "--dinov3-root",
        type=Path,
        default=None,
        help=(
            "Local DINOv3 repository used by INSID3's torch.hub loader. "
            "Defaults to INSID3's own DINOV3_ROOT environment variable."
        ),
    )
    parser.add_argument("--model-size", choices=sorted(WEIGHTS), default="small")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--svd-components", type=int, default=500)
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--merge-threshold", type=float, default=0.2)
    parser.add_argument("--crf", action="store_true", help="Enable the optional CRF refiner.")
    parser.add_argument("--crf-size", type=int, default=640)
    parser.add_argument("--device", default="auto")
    add_common_args(parser)
    args = parser.parse_args()

    root = args.insid3_root.resolve()
    if not (root / "models" / "__init__.py").is_file():
        raise SystemExit(f"INSID3 repo not found under {root}.")
    weights = root / WEIGHTS[args.model_size]
    if not weights.is_file():
        raise SystemExit(
            f"Missing DINOv3 weights: {weights}\n"
            "DINOv3 checkpoints are gated by Meta/Hugging Face; place them under "
            "<insid3-root>/pretrain/ first."
        )
    if args.dinov3_root is not None:
        os.environ["DINOV3_ROOT"] = str(args.dinov3_root.resolve())

    old_cwd = os.getcwd()
    os.chdir(root)
    sys.path.insert(0, str(root))
    try:
        from models import build_insid3

        device = pick_device(args.device)
        model = build_insid3(
            model_size=args.model_size,
            image_size=args.image_size,
            svd_components=args.svd_components,
            tau=args.tau,
            merge_threshold=args.merge_threshold,
            mask_refiner="crf" if args.crf else "bilinear",
            crf_size=args.crf_size,
            resize_to_orig_size=True,
            device=str(device),
        ).eval()

        def predict_one(support_row, query_row):
            model.set_reference(str(support_row["image"]), str(support_row["mask"]))
            model.set_target(str(query_row["image"]))
            prediction = model.segment()
            return prediction.detach().cpu().numpy()

        run_tasks(args, "INSID3", predict_one)
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
