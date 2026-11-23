"""Official GF-SAM (NeurIPS 2024, training-free) over FAME episodes."""
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


def build_model(
    gfsam_root: Path,
    sam_repo: Path,
    sam_checkpoint: Path,
    dinov2_checkpoint: Path,
    img_size: int,
    device,
):
    sys.path.insert(0, str(sam_repo))
    sys.path.insert(0, str(gfsam_root))
    from matcher.GFSAM import build_model as build_gfsam

    model_args = argparse.Namespace(
        img_size=img_size,
        dinov2_size="vit_large",
        sam_size="vit_h",
        dinov2_weights=str(dinov2_checkpoint),
        sam_weights=str(sam_checkpoint),
        device=device,
    )
    return build_gfsam(model_args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official GF-SAM on FAME episodes.")
    parser.add_argument("--gfsam-root", type=Path, required=True)
    parser.add_argument(
        "--sam-repo",
        type=Path,
        required=True,
        help="Repository that provides the segment_anything package (e.g. Matcher).",
    )
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--dinov2-checkpoint", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    add_common_args(parser)
    args = parser.parse_args()

    device = pick_device(args.device)
    model = build_model(
        args.gfsam_root,
        args.sam_repo,
        args.sam_checkpoint,
        args.dinov2_checkpoint,
        args.img_size,
        device,
    )
    transform = transforms.Compose(
        [
            transforms.Resize(size=(args.img_size, args.img_size)),
            transforms.ToTensor(),
        ]
    )

    def predict_one(support_row, query_row):
        query_shape = image_shape(query_row["image"])
        support = Image.fromarray(read_rgb(support_row["image"]))
        query = Image.fromarray(read_rgb(query_row["image"]))

        support_t = transform(support)
        query_t = transform(query)
        support_mask = read_mask(support_row["mask"])
        support_mask = (
            np.asarray(
                Image.fromarray(support_mask.astype(np.uint8) * 255).resize(
                    (args.img_size, args.img_size), Image.NEAREST
                )
            )
            > 127
        )
        support_mask_t = torch.from_numpy(support_mask).float()
        support_mask_t = F.interpolate(
            support_mask_t[None, None], support_t.shape[-2:], mode="nearest"
        )

        with torch.no_grad():
            model.clear()
            model.set_reference(
                support_t[None, None].to(device), support_mask_t.to(device)
            )
            model.set_target(query_t[None].to(device))
            prediction, _ = model.predict()

        prediction = prediction.float()
        if prediction.dim() == 2:
            prediction = prediction[None, None]
        elif prediction.dim() == 3:
            prediction = prediction[None]
        prediction = F.interpolate(
            prediction, query_shape, mode="nearest"
        )[0, 0].cpu().numpy()
        return prediction > 0

    run_tasks(args, "GF-SAM", predict_one)


if __name__ == "__main__":
    main()
