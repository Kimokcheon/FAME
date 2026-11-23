"""Official SANSA (SAM2 + AdaptFormer, universal few-shot segmenter) over FAME episodes."""
from __future__ import annotations

import argparse
import os
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

from _common import add_common_args, image_shape, pick_device, read_mask, read_rgb, run_tasks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official SANSA on FAME episodes.")
    parser.add_argument("--sansa-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Defaults to <sansa-root>/pretrain/sansa_universal.pth.",
    )
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--sam2-version", default="large")
    parser.add_argument("--channel-factor", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    add_common_args(parser)
    args = parser.parse_args()

    args.sansa_root = args.sansa_root.resolve()
    args.checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else args.sansa_root / "pretrain" / "sansa_universal.pth"
    )
    args.release_root = args.release_root.resolve()
    args.out_root = args.out_root.resolve()
    if not args.checkpoint.exists():
        raise SystemExit(f"Missing SANSA checkpoint: {args.checkpoint}")

    # SANSA's SAM2 path config is resolved relative to the repository root.
    os.chdir(args.sansa_root)
    sys.path.insert(0, str(args.sansa_root))
    from models.sansa.sansa import build_sansa
    from util.promptable_utils import build_prompt_dict

    device = pick_device(args.device)
    model = build_sansa(
        sam2_version=args.sam2_version,
        adaptformer_stages=[2, 3],
        channel_factor=args.channel_factor,
        device=str(device),
    ).to(device)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    print(
        f"[SANSA] loaded | missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(size=(args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    def predict_one(support_row, query_row):
        query_shape = image_shape(query_row["image"])
        support = Image.fromarray(read_rgb(support_row["image"]))
        query = Image.fromarray(read_rgb(query_row["image"]))
        support_mask = read_mask(support_row["mask"])

        support_t = transform(support)
        query_t = transform(query)
        support_mask_t = torch.from_numpy(support_mask).float()[None, None]
        support_mask_t = F.interpolate(
            support_mask_t, (args.img_size, args.img_size), mode="nearest"
        )

        images = torch.stack([support_t, query_t], dim=0).unsqueeze(0).to(device)
        prompt = build_prompt_dict(
            support_mask_t.to(device), "mask", n_shots=1, train_mode=False,
            device=device,
        )
        with torch.no_grad():
            outputs = model(images, prompt)
            probability = outputs["pred_masks"][-1].sigmoid()

        probability = F.interpolate(
            probability[None, None].float(),
            query_shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return probability.cpu().numpy() > 0.5

    run_tasks(args, "SANSA", predict_one)


if __name__ == "__main__":
    main()
