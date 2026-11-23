"""Official UniverSeg checkpoint over FAME episodes (128 input)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _common import (  # noqa: E402
    add_common_args,
    pick_device,
    read_gray01,
    read_mask,
    run_tasks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official UniverSeg on FAME episodes.")
    parser.add_argument(
        "--universeg-root",
        type=Path,
        help="Official UniverSeg repository; omit to use the installed universeg package.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip loading the released UniverSeg checkpoint.",
    )
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    add_common_args(parser)
    args = parser.parse_args()

    if args.universeg_root is not None:
        sys.path.insert(0, str(args.universeg_root))
    from universeg import universeg

    device = pick_device(args.device)
    model = universeg(pretrained=not args.no_pretrained).to(device).eval()

    def predict_one(support_row, query_row):
        size = (args.img_size, args.img_size)
        query = read_gray01(query_row["image"], size)
        support = read_gray01(support_row["image"], size)
        support_mask = read_mask(support_row["mask"]).astype(np.float32)
        if support_mask.shape != size:
            support_mask = np.asarray(
                torch.nn.functional.interpolate(
                    torch.from_numpy(support_mask)[None, None],
                    size,
                    mode="nearest",
                )[0, 0]
            )

        query_t = torch.from_numpy(query).float()[None, None].to(device)
        support_t = torch.from_numpy(support).float()[None, None, None].to(device)
        mask_t = torch.from_numpy(support_mask).float()[None, None, None].to(device)

        with torch.no_grad():
            logits = model(query_t, support_t, mask_t)
            probability = torch.sigmoid(logits)[0, 0].float().cpu().numpy()

        # run_tasks resizes the mask to the original query resolution.
        return probability > 0.5

    run_tasks(args, "UniverSeg", predict_one)


if __name__ == "__main__":
    main()
