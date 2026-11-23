"""Official Tyche v1 universal checkpoint over FAME episodes (128 input, K=8 samples)."""
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
    parser = argparse.ArgumentParser(description="Run official Tyche on FAME episodes.")
    parser.add_argument("--tyche-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--k", type=int, default=8, help="Number of sampled predictions to average.")
    parser.add_argument("--device", default="auto")
    add_common_args(parser)
    args = parser.parse_args()

    sys.path.insert(0, str(args.tyche_root))
    from tyche.model import TysegXC

    device = pick_device(args.device)
    model = TysegXC(encoder_blocks=[64, 64, 64, 64])
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu")["model"], strict=True
    )
    model.to(device).eval()

    def predict_one(support_row, query_row):
        query = read_gray01(query_row["image"], (args.size, args.size))
        support = read_gray01(support_row["image"], (args.size, args.size))
        support_mask = read_mask(support_row["mask"]).astype(np.float32)
        if support_mask.shape != (args.size, args.size):
            support_mask = np.asarray(
                torch.nn.functional.interpolate(
                    torch.from_numpy(support_mask)[None, None],
                    (args.size, args.size),
                    mode="nearest",
                )[0, 0]
            )

        query_t = torch.from_numpy(query).float()[None, None].to(device)
        support_t = torch.from_numpy(support).float()[None, None, None].to(device)
        mask_t = torch.from_numpy(support_mask).float()[None, None, None].to(device)

        with torch.no_grad():
            model_inputs = model.get_model_inputs(
                query_t, support_t, mask_t, target_size=args.k
            )
            logits = model.forward(**model_inputs)
            probability = torch.sigmoid(model.format_pred(logits)).mean(dim=1)[0]
            probability = probability.float().cpu().numpy()

        # run_tasks resizes the mask to the original query resolution.
        return probability > 0.5

    run_tasks(args, "Tyche", predict_one)


if __name__ == "__main__":
    main()
