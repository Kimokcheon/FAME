"""Episodic U-Net baseline over FAME episodes.

A small U-Net is trained from scratch on the selected support pair of each task
with heavy augmentation, then applied to every query image with flip TTA. This
mirrors the U-Net row of the FAME interaction-based baselines.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _common import (  # noqa: E402
    add_common_args,
    pick_device,
    read_mask,
    read_rgb,
    run_tasks,
)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, base: int = 32):
        super().__init__()
        channels = [base, base * 2, base * 4, base * 8]
        self.down1 = DoubleConv(in_channels, channels[0])
        self.down2 = DoubleConv(channels[0], channels[1])
        self.down3 = DoubleConv(channels[1], channels[2])
        self.down4 = DoubleConv(channels[2], channels[3])
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.conv3 = DoubleConv(channels[2] * 2, channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.conv2 = DoubleConv(channels[1] * 2, channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.conv1 = DoubleConv(channels[0] * 2, channels[0])
        self.head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(self.pool(down1))
        down3 = self.down3(self.pool(down2))
        down4 = self.down4(self.pool(down3))
        up3 = self.conv3(torch.cat([self.up3(down4), down3], 1))
        up2 = self.conv2(torch.cat([self.up2(up3), down2], 1))
        up1 = self.conv1(torch.cat([self.up1(up2), down1], 1))
        return self.head(up1)


def to_tensor(image: np.ndarray, resolution: int) -> torch.Tensor:
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    return F.interpolate(
        tensor, (resolution, resolution), mode="bilinear", align_corners=False
    )


def mask_to_tensor(mask: np.ndarray, resolution: int) -> torch.Tensor:
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.interpolate(tensor, (resolution, resolution), mode="nearest")


def augment(image: torch.Tensor, mask: torch.Tensor, generator: torch.Generator):
    """Random flips, 90-degree rotations, scale-translate, and intensity jitter."""
    if torch.rand(1, generator=generator).item() < 0.5:
        image, mask = torch.flip(image, [-1]), torch.flip(mask, [-1])
    if torch.rand(1, generator=generator).item() < 0.5:
        image, mask = torch.flip(image, [-2]), torch.flip(mask, [-2])
    rotations = int(torch.randint(0, 4, (1,), generator=generator).item())
    if rotations:
        image = torch.rot90(image, rotations, [-2, -1])
        mask = torch.rot90(mask, rotations, [-2, -1])

    scale = 0.85 + 0.30 * torch.rand(1, generator=generator).item()
    translate_x = (torch.rand(1, generator=generator).item() - 0.5) * 0.2
    translate_y = (torch.rand(1, generator=generator).item() - 0.5) * 0.2
    theta = torch.tensor(
        [[[scale, 0.0, translate_x], [0.0, scale, translate_y]]],
        dtype=torch.float32,
    )
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    image = F.grid_sample(
        image, grid, align_corners=False, padding_mode="border"
    )
    mask = F.grid_sample(mask, grid, align_corners=False, padding_mode="zeros")

    if torch.rand(1, generator=generator).item() < 0.7:
        image = (
            image * (0.8 + 0.4 * torch.rand(1, generator=generator).item())
        ).clamp(0, 1)
    if torch.rand(1, generator=generator).item() < 0.5:
        image = (
            image + (torch.rand(1, generator=generator).item() - 0.5) * 0.2
        ).clamp(0, 1)
    return image, mask


def soft_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probability = logits.sigmoid()
    numerator = 2 * (probability * target).sum((2, 3)) + eps
    denominator = probability.sum((2, 3)) + target.sum((2, 3)) + eps
    return (1 - numerator / denominator).mean()


def train_episode(
    support: torch.Tensor,
    support_mask: torch.Tensor,
    resolution: int,
    steps: int,
    device,
    base: int,
    seed: int,
) -> UNet:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = UNet(base=base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=1e-5
    )
    positive_weight = torch.tensor(
        [
            max(
                1.0,
                (1 - support_mask.mean().item())
                / max(support_mask.mean().item(), 1e-3),
            )
        ],
        device=device,
    ).clamp(max=20.0)

    for step in range(steps):
        images, masks = [], []
        for _ in range(4):
            image, mask = augment(support.clone(), support_mask.clone(), generator)
            images.append(image)
            masks.append(mask)
        images = torch.cat(images).to(device)
        masks = torch.cat(masks).to(device)
        logits = model(images)
        loss = F.binary_cross_entropy_with_logits(
            logits, masks, pos_weight=positive_weight
        ) + soft_dice(logits, masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (step + 1) % max(1, steps // 4) == 0:
            print(f"  step {step + 1}/{steps} loss={loss.item():.4f}", flush=True)

    model.eval()
    return model


@torch.no_grad()
def predict(model: UNet, image: torch.Tensor, device) -> torch.Tensor:
    probabilities = []
    for flip_x in (False, True):
        for flip_y in (False, True):
            tensor = image
            if flip_x:
                tensor = torch.flip(tensor, [-1])
            if flip_y:
                tensor = torch.flip(tensor, [-2])
            probability = model(tensor.to(device)).sigmoid()
            if flip_y:
                probability = torch.flip(probability, [-2])
            if flip_x:
                probability = torch.flip(probability, [-1])
            probabilities.append(probability)
    return torch.stack(probabilities).mean(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the episodic U-Net baseline on FAME episodes.")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--res", type=int, default=256)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    add_common_args(parser)
    args = parser.parse_args()

    device = pick_device(args.device)
    cache: dict[tuple[str, str], UNet | None] = {}

    def get_model(support_row):
        key = (str(support_row["image"]), str(support_row["mask"]))
        if key not in cache:
            support = to_tensor(read_rgb(support_row["image"]), args.res)
            support_mask = mask_to_tensor(read_mask(support_row["mask"]), args.res)
            if support_mask.sum() == 0:
                cache[key] = None
            else:
                cache[key] = train_episode(
                    support,
                    support_mask,
                    args.res,
                    args.steps,
                    device,
                    args.base,
                    args.seed,
                )
        return cache[key]

    def predict_one(support_row, query_row):
        query_image = read_rgb(query_row["image"])
        model = get_model(support_row)
        if model is None:
            return np.zeros(query_image.shape[:2], dtype=bool)

        query = to_tensor(query_image, args.res)
        probability = predict(model, query, device)
        probability = F.interpolate(
            probability,
            query_image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return probability.cpu().numpy() > 0.5

    run_tasks(args, "U-Net", predict_one, reset=cache.clear)


if __name__ == "__main__":
    main()
