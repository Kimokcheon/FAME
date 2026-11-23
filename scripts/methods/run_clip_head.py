"""CLIP-head baselines (MaskCLIP / SCLIP / ProxyCLIP) over FAME episodes.

All three share an OpenAI CLIP ViT-B/16 backbone and differ in how the dense
patch features of the last transformer block are formed:
    maskclip   value projection only (MaskCLIP)
    sclip      CLIP self-attention aggregated values (SCLIP)
    proxyclip  DINOv2 proxy-attention aggregated values (ProxyCLIP)

Segmentation is a support-prototype cosine matcher with a threshold calibrated on
the support pair, followed by guided-filter refinement.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
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

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]


def build_dinov2(matcher_root: Path | None, checkpoint: Path | None, device):
    if matcher_root is None or checkpoint is None:
        raise SystemExit("--matcher-root and --dinov2-checkpoint are required for proxyclip")
    sys.path.insert(0, str(matcher_root))
    from dinov2.models import vision_transformer as vits
    import dinov2.utils.utils as dinov2_utils

    model = vits.__dict__["vit_large"](
        img_size=518, patch_size=14, init_values=1.0, block_chunks=0
    )
    dinov2_utils.load_pretrained_weights(model, str(checkpoint), "teacher")
    return model.to(device).eval()


@torch.no_grad()
def clip_dense(visual, image, mode, proxy=None):
    """Return L2-normalised dense CLIP features [1, C, h, w]."""
    features = visual.conv1(image)
    height, width = features.shape[-2:]
    features = features.flatten(2).permute(0, 2, 1)
    positional = visual.positional_embedding
    if positional.dim() == 3:
        positional = positional[0]
    cls = visual.class_embedding.to(features.dtype)[None, None].expand(
        features.shape[0], -1, -1
    )
    grid = int((positional.shape[0] - 1) ** 0.5)
    patch_pos = positional[1:].reshape(1, grid, grid, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(
        patch_pos, (height, width), mode="bicubic", align_corners=False
    )
    patch_pos = patch_pos.flatten(2).permute(0, 2, 1)
    positional = torch.cat([positional[:1][None], patch_pos], dim=1)
    features = visual.ln_pre(torch.cat([cls, features], dim=1) + positional)

    blocks = visual.transformer.resblocks
    for block in blocks[:-1]:
        features = block(features)
    last = blocks[-1]
    normed = last.ln_1(features)
    attention = last.attn
    channels = normed.shape[-1]
    heads = attention.num_heads
    head_dim = channels // heads
    batch, length, _ = normed.shape
    qkv = F.linear(normed, attention.in_proj_weight, attention.in_proj_bias)
    query, key, value = qkv.chunk(3, dim=-1)
    num_patches = length - 1

    def to_heads(tensor):
        return tensor.reshape(batch, length, heads, head_dim).permute(0, 2, 1, 3)

    query, key, value = to_heads(query), to_heads(key), to_heads(value)

    if mode == "maskclip":
        output = value
    elif mode == "sclip":
        weights = (query @ key.transpose(-2, -1)) / (head_dim ** 0.5)
        weights = weights.softmax(dim=-1)
        output = weights @ value
    elif mode == "proxyclip":
        dense = proxy(image)
        dense = F.normalize(dense, dim=-1)
        weights = (dense @ dense.transpose(-2, -1)) / (dense.shape[-1] ** 0.5)
        weights = weights.softmax(dim=-1)
        weights = F.interpolate(
            weights[None], size=(num_patches, num_patches), mode="bilinear",
            align_corners=False,
        )[0]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        output = weights[:, None] @ value[:, :, 1:, :]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if mode == "proxyclip":
        output = output.permute(0, 2, 1, 3).reshape(batch, num_patches, channels)
    else:
        output = output.permute(0, 2, 1, 3).reshape(batch, length, channels)[:, 1:]
    output = F.linear(output, attention.out_proj.weight, attention.out_proj.bias)
    output = F.normalize(output, dim=-1)
    side = int(num_patches ** 0.5)
    return output.transpose(1, 2).reshape(batch, channels, side, side)


def mask_to_grid(mask, height, width):
    mask_t = torch.from_numpy(mask.astype(np.float32))[None, None]
    mask_t = F.interpolate(mask_t, (height, width), mode="area")[0, 0]
    grid = mask_t > 0.5
    if grid.sum() == 0:
        grid = mask_t > max(0.05, float(mask_t.max()) * 0.5)
    return grid.numpy()


def guided_filter(guide_rgb, source, radius=8, eps=1e-3):
    guide = cv2.cvtColor(guide_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    source = source.astype(np.float32)
    mean_guide = cv2.boxFilter(guide, -1, (radius, radius))
    mean_source = cv2.boxFilter(source, -1, (radius, radius))
    covariance = (
        cv2.boxFilter(guide * source, -1, (radius, radius))
        - mean_guide * mean_source
    )
    variance = cv2.boxFilter(guide * guide, -1, (radius, radius)) - mean_guide * mean_guide
    slope = covariance / (variance + eps)
    intercept = mean_source - slope * mean_guide
    return (
        cv2.boxFilter(slope, -1, (radius, radius)) * guide
        + cv2.boxFilter(intercept, -1, (radius, radius))
    )


def segment(support_features, query_features, support_mask, query_rgb):
    height, width = support_features.shape[-2:]
    foreground = mask_to_grid(support_mask, height, width)
    if foreground.sum() == 0:
        return np.zeros(query_rgb.shape[:2], dtype=bool)
    background = ~foreground

    prototype_fg = F.normalize(
        support_features[0][:, foreground].mean(1), dim=0
    )
    prototype_bg = F.normalize(
        support_features[0][:, background].mean(1), dim=0
    )

    def score(features):
        fg_score = torch.einsum("c,chw->hw", prototype_fg, features[0])
        bg_score = torch.einsum("c,chw->hw", prototype_bg, features[0])
        return fg_score - bg_score

    support_score = score(support_features).cpu().numpy()
    threshold = 0.5 * (
        support_score[foreground].mean() + support_score[background].mean()
    )
    temperature = max(float(support_score.std()), 1e-3)

    query_score = F.interpolate(
        score(query_features)[None, None],
        query_rgb.shape[:2],
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()
    probability = 1.0 / (
        1.0 + np.exp(-(query_score - float(threshold)) / temperature)
    )
    probability = guided_filter(query_rgb, probability, radius=8, eps=1e-3)

    mask = probability > 0.5
    if mask.any():
        from scipy import ndimage as ndi

        mask = ndi.binary_fill_holes(mask)
        count, labels = cv2.connectedComponents(mask.astype(np.uint8))
        if count > 1:
            best, best_score = 0, -1.0
            for index in range(1, count):
                component = labels == index
                candidate = float(probability[component].mean()) * (
                    component.sum() ** 0.25
                )
                if candidate > best_score:
                    best, best_score = index, candidate
            mask = labels == best
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a CLIP-head baseline on FAME episodes."
    )
    parser.add_argument("--mode", required=True, choices=["maskclip", "sclip", "proxyclip"])
    parser.add_argument("--openclip-model", default="ViT-B-16")
    parser.add_argument("--openclip-pretrained", default="openai")
    parser.add_argument("--matcher-root", type=Path, help="Matcher repo for DINOv2 (proxyclip).")
    parser.add_argument("--dinov2-checkpoint", type=Path, help="DINOv2 ViT-L/14 weights (proxyclip).")
    parser.add_argument("--res", type=int, default=448)
    parser.add_argument("--device", default="auto")
    add_common_args(parser)
    args = parser.parse_args()

    import open_clip

    device = pick_device(args.device)
    model, _, _ = open_clip.create_model_and_transforms(
        args.openclip_model, pretrained=args.openclip_pretrained, device=device
    )
    visual = model.visual.eval()
    for parameter in visual.parameters():
        parameter.requires_grad_(False)

    proxy = None
    if args.mode == "proxyclip":
        dinov2 = build_dinov2(args.matcher_root, args.dinov2_checkpoint, device)

        @torch.no_grad()
        def proxy(image):
            image = F.interpolate(
                image, (518, 518), mode="bilinear", align_corners=False
            )
            output = dinov2.forward_features(image)
            if isinstance(output, dict):
                tokens = output.get("x_norm_patchtokens")
                if tokens is None:
                    tokens = output["x_prenorm"][:, 1:]
            elif isinstance(output, (list, tuple)):
                tokens = output[0]
            else:
                tokens = output[:, 1:] if output.shape[1] > 1 else output
            return F.normalize(tokens, dim=-1)

    def prepare(rgb):
        image = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        image = F.interpolate(
            image, (args.res, args.res), mode="bilinear", align_corners=False
        )
        return ((image - CLIP_MEAN) / CLIP_STD).to(device)

    def predict_one(support_row, query_row):
        support_rgb = read_rgb(support_row["image"])
        query_rgb = read_rgb(query_row["image"])
        support_mask = read_mask(support_row["mask"])
        with torch.no_grad():
            support_features = clip_dense(visual, prepare(support_rgb), args.mode, proxy)
            query_features = clip_dense(visual, prepare(query_rgb), args.mode, proxy)
        return segment(support_features, query_features, support_mask, query_rgb)

    run_tasks(args, args.mode, predict_one)


if __name__ == "__main__":
    main()
