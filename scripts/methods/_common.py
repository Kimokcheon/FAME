"""Shared FAME episode adapter for the method runners in this directory.

Every runner reads the released FAME protocols through ``fame_eval.protocol`` and
writes one binary mask per query case to

    <out-root>/<task>/pred_masks/<case_id>.png

which is the layout consumed by ``scripts/evaluate_split.py --prediction-root``.

Methods that only accept a single support pair use ``--support-index 0`` by
default, matching the paper's one-shot (K=1) setting. The released k10 episodes
provide ten fixed support groups; pass a different index to switch the pair.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fame_eval.metrics import case_id, dice_score, resize_binary
from fame_eval.protocol import list_tasks, load_episode


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--release-root",
        type=Path,
        required=True,
        help="FAME release root, e.g. data/FAME_benchmark.",
    )
    parser.add_argument(
        "--setting",
        default="k10_seed0",
        help="Episode setting (default: k10_seed0).",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated task names; default: all released tasks.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Prediction root; masks go to <out-root>/<task>/pred_masks/.",
    )
    parser.add_argument(
        "--support-index",
        type=int,
        default=0,
        help="Support pair used by single-support methods (paper one-shot uses 0).",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Debug: only run the first N queries per task.",
    )


def run_tasks(
    args: argparse.Namespace,
    method: str,
    predict_one: Callable[[dict[str, Any], dict[str, Any]], np.ndarray],
    *,
    reset: Callable[[], None] | None = None,
    task_aware: bool = False,
) -> None:
    """Evaluate ``predict_one(support, query)`` over all queries of all tasks.

    With ``task_aware=True`` the callable receives ``(task, support, query)`` so
    text-prompted methods can use a per-task concept.
    """
    if args.tasks:
        tasks = [name.strip() for name in args.tasks.split(",") if name.strip()]
    else:
        tasks = list_tasks(args.release_root)
    print(f"[{method}] setting={args.setting} tasks={len(tasks)} out={args.out_root}", flush=True)

    total = 0
    limit = getattr(args, "limit_samples", None)
    for task in tasks:
        started = time.time()
        support, query = load_episode(
            args.release_root, task, setting=args.setting, resolve_paths=True
        )
        if not support:
            raise SystemExit(
                f"[{method}] {task}: no support pairs in setting={args.setting}; "
                "these runners need a few-shot setting such as k10_seed0."
            )
        if limit is not None:
            query = query[:limit]
        support_row = support[args.support_index % len(support)]
        out_dir = args.out_root / task / "pred_masks"
        out_dir.mkdir(parents=True, exist_ok=True)

        for index, row in enumerate(query):
            if task_aware:
                prediction = predict_one(task, support_row, row)
            else:
                prediction = predict_one(support_row, row)
            mask = np.asarray(prediction, dtype=bool)
            shape = image_shape(row["image"])
            if mask.shape != shape:
                mask = resize_binary(mask, shape).astype(bool)
            Image.fromarray((mask * 255).astype(np.uint8)).save(
                out_dir / f"{case_id(row, index)}.png"
            )

        total += len(query)
        print(f"[{method}] {task}: {len(query)} masks in {time.time() - started:.1f}s", flush=True)
        if reset is not None:
            reset()

    print(f"[{method}] wrote {total} masks under {args.out_root}", flush=True)


def image_shape(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return height, width


def read_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def read_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 127


def read_mask_float(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    mask = read_mask(path)
    if size is not None:
        mask = resize_binary(mask, size).astype(bool)
    return mask.astype(np.float32)


def read_gray01(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Grayscale image in [0, 1], optionally resized to (height, width)."""
    with Image.open(path) as image:
        image = image.convert("RGB").convert("L")
        if size is not None:
            image = image.resize((size[1], size[0]), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    low, high = float(array.min()), float(array.max())
    if high - low < 1e-6:
        return array
    return (array - low) / (high - low)


def pick_device(preferred: str = "auto"):
    import torch

    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resize_long(array: np.ndarray, max_side: int = 512) -> tuple[np.ndarray, float]:
    import cv2

    height, width = array.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1.0:
        return array, 1.0
    resized = cv2.resize(
        array, (int(round(width * scale)), int(round(height * scale)))
    )
    return resized, scale


def ncc(a: np.ndarray, b: np.ndarray, valid: np.ndarray | None = None) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    if valid is not None:
        keep = valid.ravel().astype(bool)
        a, b = a[keep], b[keep]
    if a.size < 32:
        return -1.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 1e-6 else -1.0


def estimate_warp(
    support_rgb: np.ndarray, query_rgb: np.ndarray
) -> tuple[np.ndarray, str, float]:
    """Estimate a 3x3 homography mapping support coordinates to query coordinates."""
    import cv2

    support_small, _ = resize_long(support_rgb)
    query_small, _ = resize_long(query_rgb)
    query_h, query_w = query_small.shape[:2]
    support_h, support_w = support_small.shape[:2]

    support_gray = cv2.cvtColor(support_small, cv2.COLOR_RGB2GRAY)
    query_gray = cv2.cvtColor(query_small, cv2.COLOR_RGB2GRAY)

    candidates: list[tuple[str, np.ndarray]] = [("identity", np.eye(3, dtype=np.float32))]

    # SIFT + RANSAC works well for natural / RGB modalities.
    try:
        sift = cv2.SIFT_create(nfeatures=4000)
        keypoints_s, descriptors_s = sift.detectAndCompute(support_gray, None)
        keypoints_q, descriptors_q = sift.detectAndCompute(query_gray, None)
        if (
            descriptors_s is not None
            and descriptors_q is not None
            and len(keypoints_s) >= 8
            and len(keypoints_q) >= 8
        ):
            matches = cv2.BFMatcher().knnMatch(descriptors_s, descriptors_q, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            if len(good) >= 8:
                src = np.float32(
                    [keypoints_s[m.queryIdx].pt for m in good]
                ).reshape(-1, 1, 2)
                dst = np.float32(
                    [keypoints_q[m.trainIdx].pt for m in good]
                ).reshape(-1, 1, 2)
                warp, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                if (
                    warp is not None
                    and inliers is not None
                    and int(inliers.sum()) >= 10
                ):
                    candidates.append(("sift", warp.astype(np.float32)))
    except cv2.error:
        pass

    # ECC affine is more stable on grayscale medical images.
    try:
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
        _, warp = cv2.findTransformECC(
            query_gray.astype(np.float32),
            cv2.resize(support_gray, (query_w, query_h)).astype(np.float32),
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            5,
        )
        candidates.append(
            ("ecc", np.vstack([warp, [0, 0, 1]]).astype(np.float32))
        )
    except cv2.error:
        pass

    best_name, best_warp, best_score = "identity", np.eye(3, dtype=np.float32), -1.0
    for name, warp in candidates:
        warped = cv2.warpPerspective(
            support_small,
            warp,
            (query_w, query_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = cv2.warpPerspective(
            np.ones((support_h, support_w), np.uint8),
            warp,
            (query_w, query_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        score = ncc(
            cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY), query_gray, valid
        )
        if score > best_score:
            best_name, best_warp, best_score = name, warp, score

    # Map the downscaled transform back to original image coordinates.
    scale_y = support_rgb.shape[0] / support_h
    scale_x = support_rgb.shape[1] / support_w
    query_scale_y = query_rgb.shape[0] / query_h
    query_scale_x = query_rgb.shape[1] / query_w
    support_scale = np.diag([scale_x, scale_y, 1.0])
    query_scale = np.diag([1.0 / query_scale_x, 1.0 / query_scale_y, 1.0])
    return query_scale @ best_warp @ support_scale, best_name, best_score


def clean_coarse(mask: np.ndarray) -> np.ndarray:
    import cv2

    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros_like(mask)
    threshold = max(8, int(areas.max() * 0.15))
    for index, area in enumerate(areas, start=1):
        if area >= threshold:
            keep[labels == index] = 1
    return keep


def bbox_from_mask(mask: np.ndarray, expand: float = 0.06) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    height, width = mask.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad_x = max(2, int((x1 - x0 + 1) * expand))
    pad_y = max(2, int((y1 - y0 + 1) * expand))
    return [
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width - 1, x1 + pad_x),
        min(height - 1, y1 + pad_y),
    ]


def support_to_query_coarse(
    support_row: dict[str, Any], query_row: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, list[int] | None, str, float]:
    """Warp the support mask to the query image; return (query, coarse, box, warp, score)."""
    import cv2

    support = read_rgb(support_row["image"])
    query = read_rgb(query_row["image"])
    support_mask = read_mask(support_row["mask"]).astype(np.uint8)

    warp, warp_name, warp_score = estimate_warp(support, query)
    coarse = cv2.warpPerspective(
        support_mask,
        warp,
        (query.shape[1], query.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    coarse = clean_coarse(coarse)
    return query, coarse, bbox_from_mask(coarse), warp_name, warp_score


def pick_mask(
    masks: np.ndarray, scores: np.ndarray, coarse: np.ndarray | None
) -> int:
    """Prefer the candidate that agrees most with the support-derived coarse mask."""
    if coarse is not None and coarse.any():
        return max(
            range(len(masks)),
            key=lambda index: dice_score(masks[index] > 0, coarse > 0),
        )
    return int(np.argmax(scores))
