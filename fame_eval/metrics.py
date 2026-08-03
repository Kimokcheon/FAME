from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


MASK_EXTS = [".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


def safe_name(value: Any) -> str:
    text = str(value if value is not None else "unknown")
    keep = []
    for char in text.strip().replace("/", "_").replace(" ", "_"):
        keep.append(char if char.isalnum() or char in "_.+-" else "_")
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "unknown"


def case_id(row: dict[str, Any], index: int = 0) -> str:
    for key in ["case_id", "id", "uid", "sample_id", "image_id", "name"]:
        if row.get(key) not in (None, ""):
            return safe_name(row[key])
    for key in ["image", "image_path", "file_name"]:
        if row.get(key):
            return safe_name(Path(str(row[key])).stem)
    return f"case_{index:05d}"


def load_mask(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        key = "mask" if "mask" in data.files else data.files[0]
        arr = data[key]
    else:
        arr = np.asarray(Image.open(path).convert("L"))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def resize_binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if tuple(mask.shape[:2]) == tuple(shape):
        return mask.astype(np.uint8)
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image = image.resize((int(shape[1]), int(shape[0])), Image.Resampling.NEAREST)
    return (np.asarray(image) > 0).astype(np.uint8)


def prediction_name_candidates(cid: str) -> list[str]:
    names = [cid, safe_name(cid)]
    if "__" in cid:
        names.append(safe_name(cid.split("__", 1)[1]))
    if "_" in cid:
        names.append(safe_name(cid.split("_", 1)[1]))
    return list(dict.fromkeys(name for name in names if name))


def find_prediction(pred_dir: str | Path, cid: str) -> Path | None:
    root = Path(pred_dir)
    for name in prediction_name_candidates(cid):
        for ext in MASK_EXTS:
            path = root / f"{name}{ext}"
            if path.exists():
                return path
        for path in root.glob(f"{name}.*"):
            if path.suffix.lower() in MASK_EXTS:
                return path
    return None


def dice_score(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    tp = float(np.logical_and(gt, pred).sum())
    fp = float(np.logical_and(~gt, pred).sum())
    fn = float(np.logical_and(gt, ~pred).sum())
    denom = 2.0 * tp + fp + fn
    if denom == 0:
        return 1.0
    return float((2.0 * tp) / denom)


def per_case_metrics(gt: np.ndarray, pred: np.ndarray, *, min_pred_area_frac: float = 0.0) -> dict[str, float | int]:
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)
    gt_area = float(gt_bool.sum())
    pred_area = float(pred_bool.sum())
    n_pix = float(gt_bool.size)
    gt_present = gt_area > 0
    pred_present = (pred_area / n_pix) > float(min_pred_area_frac)
    return {
        "dice": dice_score(gt_bool, pred_bool),
        "gt_present": int(gt_present),
        "pred_present": int(pred_present),
        "empty_gt": int(not gt_present),
        "empty_pred": int(not pred_present),
        "gt_area": gt_area,
        "pred_area": pred_area,
        "pred_area_frac": float(pred_area / n_pix),
        "negative_tn": int((not gt_present) and (not pred_present)),
        "negative_fp": int((not gt_present) and pred_present),
    }


def summarize_cases(rows: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = metadata or {}
    pos = [row for row in rows if int(row["gt_present"]) == 1]
    neg = [row for row in rows if int(row["gt_present"]) == 0]
    tn = sum(int(row["negative_tn"]) for row in neg)
    fp = sum(int(row["negative_fp"]) for row in neg)
    out: dict[str, Any] = dict(metadata)
    out["num_eval_cases"] = len(rows)
    out["num_positive_cases"] = len(pos)
    out["num_negative_cases"] = len(neg)
    out["num_missing_predictions"] = sum(int(row.get("missing_prediction", 0)) for row in rows)
    out["positive_dice_mean"] = float(np.mean([float(row["dice"]) for row in pos])) if pos else float("nan")
    out["positive_dice_count"] = len(pos)
    out["negative_tn"] = int(tn)
    out["negative_fp"] = int(fp)
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    out["negative_specificity_mean"] = out["specificity"]
    return out

