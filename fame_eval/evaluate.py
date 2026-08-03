from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .metrics import case_id, find_prediction, load_mask, per_case_metrics, resize_binary
from .metrics import summarize_cases
from .protocol import read_jsonl, resolve_data_path


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_prediction_dir(
    *,
    test_jsonl: str | Path,
    pred_dir: str | Path,
    out_dir: str | Path,
    method: str,
    task: str,
    release_root: str | Path | None = None,
    support_k: int = 0,
    min_pred_area_frac: float = 0.0,
) -> dict[str, Any]:
    rows = read_jsonl(test_jsonl)
    pred_root = Path(pred_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    per_case: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        cid = case_id(row, index)
        gt_path = row.get("mask") or row.get("mask_path") or row.get("gt") or row.get("gt_path")
        if release_root is not None and isinstance(gt_path, str):
            gt_path = resolve_data_path(release_root, gt_path)
        if not gt_path:
            raise RuntimeError(f"Record {index} in {test_jsonl} has no mask path")
        gt = load_mask(gt_path)
        pred_path = find_prediction(pred_root, cid)
        missing = pred_path is None
        if missing:
            pred = gt * 0
        else:
            pred = resize_binary(load_mask(pred_path), gt.shape)
        metrics = per_case_metrics(gt, pred, min_pred_area_frac=min_pred_area_frac)
        per_case.append(
            {
                "case_id": cid,
                "task": task,
                "method": method,
                "dataset": row.get("dataset", ""),
                "organ": row.get("organ", ""),
                "modality": row.get("modality", ""),
                "gt_path": gt_path,
                "pred_path": str(pred_path) if pred_path else "",
                "missing_prediction": int(missing),
                **metrics,
            }
        )
    summary = summarize_cases(
        per_case,
        {
            "method": method,
            "task": task,
            "support_k": int(support_k),
            "test_jsonl": str(test_jsonl),
            "pred_dir": str(pred_root),
        },
    )
    write_csv(out_root / "per_case_metrics.csv", per_case)
    write_csv(out_root / "summary_all.csv", [summary])
    return summary

