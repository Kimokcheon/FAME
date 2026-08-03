from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def read_csv_one(path: str | Path) -> dict[str, Any] | None:
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def collect_summary_files(eval_root: str | Path) -> list[dict[str, Any]]:
    root = Path(eval_root)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("summary_all.csv")):
        row = read_csv_one(path)
        if row is not None:
            row["summary_path"] = str(path)
            rows.append(row)
    return rows


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

