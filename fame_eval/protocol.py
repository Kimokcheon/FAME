from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any


PROTOCOL_ARCHIVE = Path("protocols/FAME_benchmark_protocols.tar.gz")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_protocols_extracted(release_root: str | Path) -> Path:
    root = Path(release_root)
    protocol_root = root / "FAME_benchmark"
    if protocol_root.exists():
        return protocol_root
    archive = root / PROTOCOL_ARCHIVE
    if not archive.exists():
        raise FileNotFoundError(f"Missing protocol archive: {archive}")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(root)
    if not protocol_root.exists():
        raise RuntimeError(f"Protocol archive did not create {protocol_root}")
    return protocol_root


def resolve_data_path(release_root: str | Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute() or "::" in value:
        return value
    return str(Path(release_root) / path)


def resolve_record_paths(release_root: str | Path, row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ["image", "mask"]:
        if isinstance(out.get(key), str):
            out[key] = resolve_data_path(release_root, out[key])
    return out


def list_tasks(release_root: str | Path) -> list[str]:
    protocol_root = ensure_protocols_extracted(release_root)
    task_list = protocol_root / "task_lists/FAME_benchmark_tasks.txt"
    return [line.strip() for line in task_list.read_text(encoding="utf-8").splitlines() if line.strip()]


def list_ood_pairs(release_root: str | Path, setting: str = "k10_seed0") -> list[str]:
    protocol_root = ensure_protocols_extracted(release_root)
    pair_root = protocol_root / "ood_episodes" / setting
    if not pair_root.exists():
        return []
    return sorted(path.name for path in pair_root.iterdir() if path.is_dir())


def load_episode(
    release_root: str | Path,
    task: str,
    *,
    setting: str = "k10_seed0",
    resolve_paths: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    protocol_root = ensure_protocols_extracted(release_root)
    episode_dir = protocol_root / "episodes" / setting / task
    support_path = episode_dir / "support.jsonl"
    test_path = episode_dir / "test.jsonl"
    support = read_jsonl(support_path) if support_path.exists() else []
    query = read_jsonl(test_path)
    if resolve_paths:
        support = [resolve_record_paths(release_root, row) for row in support]
        query = [resolve_record_paths(release_root, row) for row in query]
    return support, query


def load_zero_shot(
    release_root: str | Path,
    task: str,
    *,
    resolve_paths: bool = False,
) -> list[dict[str, Any]]:
    protocol_root = ensure_protocols_extracted(release_root)
    rows = read_jsonl(protocol_root / "episodes" / "zero_shot" / task / "test.jsonl")
    if resolve_paths:
        rows = [resolve_record_paths(release_root, row) for row in rows]
    return rows


def load_ood_episode(
    release_root: str | Path,
    pair_id: str,
    *,
    setting: str = "k10_seed0",
    resolve_paths: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    protocol_root = ensure_protocols_extracted(release_root)
    episode_dir = protocol_root / "ood_episodes" / setting / pair_id
    support = read_jsonl(episode_dir / "support.jsonl")
    query = read_jsonl(episode_dir / "test.jsonl")
    info = json.loads((episode_dir / "ood_pair_info.json").read_text(encoding="utf-8"))
    if resolve_paths:
        support = [resolve_record_paths(release_root, row) for row in support]
        query = [resolve_record_paths(release_root, row) for row in query]
    return support, query, info

