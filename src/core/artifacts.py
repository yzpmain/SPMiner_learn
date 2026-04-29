"""Artifact path management helpers.

This module centralizes output path decisions so scripts can avoid
hard-coded paths and accidental overwrite.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.logger import get_logger

__all__ = [
    "resolve_output_path",
    "task_output_dir",
    "choose_cli_output_path",
    "write_manifest",
    "ensure_dir",
]


def _now_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _versioned_candidate(path: Path, token: str, index: int | None = None) -> Path:
    stem = path.stem
    suffix = path.suffix
    if index is None:
        return path.with_name(f"{stem}.{token}{suffix}")
    return path.with_name(f"{stem}.{token}.{index}{suffix}")


def resolve_output_path(path: str | Path, strategy: str = "version") -> Path:
    """Resolve final output path and create parent directory.

    strategy:
    - version: if file exists, append timestamp / counter suffix.
    - overwrite: keep original path.
    """
    p = Path(path)
    ensure_dir(p.parent)
    if strategy == "overwrite" or not p.exists():
        return p

    token = _now_token()
    cand = _versioned_candidate(p, token)
    if not cand.exists():
        return cand

    idx = 1
    while True:
        cand = _versioned_candidate(p, token, idx)
        if not cand.exists():
            return cand
        idx += 1


def _run_label(args: Any) -> str:
    logger = get_logger()
    base = getattr(logger, "run_name", None) or _now_token()
    tag = (getattr(args, "output_tag", "") or "").strip()
    return f"{base}_{tag}" if tag else base


def task_output_dir(args: Any, task: str, dataset: str | None = None) -> Path:
    root = Path(getattr(args, "output_root", "results"))
    ds = dataset or getattr(args, "dataset", "unknown")
    return ensure_dir(root / task / str(ds) / _run_label(args))


def choose_cli_output_path(
    args: Any,
    cli_path: str,
    *,
    default_cli_path: str,
    suggested_default_path: str | Path,
) -> Path:
    """Map default CLI output to normalized output tree, keep custom path unchanged."""
    if cli_path == default_cli_path:
        target = Path(suggested_default_path)
    else:
        target = Path(cli_path)
    return resolve_output_path(target, getattr(args, "output_strategy", "version"))


def write_manifest(manifest_path: str | Path, args: Any, outputs: dict[str, Any], **extra: Any) -> Path:
    p = Path(manifest_path)
    ensure_dir(p.parent)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": getattr(get_logger(), "run_dir", None),
        "args": vars(args) if hasattr(args, "__dict__") else str(args),
        "outputs": outputs,
    }
    payload.update(extra)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return p
