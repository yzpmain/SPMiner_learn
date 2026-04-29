"""Runtime context helpers shared by entry points."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.cli import setup_runtime
from src.core import utils

__all__ = [
    "RuntimeContext",
    "build_runtime_context",
]


@dataclass
class RuntimeContext:
    device: object


def build_runtime_context(args) -> RuntimeContext:
    """Apply runtime setup and return the active context."""
    device = setup_runtime(args)
    # Keep return value aligned with existing helpers.
    return RuntimeContext(device=device or utils.get_device())
