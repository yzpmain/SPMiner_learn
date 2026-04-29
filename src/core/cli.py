"""Shared command-line helpers for repo entry points.

This module centralizes runtime-level CLI flags and setup logic so individual
scripts can focus on stage-specific parameters.
"""

from __future__ import annotations

import argparse
import random
from typing import Optional

import numpy as np
import torch

from src.core import utils

__all__ = [
    "add_runtime_args",
    "setup_runtime",
]


def add_runtime_args(
    parser: argparse.ArgumentParser,
    *,
    include_gpu: bool = True,
    include_seed: bool = True,
    include_tag: bool = False,
    include_n_workers: bool = False,
    include_progress_write_interval: bool = False,
    include_output_policy: bool = False,
) -> argparse.ArgumentParser:
    """Register cross-cutting runtime flags on a parser.

    The returned parser is the same object, which allows chaining.
    """
    runtime_parser = parser.add_argument_group("runtime")

    if include_gpu:
        runtime_parser.add_argument(
            "--no_gpu",
            dest="use_gpu",
            action="store_false",
            help="不要使用 GPU，即使可用也在 CPU 上运行",
        )
        parser.set_defaults(use_gpu=True)

    if include_seed:
        runtime_parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="随机种子（默认不显式设置）",
        )

    if include_tag:
        runtime_parser.add_argument(
            "--tag",
            type=str,
            default="",
            help="用于标识本次运行的标签",
        )

    if include_n_workers:
        runtime_parser.add_argument(
            "--n_workers",
            type=int,
            default=None,
            help="并行 worker 数量",
        )

    if include_progress_write_interval:
        runtime_parser.add_argument(
            "--progress_write_interval",
            type=float,
            default=1.0,
            help="progress 写入 run.log 的最小间隔（秒）",
        )

    if include_output_policy:
        runtime_parser.add_argument(
            "--output_root",
            type=str,
            default="results",
            help="统一产物输出根目录（默认 results）",
        )
        runtime_parser.add_argument(
            "--output_strategy",
            type=str,
            choices=["version", "overwrite"],
            default="version",
            help="输出冲突策略：version 自动版本化，overwrite 直接覆盖",
        )
        runtime_parser.add_argument(
            "--output_tag",
            type=str,
            default="",
            help="产物目录附加标签（用于区分实验批次）",
        )

    return parser


def setup_runtime(args) -> torch.device:
    """Apply common runtime configuration and return the active device."""
    if hasattr(args, "use_gpu"):
        try:
            utils.set_use_gpu(bool(args.use_gpu))
        except Exception:
            pass

    seed: Optional[int] = getattr(args, "seed", None)
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    return utils.get_device()