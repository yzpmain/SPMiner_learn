"""Configuration dataclasses for SPMiner_learn.

Centralizes all default values that were previously scattered across
multiple config.py modules and CLI set_defaults calls. Internal business
logic uses these typed config objects instead of raw argparse.Namespace.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeConfig:
    """Runtime environment configuration."""
    use_gpu: bool = True
    seed: int | None = None
    n_workers: int = 4
    tag: str = ""
    output_root: str = "results"
    output_strategy: str = "version"
    output_tag: str = ""
    progress_write_interval: float = 1.0

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> RuntimeConfig:
        return cls(
            use_gpu=getattr(ns, "use_gpu", True),
            seed=getattr(ns, "seed", None),
            n_workers=getattr(ns, "n_workers", 4),
            tag=getattr(ns, "tag", ""),
            output_root=getattr(ns, "output_root", "results"),
            output_strategy=getattr(ns, "output_strategy", "version"),
            output_tag=getattr(ns, "output_tag", ""),
            progress_write_interval=getattr(ns, "progress_write_interval", 1.0),
        )


@dataclass
class MatchingConfig:
    """Encoder (subgraph matching) model configuration."""
    conv_type: str = "SAGE"
    method_type: str = "order"
    dataset: str = "syn"
    n_layers: int = 8
    batch_size: int = 64
    hidden_dim: int = 64
    skip: str = "learnable"
    dropout: float = 0.0
    n_batches: int = 1000000
    opt: str = "adam"
    opt_scheduler: str = "none"
    opt_restart: int = 100
    weight_decay: float = 0.0
    lr: float = 1e-4
    margin: float = 0.1
    test_set: str = ""
    eval_interval: int = 1000
    model_path: str = "ckpt/model.pt"
    val_size: int = 4096
    node_anchored: bool = False
    test: bool = False

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> MatchingConfig:
        return cls(
            conv_type=getattr(ns, "conv_type", "SAGE"),
            method_type=getattr(ns, "method_type", "order"),
            dataset=getattr(ns, "dataset", "syn"),
            n_layers=getattr(ns, "n_layers", 8),
            batch_size=getattr(ns, "batch_size", 64),
            hidden_dim=getattr(ns, "hidden_dim", 64),
            skip=getattr(ns, "skip", "learnable"),
            dropout=getattr(ns, "dropout", 0.0),
            n_batches=getattr(ns, "n_batches", 1000000),
            opt=getattr(ns, "opt", "adam"),
            opt_scheduler=getattr(ns, "opt_scheduler", "none"),
            opt_restart=getattr(ns, "opt_restart", 100),
            weight_decay=getattr(ns, "weight_decay", 0.0),
            lr=getattr(ns, "lr", 1e-4),
            margin=getattr(ns, "margin", 0.1),
            test_set=getattr(ns, "test_set", ""),
            eval_interval=getattr(ns, "eval_interval", 1000),
            model_path=getattr(ns, "model_path", "ckpt/model.pt"),
            val_size=getattr(ns, "val_size", 4096),
            node_anchored=getattr(ns, "node_anchored", False),
            test=getattr(ns, "test", False),
        )


@dataclass
class MiningConfig:
    """Decoder (SPMiner mining) configuration."""
    sample_method: str = "tree"
    radius: int = 3
    subgraph_sample_size: int = 0
    out_path: str = "results/out-patterns.p"
    min_pattern_size: int = 5
    max_pattern_size: int = 8
    min_neighborhood_size: int = 20
    max_neighborhood_size: int = 29
    n_neighborhoods: int = 10000
    n_trials: int = 1000
    out_batch_size: int = 10
    frontier_top_k: int = 5
    decode_thresh: float = 0.5
    search_strategy: str = "greedy"
    node_anchored: bool = False
    batch_size: int = 1000
    dataset: str = "enzymes"
    skip: str = "learnable"
    analyze: bool = False
    use_whole_graphs: bool = False

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> MiningConfig:
        return cls(
            sample_method=getattr(ns, "sample_method", "tree"),
            radius=getattr(ns, "radius", 3),
            subgraph_sample_size=getattr(ns, "subgraph_sample_size", 0),
            out_path=getattr(ns, "out_path", "results/out-patterns.p"),
            min_pattern_size=getattr(ns, "min_pattern_size", 5),
            max_pattern_size=getattr(ns, "max_pattern_size", 8),
            min_neighborhood_size=getattr(ns, "min_neighborhood_size", 20),
            max_neighborhood_size=getattr(ns, "max_neighborhood_size", 29),
            n_neighborhoods=getattr(ns, "n_neighborhoods", 10000),
            n_trials=getattr(ns, "n_trials", 1000),
            out_batch_size=getattr(ns, "out_batch_size", 10),
            frontier_top_k=getattr(ns, "frontier_top_k", 5),
            decode_thresh=getattr(ns, "decode_thresh", 0.5),
            search_strategy=getattr(ns, "search_strategy", "greedy"),
            node_anchored=getattr(ns, "node_anchored", False),
            batch_size=getattr(ns, "batch_size", 1000),
            dataset=getattr(ns, "dataset", "enzymes"),
            skip=getattr(ns, "skip", "learnable"),
            analyze=getattr(ns, "analyze", False),
            use_whole_graphs=getattr(ns, "use_whole_graphs", False),
        )


@dataclass
class AugmentConfig:
    """Feature augmentation configuration.

    Replaces the module-level globals in feature_preprocess.py:
    AUGMENT_METHOD, FEATURE_AUGMENT, FEATURE_AUGMENT_DIMS.
    """
    method: str = "concat"
    features: tuple[str, ...] = field(default_factory=tuple)
    feature_dims: tuple[int, ...] = field(default_factory=tuple)

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> AugmentConfig:
        return cls(
            method=getattr(ns, "augment_method", "concat"),
            features=tuple(getattr(ns, "augment_features", [])),
            feature_dims=tuple(getattr(ns, "augment_feature_dims", [])),
        )


__all__ = [
    "RuntimeConfig",
    "MatchingConfig",
    "MiningConfig",
    "AugmentConfig",
]
