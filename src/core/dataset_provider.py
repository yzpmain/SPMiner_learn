"""Dataset provider abstractions for different stages."""

from __future__ import annotations

from src.core import data, dataset_registry

__all__ = [
    "make_matching_data_source",
    "normalize_for_stage",
    "load_for_stage",
]


def make_matching_data_source(args):
    """Create a training data source for matching stage.

    Keeps legacy dataset suffix behavior intact:
    - *-balanced
    - *-imbalanced
    """
    raw_dataset = args.dataset.strip().lower()

    mode = "balanced"
    base_dataset = raw_dataset
    if raw_dataset.endswith("-balanced"):
        base_dataset = raw_dataset[: -len("-balanced")]
        mode = "balanced"
    elif raw_dataset.endswith("-imbalanced"):
        base_dataset = raw_dataset[: -len("-imbalanced")]
        mode = "imbalanced"

    if base_dataset.startswith("syn"):
        if mode == "balanced":
            return data.OTFSynDataSource(node_anchored=args.node_anchored)
        return data.OTFSynImbalancedDataSource(node_anchored=args.node_anchored)

    normalized = dataset_registry.validate_dataset_name(base_dataset, "train-disk")
    if mode == "balanced":
        return data.DiskDataSource(normalized, node_anchored=args.node_anchored)
    return data.DiskImbalancedDataSource(normalized, node_anchored=args.node_anchored)


def normalize_for_stage(dataset: str, stage: str) -> str:
    return dataset_registry.validate_dataset_name(dataset, stage)


def load_for_stage(dataset: str, stage: str):
    normalized = normalize_for_stage(dataset, stage)
    ds, task = dataset_registry.load_dataset_for_stage(normalized, stage)
    return normalized, ds, task
