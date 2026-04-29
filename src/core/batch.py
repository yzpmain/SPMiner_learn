"""Batch construction from NetworkX graphs.

Previously in utils.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from deepsnap.batch import Batch
from deepsnap.graph import Graph as DSGraph

from src.core import feature_preprocess

__all__ = [
    "batch_nx_graphs",
    "get_augmenter",
]

_AUGMENTER: feature_preprocess.FeatureAugment | None = None


def get_augmenter() -> feature_preprocess.FeatureAugment:
    global _AUGMENTER
    if _AUGMENTER is None:
        _AUGMENTER = feature_preprocess.FeatureAugment()
    return _AUGMENTER


def batch_nx_graphs(graphs: list, anchors: list | None = None) -> Batch:
    """Convert a list of NetworkX graphs to a DeepSNAP Batch."""
    augmenter = get_augmenter()

    if anchors is not None:
        for anchor, g in zip(anchors, graphs):
            for v in g.nodes:
                g.nodes[v]["node_feature"] = torch.tensor([float(v == anchor)])

    batch = Batch.from_data_list([DSGraph(g) for g in graphs])
    batch = augmenter.augment(batch)
    from src.core.device import get_device
    batch = batch.to(get_device())
    return batch
