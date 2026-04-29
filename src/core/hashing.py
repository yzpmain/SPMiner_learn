"""Graph hashing utilities.

WL-style hashing and vector hashing for subgraph isomorphism fingerprints.
Previously in utils.py.
"""

from __future__ import annotations

import random
from collections.abc import Hashable

import networkx as nx
import numpy as np

__all__ = [
    "vec_hash",
    "wl_hash",
]

cached_masks: list[int] | None = None


def vec_hash(v: list[Hashable]) -> list[int]:
    """Vector hashing with cached random masks."""
    global cached_masks
    if cached_masks is None:
        random.seed(2019)
        cached_masks = [random.getrandbits(32) for _ in range(len(v))]
    v = [int(hash(v[i]) % (2**31 - 1)) ^ mask for i, mask in enumerate(cached_masks)]
    return v


def wl_hash(g: nx.Graph, dim: int = 64, node_anchored: bool = False) -> tuple:
    """WL-style graph signature (hashable tuple).

    Fixed-dimension discrete vector aggregation. Summed and returned as tuple
    for use as dict key (isomorphism fingerprint).
    """
    g = nx.convert_node_labels_to_integers(g)
    vecs = np.zeros((len(g), dim), dtype=object)
    if node_anchored:
        for v in g.nodes:
            if g.nodes[v].get("anchor") == 1:
                vecs[v] = 1
                break
    for _ in range(len(g)):
        newvecs = np.zeros((len(g), dim), dtype=object)
        for n in g.nodes:
            newvecs[n] = vec_hash(np.sum(vecs[list(g.neighbors(n)) + [n]], axis=0))
        vecs = newvecs
    return tuple(np.sum(vecs, axis=0))
