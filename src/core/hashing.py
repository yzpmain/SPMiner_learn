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

# 按向量长度缓存的掩码表，避免不同维度向量共用掩码导致错位。
_masks_by_length: dict[int, list[int]] = {}


def vec_hash(v: list[Hashable]) -> list[int]:
    """Vector hashing with cached random masks (keyed by vector length)."""
    length = len(v)
    if length not in _masks_by_length:
        rng = random.Random(2019)
        _masks_by_length[length] = [rng.getrandbits(32) for _ in range(length)]
    masks = _masks_by_length[length]
    v = [int(hash(v[i]) % (2**31 - 1)) ^ mask for i, mask in enumerate(masks)]
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
