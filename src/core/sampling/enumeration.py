"""ESU-style subgraph enumeration.

Previously in utils.py.
"""

from __future__ import annotations

import random
from collections import defaultdict

import networkx as nx
import numpy as np
from tqdm import tqdm

from src.core.hashing import wl_hash

__all__ = [
    "enumerate_subgraph",
    "extend_subgraph",
]


def enumerate_subgraph(G: nx.Graph, k: int = 3, progress_bar: bool = False, node_anchored: bool = False):
    """Enumerate subgraphs using ESU algorithm, clustered by WL signature."""
    ps = np.arange(1.0, 0.0, -1.0 / (k + 1)) ** 1.5
    motif_counts = defaultdict(list)
    iterator = tqdm(G.nodes) if progress_bar else G.nodes
    for node in iterator:
        sg = {node}
        v_ext = set()
        neighbors = [nbr for nbr in list(G[node].keys()) if nbr > node]
        n_frac = len(neighbors) * ps[1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac) else 0)
        neighbors = random.sample(neighbors, n_samples)
        for nbr in neighbors:
            v_ext.add(nbr)
        extend_subgraph(G, k, sg, v_ext, node, motif_counts, ps, node_anchored)
    return motif_counts


def extend_subgraph(
    G: nx.Graph,
    k: int,
    sg: set,
    v_ext: set,
    node_id: int,
    motif_counts: dict,
    ps: np.ndarray,
    node_anchored: bool,
):
    """Recursive ESU extension step."""
    sg_G = G.subgraph(sg)
    if node_anchored:
        sg_G = sg_G.copy()
        nx.set_node_attributes(sg_G, 0, name="anchor")
        sg_G.nodes[node_id]["anchor"] = 1

    motif_counts[len(sg_G), wl_hash(sg_G, node_anchored=node_anchored)].append(sg_G)
    if len(sg) == k:
        return

    old_v_ext = v_ext.copy()
    while v_ext:
        w = v_ext.pop()
        new_v_ext = v_ext.copy()
        neighbors = [nbr for nbr in list(G[w].keys()) if nbr > node_id and nbr not in sg and nbr not in old_v_ext]
        n_frac = len(neighbors) * ps[len(sg) + 1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac) else 0)
        neighbors = random.sample(neighbors, n_samples)
        for nbr in neighbors:
            new_v_ext.add(nbr)
        sg.add(w)
        extend_subgraph(G, k, sg, new_v_ext, node_id, motif_counts, ps, node_anchored)
        sg.remove(w)
