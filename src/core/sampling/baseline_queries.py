"""Baseline query generation (ESU / mFinder style).

Previously in utils.py.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

import networkx as nx
from tqdm import tqdm

from src.core.hashing import wl_hash
from src.core.sampling.neighborhood import sample_neigh
from src.core.sampling.enumeration import enumerate_subgraph
from src.logger import info

__all__ = [
    "gen_baseline_queries_rand_esu",
    "gen_baseline_queries_mfinder",
]


def gen_baseline_queries_rand_esu(queries, targets, node_anchored=False):
    sizes = Counter([len(g) for g in queries])
    max_size = max(sizes.keys())
    all_subgraphs = defaultdict(lambda: defaultdict(list))
    total_n_max_subgraphs, total_n_subgraphs = 0, 0
    for target in tqdm(targets):
        subgraphs = enumerate_subgraph(target, k=max_size, progress_bar=len(targets) < 10, node_anchored=node_anchored)
        for (size, k), v in subgraphs.items():
            all_subgraphs[size][k] += v
            if size == max_size:
                total_n_max_subgraphs += len(v)
            total_n_subgraphs += len(v)
    info(f"Subgraphs explored: {total_n_subgraphs} (max-size: {total_n_max_subgraphs})")
    out = []
    for size, count in sizes.items():
        counts = all_subgraphs[size]
        for _, neighs in list(sorted(counts.items(), key=lambda x: len(x[1]), reverse=True))[:count]:
            out.append(random.choice(neighs))
    return out


def gen_baseline_queries_mfinder(queries, targets, n_samples=10000, node_anchored=False):
    sizes = Counter([len(g) for g in queries])
    out = []
    for size, count in tqdm(sizes.items()):
        counts = defaultdict(list)
        for _ in tqdm(range(n_samples)):
            graph, neigh = sample_neigh(targets, size)
            v = neigh[0]
            neigh = graph.subgraph(neigh).copy()
            nx.set_node_attributes(neigh, 0, name="anchor")
            neigh.nodes[v]["anchor"] = 1
            neigh.remove_edges_from(nx.selfloop_edges(neigh))
            counts[wl_hash(neigh, node_anchored=node_anchored)].append(neigh)
        for _, neighs in list(sorted(counts.items(), key=lambda x: len(x[1]), reverse=True))[:count]:
            out.append(random.choice(neighs))
    return out
