"""Neighborhood sampling from graph datasets.

Previously in utils.py (sample_neigh).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import scipy.stats as stats

from src.logger import warning

__all__ = ["sample_neigh", "frontier_sample_nodes", "NeighborhoodSample"]


@dataclass
class NeighborhoodSample:
    """采样邻域的结果封装。"""
    graph: nx.Graph
    nodes: list = field(default_factory=list)
    anchor: object | None = None


def frontier_sample_nodes(graph: nx.Graph, size: int) -> list:
    """BFS 前沿扩展采样连通节点集（不修改原图）。"""
    start_node = random.choice(list(graph.nodes))
    neigh = [start_node]
    frontier = list(set(graph.neighbors(start_node)) - set(neigh))
    visited = {start_node}
    while len(neigh) < size and frontier:
        new_node = random.choice(list(frontier))
        neigh.append(new_node)
        visited.add(new_node)
        frontier += list(graph.neighbors(new_node))
        frontier = [x for x in frontier if x not in visited]
    return neigh


def sample_neigh(graphs: list[nx.Graph], size: int, max_attempts: int = 100):
    """Sample a connected neighborhood from a list of graphs.

    Weighted by graph size. BFS frontier expansion from random start node.

    Returns:
        (graph, neigh_nodes) where neigh_nodes is the list of sampled nodes.
    """
    ps = np.array([len(g) for g in graphs], dtype=float)
    ps /= np.sum(ps)
    dist = stats.rv_discrete(values=(np.arange(len(graphs)), ps))

    for attempt in range(max_attempts):
        idx = dist.rvs()
        graph = graphs[idx]
        neigh = frontier_sample_nodes(graph, size)

    warning(f"Could not sample neighborhood of size {size} after {max_attempts} attempts")
    return graph, neigh
