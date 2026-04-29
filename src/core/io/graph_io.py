"""Graph file I/O utilities.

Previously in utils.py (load_snap_edgelist) and dataset_registry.py (_load_graph_from_space_delimited).
"""

from __future__ import annotations

import networkx as nx

__all__ = ["load_snap_edgelist", "load_graph_from_space_delimited"]


def load_snap_edgelist(path: str) -> nx.Graph:
    """Load undirected graph from SNAP-style edge list file.

    Returns the largest connected component for consistent sampling.
    """
    graph = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                graph.add_edge(int(parts[0]), int(parts[1]))
    if not nx.is_connected(graph):
        graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
    return graph


def load_graph_from_space_delimited(path: str) -> nx.Graph:
    """Load graph from a space-delimited edge list file."""
    graph = nx.Graph()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            toks = stripped.split()
            if len(toks) < 2:
                continue
            graph.add_edge(int(toks[0]), int(toks[1]))
    return graph
