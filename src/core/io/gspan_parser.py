"""gSpan text output parser.

Previously in utils.py (parse_gspan_output).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import networkx as nx

__all__ = ["parse_gspan_output"]


def parse_gspan_output(file_path: Path) -> List[nx.Graph]:
    """Parse gSpan text output into a list of NetworkX graphs."""
    graphs: List[nx.Graph] = []
    current = None

    for raw in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.startswith("t "):
            if current is not None and current.number_of_nodes() > 0:
                graphs.append(current)
            current = nx.Graph()
            continue

        if line.startswith("v ") and current is not None:
            toks = line.split()
            if len(toks) >= 3:
                current.add_node(int(toks[1]), label=toks[2])
            continue

        if line.startswith("e ") and current is not None:
            toks = line.split()
            if len(toks) >= 4:
                current.add_edge(int(toks[1]), int(toks[2]), label=toks[3])
            continue

        if line.lower().startswith("support") and current is not None:
            try:
                current.graph["support"] = float(line.split(":", 1)[1].strip())
            except Exception:
                pass

    if current is not None and current.number_of_nodes() > 0:
        graphs.append(current)
    return graphs
