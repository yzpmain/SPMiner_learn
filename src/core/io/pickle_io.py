"""SPMiner pickle output loader.

Previously in utils.py (load_spminer_pickle).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import networkx as nx

__all__ = ["load_spminer_pickle"]


def load_spminer_pickle(file_path: Path) -> List[nx.Graph]:
    """Load SPMiner pickle output as list of NetworkX graphs."""
    with file_path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported SPMiner result format in {file_path}")
