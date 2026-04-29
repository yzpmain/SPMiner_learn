from __future__ import annotations

import csv
import difflib
from typing import Callable, Tuple

import networkx as nx
import numpy as np
from torch_geometric.datasets import PPI, QM9, TUDataset

from src.core import combined_syn
from src.core import utils

__all__ = [
    "validate_dataset_name",
    "normalize_dataset_name",
    "load_dataset_for_stage",
    "DatasetLoadResult",
]


from dataclasses import dataclass


@dataclass
class DatasetLoadResult:
    """数据集加载结果封装。"""
    dataset: object
    task: str


# ---------------------------------------------------------------------------
# 注册表模式：将 if-elif 分支映射为 <名称, 加载器> 键值对。
# 动态前缀数据集（syn / facebook_combined / roadnet- / plant-）单独处理。
# ---------------------------------------------------------------------------

def _load_tu(name: str, root_name: str | None = None) -> Tuple[object, str]:
    """加载 TUDataset，默认 task='graph'。"""
    return TUDataset(root=f"/tmp/{name.upper()}", name=root_name or name.upper()), "graph"


def _load_facebook(name: str) -> list:
    graph = utils.load_snap_edgelist(f"data/{name}.txt")
    return [graph]


def _load_as() -> Tuple[object, str]:
    graph = utils.load_snap_edgelist("data/as20000102.txt")
    return [graph], "graph"


def _load_simple_edgelist(fname: str) -> Tuple[object, str]:
    return [_load_graph_from_space_delimited(f"data/{fname}")], "graph"


# 精确匹配数据集注册表。
_LOADER_REGISTRY: dict[str, Callable[[], Tuple[object, str]]] = {
    "enzymes": lambda: _load_tu("enzymes"),
    "proteins": lambda: _load_tu("proteins"),
    "cox2": lambda: _load_tu("cox2"),
    "aids": lambda: _load_tu("aids"),
    "reddit-binary": lambda: _load_tu("reddit-binary"),
    "imdb-binary": lambda: _load_tu("imdb-binary"),
    "firstmm_db": lambda: _load_tu("firstmm_db"),
    "dblp": lambda: (TUDataset(root="/tmp/dblp", name="DBLP_v1"), "graph-truncate"),
    "coil": lambda: _load_tu("coil", "COIL-DEL"),
    "ppi": lambda: (PPI(root="/tmp/PPI"), "graph"),
    "qm9": lambda: (QM9(root="/tmp/QM9"), "graph"),
    "atlas": lambda: (
        [g for g in nx.graph_atlas_g()[1:] if nx.is_connected(g)], "graph"
    ),
    "facebook": lambda: (_load_facebook("facebook_combined"), "graph"),
    "as20000102": _load_as,
    "diseasome": lambda: _load_simple_edgelist("bio-diseasome.mtx"),
    "usroads": lambda: _load_simple_edgelist("road-usroads.mtx"),
    "mn-roads": lambda: _load_simple_edgelist("mn-roads.mtx"),
    "infect": lambda: _load_simple_edgelist("infect-dublin.edges"),
    "ppi-pathways": lambda: _ppi_pathways_loader(),
}


def _ppi_pathways_loader() -> Tuple[object, str]:
    graph = nx.Graph()
    with open("data/ppi-pathways.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            graph.add_edge(int(row[0]), int(row[1]))
    return [graph], "graph"


def _make_plant_dataset(size: int) -> list[nx.Graph]:
    """生成植入模式的合成图集合。"""
    generator = combined_syn.get_generator([size])
    pattern = generator.generate(size=10)
    graphs = []
    np.random.seed(14853)
    for _ in range(1000):
        graph = generator.generate()
        n_old = len(graph)
        graph = nx.disjoint_union(graph, pattern)
        for _ in range(2):
            u = np.random.randint(0, n_old)
            v = np.random.randint(n_old, len(graph))
            graph.add_edge(int(u), int(v))
        graphs.append(graph)
    return graphs


# 固定名称的数据集集合（不含前缀型动态数据集）。
_FIXED_DATASETS = {
    "enzymes",
    "proteins",
    "cox2",
    "aids",
    "reddit-binary",
    "imdb-binary",
    "firstmm_db",
    "dblp",
    "ppi",
    "qm9",
    "atlas",
    "facebook",
    "as20000102",
    "diseasome",
    "usroads",
    "mn-roads",
    "infect",
    "coil",
    "ppi-pathways",
}

# 别名表：用于兼容历史命令与不同写法。
_ALIASES = {
    "as-733": "as20000102",
    "facebook_combined": "facebook",
}

# 各阶段支持的数据集（固定名部分）。
_STAGE_DATASETS = {
    "train-disk": {
        "enzymes",
        "proteins",
        "cox2",
        "aids",
        "reddit-binary",
        "imdb-binary",
        "firstmm_db",
        "dblp",
        "ppi",
        "qm9",
        "atlas",
        "facebook",
        "as20000102",
    },
    "mining": {
        "enzymes",
        "cox2",
        "reddit-binary",
        "dblp",
        "ppi",
        "coil",
        "facebook",
        "as20000102",
        "diseasome",
        "usroads",
        "mn-roads",
        "infect",
    },
    "count": {
        "enzymes",
        "cox2",
        "reddit-binary",
        "ppi",
        "coil",
        "facebook",
        "as20000102",
        "diseasome",
        "usroads",
        "mn-roads",
        "infect",
        "ppi-pathways",
    },
}


def _is_dynamic_dataset(name: str) -> bool:
    return (
        name.startswith("facebook_combined")
        or name.startswith("roadnet-")
        or name.startswith("plant-")
        or name.startswith("syn")
    )


def normalize_dataset_name(name: str) -> str:
    """归一化数据集名称（大小写、别名、空白处理）。"""
    normalized = name.strip().lower()
    return _ALIASES.get(normalized, normalized)


def _candidate_names_for_stage(stage: str) -> list[str]:
    names = sorted(_STAGE_DATASETS.get(stage, set()) | set(_ALIASES.keys()))
    names.extend(["facebook_combined_50", "roadnet-er", "plant-10", "syn"])
    return sorted(set(names))


def _format_unknown_dataset_error(name: str, stage: str) -> str:
    candidates = _candidate_names_for_stage(stage)
    close = difflib.get_close_matches(name, candidates, n=5, cutoff=0.4)
    hint = f"；你是否想输入: {', '.join(close)}" if close else ""
    return (
        f"Unknown dataset '{name}' for stage '{stage}'{hint}。"
        f" 可用示例: {', '.join(candidates[:12])} ..."
    )


def validate_dataset_name(name: str, stage: str) -> str:
    """校验指定阶段的数据集名，并返回归一化名称。"""
    normalized = normalize_dataset_name(name)

    if stage not in _STAGE_DATASETS:
        raise ValueError(f"Unknown stage: {stage}")

    if _is_dynamic_dataset(normalized):
        return normalized

    if normalized in _STAGE_DATASETS[stage]:
        return normalized

    raise ValueError(_format_unknown_dataset_error(name, stage))


def _load_graph_from_space_delimited(path: str) -> nx.Graph:
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


def load_dataset_for_stage(name: str, stage: str) -> Tuple[object, str]:
    """统一加载入口（注册表 + 前缀匹配）。

    返回值:
    - dataset: 可迭代图集合或 PyG/TU 数据集对象
    - task: graph / graph-truncate
    """
    dataset_name = validate_dataset_name(name, stage)

    # 动态前缀数据集：syn、facebook_combined、roadnet-、plant-
    if dataset_name.startswith("syn"):
        generator = combined_syn.get_generator([10])
        return [generator.generate(size=10) for _ in range(10)], "graph"

    if dataset_name.startswith("facebook_combined"):
        return _load_facebook(dataset_name), "graph"

    if dataset_name.startswith("roadnet-"):
        return _load_facebook(dataset_name), "graph"  # 格式同 facebook

    if dataset_name.startswith("plant-"):
        size = int(dataset_name.split("-")[-1])
        return _make_plant_dataset(size), "graph"

    # 精确匹配注册表
    loader = _LOADER_REGISTRY.get(dataset_name)
    if loader is not None:
        return loader()

    raise ValueError(_format_unknown_dataset_error(name, stage))
