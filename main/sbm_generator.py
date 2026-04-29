"""SBM (Stochastic Block Model) 合成图生成器。

生成具有社区结构的图集合，保存到 expdata/datasets/sbm/。
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import networkx as nx
import numpy as np

from main.config import DSET_DIR

__all__ = ["generate_sbm_dataset", "ensure_sbm_dataset"]

SBM_CACHE = DSET_DIR / "sbm" / "graphs.p"


def generate_sbm_dataset(
    n_graphs: int = 50,
    min_communities: int = 2,
    max_communities: int = 5,
    min_block_size: int = 5,
    max_block_size: int = 15,
    p_in: float = 0.3,
    p_out: float = 0.01,
    seed: int = 42,
) -> list[nx.Graph]:
    """生成具有社区结构的 SBM 图集合。

    参数:
        n_graphs: 生成图数量
        min_communities: 每个图最少社区数
        max_communities: 每个图最多社区数
        min_block_size: 每个社区最少节点数
        max_block_size: 每个社区最多节点数
        p_in: 社区内连接概率
        p_out: 社区间连接概率
        seed: 随机种子

    返回:
        list[nx.Graph]: 连通 SBM 图列表
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    graphs = []
    max_attempts = n_graphs * 10
    attempts = 0
    while len(graphs) < n_graphs and attempts < max_attempts:
        attempts += 1
        n_communities = rng.randint(min_communities, max_communities)
        sizes = [rng.randint(min_block_size, max_block_size)
                 for _ in range(n_communities)]
        # 构建概率矩阵: p_in 在对角线, p_out 在非对角线
        prob_mat = np.full((n_communities, n_communities), p_out)
        np.fill_diagonal(prob_mat, p_in)
        g = nx.stochastic_block_model(sizes, prob_mat, seed=np_rng.randint(2**31))
        if nx.is_connected(g):
            g = nx.convert_node_labels_to_integers(g)
            graphs.append(g)
    return graphs


def ensure_sbm_dataset(cache_path: Path = SBM_CACHE) -> list[nx.Graph]:
    """优先读缓存，不存在则生成并缓存。"""
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    graphs = generate_sbm_dataset()
    with open(cache_path, "wb") as f:
        pickle.dump(graphs, f)
    return graphs


if __name__ == "__main__":
    gs = ensure_sbm_dataset()
    print("SBM 数据集: {} 个图".format(len(gs)))
    for i, g in enumerate(gs[:3]):
        print("  图 {}: {} 节点, {} 边, {} 社区".format(
            i, len(g), g.number_of_edges(),
            len(set(nx.get_node_attributes(g, "block").values()))))
