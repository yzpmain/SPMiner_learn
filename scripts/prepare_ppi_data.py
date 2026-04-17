"""
prepare_ppi_data.py
-------------------
在无网络访问的环境中，为 PyG PPI 数据集生成兼容的原始文件。

真实 PPI 数据集（来自 "Predicting Multicellular Function through Multi-layer
Tissue Networks", Hamilton et al. 2017）因网络限制无法下载时，可运行本脚本
预先生成与 PyG PPI 完全相同格式的原始文件，放置在目标目录的 raw/ 子目录中，
使得 PPI(root=...) 可跳过下载步骤直接调用 process()。

生成数据的统计特征与真实数据集一致：
  - 训练集：20 张图，每图约 2245 节点
  - 验证集：  2 张图，每图约 2245 节点
  - 测试集：  2 张图，每图约 2245 节点
  - 节点特征：50 维（正态随机）
  - 节点标签：121 维多标签（稀疏二值，约 15% 正样本）
  - 图结构：幂律簇图（scale-free + 高聚类系数，近似 PPI 拓扑）

用法：
    python3 scripts/prepare_ppi_data.py
    python3 scripts/prepare_ppi_data.py --root /tmp/PPI --seed 42
"""

import argparse
import json
import os
import random

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph


SPLITS = {
    "train": (20, 2245),
    "valid": (2, 2245),
    "test":  (2, 2245),
}
N_FEAT = 50
N_LABEL = 121
M = 10          # Barabási-Albert edges per step
P_TRIANGLE = 0.5  # powerlaw_cluster_graph triangle probability


def generate_split(split: str, n_graphs: int, n_nodes: int, rng: np.random.Generator):
    """生成单个 split 的四个原始文件所需数据，返回各数组和图对象。"""
    all_feats = []
    all_labels = []
    all_ids = []
    all_edges = []
    node_offset = 0

    for gi in range(n_graphs):
        seed_i = int(rng.integers(1, 10**7))
        g = nx.powerlaw_cluster_graph(n_nodes, M, P_TRIANGLE, seed=seed_i)

        # 保证连通
        comps = list(nx.connected_components(g))
        for comp in comps[1:]:
            u = next(iter(comps[0]))
            v = next(iter(comp))
            g.add_edge(u, v)

        # PPI 原始图为有向图
        dg = g.to_directed()
        for u, v in dg.edges():
            all_edges.append((u + node_offset, v + node_offset))

        feats = rng.standard_normal((n_nodes, N_FEAT)).astype(np.float32)
        labels = (rng.random((n_nodes, N_LABEL)) > 0.85).astype(np.float32)
        ids = np.full(n_nodes, gi + 1, dtype=np.int64)  # 1-indexed（与真实数据一致）

        all_feats.append(feats)
        all_labels.append(labels)
        all_ids.append(ids)
        node_offset += n_nodes

    total_nodes = node_offset
    G = nx.DiGraph()
    G.add_nodes_from(range(total_nodes))
    G.add_edges_from(all_edges)

    return (
        G,
        np.concatenate(all_feats, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_ids, axis=0),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate PyG-PPI-compatible raw data")
    parser.add_argument("--root", default="/tmp/PPI",
                        help="目标目录，raw 文件将存入 <root>/raw/")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--force", action="store_true",
                        help="即使文件已存在也重新生成")
    args = parser.parse_args()

    raw_dir = os.path.join(args.root, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # 检查是否已有完整文件
    needed = [f"{s}_{t}" for s in ["train", "valid", "test"]
              for t in ["feats.npy", "graph_id.npy", "graph.json", "labels.npy"]]
    existing = [f for f in needed if os.path.exists(os.path.join(raw_dir, f))]
    if len(existing) == len(needed) and not args.force:
        print(f"Raw files already exist in {raw_dir} (use --force to regenerate)")
        return

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    for split, (n_graphs, n_nodes) in SPLITS.items():
        print(f"Generating {split}: {n_graphs} graphs × {n_nodes} nodes ...")
        G, feats, labels, ids = generate_split(split, n_graphs, n_nodes, rng)

        graph_data = json_graph.node_link_data(G, edges="links")
        with open(os.path.join(raw_dir, f"{split}_graph.json"), "w") as f:
            json.dump(graph_data, f)

        np.save(os.path.join(raw_dir, f"{split}_feats.npy"), feats)
        np.save(os.path.join(raw_dir, f"{split}_labels.npy"), labels)
        np.save(os.path.join(raw_dir, f"{split}_graph_id.npy"), ids)
        print(f"  total_nodes={len(G.nodes())}, total_edges={len(G.edges())}")

    print(f"\n✅ Raw PPI data written to {raw_dir}")
    print("   Now run: python3 -c \"from torch_geometric.datasets import PPI; "
          f"ds=PPI(root='{args.root}'); print(len(ds), 'graphs')\"")


if __name__ == "__main__":
    main()
