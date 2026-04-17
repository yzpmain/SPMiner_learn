"""SPMiner 解码（挖掘）入口模块。

将 pattern_growth 拆分为职责单一的子函数：
  _load_mining_model     —— 加载 checkpoint 并置于 eval 模式
  _prepare_graphs        —— 将多来源数据集统一为 nx.Graph 列表
  _sample_neighborhoods  —— 邻域采样（radial / tree 两种策略）
  _embed_neighborhoods   —— 批量推理，返回嵌入列表
  _make_search_agent     —— 按 search_strategy 创建搜索代理（工厂）
  _save_patterns         —— 序列化结果到 results/
  _visualize_patterns    —— 绘图保存到 plots/cluster/

pattern_growth 是薄壳编排函数，依次调用上述子函数。
"""

import argparse
import time
import os

import numpy as np
import torch
from tqdm import tqdm

from torch_geometric.datasets import TUDataset, PPI
import torch_geometric.utils as pyg_utils

from common import models
from common import utils
from common import combined_syn
from subgraph_mining.config import parse_decoder
from subgraph_matching.config import parse_encoder
from subgraph_mining.search_agents import create_search_agent

import matplotlib.pyplot as plt

import random
from collections import defaultdict
import networkx as nx
import pickle


def make_ppi_syn_dataset(n_graphs=20, n_nodes=500, m=10, seed=42):
    """构造合成 PPI-like 数据集（无法下载真实 PPI 时的替代方案）。

    蛋白质互作网络（PPI）具有无标度（scale-free）和高聚类系数特征，
    可用 Barabási-Albert（BA）幂律簇图来近似：
    - n_graphs : 生成图的数量（真实 PPI 训练集有 20 张图）
    - n_nodes  : 每张图的节点数（真实 PPI 平均约 2000，此处缩小以加速测试）
    - m        : BA 模型每步新增边数（越大图越稠密）

    返回：networkx.Graph 列表。
    """
    rng = random.Random(seed)
    graphs = []
    for i in range(n_graphs):
        # powerlaw_cluster_graph 比纯 BA 更接近 PPI 的高聚类系数
        p_triangle = rng.uniform(0.3, 0.7)
        g = nx.powerlaw_cluster_graph(n_nodes, m, p_triangle,
                                      seed=seed + i)
        # 确保连通
        if not nx.is_connected(g):
            components = list(nx.connected_components(g))
            for comp in components[1:]:
                u = rng.choice(list(components[0]))
                v = rng.choice(list(comp))
                g.add_edge(u, v)
        graphs.append(g)
    return graphs


def make_plant_dataset(size):
    """构造带植入模式的合成图数据集。

    用于验证挖掘算法是否能从噪声图中恢复出高频结构模式：
    - 先生成一个固定模式 pattern；
    - 再把 pattern 并到随机图中并随机连边；
    - 返回由 1000 张图组成的数据集。
    """
    generator = combined_syn.get_generator([size])
    random.seed(3001)
    np.random.seed(14853)
    pattern = generator.generate(size=10)
    nx.draw(pattern, with_labels=True)
    plt.savefig("plots/cluster/plant-pattern.png")
    plt.close()
    graphs = []
    for i in range(1000):
        graph = generator.generate()
        n_old = len(graph)
        graph = nx.disjoint_union(graph, pattern)
        for j in range(1, 3):
            u = random.randint(0, n_old - 1)
            v = random.randint(n_old, len(graph) - 1)
            graph.add_edge(u, v)
        graphs.append(graph)
    return graphs


# ---------------------------------------------------------------------------
# Pipeline 子函数
# ---------------------------------------------------------------------------

def _load_mining_model(args):
    """加载训练好的嵌入模型并设置为推理模式。

    通过 common.models.build_model 统一构建模型结构，
    再加载 args.model_path 指定的权重文件。
    """
    model = models.build_model(args)
    model.eval()
    model.load_state_dict(torch.load(args.model_path,
        map_location=utils.get_device()))
    return model


def _prepare_graphs(dataset, task, args):
    """将不同来源的数据集统一转换为 networkx.Graph 列表。

    支持 "graph-labeled"（带标签，只取 label==0 的图）和
    "graph-truncate"（只取前 1000 张图）两种特殊任务模式。

    返回：
        (graphs, labels) 其中 labels 仅在 graph-labeled 时有效，其余为 None。
    """
    labels = None
    if task == "graph-labeled":
        dataset, labels = dataset

    graphs = []
    for i, graph in enumerate(dataset):
        if task == "graph-labeled" and labels[i] != 0:
            continue
        if task == "graph-truncate" and i >= 1000:
            break
        if not type(graph) == nx.Graph:
            graph = pyg_utils.to_networkx(graph).to_undirected()
        graphs.append(graph)
    return graphs


def _sample_neighborhoods(graphs, args):
    """从图集合中采样候选邻域子图。

    支持两种采样策略：
    - "radial"：以每个节点为中心，截取指定半径内的子图。
    - "tree"  ：随机前沿扩展，采样固定大小的连通邻域。

    若 args.use_whole_graphs 为 True，则直接使用完整图，跳过采样。

    返回：
        (neighs, anchors) 其中 anchors 仅在 node_anchored 模式下有效。
    """
    if args.use_whole_graphs:
        return graphs, []

    neighs = []
    anchors = []

    if args.sample_method == "radial":
        for i, graph in enumerate(graphs):
            print(i)
            for j, node in enumerate(graph.nodes):
                if len(graphs) <= 10 and j % 100 == 0:
                    print(i, j)
                neigh = list(nx.single_source_shortest_path_length(
                    graph, node, cutoff=args.radius).keys())
                if args.subgraph_sample_size != 0:
                    neigh = random.sample(neigh,
                        min(len(neigh), args.subgraph_sample_size))
                if len(neigh) > 1:
                    neigh = graph.subgraph(neigh)
                    if args.subgraph_sample_size != 0:
                        neigh = neigh.subgraph(
                            max(nx.connected_components(neigh), key=len))
                    neigh = nx.convert_node_labels_to_integers(neigh)
                    neigh.add_edge(0, 0)
                    neighs.append(neigh)

    elif args.sample_method == "tree":
        for j in tqdm(range(args.n_neighborhoods)):
            graph, neigh = utils.sample_neigh(
                graphs,
                random.randint(args.min_neighborhood_size,
                               args.max_neighborhood_size))
            neigh = graph.subgraph(neigh)
            neigh = nx.convert_node_labels_to_integers(neigh)
            neigh.add_edge(0, 0)
            neighs.append(neigh)
            if args.node_anchored:
                anchors.append(0)   # 标签转换后 0 号节点即为 anchor

    return neighs, anchors


def _embed_neighborhoods(neighs, anchors, model, args):
    """批量推理邻域嵌入。

    将邻域列表按 batch_size 切分，逐批送入模型，结果移至 CPU 并收集。

    返回：
        embs: List[Tensor]，每个元素形状为 (batch_size, hidden_dim)。
    """
    embs = []
    if len(neighs) % args.batch_size != 0:
        print("WARNING: number of graphs not multiple of batch size")
    for i in range(len(neighs) // args.batch_size):
        top = (i + 1) * args.batch_size
        with torch.no_grad():
            batch = utils.batch_nx_graphs(
                neighs[i * args.batch_size:top],
                anchors=anchors if args.node_anchored else None)
            emb = model.emb_model(batch)
            emb = emb.to(torch.device("cpu"))
        embs.append(emb)
    return embs


def _make_search_agent(model, graphs, embs, args):
    """按 args.search_strategy 创建对应的搜索代理实例。

    通过 create_search_agent 工厂函数调度，
    新增策略只需在 search_agents.AGENT_REGISTRY 中注册，此处无需修改。
    """
    common_kwargs = dict(
        node_anchored=args.node_anchored,
        analyze=args.analyze,
        out_batch_size=args.out_batch_size,
        frontier_top_k=args.frontier_top_k,
    )
    if args.search_strategy == "mcts":
        assert args.method_type == "order", \
            "MCTS 策略仅支持 order 类型模型"
        return create_search_agent(
            "mcts",
            args.min_pattern_size, args.max_pattern_size,
            model, graphs, embs,
            **common_kwargs)
    else:
        return create_search_agent(
            args.search_strategy,
            args.min_pattern_size, args.max_pattern_size,
            model, graphs, embs,
            model_type=args.method_type,
            **common_kwargs)


def _visualize_patterns(out_graphs, args):
    """将输出模式绘制并保存为 PNG / PDF 文件。"""
    count_by_size = defaultdict(int)
    for pattern in out_graphs:
        if args.node_anchored:
            colors = ["red"] + ["blue"] * (len(pattern) - 1)
            nx.draw(pattern, node_color=colors, with_labels=True)
        else:
            nx.draw(pattern)
        base = "plots/cluster/{}-{}".format(
            len(pattern), count_by_size[len(pattern)])
        print("Saving {}.png".format(base))
        plt.savefig("{}.png".format(base))
        plt.savefig("{}.pdf".format(base))
        plt.close()
        count_by_size[len(pattern)] += 1


def _save_patterns(out_graphs, args):
    """将输出模式序列化到 args.out_path。"""
    if not os.path.exists("results"):
        os.makedirs("results")
    with open(args.out_path, "wb") as f:
        pickle.dump(out_graphs, f)


# ---------------------------------------------------------------------------
# 主流程编排
# ---------------------------------------------------------------------------

def pattern_growth(dataset, task, args):
    """SPMiner 主流程：采样 -> 嵌入 -> 搜索 -> 输出。

    作为薄壳编排函数，依次调用各 pipeline 子函数：
    1. _load_mining_model  —— 加载模型权重
    2. _prepare_graphs     —— 数据集转 nx.Graph 列表
    3. _sample_neighborhoods —— 邻域采样
    4. _embed_neighborhoods  —— 批量嵌入
    5. _make_search_agent  —— 创建搜索代理
    6. agent.run_search    —— 执行搜索
    7. _visualize_patterns —— 可视化
    8. _save_patterns      —— 序列化结果
    """
    model = _load_mining_model(args)

    n_graphs = len(dataset[0]) if isinstance(dataset, tuple) else len(dataset)
    print(n_graphs, "graphs")
    print("search strategy:", args.search_strategy)

    graphs = _prepare_graphs(dataset, task, args)

    start_time = time.time()
    neighs, anchors = _sample_neighborhoods(graphs, args)

    embs = _embed_neighborhoods(neighs, anchors, model, args)
    if args.analyze:
        embs_np = torch.stack(embs).numpy()
        plt.scatter(embs_np[:, 0], embs_np[:, 1], label="node neighborhood")

    agent = _make_search_agent(model, graphs, embs, args)
    out_graphs = agent.run_search(args.n_trials)

    elapsed = time.time() - start_time
    print(elapsed, "TOTAL TIME")
    x = int(elapsed)
    print(x // 60, "mins", x % 60, "secs")

    _visualize_patterns(out_graphs, args)
    _save_patterns(out_graphs, args)


# ---------------------------------------------------------------------------
# 数据集加载注册表（仅供 main() 使用）
# 新增数据集只需在此添加一个加载函数并注册，main() 无需修改。
# ---------------------------------------------------------------------------

def _load_tu(name, root, task="graph"):
    return TUDataset(root=root, name=name), task


# 每个条目：dataset_key -> callable(args) -> (dataset, task)
_DECODER_DATASET_LOADERS = {
    "enzymes":      lambda a: (_load_tu("ENZYMES",      "/tmp/ENZYMES")),
    "cox2":         lambda a: (_load_tu("COX2",         "/tmp/cox2")),
    "reddit-binary":lambda a: (_load_tu("REDDIT-BINARY","/tmp/REDDIT-BINARY")),
    "dblp":         lambda a: (TUDataset(root="/tmp/dblp", name="DBLP_v1"),
                                "graph-truncate"),
    "coil":         lambda a: (_load_tu("COIL-DEL",     "/tmp/coil")),
    "ppi":          lambda a: (PPI(root="/tmp/PPI"),     "graph"),
    "ppi-syn":      lambda a: (make_ppi_syn_dataset(),   "graph"),
    "facebook":     lambda a: ([utils.load_snap_edgelist(
                                    "data/facebook_combined.txt")], "graph"),
    "as-733":       lambda a: ([utils.load_snap_edgelist(
                                    "data/as20000102.txt")], "graph"),
    "as20000102":   lambda a: ([utils.load_snap_edgelist(
                                    "data/as20000102.txt")], "graph"),
}


def _load_decoder_dataset(args):
    """按 args.dataset 加载数据集，返回 (dataset, task)。

    对于无法通过注册表命中的特殊数据集（roadnet-*、diseasome 等），
    保留原有加载逻辑作为 fallback。
    """
    name = args.dataset

    if name in _DECODER_DATASET_LOADERS:
        return _DECODER_DATASET_LOADERS[name](args)

    if name.startswith("plant-"):
        size = int(name.split("-")[-1])
        return make_plant_dataset(size), "graph"

    if name.startswith("roadnet-"):
        graph = nx.Graph()
        with open("data/{}.txt".format(name), "r") as f:
            for row in f:
                if not row.startswith("#"):
                    a, b = row.split("\t")
                    graph.add_edge(int(a), int(b))
        return [graph], "graph"

    if name in ("diseasome", "usroads", "mn-roads", "infect"):
        fn = {
            "diseasome": "bio-diseasome.mtx",
            "usroads":   "road-usroads.mtx",
            "mn-roads":  "mn-roads.mtx",
            "infect":    "infect-dublin.edges",
        }
        graph = nx.Graph()
        with open("data/{}".format(fn[name]), "r") as f:
            for line in f:
                if not line.strip():
                    continue
                a, b = line.strip().split(" ")
                graph.add_edge(int(a), int(b))
        return [graph], "graph"

    raise ValueError("未识别的数据集: {}".format(name))


def main():
    """解码器 CLI 入口。

    负责：
    - 组合 encoder/decoder 参数；
    - 通过 _load_decoder_dataset 读取数据集；
    - 调用 pattern_growth 执行完整挖掘。
    """
    if not os.path.exists("plots/cluster"):
        os.makedirs("plots/cluster")

    parser = argparse.ArgumentParser(description='解码器参数')
    parse_encoder(parser)
    parse_decoder(parser)
    args = parser.parse_args()

    print("Using dataset {}".format(args.dataset))
    dataset, task = _load_decoder_dataset(args)
    pattern_growth(dataset, task, args)


if __name__ == '__main__':
    main()
