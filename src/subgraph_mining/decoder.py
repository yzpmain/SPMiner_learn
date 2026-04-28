import argparse
import time
import os

import numpy as np
import torch
from tqdm import tqdm

import torch_geometric.utils as pyg_utils

from src.core import dataset_registry
from src.core import models
from src.core import utils
from src.core import combined_syn
from src.subgraph_mining.config import parse_decoder
from src.subgraph_matching.config import parse_encoder
from src.subgraph_mining.search_agents import GreedySearchAgent, MCTSSearchAgent
from src.logger import RunLogger, info, section

import matplotlib.pyplot as plt

import random
from collections import defaultdict
import networkx as nx
import pickle

from src.core.cli import setup_runtime

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

def pattern_growth(dataset, task, args):
    """SPMiner 主流程：采样 -> 嵌入 -> 搜索 -> 输出。

    该函数是挖掘入口中的核心逻辑：
    1. 加载匹配模型（作为频繁性评分器）；
    2. 构建候选邻域集合；
    3. 批量编码邻域嵌入；
    4. 调用 Greedy/MCTS 搜索代理；
    5. 保存可视化与序列化结果。
    """
    # 初始化模型
    if args.method_type == "mlp":
        model = models.BaselineMLP(1, args.hidden_dim, args)
    else:
        model = models.OrderEmbedder(1, args.hidden_dim, args)
    model.to(utils.get_device())
    model.eval()
    model.load_state_dict(torch.load(args.model_path,
        map_location=utils.get_device()))

    if task == "graph-labeled":
        dataset, labels = dataset

    start_time = time.time()
    # 将不同来源数据统一为 networkx.Graph 列表，
    # 便于后续采样和搜索器逻辑复用。
    neighs = []
    info("Dataset: {} graphs".format(len(dataset)))
    info("Search strategy: {}".format(args.search_strategy))
    if task == "graph-labeled":
        info("Using label 0")
    graphs = []
    for i, graph in enumerate(dataset):
        if task == "graph-labeled" and labels[i] != 0: continue
        if task == "graph-truncate" and i >= 1000: break
        if not type(graph) == nx.Graph:
            graph = pyg_utils.to_networkx(graph).to_undirected()
        graphs.append(graph)
    if args.use_whole_graphs:
        neighs = graphs
    else:
        section("邻域采样")
        anchors = []
        if args.sample_method == "radial":
            for i, graph in enumerate(graphs):
                if len(dataset) > 100 or i % 10 == 0:
                    info("Radial sampling: graph {}/{} ({} nodes)".format(
                        i, len(graphs), len(graph)))
                for j, node in enumerate(graph.nodes):
                    if args.use_whole_graphs:
                        neigh = graph.nodes
                    else:
                        neigh = list(nx.single_source_shortest_path_length(graph,
                            node, cutoff=args.radius).keys())
                        if args.subgraph_sample_size != 0:
                            neigh = random.sample(neigh, min(len(neigh),
                                args.subgraph_sample_size))
                    if len(neigh) > 1:
                        neigh = graph.subgraph(neigh)
                        if args.subgraph_sample_size != 0:
                            neigh = neigh.subgraph(max(
                                nx.connected_components(neigh), key=len))
                        neigh = nx.convert_node_labels_to_integers(neigh)
                        neigh.add_edge(0, 0)
                        neighs.append(neigh)
        elif args.sample_method == "tree":
            # tree 采样：每次从数据集中随机抽图，再扩展一个连通邻域。
            for j in tqdm(range(args.n_neighborhoods)):
                graph, neigh = utils.sample_neigh(graphs,
                    random.randint(args.min_neighborhood_size,
                        args.max_neighborhood_size))
                neigh = graph.subgraph(neigh)
                neigh = nx.convert_node_labels_to_integers(neigh)
                neigh.add_edge(0, 0)
                neighs.append(neigh)
                if args.node_anchored:
                    anchors.append(0)   # after converting labels, 0 will be anchor

    section("嵌入编码")
    embs = []
    for i in range(0, len(neighs), args.batch_size):
        batch_neighs = neighs[i:i+args.batch_size]
        with torch.no_grad():
            batch = utils.batch_nx_graphs(batch_neighs,
                anchors=anchors if args.node_anchored else None)
            emb = model.emb_model(batch)
        embs.append(emb)

    if args.analyze:
        embs_np = torch.stack(embs).numpy()
        plt.scatter(embs_np[:,0], embs_np[:,1], label="node neighborhood")

    # 搜索阶段：把候选邻域嵌入交给策略代理，输出频繁模式。
    section("模式搜索")
    if args.search_strategy == "mcts":
        assert args.method_type == "order"
        agent = MCTSSearchAgent(args.min_pattern_size, args.max_pattern_size,
            model, graphs, embs, node_anchored=args.node_anchored,
            analyze=args.analyze, out_batch_size=args.out_batch_size,
            frontier_top_k=args.frontier_top_k)
    elif args.search_strategy == "greedy":
        agent = GreedySearchAgent(args.min_pattern_size, args.max_pattern_size,
            model, graphs, embs, node_anchored=args.node_anchored,
            analyze=args.analyze, model_type=args.method_type,
            out_batch_size=args.out_batch_size,
            frontier_top_k=args.frontier_top_k, max_steps=args.n_trials)
    out_graphs = agent.run_search(args.n_trials)
    elapsed = time.time() - start_time
    info("Total time: {:.1f}s ({:.1f}min)".format(elapsed, elapsed / 60))

    # 可视化输出模式：每种大小按出现顺序保存图像。
    count_by_size = defaultdict(int)
    for pattern in out_graphs:
        if args.node_anchored:
            colors = ["red"] + ["blue"]*(len(pattern)-1)
            nx.draw(pattern, node_color=colors, with_labels=True)
        else:
            nx.draw(pattern)
        info("Pattern saved → plots/cluster/{}-{}.png".format(
            len(pattern), count_by_size[len(pattern)]))
        plt.savefig("plots/cluster/{}-{}.png".format(len(pattern),
            count_by_size[len(pattern)]))
        plt.savefig("plots/cluster/{}-{}.pdf".format(len(pattern),
            count_by_size[len(pattern)]))
        plt.close()
        count_by_size[len(pattern)] += 1

    if not os.path.exists("results"):
        os.makedirs("results")
    with open(args.out_path, "wb") as f:
        pickle.dump(out_graphs, f)

def main():
    """解码器 CLI 入口。

    负责：
    - 组合 encoder/decoder 参数；
    - 读取数据集；
    - 调用 pattern_growth 执行完整挖掘。
    """
    parser = argparse.ArgumentParser(description='解码器参数')
    parse_encoder(parser)
    parse_decoder(parser)
    args = parser.parse_args()

    setup_runtime(args)

    with RunLogger(args):
        if not os.path.exists("plots/cluster"):
            os.makedirs("plots/cluster")

        section("数据加载")
        # 统一命名与入口校验：所有挖掘数据集都从注册中心加载。
        normalized_dataset = dataset_registry.validate_dataset_name(args.dataset, "mining")
        args.dataset = normalized_dataset
        info("Using dataset {}".format(args.dataset))
        dataset, task = dataset_registry.load_dataset_for_stage(args.dataset, "mining")

        pattern_growth(dataset, task, args)

if __name__ == '__main__':
    main()

