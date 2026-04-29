"""SPMiner 挖掘流水线。

封装 pattern_growth() 的职责：采样→嵌入→搜索→可视化→序列化。
"""

from __future__ import annotations

import os
import pickle
import random
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

import torch_geometric.utils as pyg_utils

from src.core import utils
from src.core import CoreFacade
from src.subgraph_mining.search_agents import GreedySearchAgent, MCTSSearchAgent
from src.logger import info, section

__all__ = ["PatternGrowthPipeline"]


class PatternGrowthPipeline:
    """SPMiner 挖掘流水线：采样邻域 → 嵌入编码 → 模式搜索 → 结果输出。"""

    def __init__(self, args, model, dataset, task):
        self.args = args
        self.model = model
        self.dataset = dataset
        self.task = task

    def run(self):
        """完整流水线入口。"""
        start_time = time.time()
        graphs = self._prepare_graphs()
        neighs = self._sample_neighborhoods(graphs)
        embs = self._encode_embeddings(neighs, graphs)
        out_graphs = self._run_search(graphs, embs)
        elapsed = time.time() - start_time
        info("Total time: {:.1f}s ({:.1f}min)".format(elapsed, elapsed / 60))
        self._visualize_patterns(out_graphs)
        self._serialize_patterns(out_graphs)
        self._write_manifest(out_graphs)
        return out_graphs

    def _prepare_graphs(self):
        """统一数据集为 networkx.Graph 列表。"""
        dataset = self.dataset
        if self.task == "graph-labeled":
            dataset, labels = dataset
        else:
            labels = None

        graphs = []
        for i, graph in enumerate(dataset):
            if self.task == "graph-labeled" and labels[i] != 0:
                continue
            if self.task == "graph-truncate" and i >= 1000:
                break
            if not isinstance(graph, nx.Graph):
                graph = pyg_utils.to_networkx(graph).to_undirected()
            graphs.append(graph)

        info("Dataset: {} graphs".format(len(graphs)))
        info("Search strategy: {}".format(self.args.search_strategy))
        if self.task == "graph-labeled":
            info("Using label 0")
        return graphs

    def _sample_neighborhoods(self, graphs):
        """构建候选邻域集合。"""
        args = self.args
        if args.use_whole_graphs:
            info("Using whole graphs (no sampling)")
            return graphs

        section("邻域采样")
        neighs = []
        anchors = []

        if args.sample_method == "radial":
            for i, graph in enumerate(graphs):
                if len(graphs) > 100 or i % 10 == 0:
                    info("Radial sampling: graph {}/{} ({} nodes)".format(
                        i, len(graphs), len(graph)))
                for node in graph.nodes:
                    neigh = list(nx.single_source_shortest_path_length(
                        graph, node, cutoff=args.radius).keys())
                    if args.subgraph_sample_size != 0:
                        neigh = random.sample(neigh, min(len(neigh), args.subgraph_sample_size))
                    if len(neigh) > 1:
                        neigh = graph.subgraph(neigh)
                        if args.subgraph_sample_size != 0:
                            neigh = neigh.subgraph(max(nx.connected_components(neigh), key=len))
                        neigh = nx.convert_node_labels_to_integers(neigh)
                        neigh.add_edge(0, 0)
                        neighs.append(neigh)

        elif args.sample_method == "tree":
            for _ in tqdm(range(args.n_neighborhoods)):
                graph, nodes = utils.sample_neigh(
                    graphs,
                    random.randint(args.min_neighborhood_size, args.max_neighborhood_size),
                )
                neigh = graph.subgraph(nodes)
                neigh = nx.convert_node_labels_to_integers(neigh)
                neigh.add_edge(0, 0)
                neighs.append(neigh)
                if args.node_anchored:
                    anchors.append(0)

        return neighs

    def _encode_embeddings(self, neighs, graphs):
        """批量编码邻域嵌入。"""
        args = self.args
        section("嵌入编码")
        embs = []
        for i in range(0, len(neighs), args.batch_size):
            batch_neighs = neighs[i:i + args.batch_size]
            with torch.no_grad():
                batch = utils.batch_nx_graphs(
                    batch_neighs,
                    anchors=None,  # anchors handled internally if node_anchored
                )
                emb = self.model.emb_model(batch)
            embs.append(emb)

        if args.analyze:
            embs_np = torch.stack(embs).numpy()
            plt.scatter(embs_np[:, 0], embs_np[:, 1], label="node neighborhood")

        return embs

    def _run_search(self, graphs, embs):
        """调用搜索代理进行频繁子图挖掘。"""
        args = self.args
        section("模式搜索")

        if args.search_strategy == "mcts":
            agent = MCTSSearchAgent(
                args.min_pattern_size, args.max_pattern_size,
                self.model, graphs, embs,
                node_anchored=args.node_anchored,
                analyze=args.analyze,
                out_batch_size=args.out_batch_size,
                frontier_top_k=args.frontier_top_k,
                analysis_out_path=args.analysis_out_path,
                analysis_plot_path=args.analysis_plot_path,
            )
        elif args.search_strategy == "greedy":
            agent = GreedySearchAgent(
                args.min_pattern_size, args.max_pattern_size,
                self.model, graphs, embs,
                node_anchored=args.node_anchored,
                analyze=args.analyze,
                model_type=args.method_type,
                out_batch_size=args.out_batch_size,
                frontier_top_k=args.frontier_top_k,
                max_steps=args.n_trials,
                analysis_out_path=args.analysis_out_path,
                analysis_plot_path=args.analysis_plot_path,
            )
        else:
            raise ValueError(f"Unknown search strategy: {args.search_strategy}")

        return agent.run_search(args.n_trials)

    def _visualize_patterns(self, patterns):
        """输出模式可视化图像。"""
        args = self.args
        count_by_size = defaultdict(int)
        for pattern in patterns:
            if args.node_anchored:
                colors = ["red"] + ["blue"] * (len(pattern) - 1)
                nx.draw(pattern, node_color=colors, with_labels=True)
            else:
                nx.draw(pattern)
            png_path = os.path.join(args.pattern_plot_dir, "{}-{}.png".format(
                len(pattern), count_by_size[len(pattern)]))
            pdf_path = os.path.join(args.pattern_plot_dir, "{}-{}.pdf".format(
                len(pattern), count_by_size[len(pattern)]))
            plt.savefig(png_path)
            plt.savefig(pdf_path)
            plt.close()
            info("Pattern saved → {}".format(png_path))
            count_by_size[len(pattern)] += 1

    def _serialize_patterns(self, patterns):
        """序列化模式结果。"""
        with open(self.args.out_path, "wb") as f:
            pickle.dump(patterns, f)
        info("Patterns saved → {}".format(self.args.out_path))

    def _write_manifest(self, patterns):
        """写入 manifest 元信息。"""
        from src.core.artifacts import write_manifest
        args = self.args
        write_manifest(
            os.path.join(args.artifact_dir, "manifest.json"),
            args,
            outputs={
                "patterns": args.out_path,
                "pattern_plot_dir": args.pattern_plot_dir,
                "analysis_pickle": args.analysis_out_path,
                "analysis_plot": args.analysis_plot_path,
            },
            task="mining",
        )
