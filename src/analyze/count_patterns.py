import argparse
import copy
import csv
import json
import os
import traceback

import numpy as np

from torch_geometric.datasets import TUDataset
import torch_geometric.utils as pyg_utils

from src.core import dataset_registry
from src.core import utils
from src.logger import RunLogger, info, warning, section, progress

from multiprocessing import Pool
import random
import networkx as nx
import networkx.algorithms.isomorphism as iso
import pickle

try:
    import orca  # type: ignore[import-not-found]
except ImportError:
    orca = None

def arg_parse():
    parser = argparse.ArgumentParser(description='统计图中的图元')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--queries_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--n_workers', type=int)
    parser.add_argument('--count_method', type=str)
    parser.add_argument('--baseline', type=str)
    parser.add_argument('--max_queries', type=int,
        help='仅统计前 max_queries 个模式，0 表示使用全部')
    parser.add_argument('--progress_every', type=int,
        help='每处理多少个任务打印一次进度，0 表示关闭')
    parser.add_argument('--node_anchored', action="store_true")
    parser.set_defaults(dataset="enzymes",
                        queries_path="results/out-patterns.p",
                        out_path="results/counts.json",
                        n_workers=4,
                        count_method="bin",
                        baseline="none",
                        max_queries=0,
                        progress_every=1000)
                        #node_anchored=True)
    return parser.parse_args()
def gen_baseline_queries(queries, targets, method="mfinder", node_anchored=False):
    if method == "mfinder":
        return utils.gen_baseline_queries_mfinder(queries, targets,
                                                  node_anchored=node_anchored)
    elif method == "rand-esu":
        return utils.gen_baseline_queries_rand_esu(queries, targets,
                                                   node_anchored=node_anchored)
    neighs = []
    max_attempts = 1000
    for i, query in enumerate(queries):
        found = False
        if len(query) == 0:
            neighs.append(query)
            found = True
        n_attempts = 0
        while not found and n_attempts < max_attempts:
            n_attempts += 1
            if method == "radial":
                graph = random.choice(targets)
                node = random.choice(list(graph.nodes))
                neigh = list(nx.single_source_shortest_path_length(graph, node,
                                                                  cutoff=3).keys())
                neigh = graph.subgraph(neigh)
                neigh = neigh.subgraph(list(sorted(nx.connected_components(
                    neigh), key=len))[-1])
                neigh = nx.convert_node_labels_to_integers(neigh)
                if len(neigh) == len(query):
                    neighs.append(neigh)
                    found = True
            elif method == "tree":
                graph = random.choice(targets)
                start_node = random.choice(list(graph.nodes))
                neigh = [start_node]
                frontier = list(set(graph.neighbors(start_node)) - set(neigh))
                while len(neigh) < len(query) and frontier:
                    new_node = random.choice(list(frontier))
                    neigh.append(new_node)
                    frontier += list(graph.neighbors(new_node))
                    frontier = [x for x in frontier if x not in neigh]
                if len(neigh) == len(query):
                    neigh = graph.subgraph(neigh)
                    neigh = nx.convert_node_labels_to_integers(neigh)
                    neighs.append(neigh)
                    found = True
        if not found:
            warning("Query {} (size={}) not matched after {} attempts".format(
                i, len(query), max_attempts))
    return neighs


def preprocess_query(query, node_anchored):
    query = query.copy()
    query.remove_edges_from(nx.selfloop_edges(query))
    degree_seq = tuple(sorted((d for _, d in query.degree()), reverse=True))
    info = {
        "graph": query,
        "n_nodes": query.number_of_nodes(),
        "n_edges": query.number_of_edges(),
        "degree_seq": degree_seq,
        "anchor_count": sum(1 for _, data in query.nodes(data=True)
                            if data.get("anchor", 0) == 1) if node_anchored else 0,
    }
    return info


def preprocess_target(target, node_anchored):
    target = copy.deepcopy(target)
    target.remove_edges_from(nx.selfloop_edges(target))
    n_nodes = target.number_of_nodes()
    degree_seq = tuple(sorted((d for _, d in target.degree()), reverse=True))
    # 修复：节点锚定时，目标图每个节点都可作为锚点，故 anchor_count = n_nodes
    anchor_count = n_nodes if node_anchored else 0
    return {
        "graph": target,
        "n_nodes": n_nodes,
        "n_edges": target.number_of_edges(),
        "degree_seq": degree_seq,
        "anchor_count": anchor_count,
    }


def dedup_isomorphic_queries(queries, node_anchored=False):
    """对查询图做同构去重，并返回原索引到去重索引的映射。"""
    unique_queries = []
    orig_to_unique = []
    node_match = (iso.categorical_node_match(["anchor"], [0])
                  if node_anchored else None)

    for query in queries:
        matched_idx = None
        for idx, uniq in enumerate(unique_queries):
            if query["n_nodes"] != uniq["n_nodes"]:
                continue
            if query["n_edges"] != uniq["n_edges"]:
                continue
            if query["degree_seq"] != uniq["degree_seq"]:
                continue
            if node_anchored and query["anchor_count"] != uniq["anchor_count"]:
                continue
            if nx.is_isomorphic(query["graph"], uniq["graph"],
                                node_match=node_match):
                matched_idx = idx
                break
        if matched_idx is None:
            unique_queries.append(query)
            orig_to_unique.append(len(unique_queries) - 1)
        else:
            orig_to_unique.append(matched_idx)
    return unique_queries, orig_to_unique


def _count_one_pair(query_info, target_info, method, node_anchored):
    """计算一个(查询, 目标)对在给定方法下的计数值。"""
    query = query_info["graph"]
    target = target_info["graph"]  # 已为副本，可直接修改

    # 必要性剪枝
    if query_info["n_nodes"] > target_info["n_nodes"]:
        return 0
    if query_info["n_edges"] > target_info["n_edges"]:
        return 0
    q_deg = query_info["degree_seq"]
    t_deg = target_info["degree_seq"]
    if len(q_deg) > len(t_deg):
        return 0
    if any(q > t for q, t in zip(q_deg, t_deg)):
        return 0
    # 节点锚定：目标图至少要有查询图所需的那么多个节点来放置锚点
    if node_anchored and query_info["anchor_count"] > target_info["n_nodes"]:
        return 0

    # NetworkX 回退
    count = 0
    if method == "bin":
        if node_anchored:
            prev_anchor = None
            for anchor in target.nodes:
                if prev_anchor is not None:
                    target.nodes[prev_anchor]["anchor"] = 0
                target.nodes[anchor]["anchor"] = 1
                prev_anchor = anchor
                matcher = iso.GraphMatcher(target, query,
                                           node_match=iso.categorical_node_match(["anchor"], [0]))
                if matcher.subgraph_is_isomorphic():
                    count += 1
            if prev_anchor is not None:
                target.nodes[prev_anchor]["anchor"] = 0
        else:
            matcher = iso.GraphMatcher(target, query)
            count = int(matcher.subgraph_is_isomorphic())
    elif method == "freq":
        if node_anchored:
            seen = set()
            prev_anchor = None
            for anchor in target.nodes:
                if prev_anchor is not None:
                    target.nodes[prev_anchor]["anchor"] = 0
                target.nodes[anchor]["anchor"] = 1
                prev_anchor = anchor
                matcher = iso.GraphMatcher(target, query,
                                           node_match=iso.categorical_node_match(["anchor"], [0]))
                for match in matcher.subgraph_isomorphisms_iter():
                    seen.add(tuple(sorted(match.keys())))
            if prev_anchor is not None:
                target.nodes[prev_anchor]["anchor"] = 0
            count = len(seen)
        else:
            seen = set()
            matcher = iso.GraphMatcher(target, query)
            for match in matcher.subgraph_isomorphisms_iter():
                seen.add(tuple(sorted(match.keys())))
            count = len(seen)
    else:
        raise ValueError(f"Unknown count method: {method}")
    return count

# 多进程共享状态：通过 Pool(initializer=) 分发到各 Worker
_worker_targets = None
_worker_method = None
_worker_node_anchored = None


def init_worker(targets, method, node_anchored):
    global _worker_targets, _worker_method, _worker_node_anchored
    _worker_targets = targets
    _worker_method = method
    _worker_node_anchored = node_anchored


def count_graphlets_helper(args):
    i, q_info = args
    total = 0
    for t_info in _worker_targets:
        total += _count_one_pair(q_info, t_info, _worker_method,
                                 _worker_node_anchored)
    return i, total


def count_graphlets(queries, targets, n_workers=1, method="bin",
                    node_anchored=False,
                    progress_every=1000):
    info("Queries: {}, targets: {}".format(len(queries), len(targets)))

    queries = [preprocess_query(query, node_anchored) for query in queries]
    targets = [preprocess_target(target, node_anchored) for target in targets]

    query_to_unique = None
    work_queries = queries
    if method == "freq":
        work_queries, query_to_unique = dedup_isomorphic_queries(
            queries, node_anchored=node_anchored)
        if len(work_queries) != len(queries):
            info("Dedup: {} → {} queries".format(len(queries), len(work_queries)))

    # 每个 Worker 持有全部 targets，任务只需传递 (i, q_info)
    def task_generator():
        for i, q_info in enumerate(work_queries):
            yield (i, q_info)

    n_matches = [0] * len(work_queries)
    total = len(work_queries)
    n_done = 0

    with Pool(processes=n_workers,
              initializer=init_worker,
              initargs=(targets, method, node_anchored)) as pool:
        for i, n in pool.imap_unordered(count_graphlets_helper, task_generator(),
                                        chunksize=1):
            n_matches[i] = n
            n_done += 1
            if progress_every > 0:
                if n_done % progress_every == 0 or n_done == total:
                    progress(n_done, total, query=i, count=n)

    if query_to_unique is None:
        return n_matches
    return [n_matches[idx] for idx in query_to_unique]


def count_exact(targets, args):
    if orca is None:
        raise ImportError("orca 未安装，无法使用 baseline=exact")
    warning("orca is only applicable for node-anchored case")
    n_matches_baseline = np.zeros(73)
    for target in targets:
        counts = np.array(orca.orbit_counts("node", 5, target))
        if args.count_method == "bin":
            counts = np.sign(counts)
        counts = np.sum(counts, axis=0)
        n_matches_baseline += counts
    # 不包含尺寸 < 5 的模式
    n_matches_baseline = list(n_matches_baseline)[15:]
    counts5 = []
    num5 = 10
    for x in list(sorted(n_matches_baseline, reverse=True))[:num5]:
        counts5.append(x)
    info("Baseline size-5 avg (log10): {:.4f}".format(
        np.mean(np.log10(np.maximum(counts5, 1e-12)))))

    atlas = [g for g in nx.graph_atlas_g()[1:] if nx.is_connected(g) and len(g) == 6]
    queries = []
    for g0 in atlas:
        for v in g0.nodes:
            gc = g0.copy()
            nx.set_node_attributes(gc, 0, name="anchor")
            gc.nodes[v]["anchor"] = 1
            is_dup = False
            for g2 in queries:
                if nx.is_isomorphic(gc, g2, node_match=(lambda a, b: a["anchor"] == b["anchor"]) if args.node_anchored else None):
                    is_dup = True
                    break
            if not is_dup:
                queries.append(gc)
    info("Unique size-6 queries: {}".format(len(queries)))
    n_matches_baseline = count_graphlets(queries, targets,
                                         n_workers=args.n_workers,
                                         method=args.count_method,
                                         node_anchored=args.node_anchored,
                                         progress_every=args.progress_every)
    counts6 = []
    num6 = 20
    for x in list(sorted(n_matches_baseline, reverse=True))[:num6]:
        counts6.append(x)
    info("Baseline size-6 avg (log10): {:.4f}".format(
        np.mean(np.log10(np.maximum(counts6, 1e-12)))))
    return counts5 + counts6


if __name__ == "__main__":
    args = arg_parse()
    with RunLogger(args):
        section("模式计数")
        info("Using {} workers".format(args.n_workers))
        info("Baseline: {}".format(args.baseline))

        # 检查必需文件
        if args.dataset != "analyze" and not os.path.exists(args.queries_path):
            raise FileNotFoundError(f"Queries file not found: {args.queries_path}")

        if args.dataset == "analyze":
            with open("results/analyze.p", "rb") as f:
                cand_patterns, _ = pickle.load(f)
                queries = [q for score, q in cand_patterns[10]][:200]
            dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        else:
            # 统一命名与入口校验：复用注册中心与挖掘/训练保持一致。
            normalized_dataset = dataset_registry.validate_dataset_name(args.dataset, "count")
            args.dataset = normalized_dataset
            dataset, _ = dataset_registry.load_dataset_for_stage(args.dataset, "count")

        targets = []
        for i in range(len(dataset)):
            graph = dataset[i]
            if not isinstance(graph, nx.Graph):
                graph = pyg_utils.to_networkx(dataset[i]).to_undirected()
            targets.append(graph)

        if args.dataset != "analyze":
            with open(args.queries_path, "rb") as f:
                queries = pickle.load(f)
            if args.max_queries > 0:
                queries = queries[:args.max_queries]

        if args.baseline == "exact":
            n_matches_baseline = count_exact(targets, args)
            queries = queries[:len(n_matches_baseline)]
            query_lens = [len(q) for q in queries]
            n_matches = count_graphlets(queries, targets,
                                        n_workers=args.n_workers,
                                        method=args.count_method,
                                        node_anchored=args.node_anchored,
                                        progress_every=args.progress_every)
        elif args.baseline == "none":
            query_lens = [len(q) for q in queries]
            n_matches = count_graphlets(queries, targets,
                                        n_workers=args.n_workers,
                                        method=args.count_method,
                                        node_anchored=args.node_anchored,
                                        progress_every=args.progress_every)
        else:
            baseline_queries = gen_baseline_queries(queries, targets,
                                                    node_anchored=args.node_anchored,
                                                    method=args.baseline)
            query_lens = [len(q) for q in baseline_queries]
            n_matches = count_graphlets(baseline_queries, targets,
                                        n_workers=args.n_workers,
                                        method=args.count_method,
                                        node_anchored=args.node_anchored,
                                        progress_every=args.progress_every)

        with open(args.out_path, "w") as f:
            json.dump((query_lens, n_matches, []), f)