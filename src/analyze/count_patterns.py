import argparse
import csv
import json
import os
import traceback
import logging

import numpy as np

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import PPI
import torch_geometric.utils as pyg_utils

from src.core import utils
from src.subgraph_mining import decoder

from multiprocessing import Pool
import random
import networkx as nx
import networkx.algorithms.isomorphism as iso
import pickle

try:
    import orca  # type: ignore[import-not-found]
except ImportError:
    orca = None

try:
    from orbitsi.search import FilterEngine, OrderEngine, SearchEngine  # type: ignore[import-not-found]
    import _evoke_cpp  # type: ignore[import-not-found]
    ORBITSI_AVAILABLE = True
except ImportError:
    FilterEngine = None
    OrderEngine = None
    SearchEngine = None
    _evoke_cpp = None
    ORBITSI_AVAILABLE = False

logger = logging.getLogger(__name__)


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
    parser.add_argument('--use_orbitsi', action="store_true",
        help='使用Orbitsi(orbit-filter + backtracking)加速同构搜索')
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


class _EvokeOrbitCounterNoConverter:
    """Minimal EVOKE counter that avoids OrbitSI package matrix-file dependency."""
    def __init__(self, graph, size=4):
        self.graph = graph
        self.size = size
        self.counts = None

    def _nx_to_cpp_graph(self):
        return {int(n): [int(nbr) for nbr in self.graph.neighbors(n)]
            for n in self.graph.nodes}

    def count_orbits(self):
        cpp_graph = self._nx_to_cpp_graph()
        self.counts = _evoke_cpp.evoke_count(cpp_graph, size=self.size,
            parallel=True)
        return self.counts

    def get_orbits(self, induced=False):
        if self.counts is None:
            self.count_orbits()
        sorted_nodes = sorted(self.counts)
        return np.array([self.counts[node] for node in sorted_nodes], dtype=int)


def _build_orbitsi_labeled_graph(graph, node_anchored=False, anchor_node=None):
    """Build a copy with integer node labels required by OrbitSI filters."""
    g = graph.copy()
    nx.set_node_attributes(g, 1, name="label")
    if node_anchored:
        for node, attrs in g.nodes(data=True):
            if attrs.get("anchor", 0) == 1:
                g.nodes[node]["label"] = 2
        if anchor_node is not None and anchor_node in g.nodes:
            g.nodes[anchor_node]["label"] = 2
    return g


def _make_target_key(match, query, node_anchored):
    """从 target->query 映射生成去重键（假设 match 为 {target_node: query_node}）。"""
    # Orbitsi / SearchEngine may return mappings in either direction:
    #  - query_node -> target_node, or
    #  - target_node -> query_node
    # Normalize to the sorted tuple of target nodes.
    query_nodes_set = set(query.nodes())
    keys = list(match.keys())
    # If keys look like query nodes, treat mapping as query->target
    if keys and keys[0] in query_nodes_set:
        target_nodes = tuple(sorted(match[q] for q in keys))
        if not node_anchored:
            return target_nodes
        anchor_query_nodes = {n for n, d in query.nodes(data=True) if d.get("anchor", 0) == 1}
        anchor_targets = tuple(sorted(match[q] for q in anchor_query_nodes if q in match))
        return (target_nodes, anchor_targets)
    # Otherwise assume mapping is target->query
    target_nodes = tuple(sorted(match.keys()))
    if not node_anchored:
        return target_nodes
    anchor_query_nodes = {n for n, d in query.nodes(data=True) if d.get("anchor", 0) == 1}
    anchor_targets = tuple(sorted(t for t, q in match.items() if q in anchor_query_nodes))
    return (target_nodes, anchor_targets)


def _orbitsi_match_count(query, target, method, node_anchored):
    """Return match count using OrbitSI pipeline. Raises on non-recoverable errors."""
    find_all = (method == "freq")
    count = 0

    if node_anchored:
        # 对于锚定情况，只需对目标图执行一次轨道分解
        query_labeled = _build_orbitsi_labeled_graph(query, node_anchored=True)
        base_target = target.copy()
        nx.set_node_attributes(base_target, 0, name="anchor")
        seen = set()
        for anchor in target.nodes:
            # 修改目标图锚点标签（原地修改本地副本）
            for n in base_target.nodes:
                base_target.nodes[n]["label"] = 2 if n == anchor else 1
            target_labeled = base_target  # 此时 base_target 已带 label 属性
            filter_engine = FilterEngine(data_graph=target_labeled,
                                         pattern_graph=query_labeled,
                                         orbit_counter_class=_EvokeOrbitCounterNoConverter,
                                         graphlet_size=4)
            pattern_orbits, candidate_sets, subgraph = filter_engine.run()
            if not candidate_sets:
                continue
            order_engine = OrderEngine(query_labeled, pattern_orbits)
            order, pivot = order_engine.run()
            search_engine = SearchEngine(subgraph, query_labeled,
                                         candidate_sets, order, pivot)
            if method == "bin":
                matches = search_engine.run(return_all=False)
                if matches:
                    count += 1
            else:  # freq
                matches = search_engine.run(return_all=True)
                for match in matches:
                    # 因锚点已通过标签过滤，match 中的目标节点自然满足锚点
                    seen.add(_make_target_key(match, query, node_anchored))
        if method == "freq":
            return len(seen)
        return count

    # 非锚定情况
    query_labeled = _build_orbitsi_labeled_graph(query, node_anchored=False)
    target_labeled = _build_orbitsi_labeled_graph(target, node_anchored=False)
    filter_engine = FilterEngine(data_graph=target_labeled,
                                 pattern_graph=query_labeled,
                                 orbit_counter_class=_EvokeOrbitCounterNoConverter,
                                 graphlet_size=4)
    pattern_orbits, candidate_sets, subgraph = filter_engine.run()
    if not candidate_sets:
        return 0
    order_engine = OrderEngine(query_labeled, pattern_orbits)
    order, pivot = order_engine.run()
    search_engine = SearchEngine(subgraph, query_labeled,
                                 candidate_sets, order, pivot)
    if method == "bin":
        matches = search_engine.run(return_all=False)
        return int(bool(matches))
    # freq
    seen = set()
    for match in search_engine.run(return_all=True):
        seen.add(_make_target_key(match, query, node_anchored))
    return len(seen)


def gen_baseline_queries(queries, targets, method="mfinder", node_anchored=False):
    if method == "mfinder":
        return utils.gen_baseline_queries_mfinder(queries, targets,
                                                  node_anchored=node_anchored)
    elif method == "rand-esu":
        return utils.gen_baseline_queries_rand_esu(queries, targets,
                                                   node_anchored=node_anchored)
    neighs = []
    for i, query in enumerate(queries):
        print(i)
        found = False
        if len(query) == 0:
            neighs.append(query)
            found = True
        while not found:
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
    return neighs


def preprocess_query(query, method, node_anchored):
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
    target = target.copy()
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


def _count_one_pair(query_info, target_info, method, node_anchored, use_orbitsi):
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

    # 尝试 OrbitSI 加速
    if use_orbitsi and ORBITSI_AVAILABLE and method in ("bin", "freq"):
        try:
            return _orbitsi_match_count(query, target, method, node_anchored)
        except Exception as e:
            logger.warning("OrbitSI failed for a query-target pair, falling back to NetworkX: %s", e)

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
                    seen.add(_make_target_key(match, query, node_anchored=True))
            if prev_anchor is not None:
                target.nodes[prev_anchor]["anchor"] = 0
            count = len(seen)
        else:
            seen = set()
            matcher = iso.GraphMatcher(target, query)
            for match in matcher.subgraph_isomorphisms_iter():
                seen.add(_make_target_key(match, query, node_anchored=False))
            count = len(seen)
    else:
        raise ValueError(f"Unknown count method: {method}")
    return count

# 多进程共享状态：通过 Pool(initializer=) 分发到各 Worker
_worker_targets = None
_worker_method = None
_worker_node_anchored = None
_worker_use_orbitsi = None


def init_worker(targets, method, node_anchored, use_orbitsi):
    global _worker_targets, _worker_method, _worker_node_anchored, _worker_use_orbitsi
    _worker_targets = targets
    _worker_method = method
    _worker_node_anchored = node_anchored
    _worker_use_orbitsi = use_orbitsi


def count_graphlets_helper(args):
    i, q_info = args
    total = 0
    for t_info in _worker_targets:
        total += _count_one_pair(q_info, t_info, _worker_method,
                                 _worker_node_anchored, _worker_use_orbitsi)
    return i, total


def count_graphlets(queries, targets, n_workers=1, method="bin",
                    node_anchored=False,
                    progress_every=1000, use_orbitsi=False):
    print(len(queries), len(targets))

    queries = [preprocess_query(query, method, node_anchored) for query in queries]
    targets = [preprocess_target(target, node_anchored) for target in targets]

    query_to_unique = None
    work_queries = queries
    if method == "freq":
        work_queries, query_to_unique = dedup_isomorphic_queries(
            queries, node_anchored=node_anchored)
        if len(work_queries) != len(queries):
            print("freq query dedup:", len(queries), "->", len(work_queries))

    # 每个 Worker 持有全部 targets，任务只需传递 (i, q_info)
    def task_generator():
        for i, q_info in enumerate(work_queries):
            yield (i, q_info)

    n_matches = [0] * len(work_queries)
    total = len(work_queries)
    n_done = 0

    with Pool(processes=n_workers,
              initializer=init_worker,
              initargs=(targets, method, node_anchored, use_orbitsi)) as pool:
        for i, n in pool.imap_unordered(count_graphlets_helper, task_generator(),
                                        chunksize=1):
            n_matches[i] = n
            n_done += 1
            if progress_every and progress_every > 0:
                if n_done % progress_every == 0 or n_done == total:
                    print(n_done, "/", total, "- query", i, "count", n, "      ", end="\r")
    print()

    unique_matches = [n_matches[i] for i in range(len(work_queries))]
    if query_to_unique is None:
        return unique_matches
    return [unique_matches[idx] for idx in query_to_unique]


def count_exact(queries, targets, args):
    if orca is None:
        raise ImportError("orca 未安装，无法使用 baseline=exact")
    print("警告：orca 仅适用于节点锚定情况")
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
        print(x)
        counts5.append(x)
    print("Average for size 5:", np.mean(np.log10(counts5)))

    atlas = [g for g in nx.graph_atlas_g()[1:] if nx.is_connected(g) and len(g) == 6]
    queries = []
    for g in atlas:
        for v in g.nodes:
            g = g.copy()
            nx.set_node_attributes(g, 0, name="anchor")
            g.nodes[v]["anchor"] = 1
            is_dup = False
            for g2 in queries:
                if nx.is_isomorphic(g, g2, node_match=(lambda a, b: a["anchor"] == b["anchor"]) if args.node_anchored else None):
                    is_dup = True
                    break
            if not is_dup:
                queries.append(g)
    print(len(queries))
    n_matches_baseline = count_graphlets(queries, targets,
                                         n_workers=args.n_workers,
                                         method=args.count_method,
                                         node_anchored=args.node_anchored,
                                         progress_every=args.progress_every,
                                         use_orbitsi=getattr(args, 'use_orbitsi', False))
    counts6 = []
    num6 = 20
    for x in list(sorted(n_matches_baseline, reverse=True))[:num6]:
        print(x)
        counts6.append(x)
    print("Average for size 6:", np.mean(np.log10(counts6)))
    return counts5 + counts6


if __name__ == "__main__":
    args = arg_parse()
    print("Using {} workers".format(args.n_workers))
    print("Baseline:", args.baseline)
    if args.use_orbitsi:
        if ORBITSI_AVAILABLE:
            print("Orbitsi backend: enabled (EVOKE)")
        else:
            print("Orbitsi backend: unavailable, fallback to NetworkX")

    # 检查必需文件
    if args.dataset != "analyze" and not os.path.exists(args.queries_path):
        raise FileNotFoundError(f"Queries file not found: {args.queries_path}")

    if args.dataset == 'syn':
        from src.core import combined_syn
        generator = combined_syn.get_generator([10])
        dataset = [generator.generate(size=10) for _ in range(10)]
    elif args.dataset == 'enzymes':
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    elif args.dataset == 'cox2':
        dataset = TUDataset(root='/tmp/cox2', name='COX2')
    elif args.dataset == 'reddit-binary':
        dataset = TUDataset(root='/tmp/REDDIT-BINARY', name='REDDIT-BINARY')
    elif args.dataset == 'coil':
        dataset = TUDataset(root='/tmp/COIL-DEL', name='COIL-DEL')
    elif args.dataset == 'ppi-pathways':
        graph = nx.Graph()
        with open("data/ppi-pathways.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                graph.add_edge(int(row[0]), int(row[1]))
        dataset = [graph]
    elif args.dataset == 'ppi':
        dataset = PPI(root='/tmp/PPI')
    elif args.dataset in ['diseasome', 'usroads', 'mn-roads', 'infect']:
        fn = {"diseasome": "bio-diseasome.mtx",
              "usroads": "road-usroads.mtx",
              "mn-roads": "mn-roads.mtx",
              "infect": "infect-dublin.edges"}
        graph = nx.Graph()
        with open("data/{}".format(fn[args.dataset]), "r") as f:
            for line in f:
                if not line.strip():
                    continue
                a, b = line.strip().split(" ")
                graph.add_edge(int(a), int(b))
        dataset = [graph]
    elif args.dataset.startswith('plant-'):
        size = int(args.dataset.split("-")[-1])
        dataset = decoder.make_plant_dataset(size)
    elif args.dataset == 'facebook':
        dataset = [utils.load_snap_edgelist("data/facebook_combined.txt")]
    elif args.dataset in ["as-733", "as20000102"]:
        dataset = [utils.load_snap_edgelist("data/as20000102.txt")]
    elif args.dataset.startswith('facebook_combined'):
        dataset = [utils.load_snap_edgelist("data/{}.txt".format(args.dataset))]
    elif args.dataset.startswith('roadnet-'):
        dataset = [utils.load_snap_edgelist("data/{}.txt".format(args.dataset))]
    elif args.dataset == "analyze":
        with open("results/analyze.p", "rb") as f:
            cand_patterns, _ = pickle.load(f)
            queries = [q for score, q in cand_patterns[10]][:200]
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

    targets = []
    for i in range(len(dataset)):
        graph = dataset[i]
        if not isinstance(graph, nx.Graph):
            graph = pyg_utils.to_networkx(dataset[i]).to_undirected()
        targets.append(graph)

    if args.dataset != "analyze":
        with open(args.queries_path, "rb") as f:
            queries = pickle.load(f)
        if args.max_queries and args.max_queries > 0:
            queries = queries[:args.max_queries]

    query_lens = [len(query) for query in queries]

    if args.baseline == "exact":
        n_matches_baseline = count_exact(queries, targets, args)
        n_matches = count_graphlets(queries[:len(n_matches_baseline)], targets,
                                    n_workers=args.n_workers,
                                    method=args.count_method,
                                    node_anchored=args.node_anchored,
                                    progress_every=args.progress_every,
                                    use_orbitsi=getattr(args, 'use_orbitsi', False))
    elif args.baseline == "none":
        n_matches = count_graphlets(queries, targets,
                                    n_workers=args.n_workers,
                                    method=args.count_method,
                                    node_anchored=args.node_anchored,
                                    progress_every=args.progress_every,
                                    use_orbitsi=getattr(args, 'use_orbitsi', False))
    else:
        baseline_queries = gen_baseline_queries(queries, targets,
                                                node_anchored=args.node_anchored,
                                                method=args.baseline)
        query_lens = [len(q) for q in baseline_queries]
        n_matches = count_graphlets(baseline_queries, targets,
                                    n_workers=args.n_workers,
                                    method=args.count_method,
                                    node_anchored=args.node_anchored,
                                    progress_every=args.progress_every,
                                    use_orbitsi=getattr(args, 'use_orbitsi', False))

    with open(args.out_path, "w") as f:
        json.dump((query_lens, n_matches, []), f)