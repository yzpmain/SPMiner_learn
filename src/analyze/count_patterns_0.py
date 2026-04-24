import argparse
import csv
import json
import traceback

import numpy as np

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import PPI
import torch_geometric.utils as pyg_utils

from src.core import data
from src.core import utils
from src.subgraph_mining import decoder

from multiprocessing import Pool
import random
from collections import defaultdict
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
    parser.add_argument('--chunksize', type=int,
        help='多进程分块大小，越大越省调度开销')
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
                        chunksize=32,
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


def _orbitsi_match_to_target_key(match, query, node_anchored):
    """Build a canonical dedup key from one OrbitSI match mapping.

    Key semantics:
    - Non-anchored: sorted tuple of matched target nodes.
    - Anchored: (sorted target nodes, sorted anchor target nodes).
    """
    if not isinstance(match, dict):
        raise TypeError("Match must be a dict for freq dedup")

    q_nodes = set(query.nodes)
    keys = set(match.keys())
    vals = set(match.values())

    # Detect mapping orientation.
    # If values overlap query nodes more, treat as target->query.
    # Otherwise treat as query->target.
    key_overlap = len(keys & q_nodes)
    val_overlap = len(vals & q_nodes)
    target_to_query = (val_overlap >= key_overlap)

    if target_to_query:
        target_nodes = tuple(sorted(int(n) for n in match.keys()))
    else:
        target_nodes = tuple(sorted(int(n) for n in match.values()))

    if not node_anchored:
        return target_nodes

    anchor_query_nodes = {n for n, data in query.nodes(data=True)
        if data.get("anchor", 0) == 1}
    if not anchor_query_nodes:
        return (target_nodes, tuple())

    anchor_target_nodes = []
    if target_to_query:
        for t_node, q_node in match.items():
            if q_node in anchor_query_nodes:
                anchor_target_nodes.append(int(t_node))
    else:
        for q_node, t_node in match.items():
            if q_node in anchor_query_nodes:
                anchor_target_nodes.append(int(t_node))
    anchor_target_nodes = tuple(sorted(anchor_target_nodes))
    return (target_nodes, anchor_target_nodes)

def _orbitsi_match_count(query, target, method, node_anchored, anchor_or_none):
    """Return match count using OrbitSI pipeline.

    Raises on any runtime issue so caller can safely fallback to NetworkX.
    """
    find_all = (method == "freq")
    n = 0

    if method == "bin" and node_anchored:
        anchors = (target.nodes if anchor_or_none is None else [anchor_or_none])
        query_labeled = _build_orbitsi_labeled_graph(query, node_anchored=True)
        for anchor in anchors:
            target_labeled = _build_orbitsi_labeled_graph(target,
                node_anchored=True, anchor_node=anchor)
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
            matches = search_engine.run(return_all=False)
            if matches:
                n += 1
        return n

    query_labeled = _build_orbitsi_labeled_graph(query,
        node_anchored=(method == "bin" and node_anchored))
    target_labeled = _build_orbitsi_labeled_graph(target,
        node_anchored=(method == "bin" and node_anchored))
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
    matches = search_engine.run(return_all=find_all)

    if method == "bin":
        return int(bool(matches))

    # freq: strict dedup on target node sets (+ anchor semantics when enabled)
    seen = set()
    for match in matches:
        seen.add(_orbitsi_match_to_target_key(match, query, node_anchored))
    return len(seen)

def gen_baseline_queries(queries, targets, method="mfinder",
    node_anchored=False):
    # 使用此函数生成 N 个尺寸为 K 的查询图
    #queries = [[0]*n for n in range(5, 21) for i in range(10)]
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
                #neigh = random.sample(neigh, min(len(neigh), 15))
                neigh = graph.subgraph(neigh)
                neigh = neigh.subgraph(list(sorted(nx.connected_components(
                    neigh), key=len))[-1])
                neigh = nx.convert_node_labels_to_integers(neigh)
                print(i, len(neigh), len(query))
                if len(neigh) == len(query):
                    neighs.append(neigh)
                    found = True
            elif method == "tree":
                # https://academic.oup.com/bioinformatics/article/20/11/1746/300212
                graph = random.choice(targets)
                start_node = random.choice(list(graph.nodes))
                neigh = [start_node]
                frontier = list(set(graph.neighbors(start_node)) - set(neigh))
                while len(neigh) < len(query) and frontier:
                    new_node = random.choice(list(frontier))
                    assert new_node not in neigh
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
    n_symmetries = None
    if method == "freq":
        ismags = nx.isomorphism.ISMAGS(query, query)
        n_symmetries = len(list(ismags.isomorphisms_iter(symmetry=False)))
    return {
        "graph": query,
        "n_nodes": query.number_of_nodes(),
        "n_edges": query.number_of_edges(),
        "degree_seq": degree_seq,
        "anchor_count": sum(1 for _, data in query.nodes(data=True)
            if data.get("anchor", 0) == 1) if node_anchored else 0,
        "n_symmetries": n_symmetries,
    }

def preprocess_target(target, node_anchored):
    target = target.copy()
    target.remove_edges_from(nx.selfloop_edges(target))
    degree_seq = tuple(sorted((d for _, d in target.degree()), reverse=True))
    return {
        "graph": target,
        "n_nodes": target.number_of_nodes(),
        "n_edges": target.number_of_edges(),
        "degree_seq": degree_seq,
        "anchor_count": sum(1 for _, data in target.nodes(data=True)
            if data.get("anchor", 0) == 1) if node_anchored else 0,
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
            # 先用便宜条件过滤，再做同构判断。
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

def count_graphlets_helper(inp):
    i, query_info, target_info, method, node_anchored, anchor_or_none, use_orbitsi = inp
    query = query_info["graph"]
    target = target_info["graph"]

    n, n_bin = 0, 0

    # 先做一层便宜的必要条件过滤，避免对明显不可能匹配的图调用
    # GraphMatcher。这不会改变支持度定义，只减少无效匹配检查。
    if query_info["n_nodes"] > target_info["n_nodes"]:
        return i, 0
    if query_info["n_edges"] > target_info["n_edges"]:
        return i, 0

    # 第二层粗筛：度序列必要条件。若 query 的前 k 大度序列在任意位置
    # 高于 target 的前 k 大度序列，则不可能存在子图同构。
    q_degree_seq = query_info["degree_seq"]
    t_degree_seq = target_info["degree_seq"]
    for q_deg, t_deg in zip(q_degree_seq, t_degree_seq[:len(q_degree_seq)]):
        if q_deg > t_deg:
            return i, 0

    if node_anchored:
        query_anchor_count = query_info["anchor_count"]
        target_anchor_count = target_info["anchor_count"]
        if query_anchor_count and target_anchor_count == 0:
            return i, 0
        if query_anchor_count > target_anchor_count:
            return i, 0

    target = target.copy()

    # Try OrbitSI first; fall back to NetworkX when unavailable or runtime fails.
    used_orbitsi = False
    if use_orbitsi and ORBITSI_AVAILABLE and method in ("bin", "freq"):
        try:
            n = _orbitsi_match_count(query, target, method, node_anchored,
                anchor_or_none)
            used_orbitsi = True
        except Exception:
            used_orbitsi = False

    if not used_orbitsi:
        if method == "bin":
            if node_anchored:
                for anchor in (target.nodes if anchor_or_none is None else
                    [anchor_or_none]):
                    nx.set_node_attributes(target, 0, name="anchor")
                    target.nodes[anchor]["anchor"] = 1
                    matcher = iso.GraphMatcher(target, query,
                        node_match=iso.categorical_node_match(["anchor"], [0]))
                    if matcher.subgraph_is_isomorphic():
                        n += 1
            else:
                matcher = iso.GraphMatcher(target, query)
                n += int(matcher.subgraph_is_isomorphic())
        elif method == "freq":
            matcher = iso.GraphMatcher(target, query)
            if use_orbitsi:
                # Keep strict dedup semantics even when OrbitSI falls back.
                seen = set()
                for match in matcher.subgraph_isomorphisms_iter():
                    seen.add(_orbitsi_match_to_target_key(match, query,
                        node_anchored))
                n += len(seen)
            else:
                # Legacy semantics: count all embeddings normalized by symmetry.
                n_symmetries = query_info["n_symmetries"]
                n += len(list(matcher.subgraph_isomorphisms_iter())) / n_symmetries
        else:
            print("计数方法不被识别")
    #n_matches.append(n / n_symmetries)
    #print(i, n / n_symmetries)
    count = n# / n_symmetries
    #if include_bin:
    #    count = (count, n_bin)
    #print(i, count)
    return i, count

def count_graphlets(queries, targets, n_workers=1, method="bin",
    node_anchored=False, min_count=0, chunksize=32, progress_every=1000,
    use_orbitsi=False):
    print(len(queries), len(targets))
    #idxs, counts = zip(*[count_graphlets_helper((i, q, targets, include_bin))
    #    for i, q in enumerate(queries)])
    #counts = list(counts)
    #return counts

    queries = [preprocess_query(query, method, node_anchored)
        for query in queries]
    targets = [preprocess_target(target, node_anchored)
        for target in targets]

    query_to_unique = None
    work_queries = queries
    if method == "freq":
        work_queries, query_to_unique = dedup_isomorphic_queries(
            queries, node_anchored=node_anchored)
        if len(work_queries) != len(queries):
            print("freq query dedup:", len(queries), "->", len(work_queries))

    n_matches = defaultdict(float)
    #for i, query in enumerate(work_queries):
    if node_anchored:
        inp = [(i, query, target, method, node_anchored, anchor, use_orbitsi) for i, query
            in enumerate(work_queries) for target in targets for anchor in (target
                ["graph"] if len(targets) < 10 else [None])]
    else:
        inp = [(i, query, target, method, node_anchored, None, use_orbitsi) for i, query
            in enumerate(work_queries) for target in targets]
    print(len(inp))
    n_done = 0
    total = len(inp)
    if chunksize is None or chunksize < 1:
        chunksize = 1
    with Pool(processes=n_workers) as pool:
        for i, n in pool.imap_unordered(count_graphlets_helper, inp,
            chunksize=chunksize):
            n_matches[i] += n
            n_done += 1
            if progress_every and progress_every > 0:
                if n_done % progress_every == 0 or n_done == total:
                    print(n_done, total, len(n_matches), i, n, "      ", end="\r")
    print()
    unique_matches = [n_matches[i] for i in range(len(work_queries))]
    if query_to_unique is None:
        return unique_matches
    return [unique_matches[idx] for idx in query_to_unique]

def count_exact(queries, targets, args):
    if orca is None:
        raise ImportError("orca 未安装，无法使用 baseline=exact")
    print("警告：orca 仅适用于节点锚定情况")
    # TODO: 非节点锚定情况
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
    num5 = 10#len([q for q in queries if len(q) == 5])
    for x in list(sorted(n_matches_baseline, reverse=True))[:num5]:
        print(x)
        counts5.append(x)
    print("Average for size 5:", np.mean(np.log10(counts5)))

    atlas = [g for g in nx.graph_atlas_g()[1:] if nx.is_connected(g)
        and len(g) == 6]
    queries = []
    for g in atlas:
        for v in g.nodes:
            g = g.copy()
            nx.set_node_attributes(g, 0, name="anchor")
            g.nodes[v]["anchor"] = 1
            is_dup = False
            for g2 in queries:
                if nx.is_isomorphic(g, g2, node_match=(lambda a, b: a["anchor"]
                    == b["anchor"]) if args.node_anchored else None):
                    is_dup = True
                    break
            if not is_dup:
                queries.append(g)
    print(len(queries))
    n_matches_baseline = count_graphlets(queries, targets,
        n_workers=args.n_workers, method=args.count_method,
        node_anchored=args.node_anchored,
        min_count=10000,
        chunksize=args.chunksize,
        progress_every=args.progress_every,
        use_orbitsi=getattr(args, 'use_orbitsi', False))
    counts6 = []
    num6 = 20#len([q for q in queries if len(q) == 6])
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

    if args.dataset == 'syn':
        # 使用合成数据集
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
                if not line.strip(): continue
                a, b = line.strip().split(" ")
                graph.add_edge(int(a), int(b))
        dataset = [graph]
    elif args.dataset.startswith('plant-'):
        size = int(args.dataset.split("-")[-1])
        dataset = decoder.make_plant_dataset(size)
    elif args.dataset == 'facebook':
        # 斯坦福 SNAP ego-Facebook 数据集
        # 请将 facebook_combined.txt 放置在 data/ 目录下
        dataset = [utils.load_snap_edgelist("data/facebook_combined.txt")]
    elif args.dataset in ["as-733", "as20000102"]:
        # 斯坦福 SNAP AS 路由图数据集（as20000102）
        # 请将 as20000102.txt 放置在 data/ 目录下
        dataset = [utils.load_snap_edgelist("data/as20000102.txt")]
    elif args.dataset.startswith('facebook_combined'):
        
        dataset = [utils.load_snap_edgelist("data/{}.txt".format(args.dataset))]
    elif args.dataset.startswith('roadnet-'):
        # roadnet-* 使用 data/<dataset>.txt 的 tab/space 边列表。
        dataset = [utils.load_snap_edgelist("data/{}.txt".format(args.dataset))]
    elif args.dataset == "analyze":
        with open("results/analyze.p", "rb") as f:
            cand_patterns, _ = pickle.load(f)
            queries = [q for score, q in cand_patterns[10]][:200]
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

    targets = []
    for i in range(len(dataset)):
        graph = dataset[i]
        if not type(graph) == nx.Graph:
            graph = pyg_utils.to_networkx(dataset[i]).to_undirected()
        targets.append(graph)

    if args.dataset != "analyze":
        with open(args.queries_path, "rb") as f:
            queries = pickle.load(f)
        if args.max_queries and args.max_queries > 0:
            queries = queries[:args.max_queries]

    # 仅过滤出前几个非同构的 6 阶模体
    #filt_q = []
    #for q in queries:
    #    if len([qc for qc in filt_q if nx.is_isomorphic(q, qc)]) == 0:
    #        filt_q.append(q)
    #queries = filt_q[:]
    #print(len(queries))
            
    query_lens = [len(query) for query in queries]

    if args.baseline == "exact":
        n_matches_baseline = count_exact(queries, targets, args)
        n_matches = count_graphlets(queries[:len(n_matches_baseline)], targets,
            n_workers=args.n_workers, method=args.count_method,
            node_anchored=args.node_anchored,
            chunksize=args.chunksize,
            progress_every=args.progress_every,
            use_orbitsi=getattr(args, 'use_orbitsi', False))
    elif args.baseline == "none":
        n_matches = count_graphlets(queries, targets,
            n_workers=args.n_workers, method=args.count_method,
            node_anchored=args.node_anchored,
            chunksize=args.chunksize,
            progress_every=args.progress_every,
            use_orbitsi=getattr(args, 'use_orbitsi', False))
    else:
        baseline_queries = gen_baseline_queries(queries, targets,
            node_anchored=args.node_anchored, method=args.baseline)
        query_lens = [len(q) for q in baseline_queries]
        n_matches = count_graphlets(baseline_queries, targets,
            n_workers=args.n_workers, method=args.count_method,
            node_anchored=args.node_anchored,
            chunksize=args.chunksize,
            progress_every=args.progress_every,
            use_orbitsi=getattr(args, 'use_orbitsi', False))
    with open(args.out_path, "w") as f:
        json.dump((query_lens, n_matches, []), f)
