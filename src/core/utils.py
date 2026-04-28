from collections import defaultdict, Counter

from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch
import torch
import torch.optim as optim
import networkx as nx
import numpy as np
import random
import pickle
import re
from pathlib import Path
from typing import List
import scipy.stats as stats
from tqdm import tqdm

from src.core import feature_preprocess
from src.logger import info, warning

def sample_neigh(graphs, size, max_attempts=100):
    """在图集合中按图大小加权采样一个连通邻域。

    采样步骤：
    1. 按 |V| 作为权重选择一张图（大图被选中概率更高）。
    2. 在图中随机选择起点并进行前沿扩展，直到达到给定 size。
    3. 若前沿耗尽导致节点不足，则重新采样。

    参数：
        graphs: networkx 图列表。
        size: 目标邻域节点数。
        max_attempts: 最大尝试次数，防止无限循环。
    返回：
        (graph, neigh_nodes) 其中 neigh_nodes 为采样到的节点列表。
    """
    ps = np.array([len(g) for g in graphs], dtype=float)
    ps /= np.sum(ps)
    dist = stats.rv_discrete(values=(np.arange(len(graphs)), ps))

    for attempt in range(max_attempts):
        idx = dist.rvs()
        #graph = random.choice(graphs)
        graph = graphs[idx]
        start_node = random.choice(list(graph.nodes))
        neigh = [start_node]
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = set([start_node])
        while len(neigh) < size and frontier:
            new_node = random.choice(list(frontier))
            #new_node = max(sorted(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)
            frontier += list(graph.neighbors(new_node))
            frontier = [x for x in frontier if x not in visited]
        if len(neigh) == size:
            return graph, neigh

    # 如果尝试多次都失败，返回最大的可连通子图
    warning("Could not sample neighborhood of size {} after {} attempts".format(
        size, max_attempts))
    return graph, neigh

cached_masks = None
def vec_hash(v):
    global cached_masks
    if cached_masks is None:
        random.seed(2019)
        cached_masks = [random.getrandbits(32) for i in range(len(v))]
    v = [int(hash(v[i]) % (2**31 - 1)) ^ mask for i, mask in enumerate(cached_masks)]
    return v
def wl_hash(g, dim=64, node_anchored=False):
    """计算图的 WL 风格哈希签名。

    该实现使用固定维度的离散向量做迭代聚合，最终把节点向量求和后
    转成 tuple 作为“结构签名”，用于把同构/近同构候选归并计数。

    参数：
        g: networkx.Graph。
        dim: 哈希向量维度。
        node_anchored: 是否使用 anchor 节点信息。
    返回：
        可哈希的 tuple 签名。
    """
    g = nx.convert_node_labels_to_integers(g)
    vecs = np.zeros((len(g), dim), dtype=object)
    if node_anchored:
        for v in g.nodes:
            if g.nodes[v]["anchor"] == 1:
                vecs[v] = 1
                break
    for i in range(len(g)):
        newvecs = np.zeros((len(g), dim), dtype=object)
        for n in g.nodes:
            newvecs[n] = vec_hash(np.sum(vecs[list(g.neighbors(n)) + [n]],
                axis=0))
        vecs = newvecs
    return tuple(np.sum(vecs, axis=0))

def gen_baseline_queries_rand_esu(queries, targets, node_anchored=False):
    sizes = Counter([len(g) for g in queries])
    max_size = max(sizes.keys())
    all_subgraphs = defaultdict(lambda: defaultdict(list))
    total_n_max_subgraphs, total_n_subgraphs = 0, 0
    for target in tqdm(targets):
        subgraphs = enumerate_subgraph(target, k=max_size,
            progress_bar=len(targets) < 10, node_anchored=node_anchored)
        for (size, k), v in subgraphs.items():
            all_subgraphs[size][k] += v
            if size == max_size: total_n_max_subgraphs += len(v)
            total_n_subgraphs += len(v)
    info("Subgraphs explored: {} (max-size: {})".format(
        total_n_subgraphs, total_n_max_subgraphs))
    out = []
    for size, count in sizes.items():
        counts = all_subgraphs[size]
        for _, neighs in list(sorted(counts.items(), key=lambda x: len(x[1]),
            reverse=True))[:count]:
            out.append(random.choice(neighs))
    return out

def enumerate_subgraph(G, k=3, progress_bar=False, node_anchored=False):
    """基于 ESU 思想枚举子图并按 WL 签名聚类。"""
    ps = np.arange(1.0, 0.0, -1.0/(k+1)) ** 1.5
    #ps = [1.0]*(k+1)
    motif_counts = defaultdict(list)
    for node in tqdm(G.nodes) if progress_bar else G.nodes:
        sg = set()
        sg.add(node)
        v_ext = set()
        neighbors = [nbr for nbr in list(G[node].keys()) if nbr > node]
        n_frac = len(neighbors) * ps[1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac)
            else 0)
        neighbors = random.sample(neighbors, n_samples)
        for nbr in neighbors:
            v_ext.add(nbr)
        extend_subgraph(G, k, sg, v_ext, node, motif_counts, ps, node_anchored)
    return motif_counts

def extend_subgraph(G, k, sg, v_ext, node_id, motif_counts, ps, node_anchored):
    """递归扩展当前子图并记录到 motif_counts。"""
    # 基础情形
    sg_G = G.subgraph(sg)
    if node_anchored:
        sg_G = sg_G.copy()
        nx.set_node_attributes(sg_G, 0, name="anchor")
        sg_G.nodes[node_id]["anchor"] = 1

    motif_counts[len(sg), wl_hash(sg_G,
        node_anchored=node_anchored)].append(sg_G)
    if len(sg) == k:
        return
    # 递归步骤：
    old_v_ext = v_ext.copy()
    while len(v_ext) > 0:
        w = v_ext.pop()
        new_v_ext = v_ext.copy()
        neighbors = [nbr for nbr in list(G[w].keys()) if nbr > node_id and nbr
            not in sg and nbr not in old_v_ext]
        n_frac = len(neighbors) * ps[len(sg) + 1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac)
            else 0)
        neighbors = random.sample(neighbors, n_samples)
        for nbr in neighbors:
            #if nbr > node_id and nbr not in sg and nbr not in old_v_ext:
            new_v_ext.add(nbr)
        sg.add(w)
        extend_subgraph(G, k, sg, new_v_ext, node_id, motif_counts, ps,
            node_anchored)
        sg.remove(w)

def gen_baseline_queries_mfinder(queries, targets, n_samples=10000,
    node_anchored=False):
    """基于随机邻域采样构造 mfinder 风格基线查询集。"""
    sizes = Counter([len(g) for g in queries])
    #sizes = {}
    #for i in range(5, 17):
    #    sizes[i] = 10
    out = []
    for size, count in tqdm(sizes.items()):
        print(size)
        counts = defaultdict(list)
        for i in tqdm(range(n_samples)):
            graph, neigh = sample_neigh(targets, size)
            v = neigh[0]
            neigh = graph.subgraph(neigh).copy()
            nx.set_node_attributes(neigh, 0, name="anchor")
            neigh.nodes[v]["anchor"] = 1
            neigh.remove_edges_from(nx.selfloop_edges(neigh))
            counts[wl_hash(neigh, node_anchored=node_anchored)].append(neigh)
        #bads, t = 0, 0
        #for ka, nas in counts.items():
        #    for kb, nbs in counts.items():
        #        if ka != kb:
        #            for a in nas:
        #                for b in nbs:
        #                    if nx.is_isomorphic(a, b):
        #                        bads += 1
        #                        print("bad", bads, t)
        #                    t += 1

        for _, neighs in list(sorted(counts.items(), key=lambda x: len(x[1]),
            reverse=True))[:count]:
            print(len(neighs))
            out.append(random.choice(neighs))
    return out

def load_snap_edgelist(path):
    """从 SNAP 风格的边列表文件中加载无向图。

    支持空格或制表符分隔的边，自动跳过空行和以 '#' 开头的注释行。
    返回最大连通子图，以确保采样操作的一致性。

    参数：
        path: 边列表文件路径（每行格式为 "节点1 节点2"）
    返回：
        最大连通子图（networkx.Graph）
    """
    graph = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                graph.add_edge(int(parts[0]), int(parts[1]))
    # 取最大连通子图，保证子图采样的连通性
    if not nx.is_connected(graph):
        graph = graph.subgraph(
            max(nx.connected_components(graph), key=len)
        ).copy()
    return graph

device_cache = None
def get_device():
    """懒加载运行设备（优先 CUDA）。"""
    global device_cache
    if device_cache is None:
        # Respect optional override via set_use_gpu().
        try:
            use_gpu = USE_GPU
        except NameError:
            use_gpu = True
        if use_gpu and torch.cuda.is_available():
            device_cache = torch.device("cuda")
        else:
            device_cache = torch.device("cpu")
    return device_cache


# Global switch to allow forcing CPU even if CUDA is available.
USE_GPU = True

def set_use_gpu(flag: bool):
    """设置是否允许使用 GPU。

    调用后会清空内部缓存以便下一次调用 `get_device()` 根据新设置返回设备。
    """
    global USE_GPU, device_cache
    USE_GPU = bool(flag)
    device_cache = None

def parse_optimizer(parser):
    """向解析器注册优化器相关参数。"""
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str,
            help='优化器类型')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
            help='优化器调度器类型，默认为无')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
            help='重启前的训练轮数（默认为 0，即不重启）')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
            help='衰减前的训练轮数')
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
            help='学习率衰减比率')
    opt_parser.add_argument('--lr', dest='lr', type=float,
            help='学习率')
    opt_parser.add_argument('--clip', dest='clip', type=float,
            help='梯度裁剪')
    opt_parser.add_argument('--weight_decay', type=float,
            help='优化器权重衰减')

def build_optimizer(args, params):
    """按配置创建优化器与学习率调度器。"""
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95,
            weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer

_AUGMENTER: feature_preprocess.FeatureAugment | None = None

def _get_augmenter() -> feature_preprocess.FeatureAugment:
    global _AUGMENTER
    if _AUGMENTER is None:
        _AUGMENTER = feature_preprocess.FeatureAugment()
    return _AUGMENTER


def batch_nx_graphs(graphs, anchors=None):
    augmenter = _get_augmenter()
    
    if anchors is not None:
        for anchor, g in zip(anchors, graphs):
            for v in g.nodes:
                g.nodes[v]["node_feature"] = torch.tensor([float(v == anchor)])

    batch = Batch.from_data_list([DSGraph(g) for g in graphs])
    batch = augmenter.augment(batch)
    device = get_device()
    batch = batch.to(device)
    return batch


def parse_gspan_output(file_path: Path) -> List[nx.Graph]:
    """解析 gSpan 文本输出为 NetworkX 图列表。"""
    graphs: List[nx.Graph] = []
    current = None

    for raw in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.startswith("t "):
            if current is not None and current.number_of_nodes() > 0:
                graphs.append(current)
            current = nx.Graph()
            continue

        if line.startswith("v ") and current is not None:
            toks = line.split()
            if len(toks) >= 3:
                current.add_node(int(toks[1]), label=toks[2])
            continue

        if line.startswith("e ") and current is not None:
            toks = line.split()
            if len(toks) >= 4:
                current.add_edge(int(toks[1]), int(toks[2]), label=toks[3])
            continue

        if line.lower().startswith("support") and current is not None:
            try:
                current.graph["support"] = float(line.split(":", 1)[1].strip())
            except Exception:
                pass

    if current is not None and current.number_of_nodes() > 0:
        graphs.append(current)
    return graphs


def load_spminer_pickle(file_path: Path) -> List[nx.Graph]:
    """加载 SPMiner pickle 输出为 NetworkX 图列表。"""
    with file_path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported SPMiner result format in {file_path}")
