import os
import pickle
import random

from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset
import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
import torch_geometric.utils as pyg_utils
from tqdm import tqdm
import scipy.stats as stats

from src.core import combined_syn
from src.core import dataset_registry
from src.core import feature_preprocess
from src.core import utils
from src.core.sampling.neighborhood import frontier_sample_nodes
from src.logger import info

__all__ = [
    "DataSource",
    "OTFSynDataSource",
    "OTFSynImbalancedDataSource",
    "DiskDataSource",
    "DiskImbalancedDataSource",
    "load_dataset",
]


def _imbalanced_add_anchor(g, node_anchored):
    """为不平衡数据源的 anchor 节点添加特征标记。"""
    anchor = random.choice(list(g.G.nodes))
    for v in g.G.nodes:
        g.G.nodes[v]["node_feature"] = (torch.ones(1) if anchor == v
            or not node_anchored else torch.zeros(1))
    return g


def _imbalanced_gen_batch_impl(self, graphs_a, graphs_b, _, train, cache_fn):
    """OTFSynImbalancedDataSource 和 DiskImbalancedDataSource 共享的 gen_batch 实现。"""
    pos_a, pos_b, neg_a, neg_b = [], [], [], []
    if not os.path.exists(cache_fn):
        graphs_a = graphs_a.apply_transform(
            lambda g: _imbalanced_add_anchor(g, self.node_anchored))
        graphs_b = graphs_b.apply_transform(
            lambda g: _imbalanced_add_anchor(g, self.node_anchored))
        for graph_a, graph_b in tqdm(list(zip(graphs_a.G, graphs_b.G))):
            matcher = nx.algorithms.isomorphism.GraphMatcher(graph_a, graph_b,
                node_match=(lambda a, b: (a["node_feature"][0] > 0.5) ==
                    (b["node_feature"][0] > 0.5)) if self.node_anchored else None)
            if matcher.subgraph_is_isomorphic():
                pos_a.append(graph_a)
                pos_b.append(graph_b)
            else:
                neg_a.append(graph_a)
                neg_b.append(graph_b)
        if not os.path.exists("data/cache"):
            os.makedirs("data/cache")
        with open(cache_fn, "wb") as f:
            pickle.dump((pos_a, pos_b, neg_a, neg_b), f)
        info("Cache saved → {}".format(cache_fn))
    else:
        with open(cache_fn, "rb") as f:
            info("Cache loaded ← {}".format(cache_fn))
            pos_a, pos_b, neg_a, neg_b = pickle.load(f)
    if pos_a:
        pos_a = utils.batch_nx_graphs(pos_a)
        pos_b = utils.batch_nx_graphs(pos_b)
    neg_a = utils.batch_nx_graphs(neg_a)
    neg_b = utils.batch_nx_graphs(neg_b)
    self.batch_idx += 1
    return pos_a, pos_b, neg_a, neg_b


def load_dataset(name):
    """ 加载 PyTorch Geometric 中提供的真实世界数据集。

    用作 DiskDataSource 的辅助函数。
    """
    # 统一通过注册中心做命名归一化与合法性校验，避免多处 if-elif 漂移。
    dataset, task = dataset_registry.load_dataset_for_stage(name, "train-disk")

    # 单图数据集（如 facebook/as20000102）在训练场景中不做随机切分。
    if isinstance(dataset, list) and len(dataset) == 1 and isinstance(dataset[0], nx.Graph):
        return [dataset[0]], [dataset[0]], task

    if task == "graph":
        train_len = int(0.8 * len(dataset))
        train, test = [], []
        dataset = list(dataset)
        random.shuffle(dataset)
        has_name = hasattr(dataset[0], "name")
        for i, graph in tqdm(enumerate(dataset)):
            if not type(graph) == nx.Graph:
                if has_name: del graph.name
                graph = pyg_utils.to_networkx(graph).to_undirected()
            if i < train_len:
                train.append(graph)
            else:
                test.append(graph)
    return train, test, task

class DataSource:
    def gen_batch(batch_target, batch_neg_target, batch_neg_query, train):
        raise NotImplementedError

class OTFSynDataSource(DataSource):
    """ 用于训练子图模型的在线生成合成数据。

    每次迭代时，使用预定义的生成器（参见 combined_syn.py）
    动态生成新的图批次（正例和负例）。

    使用 DeepSNAP 变换来生成正例和负例。
    """
    def __init__(self, max_size=29, min_size=5, n_workers=4,
        max_queue_size=256, node_anchored=False):
        self.closed = False
        self.max_size = max_size
        self.min_size = min_size
        self.node_anchored = node_anchored
        self.generator = combined_syn.get_generator(np.arange(
            self.min_size + 1, self.max_size + 1))

    def gen_data_loaders(self, size, batch_size, train=True,
        use_distributed_sampling=False):
        loaders = []
        device = utils.get_device()
        pin_memory = True if device.type == 'cuda' else False
        for i in range(2):
            dataset = combined_syn.get_dataset("graph", size // 2,
                np.arange(self.min_size + 1, self.max_size + 1))
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=hvd.size(), rank=hvd.rank()) if \
                    use_distributed_sampling else None
            loaders.append(TorchDataLoader(dataset,
                collate_fn=Batch.collate([]), batch_size=batch_size // 2 if i
                == 0 else batch_size // 2,
                sampler=sampler, shuffle=False, pin_memory=pin_memory))
        loaders.append([None]*(size // batch_size))
        return loaders

    def gen_batch(self, batch_target, batch_neg_target, batch_neg_query,
        train):
        def sample_subgraph(graph, offset=0, use_precomp_sizes=False,
            filter_negs=False, supersample_small_graphs=False, neg_target=None,
            hard_neg_idxs=None):
            if neg_target is not None: graph_idx = graph.G.graph["idx"]
            use_hard_neg = (hard_neg_idxs is not None and graph.G.graph["idx"]
                in hard_neg_idxs)
            done = False
            n_tries = 0
            while not done:
                if use_precomp_sizes:
                    size = graph.G.graph["subgraph_size"]
                else:
                    if train and supersample_small_graphs:
                        sizes = np.arange(self.min_size + offset,
                            len(graph.G) + offset)
                        ps = (sizes - self.min_size + 2) ** (-1.1)
                        ps /= ps.sum()
                        size = stats.rv_discrete(values=(sizes, ps)).rvs()
                    else:
                        d = 1 if train else 0
                        size = random.randint(self.min_size + offset - d,
                            len(graph.G) - 1 + offset)
                neigh = frontier_sample_nodes(graph.G, size)
                if self.node_anchored:
                    anchor = neigh[0]
                    for v in graph.G.nodes:
                        graph.G.nodes[v]["node_feature"] = (torch.ones(1) if
                            anchor == v else torch.zeros(1))
                neigh = graph.G.subgraph(neigh)
                if use_hard_neg and train:
                    neigh = neigh.copy()
                    non_edges = list(nx.non_edges(neigh))
                    if len(non_edges) > 0:
                        for u, v in random.sample(non_edges, random.randint(1,
                            min(len(non_edges), 5))):
                            neigh.add_edge(u, v)

                if (filter_negs and train and len(neigh) <= 6 and neg_target is
                    not None):
                    matcher = nx.algorithms.isomorphism.GraphMatcher(
                        neg_target[graph_idx], neigh)
                    if not matcher.subgraph_is_isomorphic(): done = True
                else:
                    done = True

            return graph, DSGraph(neigh)

        augmenter = feature_preprocess.FeatureAugment()

        pos_target = batch_target
        pos_target, pos_query = pos_target.apply_transform_multi(sample_subgraph)
        neg_target = batch_neg_target
        # TODO: 使用困难负例
        hard_neg_idxs = set(random.sample(range(len(neg_target.G)),
            int(len(neg_target.G) * 1/2)))
        #hard_neg_idxs = set()
        batch_neg_query = Batch.from_data_list(
            [DSGraph(self.generator.generate(size=len(g))
                if i not in hard_neg_idxs else g)
                for i, g in enumerate(neg_target.G)])
        for i, g in enumerate(batch_neg_query.G):
            g.graph["idx"] = i
        _, neg_query = batch_neg_query.apply_transform_multi(sample_subgraph,
            hard_neg_idxs=hard_neg_idxs)
        if self.node_anchored:
            def add_anchor(g, anchors=None):
                if anchors is not None:
                    anchor = anchors[g.G.graph["idx"]]
                else:
                    anchor = random.choice(list(g.G.nodes))
                for v in g.G.nodes:
                    if "node_feature" not in g.G.nodes[v]:
                        g.G.nodes[v]["node_feature"] = (torch.ones(1) if anchor == v
                            else torch.zeros(1))
                return g
            neg_target = neg_target.apply_transform(add_anchor)
        device = utils.get_device()
        pos_target = augmenter.augment(pos_target).to(device)
        pos_query = augmenter.augment(pos_query).to(device)
        neg_target = augmenter.augment(neg_target).to(device)
        neg_query = augmenter.augment(neg_query).to(device)
        #print(len(pos_target.G[0]), len(pos_query.G[0]))
        return pos_target, pos_query, neg_target, neg_query

class OTFSynImbalancedDataSource(OTFSynDataSource):
    """ 不平衡的在线合成数据。

    与平衡数据集不同，此数据源不使用 1:1 的正负例比例。
    而是从在线生成器中随机采样 2 个图，并记录该对的真实标签（是否为子图关系）。
    因此数据是不平衡的（子图关系较为罕见）。
    该设置是一种具有挑战性的模型推理场景。
    """
    def __init__(self, max_size=29, min_size=5, n_workers=4,
        max_queue_size=256, node_anchored=False):
        super().__init__(max_size=max_size, min_size=min_size,
            n_workers=n_workers, node_anchored=node_anchored)
        self.batch_idx = 0

    def gen_batch(self, graphs_a, graphs_b, _, train):
        fn = "data/cache/imbalanced-{}-{}".format(str(self.node_anchored),
            self.batch_idx)
        return _imbalanced_gen_batch_impl(self, graphs_a, graphs_b, _, train, fn)

class DiskDataSource(DataSource):
    """ 使用保存在数据集文件中的图集合来训练子图模型。

    每次迭代时，通过从给定数据集中采样子图来生成新的图批次（正例和负例）。

    支持的数据集请参见 load_dataset 函数。
    """
    def __init__(self, dataset_name, node_anchored=False, min_size=5,
        max_size=29):
        self.node_anchored = node_anchored
        self.dataset = load_dataset(dataset_name)
        self.min_size = min_size
        self.max_size = max_size

    def gen_data_loaders(self, size, batch_size, train=True,
        use_distributed_sampling=False):
        loaders = [[batch_size]*(size // batch_size) for i in range(3)]
        return loaders

    def gen_batch(self, a, b, c, train, max_size=15, min_size=5, seed=None,
        filter_negs=False, sample_method="tree-pair"):
        batch_size = a
        train_set, test_set, task = self.dataset
        graphs = train_set if train else test_set
        if seed is not None:
            random.seed(seed)

        pos_a, pos_b = [], []
        pos_a_anchors, pos_b_anchors = [], []
        for i in range(batch_size // 2):
            if sample_method == "tree-pair":
                size = random.randint(min_size+1, max_size)
                graph, a = utils.sample_neigh(graphs, size)
                b = a[:random.randint(min_size, len(a) - 1)]
            elif sample_method == "subgraph-tree":
                graph = None
                while graph is None or len(graph) < min_size + 1:
                    graph = random.choice(graphs)
                a = graph.nodes
                _, b = utils.sample_neigh([graph], random.randint(min_size,
                    len(graph) - 1))
            if self.node_anchored:
                anchor = list(graph.nodes)[0]
                pos_a_anchors.append(anchor)
                pos_b_anchors.append(anchor)
            neigh_a, neigh_b = graph.subgraph(a), graph.subgraph(b)
            pos_a.append(neigh_a)
            pos_b.append(neigh_b)

        neg_a, neg_b = [], []
        neg_a_anchors, neg_b_anchors = [], []
        while len(neg_a) < batch_size // 2:
            if sample_method == "tree-pair":
                size = random.randint(min_size+1, max_size)
                graph_a, a = utils.sample_neigh(graphs, size)
                graph_b, b = utils.sample_neigh(graphs, random.randint(min_size,
                    size - 1))
            elif sample_method == "subgraph-tree":
                graph_a = None
                while graph_a is None or len(graph_a) < min_size + 1:
                    graph_a = random.choice(graphs)
                a = graph_a.nodes
                graph_b, b = utils.sample_neigh(graphs, random.randint(min_size,
                    len(graph_a) - 1))
            if self.node_anchored:
                neg_a_anchors.append(list(graph_a.nodes)[0])
                neg_b_anchors.append(list(graph_b.nodes)[0])
            neigh_a, neigh_b = graph_a.subgraph(a), graph_b.subgraph(b)
            if filter_negs:
                matcher = nx.algorithms.isomorphism.GraphMatcher(neigh_a, neigh_b)
                if matcher.subgraph_is_isomorphic(): # a <= b (b is subgraph of a)
                    continue
            neg_a.append(neigh_a)
            neg_b.append(neigh_b)

        pos_a = utils.batch_nx_graphs(pos_a, anchors=pos_a_anchors if
            self.node_anchored else None)
        pos_b = utils.batch_nx_graphs(pos_b, anchors=pos_b_anchors if
            self.node_anchored else None)
        neg_a = utils.batch_nx_graphs(neg_a, anchors=neg_a_anchors if
            self.node_anchored else None)
        neg_b = utils.batch_nx_graphs(neg_b, anchors=neg_b_anchors if
            self.node_anchored else None)
        return pos_a, pos_b, neg_a, neg_b

class DiskImbalancedDataSource(OTFSynDataSource):
    """ 不平衡的在线真实数据。

    与平衡数据集不同，此数据源不使用 1:1 的正负例比例。
    而是从在线生成器中随机采样 2 个图，并记录该对的真实标签（是否为子图关系）。
    因此数据是不平衡的（子图关系较为罕见）。
    该设置是一种具有挑战性的模型推理场景。
    """
    def __init__(self, dataset_name, max_size=29, min_size=5, n_workers=4,
        max_queue_size=256, node_anchored=False):
        super().__init__(max_size=max_size, min_size=min_size,
            n_workers=n_workers, node_anchored=node_anchored)
        self.batch_idx = 0
        self.dataset = load_dataset(dataset_name)
        self.train_set, self.test_set, _ = self.dataset
        self.dataset_name = dataset_name

    def gen_data_loaders(self, size, batch_size, train=True,
        use_distributed_sampling=False):
        loaders = []
        device = utils.get_device()
        pin_memory = True if device.type == 'cuda' else False
        for i in range(2):
            neighs = []
            for j in range(size // 2):
                graph, neigh = utils.sample_neigh(self.train_set if train else
                    self.test_set, random.randint(self.min_size, self.max_size))
                neighs.append(graph.subgraph(neigh))
            dataset = GraphDataset(neighs)
            loaders.append(TorchDataLoader(dataset,
                collate_fn=Batch.collate([]), batch_size=batch_size // 2 if i
                == 0 else batch_size // 2,
                sampler=None, shuffle=False, pin_memory=pin_memory))
        loaders.append([None]*(size // batch_size))
        return loaders

    def gen_batch(self, graphs_a, graphs_b, _, train):
        fn = "data/cache/imbalanced-{}-{}-{}".format(self.dataset_name.lower(),
            str(self.node_anchored), self.batch_idx)
        return _imbalanced_gen_batch_impl(self, graphs_a, graphs_b, _, train, fn)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14})
    for name in ["enzymes", "reddit-binary", "cox2"]:
        data_source = DiskDataSource(name)
        train, test, _ = data_source.dataset
        i = 11
        neighs = [utils.sample_neigh(train, i) for j in range(10000)]
        clustering = [nx.average_clustering(graph.subgraph(nodes)) for graph,
            nodes in neighs]
        path_length = [nx.average_shortest_path_length(graph.subgraph(nodes))
            for graph, nodes in neighs]
        #plt.subplot(1, 2, i-9)
        plt.scatter(clustering, path_length, s=10, label=name)
    plt.legend()
    plt.savefig("plots/clustering-vs-path-length.png")
