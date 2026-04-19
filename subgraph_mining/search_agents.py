import numpy as np
import torch
from tqdm import tqdm
from common import utils

import matplotlib.pyplot as plt

import random
import scipy.stats as stats
from collections import defaultdict
import networkx as nx
import pickle

class SearchAgent:
    """ 用于在嵌入空间中识别频繁子图的搜索策略类。

    该问题被建模为搜索过程。第一个动作选择一个种子节点作为生长起点。
    后续动作选择数据集中的一个节点连接到现有子图模式，
    每次将模式大小增加 1。

    详细原理和算法请参阅论文。
    """
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, model_type="order",
        out_batch_size=20, frontier_top_k=0):
        """ 通过在嵌入空间中游走进行子图模式搜索。

        参数说明：
            min_pattern_size: 待识别频繁子图的最小尺寸。
            max_pattern_size: 待识别频繁子图的最大尺寸。
            model: 已训练的子图匹配模型（PyTorch nn.Module）。
            dataset: 用于挖掘频繁子图模式的 DeepSNAP 数据集。
            embs: 采样节点邻域的嵌入（参见论文）。
            node_anchored: 是否识别节点锚定的子图模式。
                节点锚定搜索过程必须使用节点锚定模型（在子图匹配 config.py 中指定）。
            analyze: 是否启用分析可视化。
            model_type: 子图匹配模型类型（须与 model 参数保持一致）。
            out_batch_size: 挖掘算法为每种尺寸输出的频繁子图数量。
                这些被预测为数据集中出现频率最高的 out_batch_size 个子图。
            frontier_top_k: 每一步保留的 frontier 候选上限。0 表示不剪枝。
        """
        self.min_pattern_size = min_pattern_size
        self.max_pattern_size = max_pattern_size
        self.model = model
        self.dataset = dataset
        self.embs = embs
        self.node_anchored = node_anchored
        self.analyze = analyze
        self.model_type = model_type
        self.out_batch_size = out_batch_size
        self.frontier_top_k = frontier_top_k
        self.cand_emb_cache = {}

    def run_search(self, n_trials=1000): 
        """统一搜索驱动器。

        子类只需实现 init_search / step / finish_search，
        即可复用这套主循环。
        """
        self.cand_patterns = defaultdict(list)
        self.counts = defaultdict(lambda: defaultdict(list))
        self.n_trials = n_trials

        self.init_search()
        while not self.is_search_done():
            self.step()
        return self.finish_search()

    def init_search():
        raise NotImplementedError

    def step(self):
        """ 执行一步搜索的抽象方法。
        每一步向子图模式中添加一个新节点。
        run_search 至少调用 min_pattern_size 次 step 以生成至少该尺寸的模式。
        由具体搜索策略实现类继承。
        """
        raise NotImplementedError

    def _candidate_cache_key(self, graph_idx, nodes, anchor_node=None):
        """为候选子图生成稳定缓存键。"""
        return graph_idx, frozenset(nodes), anchor_node if self.node_anchored else None

    def _get_candidate_embs(self, graph_idx, graph, neigh, frontier):
        """批量获取 frontier 对应候选子图的 embedding，并缓存重复状态。"""
        cache_keys = []
        cand_graphs = []
        anchors = []
        cand_nodes = []
        cand_embs = [None] * len(frontier)
        anchor_node = neigh[0] if self.node_anchored else None

        for idx, cand_node in enumerate(frontier):
            nodes = list(neigh) + [cand_node]
            cache_key = self._candidate_cache_key(graph_idx, nodes, anchor_node)
            cache_keys.append(cache_key)
            cand_nodes.append(cand_node)
            if cache_key in self.cand_emb_cache:
                cand_embs[idx] = self.cand_emb_cache[cache_key]
            else:
                cand_graphs.append(graph.subgraph(nodes))
                if self.node_anchored:
                    anchors.append(anchor_node)

        if cand_graphs:
            new_embs = self.model.emb_model(utils.batch_nx_graphs(
                cand_graphs, anchors=anchors if self.node_anchored else None))
            for cand_node, cache_key, emb in zip(
                [n for i, n in enumerate(cand_nodes) if cand_embs[i] is None],
                [k for i, k in enumerate(cache_keys) if cand_embs[i] is None],
                new_embs,
            ):
                emb = emb.detach().cpu()
                self.cand_emb_cache[cache_key] = emb

        for idx, cache_key in enumerate(cache_keys):
            if cand_embs[idx] is None:
                cand_embs[idx] = self.cand_emb_cache[cache_key]
        return cand_embs

    def _prune_frontier(self, graph, frontier):
        """按候选节点度数保留前 K 个 frontier 节点。"""
        if self.frontier_top_k and len(frontier) > self.frontier_top_k:
            frontier = sorted(frontier,
                key=lambda node: (graph.degree(node), -node),
                reverse=True)[:self.frontier_top_k]
        return frontier

class MCTSSearchAgent(SearchAgent):
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, model_type="order",
        out_batch_size=20, c_uct=0.7, frontier_top_k=0):
        """ 子图模式搜索的 MCTS 实现。
        使用 MCTS 策略搜索最常见的模式。

        参数说明：
            c_uct: UCT 准则中使用的探索常数（参见论文）。
        """
        super().__init__(min_pattern_size, max_pattern_size, model, dataset,
            embs, node_anchored=node_anchored, analyze=analyze,
            model_type=model_type, out_batch_size=out_batch_size,
            frontier_top_k=frontier_top_k)
        self.c_uct = c_uct
        assert not analyze

    def init_search(self):
        """初始化 MCTS 运行时缓存。"""
        self.wl_hash_to_graphs = defaultdict(list)
        self.cum_action_values = defaultdict(lambda: defaultdict(float))
        self.visit_counts = defaultdict(lambda: defaultdict(float))
        self.visited_seed_nodes = set()
        self.max_size = self.min_pattern_size

    def is_search_done(self):
        return self.max_size == self.max_pattern_size + 1

    # 返回从 start_node 起至少有 n 个可达节点
    def has_min_reachable_nodes(self, graph, start_node, n):
        for depth_limit in range(n+1):
            edges = nx.bfs_edges(graph, start_node, depth_limit=depth_limit)
            nodes = set([v for u, v in edges])
            if len(nodes) + 1 >= n:
                return True
        return False

    def step(self):
        """执行一轮 MCTS 扩展与价值回传。"""
        ps = np.array([len(g) for g in self.dataset], dtype=float)
        ps /= np.sum(ps)
        graph_dist = stats.rv_discrete(values=(np.arange(len(self.dataset)), ps))

        print("Size", self.max_size)
        print(len(self.visited_seed_nodes), "distinct seeds")
        for simulation_n in tqdm(range(self.n_trials //
            (self.max_pattern_size+1-self.min_pattern_size))):
            # 选择种子节点
            best_graph_idx, best_start_node, best_score = None, None, -float("inf")
            for cand_graph_idx, cand_start_node in self.visited_seed_nodes:
                state = cand_graph_idx, cand_start_node
                my_visit_counts = sum(self.visit_counts[state].values())
                q_score = (sum(self.cum_action_values[state].values()) /
                    (my_visit_counts or 1))
                uct_score = self.c_uct * np.sqrt(np.log(simulation_n or 1) /
                    (my_visit_counts or 1))
                node_score = q_score + uct_score
                if node_score > best_score:
                    best_score = node_score
                    best_graph_idx = cand_graph_idx
                    best_start_node = cand_start_node
            # 如果现有种子节点优于选择新种子节点
            if best_score >= self.c_uct * np.sqrt(np.log(simulation_n or 1)):
                graph_idx, start_node = best_graph_idx, best_start_node
                assert best_start_node in self.dataset[graph_idx].nodes
                graph = self.dataset[graph_idx]
            else:
                found = False
                while not found:
                    graph_idx = np.arange(len(self.dataset))[graph_dist.rvs()]
                    graph = self.dataset[graph_idx]
                    start_node = random.choice(list(graph.nodes))
                    # 不选择孤立节点或小的连通分量
                    if self.has_min_reachable_nodes(graph, start_node,
                        self.min_pattern_size):
                        found = True
                self.visited_seed_nodes.add((graph_idx, start_node))
            neigh = [start_node]
            frontier = list(set(graph.neighbors(start_node)) - set(neigh))
            visited = set([start_node])
            neigh_g = nx.Graph()
            neigh_g.add_node(start_node, anchor=1)
            cur_state = graph_idx, start_node
            state_list = [cur_state]
            while frontier and len(neigh) < self.max_size:
                frontier = self._prune_frontier(graph, frontier)
                cand_embs = self._get_candidate_embs(graph_idx, graph, neigh,
                    frontier)
                best_v_score, best_node_score, best_node = 0, -float("inf"), None
                for cand_node, cand_emb in zip(frontier, cand_embs):
                    cand_emb = cand_emb.to(utils.get_device())
                    score, n_embs = 0, 0
                    for emb_batch in self.embs:
                        score += torch.sum(self.model.predict((
                            emb_batch.to(utils.get_device()), cand_emb))).item()
                        n_embs += len(emb_batch)
                    v_score = -np.log(score/n_embs + 1) + 1
                    # 获取下一状态的 WL 哈希值
                    neigh_g = graph.subgraph(neigh + [cand_node]).copy()
                    neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
                    for v in neigh_g.nodes:
                        neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
                    next_state = utils.wl_hash(neigh_g,
                        node_anchored=self.node_anchored)
                    # 计算节点分数
                    parent_visit_counts = sum(self.visit_counts[cur_state].values())
                    my_visit_counts = sum(self.visit_counts[next_state].values())
                    q_score = (sum(self.cum_action_values[next_state].values()) /
                        (my_visit_counts or 1))
                    uct_score = self.c_uct * np.sqrt(np.log(parent_visit_counts or
                        1) / (my_visit_counts or 1))
                    node_score = q_score + uct_score
                    if node_score > best_node_score:
                        best_node_score = node_score
                        best_v_score = v_score
                        best_node = cand_node
                frontier = list(((set(frontier) |
                    set(graph.neighbors(best_node))) - visited) -
                    set([best_node]))
                visited.add(best_node)
                neigh.append(best_node)

                # 更新访问次数和 WL 缓存
                neigh_g = graph.subgraph(neigh).copy()
                neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
                for v in neigh_g.nodes:
                    neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
                prev_state = cur_state
                cur_state = utils.wl_hash(neigh_g, node_anchored=self.node_anchored)
                state_list.append(cur_state)
                self.wl_hash_to_graphs[cur_state].append(neigh_g)

            # 反向传播价值：将末端评分 best_v_score 沿路径累计。
            for i in range(0, len(state_list) - 1):
                self.cum_action_values[state_list[i]][
                    state_list[i+1]] += best_v_score
                self.visit_counts[state_list[i]][state_list[i+1]] += 1
        self.max_size += 1

    def finish_search(self):
        """按访问统计选出每个尺寸的高频候选模式。"""
        counts = defaultdict(lambda: defaultdict(int))
        for _, v in self.visit_counts.items():
            for s2, count in v.items():
                counts[len(random.choice(self.wl_hash_to_graphs[s2]))][s2] += count

        cand_patterns_uniq = []
        for pattern_size in range(self.min_pattern_size, self.max_pattern_size+1):
            for wl_hash, count in sorted(counts[pattern_size].items(), key=lambda
                x: x[1], reverse=True)[:self.out_batch_size]:
                cand_patterns_uniq.append(random.choice(
                    self.wl_hash_to_graphs[wl_hash]))
                print("- outputting", count, "motifs of size", pattern_size)
        return cand_patterns_uniq

class GreedySearchAgent(SearchAgent):
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, rank_method="counts",
        model_type="order", out_batch_size=20, n_beams=1,
        frontier_top_k=0, max_steps=1000):
        """子图模式搜索的贪心实现。
        算法在每一步贪心地选择下一个节点进行扩展，同时保持模式
        被预测为频繁的。选择下一动作的标准取决于子图匹配模型预测的分数
        （实际分数由 rank_method 参数决定）。

        参数说明：
            rank_method: 贪心搜索启发式需要一个分数来对可能的下一动作排序。
                如果 rank_method=='counts'，使用搜索树中该模式的计数；
                如果 rank_method=='margin'，使用匹配模型预测的该模式的 margin 分数；
                如果 rank_method=='hybrid'，同时考虑计数和 margin 对动作排序。
        """
        super().__init__(min_pattern_size, max_pattern_size, model, dataset,
            embs, node_anchored=node_anchored, analyze=analyze,
            model_type=model_type, out_batch_size=out_batch_size,
            frontier_top_k=frontier_top_k)
        self.rank_method = rank_method
        self.n_beams = n_beams
        self.max_steps = max_steps
        self.step_count = 0
        print("Rank Method:", rank_method)
        print("Max Steps:", max_steps)

    def init_search(self):
        """初始化贪心搜索 beam。"""
        self.step_count = 0
        ps = np.array([len(g) for g in self.dataset], dtype=float)
        ps /= np.sum(ps)
        graph_dist = stats.rv_discrete(values=(np.arange(len(self.dataset)), ps))

        beams = []
        for trial in range(self.n_trials):
            graph_idx = np.arange(len(self.dataset))[graph_dist.rvs()]
            graph = self.dataset[graph_idx]
            start_node = random.choice(list(graph.nodes))
            neigh = [start_node]
            frontier = list(set(graph.neighbors(start_node)) - set(neigh))
            visited = set([start_node])
            beams.append([(0, neigh, frontier, visited, graph_idx)])
        self.beam_sets = beams
        self.analyze_embs = []

    def is_search_done(self):
        return len(self.beam_sets) == 0 or self.step_count >= self.max_steps

    def step(self):
        """执行一轮贪心扩展。

        对每个 beam 状态，枚举 frontier 候选节点，
        基于匹配模型分数选择最优扩展。
        """
        self.step_count += 1
        print(f"Step {self.step_count}/{self.max_steps}, Active beams: {len(self.beam_sets)}")

        new_beam_sets = []
        print("seeds come from", len(set(b[0][-1] for b in self.beam_sets)),
            "distinct graphs")
        analyze_embs_cur = []
        for beam_set in tqdm(self.beam_sets):
            new_beams = []
            for _, neigh, frontier, visited, graph_idx in beam_set:
                graph = self.dataset[graph_idx]
                if len(neigh) >= self.max_pattern_size or not frontier: continue
                frontier = self._prune_frontier(graph, frontier)
                cand_embs = self._get_candidate_embs(graph_idx, graph, neigh,
                    frontier)
                best_score, best_node = float("inf"), None
                for cand_node, cand_emb in zip(frontier, cand_embs):
                    cand_emb = cand_emb.to(utils.get_device())
                    score, n_embs = 0, 0
                    for emb_batch in self.embs:
                        n_embs += len(emb_batch)
                        if self.model_type == "order":
                            score -= torch.sum(torch.argmax(
                                self.model.clf_model(self.model.predict((
                                emb_batch.to(utils.get_device()),
                                cand_emb)).unsqueeze(1)), axis=1)).item()
                        elif self.model_type == "mlp":
                            score += torch.sum(self.model(
                                emb_batch.to(utils.get_device()),
                                cand_emb.unsqueeze(0).expand(len(emb_batch), -1)
                                )[:,0]).item()
                        else:
                            print("未识别的模型类型")
                    if score < best_score:
                        best_score = score
                        best_node = cand_node
                    new_frontier = list(((set(frontier) |
                        set(graph.neighbors(cand_node))) - visited) -
                        set([cand_node]))
                    new_beams.append((
                        score, neigh + [cand_node],
                        new_frontier, visited | set([cand_node]), graph_idx))
            new_beams = list(sorted(new_beams, key=lambda x:
                x[0]))[:self.n_beams]
            for score, neigh, frontier, visited, graph_idx in new_beams[:1]:
                graph = self.dataset[graph_idx]
                # 添加到记录
                neigh_g = graph.subgraph(neigh).copy()
                neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
                for v in neigh_g.nodes:
                    neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
                self.cand_patterns[len(neigh_g)].append((score, neigh_g))
                if self.rank_method in ["counts", "hybrid"]:
                    self.counts[len(neigh_g)][utils.wl_hash(neigh_g,
                        node_anchored=self.node_anchored)].append(neigh_g)
                if self.analyze and len(neigh) >= 3:
                    emb = self.model.emb_model(utils.batch_nx_graphs(
                        [neigh_g], anchors=[neigh[0]] if self.node_anchored
                        else None)).squeeze(0)
                    analyze_embs_cur.append(emb.detach().cpu().numpy())
            if len(new_beams) > 0:
                new_beam_sets.append(new_beams)
        self.beam_sets = new_beam_sets
        self.analyze_embs.append(analyze_embs_cur)

    def finish_search(self):
        """根据 rank_method 汇总并去重输出模式。"""
        if self.analyze:
            print("Saving analysis info in results/analyze.p")
            with open("results/analyze.p", "wb") as f:
                pickle.dump((self.cand_patterns, self.analyze_embs), f)
            xs, ys = [], []
            for embs_ls in self.analyze_embs:
                for emb in embs_ls:
                    xs.append(emb[0])
                    ys.append(emb[1])
            print("Saving analysis plot in results/analyze.png")
            plt.scatter(xs, ys, color="red", label="motif")
            plt.legend()
            plt.savefig("plots/analyze.png")
            plt.close()

        cand_patterns_uniq = []
        for pattern_size in range(self.min_pattern_size, self.max_pattern_size+1):
            if self.rank_method == "hybrid":
                cur_rank_method = "margin" if len(max(
                    self.counts[pattern_size].values(), key=len)) < 3 else "counts"
            else:
                cur_rank_method = self.rank_method

            if cur_rank_method == "margin":
                wl_hashes = set()
                cands = self.cand_patterns[pattern_size]
                cand_patterns_uniq_size = []
                for pattern in sorted(cands, key=lambda x: x[0]):
                    wl_hash = utils.wl_hash(pattern[1],
                        node_anchored=self.node_anchored)
                    if wl_hash not in wl_hashes:
                        wl_hashes.add(wl_hash)
                        cand_patterns_uniq_size.append(pattern[1])
                        if len(cand_patterns_uniq_size) >= self.out_batch_size:
                            cand_patterns_uniq += cand_patterns_uniq_size
                            break
            elif cur_rank_method == "counts":
                for _, neighs in list(sorted(self.counts[pattern_size].items(),
                    key=lambda x: len(x[1]), reverse=True))[:self.out_batch_size]:
                    cand_patterns_uniq.append(random.choice(neighs))
            else:
                print("未识别的排名方法")
        return cand_patterns_uniq
