import argparse
from common import utils

def parse_decoder(parser):
    """注册 SPMiner 解码阶段参数。

    该函数仅负责把“子图挖掘阶段”所需参数挂到现有 argparse 解析器上，
    不直接执行任何挖掘逻辑。它通常与编码器参数组合使用。

    参数：
        parser: 外部创建的 argparse.ArgumentParser。
    """
    dec_parser = parser.add_argument_group()
    dec_parser.add_argument('--sample_method', type=str,
                        help='"tree"（树形）或 "radial"（辐射形）')
    dec_parser.add_argument('--motif_dataset', type=str,
                        help='模体数据集')
    dec_parser.add_argument('--radius', type=int,
                        help='节点邻域半径')
    dec_parser.add_argument('--subgraph_sample_size', type=int,
                        help='每个邻域中采样的节点数')
    dec_parser.add_argument('--out_path', type=str,
                        help='候选模体输出路径')
    dec_parser.add_argument('--n_clusters', type=int)
    dec_parser.add_argument('--min_pattern_size', type=int)
    dec_parser.add_argument('--max_pattern_size', type=int)
    dec_parser.add_argument('--min_neighborhood_size', type=int)
    dec_parser.add_argument('--max_neighborhood_size', type=int)
    dec_parser.add_argument('--n_neighborhoods', type=int)
    dec_parser.add_argument('--n_trials', type=int,
                        help='搜索试验次数')
    dec_parser.add_argument('--out_batch_size', type=int,
                        help='每种图大小输出的模体数量')
    dec_parser.add_argument('--frontier_top_k', type=int,
                        help='每步仅保留度数最高的前 K 个 frontier 候选，0 表示不剪枝')
    dec_parser.add_argument('--analyze', action="store_true")
    dec_parser.add_argument('--search_strategy', type=str,
                        help='"greedy"（贪心）或 "mcts"')
    dec_parser.add_argument('--use_whole_graphs', action="store_true",
        help="是否对完整图聚类而非采样节点邻域")

    # 这些默认值对应论文/仓库中常用的挖掘配置，
    # 其中 n_neighborhoods 与 n_trials 会显著影响耗时。
    dec_parser.set_defaults(out_path="results/out-patterns.p",
                        n_neighborhoods=10000,
                        n_trials=1000,
                        decode_thresh=0.5,
                        radius=3,
                        subgraph_sample_size=0,
                        sample_method="tree",
                        skip="learnable",
                        min_pattern_size=5,
                        max_pattern_size=20,
                        min_neighborhood_size=20,
                        max_neighborhood_size=29,
                        search_strategy="greedy",
                        out_batch_size=10,
                        frontier_top_k=5,
                        node_anchored=True)

    # 挖掘阶段会复用编码器参数中的 dataset 与 batch_size，
    # 这里覆盖为更适合解码场景的默认值。
    parser.set_defaults(dataset="enzymes",
                        batch_size=1000)
