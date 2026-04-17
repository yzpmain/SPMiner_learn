"""公共编码器参数注册。

将两个阶段（子图匹配训练 & SPMiner 解码）均需要的编码器结构参数
提取到此处，供 subgraph_matching/config.py 和 subgraph_mining/config.py
统一调用，消除重复 add_argument 代码。
"""
import argparse


def add_encoder_args(parser):
    """向解析器注册编码器结构参数（两阶段共用）。

    包含：卷积类型、网络层数、隐层维度、skip 连接、dropout、
    模型路径、method_type、节点锚定标志。

    参数：
        parser: argparse.ArgumentParser 或其 argument_group。
    """
    enc = parser.add_argument_group("编码器结构参数（共用）")
    enc.add_argument('--conv_type', type=str,
                     help='卷积类型（GCN / GIN / SAGE / graph / GAT / gated / PNA）')
    enc.add_argument('--method_type', type=str,
                     help='嵌入类型（order / mlp）')
    enc.add_argument('--n_layers', type=int,
                     help='图卷积层数')
    enc.add_argument('--hidden_dim', type=int,
                     help='隐层维度')
    enc.add_argument('--skip', type=str,
                     help='"all"、"learnable" 或 "last"')
    enc.add_argument('--dropout', type=float,
                     help='Dropout 比率')
    enc.add_argument('--model_path', type=str,
                     help='模型保存/加载路径')
    enc.add_argument('--node_anchored', action="store_true",
                     help='是否使用节点锚定')
