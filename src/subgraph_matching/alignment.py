"""为查询子图在目标图中的匹配构建对齐矩阵。
子图匹配模型需要使用节点锚定选项进行训练（默认）。"""

import argparse
import os
import pickle
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from src.core import data
from src.core import models
from src.core import utils
from src.subgraph_matching.config import parse_encoder
from src.subgraph_matching.test import validation
from src.subgraph_matching.train import build_model
from src.logger import RunLogger, info

def gen_alignment_matrix(model, query, target, method_type="order"):
    """为给定的查询图和目标图生成子图匹配对齐矩阵。
    矩阵中每个条目 (u, v) 包含模型对以 u 为锚点的查询图
    是以 v 为锚点的目标图的子图的置信度分数。

    参数说明：
        model: 子图匹配模型，必须使用节点锚定设置训练（--node_anchored，默认）
        query: 查询图（networkx Graph）
        target: 目标图（networkx Graph）
        method_type: 模型使用的方法。
            "order" 表示序嵌入，"mlp" 表示 MLP 模型
    """

    mat = np.zeros((len(query), len(target)))
    for i, u in enumerate(query.nodes):
        for j, v in enumerate(target.nodes):
            batch = utils.batch_nx_graphs([query, target], anchors=[u, v])
            embs = model.emb_model(batch)
            pred = model(embs[1].unsqueeze(0), embs[0].unsqueeze(0))
            raw_pred = model.predict(pred)
            if method_type == "order":
                raw_pred = torch.log(raw_pred)
            elif method_type == "mlp":
                raw_pred = raw_pred[0][1]
            mat[i][j] = raw_pred.item()
    return mat

def main():
    parser = argparse.ArgumentParser(description='对齐矩阵参数')
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    parser.add_argument('--query_path', type=str, help='查询图的路径',
        default="")
    parser.add_argument('--target_path', type=str, help='目标图的路径',
        default="")
    args = parser.parse_args()
    args.test = True

    if not os.path.exists("plots/"):
        os.makedirs("plots/")
    if not os.path.exists("results/"):
        os.makedirs("results/")

    with RunLogger(args):
        if args.query_path:
            with open(args.query_path, "rb") as f:
                query = pickle.load(f)
        else:
            query = nx.gnp_random_graph(8, 0.25)
        if args.target_path:
            with open(args.target_path, "rb") as f:
                target = pickle.load(f)
        else:
            target = nx.gnp_random_graph(16, 0.25)

        model = build_model(args)
        mat = gen_alignment_matrix(model, query, target,
            method_type=args.method_type)

        np.save("results/alignment.npy", mat)
        info("Alignment matrix saved → results/alignment.npy")

        plt.imshow(mat, interpolation="nearest")
        plt.savefig("plots/alignment.png")
        info("Alignment matrix plot saved → plots/alignment.png")

if __name__ == '__main__':
    main()

