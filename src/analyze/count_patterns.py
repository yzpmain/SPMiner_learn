"""
统计查询图（graphlet / pattern）在一组目标图中的出现次数。
支持 bin（是否存在）和 freq（不同嵌入数）两种计数方式，以及节点锚定模式。
使用多进程并行加速。
"""

import argparse
import copy
import csv
import json
import os
import traceback

import numpy as np

from torch_geometric.datasets import TUDataset
import torch_geometric.utils as pyg_utils

from src.core import CoreFacade
from src.core import utils
from src.core.artifacts import choose_cli_output_path, task_output_dir, write_manifest
from src.logger import RunLogger, info, warning, section, progress

from multiprocessing import Pool
import random
import networkx as nx
import networkx.algorithms.isomorphism as iso
import pickle

from src.core.cli import add_runtime_args, setup_runtime

__all__ = [
    "count_graphlets",
    "preprocess_query",
    "preprocess_target",
    "dedup_isomorphic_queries",
]

def arg_parse():
    parser = argparse.ArgumentParser(description='统计图中的图元')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--queries_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--count_method', type=str)
    parser.add_argument('--analysis_path', type=str,
                        help='analyze 模式输入文件路径')
    parser.add_argument('--max_queries', type=int,
                        help='仅统计前 max_queries 个模式，0 表示使用全部')
    parser.add_argument('--progress_every', type=int,
                        help='每处理多少个任务打印一次进度，0 表示关闭')
    parser.add_argument('--node_anchored', action="store_true")
    add_runtime_args(parser, include_gpu=False, include_seed=True,
                     include_n_workers=True, include_progress_write_interval=True,
                     include_output_policy=True)
    parser.set_defaults(dataset="enzymes",
                        queries_path="results/out-patterns.p",
                        out_path="results/counts.json",
                        analysis_path="results/analyze.p",
                        n_workers=4,
                        count_method="bin",
                        max_queries=0,
                        progress_every=1000)
    return parser.parse_args()


def preprocess_query(query, node_anchored):
    query = query.copy()
    query.remove_edges_from(nx.selfloop_edges(query))
    degree_seq = tuple(sorted((d for _, d in query.degree()), reverse=True))
    anchor_count = (sum(1 for _, data in query.nodes(data=True)
                        if data.get("anchor", 0) == 1)
                    if node_anchored else 0)
    info = {
        "graph": query,
        "n_nodes": query.number_of_nodes(),
        "n_edges": query.number_of_edges(),
        "degree_seq": degree_seq,
        "anchor_count": anchor_count,
    }
    return info


def preprocess_target(target, node_anchored):
    target = copy.deepcopy(target)
    target.remove_edges_from(nx.selfloop_edges(target))
    n_nodes = target.number_of_nodes()
    degree_seq = tuple(sorted((d for _, d in target.degree()), reverse=True))
    anchor_count = n_nodes if node_anchored else 0
    return {
        "graph": target,
        "n_nodes": n_nodes,
        "n_edges": target.number_of_edges(),
        "degree_seq": degree_seq,
        "anchor_count": anchor_count,
    }


def dedup_isomorphic_queries(queries, node_anchored=False):
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
    query = query_info["graph"]
    target = target_info["graph"]

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
    if node_anchored and query_info["anchor_count"] > target_info["n_nodes"]:
        return 0

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


if __name__ == "__main__":
    args = arg_parse()
    setup_runtime(args)
    with RunLogger(args):
        section("模式计数")
        info("Using {} workers".format(args.n_workers))

        artifact_dir = task_output_dir(args, "count", args.dataset)
        args.out_path = str(choose_cli_output_path(
            args,
            args.out_path,
            default_cli_path="results/counts.json",
            suggested_default_path=artifact_dir / "counts.json",
        ))
        if args.analysis_path == "results/analyze.p":
            args.analysis_path = str(artifact_dir / "analyze.p")

        if args.dataset != "analyze" and not os.path.exists(args.queries_path):
            raise FileNotFoundError(f"Queries file not found: {args.queries_path}")

        if args.dataset == "analyze":
            with open(args.analysis_path, "rb") as f:
                cand_patterns, _ = pickle.load(f)
                # 取第一个可用尺寸的模式（优先取 >= 3 的尺寸）
                avail_sizes = sorted(cand_patterns.keys())
                target_size = next((s for s in avail_sizes if s >= 3), avail_sizes[0]) if avail_sizes else 10
                queries = [q for score, q in cand_patterns[target_size]][:200]
            dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        else:
            normalized_dataset, dataset, _ = CoreFacade.load_stage_dataset(args.dataset, "count")
            args.dataset = normalized_dataset

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

        # 直接对查询进行计数（已移除基线对比功能）
        query_lens = [len(q) for q in queries]
        n_matches = count_graphlets(queries, targets,
                                    n_workers=args.n_workers,
                                    method=args.count_method,
                                    node_anchored=args.node_anchored,
                                    progress_every=args.progress_every)

        with open(args.out_path, "w") as f:
            json.dump((query_lens, n_matches, []), f)
        info("Counts saved → {}".format(args.out_path))
        write_manifest(
            artifact_dir / "manifest.json",
            args,
            outputs={
                "counts": args.out_path,
                "queries": args.queries_path,
                "analysis_input": args.analysis_path if args.dataset == "analyze" else None,
            },
            task="count",
        )