"""Compare old vs current NetworkX frequency counting semantics.

Old semantics (restored from git commit 4bc8356):
- count all subgraph isomorphisms
- divide by query automorphism count (n_symmetries)

Current semantics (src/analyze/count_patterns.py):
- count unique target-node sets returned by GraphMatcher

This script compares both per query pattern and writes a CSV report.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import networkx as nx
import networkx.algorithms.isomorphism as iso

from src.analyze import count_patterns as cp
from src.core import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare old vs new NetworkX freq semantics")
    parser.add_argument("--dataset", type=str, default="facebook_combined_50")
    parser.add_argument("--queries_path", type=str, default="results/facebook50-patterns.p")
    parser.add_argument("--out_csv", type=str, default="results/facebook50-networkx-old-vs-new.csv")
    parser.add_argument("--node_anchored", action="store_true")
    return parser.parse_args()


def load_dataset(dataset_name: str):
    if dataset_name.startswith("facebook_combined"):
        return [utils.load_snap_edgelist(f"data/{dataset_name}.txt")]
    if dataset_name == "facebook":
        return [utils.load_snap_edgelist("data/facebook_combined.txt")]
    raise ValueError(f"Unsupported dataset for this comparison: {dataset_name}")


def old_freq_pair(query_info, target_info, node_anchored: bool) -> float:
    query = query_info["graph"]
    target = target_info["graph"]

    if query_info["n_nodes"] > target_info["n_nodes"]:
        return 0.0
    if query_info["n_edges"] > target_info["n_edges"]:
        return 0.0
    q_deg = query_info["degree_seq"]
    t_deg = target_info["degree_seq"]
    for q, t in zip(q_deg, t_deg[: len(q_deg)]):
        if q > t:
            return 0.0

    if node_anchored:
        query_anchor_count = query_info["anchor_count"]
        target_anchor_count = target_info["anchor_count"]
        if query_anchor_count and target_anchor_count == 0:
            return 0.0
        if query_anchor_count > target_anchor_count:
            return 0.0

    matcher = iso.GraphMatcher(target, query)
    n_symmetries = query_info.get("n_symmetries") or 1
    return len(list(matcher.subgraph_isomorphisms_iter())) / n_symmetries


def new_freq_pair(query_info, target_info, node_anchored: bool) -> float:
    return float(cp._count_one_pair(query_info, target_info, "freq", node_anchored))


def compare(dataset_name: str, queries_path: str, out_csv: str, node_anchored: bool) -> None:
    with open(queries_path, "rb") as f:
        queries = pickle.load(f)

    targets = []
    for graph in load_dataset(dataset_name):
        if not isinstance(graph, nx.Graph):
            graph = nx.Graph(graph)
        targets.append(graph)

    query_infos = [cp.preprocess_query(q, "freq", node_anchored) for q in queries]
    target_infos = [cp.preprocess_target(t, node_anchored) for t in targets]

    work_queries, query_to_unique = cp.dedup_isomorphic_queries(query_infos, node_anchored=node_anchored)

    rows = []
    old_unique = []
    new_unique = []

    for idx, q_info in enumerate(work_queries):
        old_count = 0.0
        new_count = 0.0
        for t_info in target_infos:
            old_count += old_freq_pair(q_info, t_info, node_anchored)
            new_count += new_freq_pair(q_info, t_info, node_anchored)
        old_unique.append(old_count)
        new_unique.append(new_count)

    old_counts = [old_unique[idx] for idx in query_to_unique]
    new_counts = [new_unique[idx] for idx in query_to_unique]

    for i, q_info in enumerate(query_infos):
        rows.append({
            "query_index": i,
            "n_nodes": q_info["n_nodes"],
            "n_edges": q_info["n_edges"],
            "old_count": old_counts[i],
            "new_count": new_counts[i],
            "diff": new_counts[i] - old_counts[i],
            "ratio": (new_counts[i] / old_counts[i]) if old_counts[i] else None,
        })

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_index", "n_nodes", "n_edges", "old_count", "new_count", "diff", "ratio"])
        writer.writeheader()
        writer.writerows(rows)

    old_total = sum(old_counts)
    new_total = sum(new_counts)
    worst = max(rows, key=lambda r: abs(r["diff"])) if rows else None

    print(json.dumps({
        "queries": len(rows),
        "old_total": old_total,
        "new_total": new_total,
        "delta": new_total - old_total,
        "worst_query": worst,
        "out_csv": str(out_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    args = parse_args()
    compare(args.dataset, args.queries_path, args.out_csv, args.node_anchored)
