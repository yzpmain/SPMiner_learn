from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch_geometric.utils as pyg_utils

from src.analyze.count_patterns import count_graphlets
from src.core import dataset_registry
from src.core.utils import load_spminer_pickle, parse_gspan_output

__all__ = [
    "key_of_file",
    "quick_sig",
    "match_isomorphic_patterns",
    "evaluate_pair",
    "collect_files_from_summary",
    "summarize_supports",
    "exact_support_counts",
    "build_accuracy_table",
    "add_runtime_metrics",
]


PAIR_RE = re.compile(r"n(?P<n>\d+)_k(?P<k>\d+)")


def key_of_file(file_path: Path) -> Tuple[int, int] | None:
    m = PAIR_RE.search(file_path.stem)
    if not m:
        return None
    return int(m.group("n")), int(m.group("k"))


def quick_sig(g: nx.Graph) -> Tuple[int, int, Tuple[int, ...]]:
    deg = tuple(sorted((d for _, d in g.degree()), reverse=True))
    return g.number_of_nodes(), g.number_of_edges(), deg


def match_isomorphic_patterns(spminer_graphs: List[nx.Graph], gspan_graphs: List[nx.Graph]) -> List[Tuple[int, int]]:
    if not spminer_graphs or not gspan_graphs:
        return []

    adjacency: Dict[int, List[int]] = {}
    gspan_sigs = [quick_sig(g) for g in gspan_graphs]
    spminer_sigs = [quick_sig(g) for g in spminer_graphs]

    for i, sg in enumerate(spminer_graphs):
        for j, gg in enumerate(gspan_graphs):
            if spminer_sigs[i] != gspan_sigs[j]:
                continue
            if nx.is_isomorphic(sg, gg):
                adjacency.setdefault(i, []).append(j)

    bip = nx.Graph()
    left_nodes = [("s", i) for i in range(len(spminer_graphs))]
    right_nodes = [("g", j) for j in range(len(gspan_graphs))]
    bip.add_nodes_from(left_nodes, bipartite=0)
    bip.add_nodes_from(right_nodes, bipartite=1)

    for i, js in adjacency.items():
        for j in js:
            bip.add_edge(("s", i), ("g", j))

    matching = nx.algorithms.bipartite.maximum_matching(bip, top_nodes=set(left_nodes))
    pairs = []
    for i in range(len(spminer_graphs)):
        left = ("s", i)
        if left in matching:
            pairs.append((i, int(matching[left][1])))
    return pairs


def evaluate_pair(spminer_graphs: List[nx.Graph], gspan_graphs: List[nx.Graph], top_k: int) -> dict:
    if top_k > 0:
        spminer_graphs = spminer_graphs[:top_k]
        gspan_graphs = gspan_graphs[:top_k]

    n_sp = len(spminer_graphs)
    n_gs = len(gspan_graphs)
    matched_pairs = match_isomorphic_patterns(spminer_graphs, gspan_graphs)
    matched = len(matched_pairs)

    precision = matched / n_sp if n_sp > 0 else 0.0
    recall = matched / n_gs if n_gs > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = recall
    union = n_sp + n_gs - matched
    jaccard = matched / union if union > 0 else 1.0

    return {
        "spminer_count": n_sp,
        "gspan_count": n_gs,
        "isomorphic_matches": matched,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "jaccard": round(jaccard, 6),
    }


def collect_files_from_summary(summary_csv: Path) -> Tuple[Dict[Tuple[int, int], Path], Dict[Tuple[int, int], Path]]:
    df = pd.read_csv(summary_csv)
    sp_files: Dict[Tuple[int, int], Path] = {}
    gs_files: Dict[Tuple[int, int], Path] = {}

    for _, row in df.iterrows():
        src = str(row["source"]).strip().lower()
        file_path = Path(str(row["file"]))
        key = key_of_file(file_path)
        if key is None:
            continue
        if src == "spminer":
            sp_files[key] = file_path
        elif src == "gspan":
            gs_files[key] = file_path

    return sp_files, gs_files


def _support_values(graphs: List[nx.Graph]) -> List[float]:
    values: List[float] = []
    for graph in graphs:
        support = graph.graph.get("support")
        if support is None:
            continue
        try:
            values.append(float(support))
        except Exception:
            continue
    return values


def summarize_supports(graphs: List[nx.Graph], prefix: str) -> dict:
    values = _support_values(graphs)
    if not values:
        return {
            f"{prefix}_support_count": 0,
            f"{prefix}_support_min": np.nan,
            f"{prefix}_support_max": np.nan,
            f"{prefix}_support_mean": np.nan,
            f"{prefix}_support_median": np.nan,
        }

    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_support_count": int(arr.size),
        f"{prefix}_support_min": float(np.min(arr)),
        f"{prefix}_support_max": float(np.max(arr)),
        f"{prefix}_support_mean": float(np.mean(arr)),
        f"{prefix}_support_median": float(np.median(arr)),
    }


def _load_target_graphs(dataset_name: str) -> List[nx.Graph]:
    normalized = dataset_registry.validate_dataset_name(dataset_name, "count")
    dataset, _ = dataset_registry.load_dataset_for_stage(normalized, "count")
    targets: List[nx.Graph] = []
    for item in dataset:
        if isinstance(item, nx.Graph):
            graph = item.copy()
        else:
            graph = pyg_utils.to_networkx(item).to_undirected()
        targets.append(graph)
    return targets


def exact_support_counts(
    graphs: List[nx.Graph],
    dataset_name: str,
    node_anchored: bool,
    n_workers: int | None = None,
) -> List[int]:
    if not graphs:
        return []

    targets = _load_target_graphs(dataset_name)
    worker_count = n_workers if n_workers and n_workers > 0 else max(1, min(4, os.cpu_count() or 1))
    return count_graphlets(
        graphs,
        targets,
        n_workers=worker_count,
        method="bin",
        node_anchored=node_anchored,
        progress_every=0,
    )


def build_accuracy_table(
    benchmark_df: pd.DataFrame,
    dataset_name: str | None,
    top_k: int,
    node_anchored: bool = False,
    exact_frequency: bool = True,
    frequency_workers: int | None = None,
) -> pd.DataFrame:
    rows = []
    for _, row in benchmark_df.iterrows():
        sp_path = Path(str(row["spminer_result_file"]))
        gs_path = Path(str(row["gspan_result_file"]))

        row_dataset_name = str(row.get("frequency_dataset", dataset_name or "")).strip()

        row_out = {
            "run_id": row.get("run_id", np.nan),
            "graph_size": row.get("graph_size", np.nan),
            "k": row.get("k", np.nan),
            "spminer_result_file": str(sp_path),
            "gspan_result_file": str(gs_path),
        }

        try:
            sp_graphs = load_spminer_pickle(sp_path)
        except Exception as exc:
            row_out["analysis_status"] = f"spminer_load_failed: {exc}"
            rows.append(row_out)
            continue

        try:
            gs_graphs = parse_gspan_output(gs_path)
        except Exception as exc:
            row_out["analysis_status"] = f"gspan_load_failed: {exc}"
            rows.append(row_out)
            continue

        metrics = evaluate_pair(sp_graphs, gs_graphs, top_k)
        row_out.update(metrics)
        row_out.update(summarize_supports(sp_graphs[:top_k] if top_k > 0 else sp_graphs, "spminer"))
        row_out.update(summarize_supports(gs_graphs[:top_k] if top_k > 0 else gs_graphs, "gspan"))
        row_out["analysis_status"] = "ok"

        if exact_frequency and row_dataset_name:
            effective_node_anchored = node_anchored
            if effective_node_anchored:
                has_anchor = any(
                    any(data.get("anchor", 0) == 1 for _, data in graph.nodes(data=True))
                    for graph in sp_graphs[:top_k] if graph is not None
                )
                effective_node_anchored = has_anchor

            try:
                selected_sp_graphs = sp_graphs[:top_k] if top_k > 0 else sp_graphs
                selected_gs_graphs = gs_graphs[:top_k] if top_k > 0 else gs_graphs
                exact_counts = exact_support_counts(
                    selected_sp_graphs,
                    row_dataset_name,
                    node_anchored=effective_node_anchored,
                    n_workers=frequency_workers,
                )
                row_out["spminer_exact_support_count"] = int(len(exact_counts))
                row_out["spminer_exact_support_mean"] = float(np.mean(exact_counts)) if exact_counts else np.nan
                row_out["spminer_exact_support_max"] = float(np.max(exact_counts)) if exact_counts else np.nan
                row_out["spminer_exact_support_min"] = float(np.min(exact_counts)) if exact_counts else np.nan

                matched_pairs = match_isomorphic_patterns(selected_sp_graphs, selected_gs_graphs)
                gaps = []
                ratios = []
                mape_terms = []
                for sp_idx, gs_idx in matched_pairs:
                    if sp_idx >= len(exact_counts) or gs_idx >= len(selected_gs_graphs):
                        continue
                    exact = float(exact_counts[sp_idx])
                    support = selected_gs_graphs[gs_idx].graph.get("support")
                    if support is None:
                        continue
                    support = float(support)
                    gap = abs(support - exact)
                    gaps.append(gap)
                    mape_terms.append(gap / exact if exact > 0 else np.nan)
                    if exact > 0:
                        ratios.append(support / exact)

                row_out["frequency_pair_count"] = len(gaps)
                row_out["frequency_mae"] = float(np.mean(gaps)) if gaps else np.nan
                row_out["frequency_rmse"] = float(np.sqrt(np.mean(np.square(gaps)))) if gaps else np.nan
                row_out["frequency_ratio_mean"] = float(np.mean(ratios)) if ratios else np.nan
                mape_terms = [value for value in mape_terms if np.isfinite(value)]
                row_out["frequency_mape"] = float(np.mean(mape_terms)) if mape_terms else np.nan
            except Exception as exc:
                row_out["exact_frequency_status"] = f"failed: {exc}"

        rows.append(row_out)

    if not rows:
        return pd.DataFrame(columns=["graph_size", "k"])

    out_df = pd.DataFrame(rows)
    sort_cols = [c for c in ["graph_size", "k"] if c in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols).reset_index(drop=True)
    return out_df


def add_runtime_metrics(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if {"gspan_time", "spminer_time"}.issubset(enriched.columns):
        enriched["time_speedup"] = enriched.apply(
            lambda row: (row["gspan_time"] / row["spminer_time"])
            if pd.notna(row["gspan_time"]) and pd.notna(row["spminer_time"]) and row["spminer_time"] > 0
            else np.nan,
            axis=1,
        )
        enriched["time_delta"] = enriched["gspan_time"] - enriched["spminer_time"]
    if {"gspan_mem", "spminer_mem"}.issubset(enriched.columns):
        enriched["memory_ratio"] = enriched.apply(
            lambda row: (row["gspan_mem"] / row["spminer_mem"])
            if pd.notna(row["gspan_mem"]) and pd.notna(row["spminer_mem"]) and row["spminer_mem"] > 0
            else np.nan,
            axis=1,
        )
        enriched["memory_delta"] = enriched["gspan_mem"] - enriched["spminer_mem"]
    return enriched