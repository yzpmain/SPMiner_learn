"""ER 随机图基线生成与挖掘。

对每个数据集生成同等规模 (|V|, |E| 相同) 的 ER 随机图，
运行 SPMiner 挖掘，比较模式频次。
"""

from __future__ import annotations

import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from src.subgraph_mining.pipeline import PatternGrowthPipeline
from src.analyze.count_patterns import count_graphlets
from src.logger import info, section

__all__ = ["generate_er_graphs", "run_er_baseline"]


# ---------------------------------------------------------------------------
# 内联辅助函数 (避免循环引用 experiment.py)
# ---------------------------------------------------------------------------
def _compute_stats(patterns, counts, elapsed):
    size_counter = Counter(len(g) for g in patterns)
    freq_by_size = defaultdict(list)
    for pat, cnt in zip(patterns, counts):
        freq_by_size[len(pat)].append(cnt)
    sorted_sizes = sorted(freq_by_size.keys())
    return {
        "n_patterns": len(patterns),
        "n_unique_patterns": len(set(tuple(sorted(g.edges())) for g in patterns)),
        "size_distribution": dict(size_counter),
        "mean_pattern_size": float(np.mean([len(g) for g in patterns])) if patterns else 0,
        "freq_by_size": {str(s): {"mean": float(np.mean(freq_by_size[s])),
            "median": float(np.median(freq_by_size[s])),
            "max": float(np.max(freq_by_size[s])),
            "min": float(np.min(freq_by_size[s]))} for s in sorted_sizes},
        "total_counts": int(np.sum(counts)),
        "time_seconds": elapsed,
    }


def _save_results(out_dir: Path, patterns, counts, stats):
    with open(out_dir / "patterns.p", "wb") as f:
        pickle.dump(patterns, f)
    with open(out_dir / "counts.json", "w") as f:
        json.dump(([len(g) for g in patterns], counts, []), f)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    info("基线结果保存至 {}".format(out_dir))


# ---------------------------------------------------------------------------
# ER 生成
# ---------------------------------------------------------------------------
def generate_er_graphs(
    real_graphs: list[nx.Graph],
    seed: int = 42,
    max_attempts: int = 50,
) -> list[nx.Graph]:
    """为每个真实图生成同规模 ER 随机图 (G(n,m))。"""
    rng = np.random.RandomState(seed)
    er_graphs = []
    n_skipped = 0
    for g in real_graphs:
        n, m = len(g), g.number_of_edges()
        if n < 2 or m < 1:
            n_skipped += 1
            continue
        found = False
        for _ in range(max_attempts):
            eg = nx.gnm_random_graph(n, m, seed=rng.randint(2**31))
            if nx.is_connected(eg):
                er_graphs.append(eg)
                found = True
                break
        if not found:
            eg = nx.gnm_random_graph(n, m, seed=rng.randint(2**31))
            er_graphs.append(eg)
    info("ER 基线: 生成 {} 个图 (跳过 {} 个)".format(len(er_graphs), n_skipped))
    return er_graphs


# ---------------------------------------------------------------------------
# 基线挖掘
# ---------------------------------------------------------------------------
def run_er_baseline(
    dataset_name: str,
    model,
    args,
    real_graphs: list[nx.Graph],
    out_dir: Path,
    dry_run: bool = False,
    count_method: str = "bin",
    count_sample_size: int = 100,
) -> dict:
    """对 ER 随机图运行 SPMiner 挖掘作为基线。"""
    section("ER 基线对比: {}".format(dataset_name))
    baseline_dir = out_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    info("生成 ER 随机图 ...")
    er_graphs = generate_er_graphs(real_graphs)
    info("ER 图: {} 个".format(len(er_graphs)))

    info("开始 ER 基线挖掘 ...")
    t0 = time.time()
    pipeline = PatternGrowthPipeline(args, model, er_graphs, "graph")
    patterns = pipeline.run()
    mining_time = time.time() - t0
    info("ER 基线挖掘: {} 个模式, 耗时 {:.1f}s".format(len(patterns), mining_time))

    if not patterns:
        stats = _compute_stats([], [], mining_time)
        _save_results(baseline_dir, patterns, [], stats)
        return stats

    info("ER 基线计数 ...")
    targets = er_graphs
    if count_method == "sample" and len(targets) > count_sample_size:
        import random
        rng = random.Random(42)
        targets = rng.sample(targets, count_sample_size)
    t0 = time.time()
    counts = count_graphlets(
        patterns, targets,
        n_workers=args.n_workers,
        method=count_method if count_method != "sample" else "bin",
        node_anchored=args.node_anchored,
        progress_every=0,
    )
    count_time = time.time() - t0
    stats = _compute_stats(patterns, counts, mining_time + count_time)
    _save_results(baseline_dir, patterns, counts, stats)
    info("ER 基线完成")
    return stats
