"""实验汇总分析与可视化。

读取 expdata/outputs/<dataset>/summary.json 生成汇总表和图表。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from main.config import EXPERIMENT_DATASETS, OUT_DIR, PLOT_DIR, DATA_ROOT

__all__ = [
    "build_summary_table",
    "plot_size_distribution",
    "plot_top_patterns",
    "plot_network_properties",
    "analyze_all",
]


def _load_all_results() -> dict:
    """加载所有数据集的实验结果。"""
    results = {}
    for name in EXPERIMENT_DATASETS:
        summary_path = OUT_DIR / name / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                results[name] = json.load(f)
    return results


def build_summary_table(all_results: dict | None = None) -> str:
    """生成 Markdown 格式的 thesis 汇总表。"""
    if all_results is None:
        all_results = _load_all_results()
    if not all_results:
        return "(无结果数据)"

    lines = []
    lines.append("| 数据集 | 类型 | 模式数 | 平均尺寸 | 总频次 | 耗时(s) |")
    lines.append("|--------|------|--------|----------|--------|---------|")
    for name, cfg in EXPERIMENT_DATASETS.items():
        res = all_results.get(name, {})
        if not res:
            continue
        lines.append("| {} | {} | {} | {:.1f} | {} | {:.1f} |".format(
            cfg["label"],
            "人工" if cfg["type"] == "synthetic" else "真实",
            res.get("n_patterns", 0),
            res.get("mean_pattern_size", 0),
            res.get("total_counts", 0),
            res.get("time_seconds", 0),
        ))
    return "\n".join(lines)


def plot_size_distribution(all_results: dict | None = None) -> Path:
    """各数据集模式尺寸分布 (分组柱状图)。"""
    if all_results is None:
        all_results = _load_all_results()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = []
    all_data = []
    for name, cfg in EXPERIMENT_DATASETS.items():
        res = all_results.get(name)
        if not res:
            continue
        sd = res.get("size_distribution", {})
        if not sd:
            continue
        x_labels.append(cfg["label"])
        sizes = []
        # 按尺寸扩展为列表
        for size_str, count in sd.items():
            sizes.extend([int(size_str)] * count)
        all_data.append(sizes)

    if all_data:
        ax.boxplot(all_data, labels=x_labels)
        ax.set_xlabel("数据集")
        ax.set_ylabel("模式尺寸")
        ax.set_title("各数据集模式尺寸分布")
        plt.xticks(rotation=15)
        path = PLOT_DIR / "size_distribution.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path
    plt.close()
    return PLOT_DIR / "size_distribution.png"


def plot_top_patterns(all_results: dict | None = None, top_k: int = 5) -> Path:
    """各数据集 Top-K 模式频次对比。"""
    if all_results is None:
        all_results = _load_all_results()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    names = []
    freqs = []
    for name, cfg in EXPERIMENT_DATASETS.items():
        res = all_results.get(name)
        if not res:
            continue
        fb = res.get("freq_by_size", {})
        if fb:
            # 取所有尺寸中最大 median 频次
            max_median = max(v["median"] for v in fb.values())
            names.append(cfg["label"])
            freqs.append(max_median)

    if freqs:
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:len(freqs)]
        ax.bar(names, freqs, color=colors)
        ax.set_ylabel("中位数频次 (log)")
        ax.set_yscale("log")
        ax.set_title("各数据集 Top 模式频次对比")
        plt.xticks(rotation=15)
        path = PLOT_DIR / "top_patterns.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path
    plt.close()
    return PLOT_DIR / "top_patterns.png"


def plot_network_properties() -> Path:
    """网络基本属性对比。"""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 统计各数据集的网络属性
    props = {}
    for name, cfg in EXPERIMENT_DATASETS.items():
        counts_path = OUT_DIR / name / "counts.json"
        if not counts_path.exists():
            continue
        # 从计数文件获取模式信息
        try:
            import pickle
            patterns_path = OUT_DIR / name / "patterns.p"
            if patterns_path.exists():
                with open(patterns_path, "rb") as f:
                    patterns = pickle.load(f)
                sizes = [len(g) for g in patterns]
                edges = [g.number_of_edges() for g in patterns]
                props[cfg["label"]] = {
                    "avg_nodes": np.mean(sizes) if sizes else 0,
                    "avg_edges": np.mean(edges) if edges else 0,
                    "density": np.mean([2*e/(s*(s-1)) if s > 1 else 0
                                        for s, e in zip(sizes, edges)]),
                }
        except Exception:
            continue

    if not props:
        return PLOT_DIR / "network_properties.png"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = list(props.keys())
    metrics = [
        ("avg_nodes", "平均节点数"),
        ("avg_edges", "平均边数"),
        ("density", "平均密度"),
    ]
    for ax, (key, ylabel) in zip(axes, metrics):
        vals = [props[l][key] for l in labels]
        ax.bar(labels, vals, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(labels, rotation=15)

    plt.tight_layout()
    path = PLOT_DIR / "network_properties.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def analyze_all():
    """生成全部汇总输出。"""
    results = _load_all_results()
    if not results:
        print("无实验结果。请先运行 python -m main.run_all")
        return

    print(build_summary_table(results))
    print()
    print("生成图表 ...")
    plot_size_distribution(results)
    plot_top_patterns(results)
    plot_network_properties()
    print("图表保存至 {}".format(PLOT_DIR))


if __name__ == "__main__":
    analyze_all()
