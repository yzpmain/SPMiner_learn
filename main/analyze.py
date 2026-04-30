"""实验分析与报告生成。

按数据集生成独立分析报告 (Markdown + 图表)，含基线对比。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
for font in ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
    try:
        matplotlib.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue
import matplotlib.pyplot as plt
import numpy as np

from main.config import EXPERIMENT_DATASETS, OUT_DIR, PLOT_DIR

__all__ = ["generate_report", "generate_all_reports"]


def _load_summary(dataset_name: str) -> dict | None:
    p = OUT_DIR / dataset_name / "summary.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _load_baseline(dataset_name: str) -> dict | None:
    p = OUT_DIR / dataset_name / "baseline" / "summary.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _plot_size_distribution(stats: dict, out_path: Path):
    """模式尺寸分布柱状图。"""
    sd = stats.get("size_distribution", {})
    if not sd:
        return
    sizes = sorted(int(k) for k in sd)
    counts = [sd[str(s)] for s in sizes]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(sizes, counts, width=0.6, edgecolor="black")
    ax.set_xlabel("模式尺寸")
    ax.set_ylabel("数量")
    ax.set_title("模式尺寸分布")
    ax.set_xticks(sizes)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_baseline_comparison(stats: dict, baseline: dict, out_path: Path):
    """SPMiner vs ER 基线频次对比。"""
    real_fb = stats.get("freq_by_size", {})
    base_fb = baseline.get("freq_by_size", {})
    if not real_fb:
        return
    sizes = sorted(int(k) for k in real_fb)
    real_medians = [real_fb[str(s)]["median"] for s in sizes if str(s) in real_fb]
    base_medians = [base_fb.get(str(s), {}).get("median", 0) for s in sizes if str(s) in real_fb]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(sizes))
    w = 0.35
    ax.bar(x - w/2, real_medians, w, label="SPMiner (真实)", edgecolor="black")
    ax.bar(x + w/2, base_medians, w, label="ER 基线", edgecolor="black")
    ax.set_xlabel("模式尺寸")
    ax.set_ylabel("频次中位数")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.set_title("SPMiner vs ER 基线频次对比")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_report(dataset_name: str) -> str | None:
    """为单个数据集生成 Markdown 分析报告。"""
    cfg = EXPERIMENT_DATASETS.get(dataset_name)
    if cfg is None:
        return None

    stats = _load_summary(dataset_name)
    if stats is None:
        return None

    baseline = _load_baseline(dataset_name)
    out_dir = OUT_DIR / dataset_name
    report_dir = out_dir / "report_plots"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 生成图表
    _plot_size_distribution(stats, report_dir / "size_distribution.png")
    if baseline:
        _plot_baseline_comparison(stats, baseline, report_dir / "baseline_compare.png")

    # 拼报告
    lines = []
    lines.append("# 实验报告: {}".format(cfg["label"]))
    lines.append("")
    lines.append("| 属性 | 值 |")
    lines.append("|------|-----|")
    lines.append("| 数据集 | {} |".format(dataset_name))
    lines.append("| 类型 | {} |".format("人工合成" if cfg["type"] == "synthetic" else "真实网络"))
    lines.append("| 模式数 | {} |".format(stats.get("n_patterns", 0)))
    lines.append("| 平均尺寸 | {:.2f} |".format(stats.get("mean_pattern_size", 0)))
    lines.append("| 总频次 | {} |".format(stats.get("total_counts", 0)))
    lines.append("| 耗时 | {:.1f}s |".format(stats.get("time_seconds", 0)))
    lines.append("")

    # 尺寸分布
    sd = stats.get("size_distribution", {})
    if sd:
        lines.append("## 模式尺寸分布")
        lines.append("")
        lines.append("| 尺寸 | 数量 |")
        lines.append("|------|------|")
        for s in sorted(int(k) for k in sd):
            lines.append("| {} | {} |".format(s, sd[str(s)]))
        lines.append("")
        lines.append("![尺寸分布](report_plots/size_distribution.png)")
        lines.append("")

    # 频次统计
    fb = stats.get("freq_by_size", {})
    if fb:
        lines.append("## 频次统计")
        lines.append("")
        lines.append("| 尺寸 | 均值 | 中位数 | 最大 | 最小 |")
        lines.append("|------|------|--------|------|------|")
        for s in sorted(int(k) for k in fb):
            v = fb[str(s)]
            lines.append("| {} | {:.1f} | {:.1f} | {} | {} |".format(
                s, v["mean"], v["median"], v["max"], v["min"]))
        lines.append("")

    # 基线对比
    if baseline:
        lines.append("## 基线对比 (ER 随机图)")
        lines.append("")
        lines.append("| 指标 | SPMiner (真实) | ER 基线 |")
        lines.append("|------|---------------|---------|")
        lines.append("| 模式数 | {} | {} |".format(
            stats.get("n_patterns", 0), baseline.get("n_patterns", 0)))
        lines.append("| 总频次 | {} | {} |".format(
            stats.get("total_counts", 0), baseline.get("total_counts", 0)))
        lines.append("")
        lines.append("![基线对比](report_plots/baseline_compare.png)")
        lines.append("")

    report = "\n".join(lines)
    report_path = out_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report


def generate_all_reports():
    """为所有有结果的数据集生成报告。"""
    print("生成分析报告 ...")
    for name in EXPERIMENT_DATASETS:
        report = generate_report(name)
        if report:
            print("  {} → report.md".format(name))
            # 前 15 行摘要
            first_lines = report.split("\n")[:8]
            for l in first_lines:
                print("    {}".format(l))
            print()


if __name__ == "__main__":
    generate_all_reports()
