"""单数据集实验流水线。

加载数据 → 加载模型 → 挖掘模式 → 计数 → 统计 → 保存。
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from src.core import CoreFacade
from src.core import utils
from src.core.cli import setup_runtime
from src.subgraph_mining.pipeline import PatternGrowthPipeline
from src.analyze.count_patterns import count_graphlets
from src.logger import info, section

from main.config import EXPERIMENT_DATASETS, MINE_CONFIG, DRY_RUN_CONFIG
from main.sbm_generator import ensure_sbm_dataset
from main.download_as733 import ensure_as733

__all__ = ["run_dataset"]


# ---------------------------------------------------------------------------
# 自定义数据集加载
# ---------------------------------------------------------------------------
def _load_custom_dataset(name: str) -> tuple[list[nx.Graph], str]:
    """加载非注册表数据集。"""
    if name == "sbm":
        return ensure_sbm_dataset(), "graph"
    if name == "as733":
        return ensure_as733(), "graph"
    raise ValueError("未知的自定义数据集: {}".format(name))


def _load_dataset(name: str) -> tuple[list[nx.Graph], str]:
    """统一加载接口：注册表或自定义。"""
    cfg = EXPERIMENT_DATASETS.get(name)
    if cfg is None:
        raise ValueError("未知数据集: {}".format(name))
    if cfg["loader"] == "registry":
        _, dataset, task = CoreFacade.load_stage_dataset(name, "mining")
        return dataset, task
    return _load_custom_dataset(name)


# ---------------------------------------------------------------------------
# 构建最小 args 对象 (argparse.Namespace)
# ---------------------------------------------------------------------------
def _make_args(
    dataset_name: str,
    model_path: str,
    out_dir: Path,
    overrides: dict | None = None,
) -> argparse.Namespace:
    """构造 PatternGrowthPipeline 和模型加载所需的 args。

    融合 MINE_CONFIG + 数据集特定参数 + CLI 覆盖。
    """
    params = dict(MINE_CONFIG)
    if overrides:
        params.update(overrides)

    # 模型架构参数 (必须与 checkpoint 训练时一致)
    # 默认使用 SAGE + order 的常见配置
    model_params = {
        "conv_type": "SAGE",
        "n_layers": 8,
        "hidden_dim": 64,
        "skip": "learnable",
        "dropout": 0.0,
        "method_type": "order",
        "margin": 0.1,
        "use_gpu": True,
        "model_path": model_path,
    }

    ns = argparse.Namespace(**model_params, **params)
    ns.dataset = dataset_name
    ns.node_anchored = params.get("node_anchored", True)
    ns.test = True  # 推理模式
    ns.analyze = False
    ns.use_whole_graphs = False
    ns.radius = 3
    ns.subgraph_sample_size = 0
    ns.out_path = str(out_dir / "patterns.p")
    ns.artifact_dir = str(out_dir)
    ns.pattern_plot_dir = str(out_dir / "plots")
    ns.analysis_out_path = str(out_dir / "analyze.p")
    ns.analysis_plot_path = str(out_dir / "analyze.png")
    ns.skip = model_params["skip"]
    ns.augment_method = "concat"
    ns.augment_features = ""
    ns.augment_feature_dims = ""
    # 优化器参数 (build_model 需要)
    ns.opt = "adam"
    ns.lr = 1e-4
    ns.opt_scheduler = "none"
    ns.opt_restart = 100
    ns.weight_decay = 0.0
    return ns


# ---------------------------------------------------------------------------
# 数据转换
# ---------------------------------------------------------------------------
def _to_nx_list(dataset) -> list[nx.Graph]:
    """统一转为 nx.Graph 列表。"""
    import torch_geometric.utils as pyg_utils
    graphs = []
    for g in dataset:
        if not isinstance(g, nx.Graph):
            g = pyg_utils.to_networkx(g).to_undirected()
        graphs.append(g)
    return graphs


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def _compute_stats(
    patterns: list[nx.Graph],
    counts: list[int],
    elapsed: float,
) -> dict:
    """收集实验统计指标。"""
    size_counter = Counter(len(g) for g in patterns)
    # 按大小分组频次
    freq_by_size = defaultdict(list)
    for pat, cnt in zip(patterns, counts):
        freq_by_size[len(pat)].append(cnt)

    sorted_sizes = sorted(freq_by_size.keys())
    return {
        "n_patterns": len(patterns),
        "n_unique_patterns": len(set(
            tuple(sorted(g.edges())) for g in patterns)),
        "size_distribution": dict(size_counter),
        "mean_pattern_size": float(np.mean([len(g) for g in patterns])) if patterns else 0,
        "freq_by_size": {
            str(s): {
                "mean": float(np.mean(freq_by_size[s])),
                "median": float(np.median(freq_by_size[s])),
                "max": float(np.max(freq_by_size[s])),
                "min": float(np.min(freq_by_size[s])),
            }
            for s in sorted_sizes
        },
        "total_counts": int(np.sum(counts)),
        "time_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
def run_dataset(
    dataset_name: str,
    model_path: str,
    out_dir: Path,
    dry_run: bool = False,
    min_size: int | None = None,
    max_size: int | None = None,
    count_method: str = "bin",
    count_sample_size: int = 100,
) -> dict:
    """单数据集实验入口。

    参数:
        count_method: 计数模式 bin/freq/sample
        count_sample_size: sample 模式下的目标图采样数

    返回:
        dict: 实验统计结果
    """
    overrides = dict(DRY_RUN_CONFIG if dry_run else {})
    if min_size is not None:
        overrides["min_pattern_size"] = min_size
    if max_size is not None:
        overrides["max_pattern_size"] = max_size

    section("数据集: {}".format(dataset_name))
    args = _make_args(dataset_name, model_path, out_dir, overrides)
    setup_runtime(args)

    # 1. 准备数据
    info("加载数据 ...")
    dataset, task = _load_dataset(dataset_name)
    info("加载完成: {} 个图".format(len(dataset) if hasattr(dataset, "__len__") else "?"))

    # 2. 加载模型
    info("加载模型: {}".format(model_path))
    model = CoreFacade.build_model(args, for_inference=True, load_weights=True)
    model.eval()

    # 3. 准备输出目录
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 4. 挖掘模式
    info("开始挖掘 (min={}, max={}, {} trials)...".format(
        args.min_pattern_size, args.max_pattern_size, args.n_trials))
    t0 = time.time()
    pipeline = PatternGrowthPipeline(args, model, dataset, task)
    patterns = pipeline.run()
    mining_time = time.time() - t0
    info("挖掘完成: {} 个模式, 耗时 {:.1f}s".format(len(patterns), mining_time))

    if not patterns:
        info("未挖掘到模式，跳过计数")
        stats = _compute_stats([], [], mining_time)
        _save_results(out_dir, patterns, [], stats)
        return stats

    # 5. 计数
    targets = _to_nx_list(dataset)
    info("计数模式频率 (method={}) ...".format(count_method))
    if count_method == "sample" and len(targets) > count_sample_size:
        import random
        rng = random.Random(42)
        sampled = rng.sample(targets, count_sample_size)
        info("  采样 {}/{} 个目标图".format(count_sample_size, len(targets)))
        targets = sampled
    t0 = time.time()
    counts = count_graphlets(
        patterns, targets,
        n_workers=args.n_workers,
        method=count_method if count_method != "sample" else "bin",
        node_anchored=args.node_anchored,
        progress_every=0,
    )
    count_time = time.time() - t0
    info("计数完成: 耗时 {:.1f}s".format(count_time))

    # 6. 统计
    stats = _compute_stats(patterns, counts, mining_time + count_time)
    stats["dataset"] = dataset_name
    stats["label"] = EXPERIMENT_DATASETS[dataset_name]["label"]

    _save_results(out_dir, patterns, counts, stats)
    _save_plots(plot_dir, patterns, counts, stats)

    return stats


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------
def _save_results(out_dir: Path, patterns: list, counts: list, stats: dict):
    with open(out_dir / "patterns.p", "wb") as f:
        pickle.dump(patterns, f)
    with open(out_dir / "counts.json", "w") as f:
        json.dump(([len(g) for g in patterns], counts, []), f)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    info("结果保存至 {}".format(out_dir))


def _setup_cjk_font():
    """配置 matplotlib 中文字体支持。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Windows 常见中文字体
    for font in ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
        try:
            plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            # 验证可用
            from matplotlib.font_manager import findfont, FontProperties
            findfont(FontProperties(family=font))
            return plt
        except Exception:
            continue
    return plt

def _save_plots(plot_dir: Path, patterns: list, counts: list, stats: dict):
    plt = _setup_cjk_font()

    # 模式尺寸分布
    sizes = [len(g) for g in patterns]
    if sizes:
        plt.figure()
        plt.hist(sizes, bins=range(min(sizes), max(sizes) + 2),
                 align="left", edgecolor="black")
        plt.xlabel("模式尺寸")
        plt.ylabel("数量")
        plt.title("模式尺寸分布")
        plt.savefig(plot_dir / "size_distribution.png", dpi=150)
        plt.close()

    # 频次直方图
    if counts:
        plt.figure()
        plt.hist(counts, bins=20, edgecolor="black")
        plt.xlabel("出现频次")
        plt.ylabel("模式数")
        plt.yscale("log")
        plt.title("模式频次分布")
        plt.savefig(plot_dir / "frequency.png", dpi=150)
        plt.close()
