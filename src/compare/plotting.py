from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

__all__ = ["plot_results"]


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _plot_grouped_lines(df: pd.DataFrame, x_col: str, y_col: str, dataset: str, title: str, ylabel: str, out_path: Path) -> None:
    if x_col not in df.columns or y_col not in df.columns:
        return

    plt.figure(figsize=(10, 6))
    if "graph_size" in df.columns and df["graph_size"].nunique(dropna=True) > 1:
        for size in sorted(df["graph_size"].dropna().unique()):
            sub = df[df["graph_size"] == size].sort_values(x_col)
            plt.plot(sub[x_col], sub[y_col], marker="o", linewidth=2, label=f"{size}节点")
    else:
        sub = df.sort_values(x_col)
        plt.plot(sub[x_col], sub[y_col], marker="o", linewidth=2, label=y_col)

    plt.xlabel("子图大小 k (节点数)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{title} ({dataset})", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_results(df: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    _configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    if {"gspan_time", "spminer_time"}.issubset(df.columns):
        plt.figure(figsize=(10, 6))
        if "graph_size" in df.columns and df["graph_size"].nunique(dropna=True) > 1:
            for size in sorted(df["graph_size"].dropna().unique()):
                sub = df[df["graph_size"] == size].sort_values("k")
                plt.plot(sub["k"], sub["gspan_time"], marker="o", linewidth=2, label=f"gSpan | {size}节点")
                plt.plot(sub["k"], sub["spminer_time"], marker="s", linewidth=2, label=f"SPMiner | {size}节点")
        else:
            sub = df.sort_values("k")
            plt.plot(sub["k"], sub["gspan_time"], marker="o", label="gSpan", linewidth=2)
            plt.plot(sub["k"], sub["spminer_time"], marker="s", label="SPMiner", linewidth=2)
        plt.xlabel("子图大小 k (节点数)", fontsize=12)
        plt.ylabel("运行时间 (秒)", fontsize=12)
        plt.title(f"SPMiner vs gSpan 运行时间对比 ({dataset})", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"time_comparison_{dataset}.png", dpi=300)
        plt.close()

    if {"gspan_mem", "spminer_mem"}.issubset(df.columns):
        plt.figure(figsize=(10, 6))
        if "graph_size" in df.columns and df["graph_size"].nunique(dropna=True) > 1:
            for size in sorted(df["graph_size"].dropna().unique()):
                sub = df[df["graph_size"] == size].sort_values("k")
                plt.plot(sub["k"], sub["gspan_mem"], marker="o", linewidth=2, label=f"gSpan | {size}节点")
                plt.plot(sub["k"], sub["spminer_mem"], marker="s", linewidth=2, label=f"SPMiner | {size}节点")
        else:
            sub = df.sort_values("k")
            plt.plot(sub["k"], sub["gspan_mem"], marker="o", label="gSpan", linewidth=2)
            plt.plot(sub["k"], sub["spminer_mem"], marker="s", label="SPMiner", linewidth=2)
        plt.xlabel("子图大小 k (节点数)", fontsize=12)
        plt.ylabel("最大内存占用 (MB)", fontsize=12)
        plt.title(f"SPMiner vs gSpan 内存占用对比 ({dataset})", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"mem_comparison_{dataset}.png", dpi=300)
        plt.close()

    if "time_speedup" in df.columns:
        _plot_grouped_lines(
            df,
            "k",
            "time_speedup",
            dataset,
            "SPMiner 相对 gSpan 加速比",
            "加速比 (gSpan / SPMiner)",
            out_dir / f"speedup_comparison_{dataset}.png",
        )

    if {"gspan_support_mean", "spminer_exact_support_mean"}.issubset(df.columns):
        plt.figure(figsize=(10, 6))
        if "graph_size" in df.columns and df["graph_size"].nunique(dropna=True) > 1:
            for size in sorted(df["graph_size"].dropna().unique()):
                sub = df[df["graph_size"] == size].sort_values("k")
                plt.plot(sub["k"], sub["gspan_support_mean"], marker="o", linewidth=2, label=f"gSpan support | {size}节点")
                plt.plot(sub["k"], sub["spminer_exact_support_mean"], marker="s", linewidth=2, label=f"SPMiner exact | {size}节点")
        else:
            sub = df.sort_values("k")
            plt.plot(sub["k"], sub["gspan_support_mean"], marker="o", linewidth=2, label="gSpan support")
            plt.plot(sub["k"], sub["spminer_exact_support_mean"], marker="s", linewidth=2, label="SPMiner exact")
        plt.xlabel("子图大小 k (节点数)", fontsize=12)
        plt.ylabel("频率 / 支持度", fontsize=12)
        plt.title(f"SPMiner vs gSpan 真实频率对比 ({dataset})", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"frequency_comparison_{dataset}.png", dpi=300)
        plt.close()

    if "frequency_mae" in df.columns:
        plt.figure(figsize=(10, 6))
        if "graph_size" in df.columns and df["graph_size"].nunique(dropna=True) > 1:
            for size in sorted(df["graph_size"].dropna().unique()):
                sub = df[df["graph_size"] == size].sort_values("k")
                plt.plot(sub["k"], sub["frequency_mae"], marker="o", linewidth=2, label=f"{size}节点")
        else:
            sub = df.sort_values("k")
            plt.plot(sub["k"], sub["frequency_mae"], marker="o", linewidth=2, label="MAE")
        plt.xlabel("子图大小 k (节点数)", fontsize=12)
        plt.ylabel("频率 MAE", fontsize=12)
        plt.title(f"SPMiner 与 gSpan 频率误差 ({dataset})", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"frequency_error_{dataset}.png", dpi=300)
        plt.close()