"""SPMiner_learn 公开 API。

外部 Python 代码通过本包导入核心功能，无需关心内部模块结构：

用法示例:

    from src import (
        RuntimeConfig, MatchingConfig, MiningConfig,
        train_encoder, mine_patterns, count_graphlets,
        CoreFacade, OrderEmbedder,
    )

    # 训练编码器
    patterns = train_encoder(
        dataset="facebook",
        node_anchored=True,
        n_batches=100, eval_interval=10,
        batch_size=32, model_path="model.pt",
    )

    # 挖掘频繁子图
    patterns = mine_patterns(
        dataset="facebook",
        model_path="model.pt",
        n_neighborhoods=200, n_trials=20,
        out_path="patterns.p",
    )

    # 低阶 API：先构造 config，再传入函数
    from src.subgraph_matching.train import train_loop
    from src.core.cli import setup_runtime

    config = MatchingConfig(dataset="enzymes", n_layers=4)
    # 通过 vars() 转成 argparse.Namespace 兼容层...
"""

from __future__ import annotations

import os
import platform

# ---------------------------------------------------------------------------
# OpenMP 运行时保护（Windows 兼容）—— 必须在其他导入前执行
# ---------------------------------------------------------------------------
def _configure_openmp_runtime() -> None:
    if platform.system().lower() != "windows":
        return
    if os.environ.get("SPMINER_STRICT_OPENMP", "0") == "1":
        return
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_configure_openmp_runtime()

# ---------------------------------------------------------------------------
# 配置（核心模块，无副作用）
# ---------------------------------------------------------------------------
from src.core.config import RuntimeConfig, MatchingConfig, MiningConfig, AugmentConfig

# ---------------------------------------------------------------------------
# 核心外观（数据集、模型、产物管理）
# ---------------------------------------------------------------------------
from src.core.facade import CoreFacade

# ---------------------------------------------------------------------------
# 模型定义
# ---------------------------------------------------------------------------
from src.core.models import OrderEmbedder, BaselineMLP, SkipLastGNN

# ---------------------------------------------------------------------------
# 训练与评估
# ---------------------------------------------------------------------------
from src.subgraph_matching.train import train_loop as train_encoder
from src.subgraph_matching.test import validation as evaluate
from src.subgraph_matching.alignment import gen_alignment_matrix

# ---------------------------------------------------------------------------
# 挖掘
# ---------------------------------------------------------------------------
from src.subgraph_mining.pipeline import PatternGrowthPipeline
from src.subgraph_mining.decoder import pattern_growth as mine_patterns
from src.subgraph_mining.search_agents import SearchAgent, GreedySearchAgent, MCTSSearchAgent

# ---------------------------------------------------------------------------
# 计数与分析
# ---------------------------------------------------------------------------
from src.analyze.count_patterns import count_graphlets, preprocess_query, preprocess_target

# ---------------------------------------------------------------------------
# 公开 API 列表
# ---------------------------------------------------------------------------
__all__ = [
    # 配置
    "RuntimeConfig",
    "MatchingConfig",
    "MiningConfig",
    "AugmentConfig",

    # 核心外观
    "CoreFacade",

    # 模型
    "OrderEmbedder",
    "BaselineMLP",
    "SkipLastGNN",

    # 训练与评估
    "train_encoder",
    "evaluate",
    "gen_alignment_matrix",

    # 挖掘
    "mine_patterns",
    "PatternGrowthPipeline",
    "SearchAgent",
    "GreedySearchAgent",
    "MCTSSearchAgent",

    # 分析
    "count_graphlets",
    "preprocess_query",
    "preprocess_target",
]
