"""实验配置：数据集定义、路径、挖掘超参数。"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

# ---------------------------------------------------------------------------
# 根目录
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "expdata"
DSET_DIR = DATA_ROOT / "datasets"
OUT_DIR = DATA_ROOT / "outputs"
PLOT_DIR = DATA_ROOT / "plots"

# ---------------------------------------------------------------------------
# 数据集定义
# ---------------------------------------------------------------------------
EXPERIMENT_DATASETS = OrderedDict([
    ("sbm",      {"type": "synthetic", "loader": "custom",   "label": "SBM 合成网络"}),
    ("facebook", {"type": "real",      "loader": "registry", "label": "Facebook 社交网络"}),
    ("ppi",      {"type": "real",      "loader": "registry", "label": "PPI 蛋白质网络"}),
    ("as733",    {"type": "real",      "loader": "custom",   "label": "AS733 互联网拓扑"}),
])

# ---------------------------------------------------------------------------
# 挖掘超参数（可通过 CLI 覆盖）
# ---------------------------------------------------------------------------
MINE_CONFIG = {
    "search_strategy": "greedy",
    "min_pattern_size": 4,
    "max_pattern_size": 10,
    "n_neighborhoods": 2000,
    "n_trials": 500,
    "out_batch_size": 10,
    "node_anchored": True,
    "batch_size": 1000,
    "sample_method": "tree",
    "min_neighborhood_size": 20,
    "max_neighborhood_size": 29,
    "frontier_top_k": 5,
    "n_workers": 4,
}

# dry-run 模式用少量参数快速验证
DRY_RUN_CONFIG = {
    "n_neighborhoods": 50,
    "n_trials": 5,
    "batch_size": 50,
}
