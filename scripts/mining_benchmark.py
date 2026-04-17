"""不同大小数据集上的子图挖掘基准测试脚本。

流程：
  1. 若尚未存在训练好的模型，则先训练一个编码器；
  2. 在 plant-N（N = 5, 10, 20, 50）四种合成数据集上运行挖掘；
  3. 每个数据集记录：图数量、邻域数、嵌入耗时、搜索耗时、发现的模式数量及大小分布；
  4. 将摘要打印到控制台并写入 results/benchmark_summary.txt。

用法：
    python3 scripts/mining_benchmark.py [--skip-train]

选项：
    --skip-train   若已有 ckpt/benchmark_model.pt，跳过训练阶段直接进行挖掘测试。
"""

import argparse
import os
import pickle
import subprocess
import sys
import time
from collections import Counter

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "ckpt", "benchmark_model.pt")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
PLOTS_DIR = os.path.join(REPO_ROOT, "plots", "cluster")


def run_cmd(cmd, cwd=None, check=True):
    """执行 shell 命令并将输出直接打印到标准输出。"""
    print(f"\n[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or REPO_ROOT)
    if check and result.returncode != 0:
        sys.exit(f"命令失败，退出码 {result.returncode}")
    return result.returncode


def ensure_dirs():
    os.makedirs(os.path.join(REPO_ROOT, "ckpt"), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 阶段一：训练
# ---------------------------------------------------------------------------

TRAIN_ARGS = [
    sys.executable, "-m", "subgraph_matching.train",
    "--dataset",      "syn",
    "--n_batches",    "500",          # 500 个 mini-batch，足以收敛
    "--batch_size",   "64",
    "--val_size",     "256",
    "--eval_interval","50",
    "--n_layers",     "3",
    "--hidden_dim",   "64",
    "--model_path",   MODEL_PATH,
    "--n_workers",    "2",
]


def train_model():
    print("=" * 60)
    print("阶段一：训练嵌入模型")
    print("=" * 60)
    t0 = time.time()
    run_cmd(TRAIN_ARGS)
    elapsed = time.time() - t0
    print(f"\n训练完成，耗时 {elapsed:.1f}s，模型保存至 {MODEL_PATH}")


# ---------------------------------------------------------------------------
# 阶段二：挖掘（单次运行）
# ---------------------------------------------------------------------------

# 各数据集规模对应的邻域/试验数（越小的图集邻域越多以保证覆盖率）
DATASET_CONFIGS = [
    # (dataset_name, n_neighborhoods, n_trials, label)
    ("plant-5",  400, 200, "plant-5  (图大小≈5 节点)"),
    ("plant-10", 400, 200, "plant-10 (图大小≈10节点)"),
    ("plant-20", 400, 200, "plant-20 (图大小≈20节点)"),
    ("plant-50", 400, 200, "plant-50 (图大小≈50节点)"),
]


def mine_dataset(dataset_name, n_neighborhoods, n_trials, out_path):
    """对单个数据集运行挖掘，返回总耗时（秒）。"""
    cmd = [
        sys.executable, "-m", "subgraph_mining.decoder",
        "--dataset",            dataset_name,
        "--model_path",         MODEL_PATH,
        "--n_neighborhoods",    str(n_neighborhoods),
        "--n_trials",           str(n_trials),
        "--batch_size",         "64",
        "--n_layers",           "3",
        "--hidden_dim",         "64",
        "--min_pattern_size",   "3",
        "--max_pattern_size",   "10",
        "--min_neighborhood_size", "5",
        "--max_neighborhood_size", "20",
        "--out_batch_size",     "5",
        "--frontier_top_k",     "5",
        "--out_path",           out_path,
        "--node_anchored",
    ]
    t0 = time.time()
    rc = run_cmd(cmd, check=False)
    elapsed = time.time() - t0
    if rc != 0:
        print(f"  ⚠️  挖掘返回非零退出码 {rc}，继续统计已有结果")
    return elapsed


# ---------------------------------------------------------------------------
# 阶段三：统计分析
# ---------------------------------------------------------------------------

def analyze_patterns(out_path):
    """加载挖掘结果 pickle 并返回统计信息字典。"""
    if not os.path.exists(out_path):
        return {"error": "结果文件不存在"}
    with open(out_path, "rb") as f:
        patterns = pickle.load(f)
    if not patterns:
        return {"n_patterns": 0}
    sizes = [len(p.nodes()) for p in patterns]
    size_dist = dict(sorted(Counter(sizes).items()))
    return {
        "n_patterns": len(patterns),
        "size_min":   min(sizes),
        "size_max":   max(sizes),
        "size_mean":  round(sum(sizes) / len(sizes), 2),
        "size_dist":  size_dist,
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="子图挖掘多规模基准测试")
    parser.add_argument("--skip-train", action="store_true",
                        help="跳过训练，直接使用已有模型")
    args = parser.parse_args()

    ensure_dirs()

    # ---- 训练 ----
    if args.skip_train and os.path.exists(MODEL_PATH):
        print(f"[跳过训练] 使用已有模型: {MODEL_PATH}")
    else:
        if args.skip_train:
            print(f"[警告] --skip-train 但模型不存在，强制训练")
        train_model()

    # ---- 挖掘 + 统计 ----
    print("\n" + "=" * 60)
    print("阶段二：多规模挖掘测试")
    print("=" * 60)

    rows = []
    for dataset_name, n_neigh, n_trials, label in DATASET_CONFIGS:
        out_path = os.path.join(RESULTS_DIR,
                                f"patterns_{dataset_name}.p")
        print(f"\n--- {label} ---")
        elapsed = mine_dataset(dataset_name, n_neigh, n_trials, out_path)
        stats = analyze_patterns(out_path)
        stats["dataset"] = dataset_name
        stats["label"]   = label
        stats["time_s"]  = round(elapsed, 1)
        rows.append(stats)
        print(f"  耗时: {elapsed:.1f}s | 模式数: {stats.get('n_patterns', '?')} | "
              f"大小分布: {stats.get('size_dist', '?')}")

    # ---- 生成摘要 ----
    summary_path = os.path.join(RESULTS_DIR, "benchmark_summary.txt")
    lines = [
        "=" * 70,
        "子图挖掘多规模基准测试结果摘要",
        "=" * 70,
        f"{'数据集':<30} {'耗时(s)':>8} {'模式数':>6} "
        f"{'最小':>5} {'最大':>5} {'均值':>6}",
        "-" * 70,
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"{r['label']:<30} {'ERROR':>8}")
        else:
            lines.append(
                f"{r['label']:<30} {r['time_s']:>8.1f} "
                f"{r.get('n_patterns',0):>6} "
                f"{r.get('size_min','—'):>5} "
                f"{r.get('size_max','—'):>5} "
                f"{r.get('size_mean','—'):>6}"
            )
    lines += ["=" * 70, ""]
    lines.append("各数据集模式大小分布：")
    for r in rows:
        lines.append(f"  {r['dataset']}: {r.get('size_dist', 'N/A')}")

    summary = "\n".join(lines)
    print("\n" + summary)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\n摘要已写入: {summary_path}")


if __name__ == "__main__":
    main()
