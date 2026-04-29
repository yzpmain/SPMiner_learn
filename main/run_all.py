"""遍历所有数据集执行实验的主入口。"""

from __future__ import annotations

import argparse

from src.logger import RunLogger, info, section

from main.config import EXPERIMENT_DATASETS, OUT_DIR
from main.experiment import run_dataset

__all__ = ["main"]


def main():
    parser = argparse.ArgumentParser(description="SPMiner 实验主入口")
    parser.add_argument("--model-path", required=True,
                        help="预训练模型 .pt 文件路径")
    parser.add_argument("--datasets", nargs="+",
                        default=list(EXPERIMENT_DATASETS.keys()),
                        choices=list(EXPERIMENT_DATASETS.keys()) + ["all"],
                        help="要运行的数据集子集 (默认全部)")
    parser.add_argument("--dry-run", action="store_true",
                        help="快速测试模式 (少量 neighborhoods/trials)")
    parser.add_argument("--min-size", type=int, default=None,
                        help="最小模式尺寸 (覆盖默认)")
    parser.add_argument("--max-size", type=int, default=None,
                        help="最大模式尺寸 (覆盖默认)")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="搜索试验次数 (覆盖默认)")
    parser.add_argument("--n-neighborhoods", type=int, default=None,
                        help="邻域采样数量 (覆盖默认)")
    args = parser.parse_args()

    # 构造覆盖参数
    overrides = {}
    if args.n_trials is not None:
        overrides["n_trials"] = args.n_trials
    if args.n_neighborhoods is not None:
        overrides["n_neighborhoods"] = args.n_neighborhoods

    dataset_names = args.datasets
    if "all" in dataset_names:
        dataset_names = list(EXPERIMENT_DATASETS.keys())

    with RunLogger(args):
        all_results = {}
        for name in dataset_names:
            cfg = EXPERIMENT_DATASETS[name]
            section("运行实验: {} ({})".format(cfg["label"], name))
            out_dir = OUT_DIR / name
            result = run_dataset(
                dataset_name=name,
                model_path=args.model_path,
                out_dir=out_dir,
                dry_run=args.dry_run,
                min_size=args.min_size,
                max_size=args.max_size,
            )
            all_results[name] = result

        # 保存汇总
        _save_summary(all_results)
        info("全部实验完成。结果保存至 {}".format(OUT_DIR.parent))


def _save_summary(all_results: dict):
    """简易汇总：各数据集的基本统计。"""
    import csv
    from main.config import DATA_ROOT

    csv_path = DATA_ROOT / "summary.csv"
    fields = ["dataset", "label", "n_patterns", "mean_pattern_size",
              "total_counts", "time_seconds"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, res in all_results.items():
            row = {k: res.get(k, "") for k in fields}
            row["dataset"] = name
            row["label"] = EXPERIMENT_DATASETS.get(name, {}).get("label", name)
            w.writerow(row)
    info("汇总表保存至 {}".format(csv_path))


if __name__ == "__main__":
    main()
