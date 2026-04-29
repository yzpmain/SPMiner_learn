import argparse
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import csv

from src.core.artifacts import choose_cli_output_path, task_output_dir, write_manifest
from src.core.cli import add_runtime_args
from src.logger import RunLogger, info, section

__all__: list[str] = []

def arg_parse():
    parser = argparse.ArgumentParser(description='统计图中的图元')
    #parser.add_argument('--graphlets_path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--counts_path', type=str)
    parser.add_argument('--out_path', type=str)
    add_runtime_args(parser, include_gpu=False, include_seed=False,
        include_tag=True, include_n_workers=False,
        include_progress_write_interval=False, include_output_policy=True)
    parser.set_defaults(dataset="analysis")
    parser.set_defaults(counts_path="results/counts.json")
    #parser.set_defaults(graphlets_path="out/out-graphlets.p")
    parser.set_defaults(out_path="results/analysis.csv")
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()

    with RunLogger(args):
        section("计数分析")
        artifact_dir = task_output_dir(args, "analyze", args.dataset)
        args.out_path = str(choose_cli_output_path(
            args,
            args.out_path,
            default_cli_path="results/analysis.csv",
            suggested_default_path=artifact_dir / "analysis.csv",
        ))
        plot_path = artifact_dir / "pattern-counts.png"

        all_counts = {}
        for fn in os.listdir(args.counts_path):
            if not fn.endswith(".json"): continue

            with open(os.path.join(args.counts_path, fn), "r") as f:
                graphlet_lens, n_matches, n_matches_bl = json.load(f)
                name = fn[:-5]
                all_counts[name] = graphlet_lens, n_matches

        all_labels, all_xs, all_ys, all_ub_ys, all_lb_ys = [], [], [], [], []
        summary_rows = []
        for name, (sizes, counts) in all_counts.items():
            all_labels.append(name)

            matches_by_size = defaultdict(list)
            for i in range(len(sizes)):
                matches_by_size[sizes[i]].append(counts[i])

        #print("By size:")
            ys = []
            ub_ys, lb_ys = [], []
            for size in sorted(matches_by_size.keys()):
            #a, b = (stats.t.interval(0.95, len(matches_by_size[size]) - 1,
            #    loc=np.mean(np.log10(matches_by_size[size])),
            #    scale=stats.sem(np.log10(matches_by_size[size]))))
            #s = np.std(np.log10(matches_by_size[size]), ddof=1)
            #m = np.mean(np.log10(matches_by_size[size]))
            #a, b = m - s, m + s
            # 避免 0 计数导致 log10 警告/inf。
                safe_counts = np.maximum(np.array(matches_by_size[size], dtype=float), 1e-12)
                a, b = np.percentile(np.log10(safe_counts), [25, 75])

                ub_ys.append(b)
                lb_ys.append(a)
            #ys.append(np.mean(np.log10(matches_by_size[size])))
                ys.append(np.median(np.log10(safe_counts)))

                summary_rows.append({
                    "name": name,
                    "graph_size": int(size),
                    "n_patterns": int(len(matches_by_size[size])),
                    "median_frequency": float(np.power(10, ys[-1])),
                    "q1_frequency": float(np.power(10, a)),
                    "q3_frequency": float(np.power(10, b)),
                })

            all_xs.append(list(sorted(matches_by_size.keys())))
            all_ys.append(ys)
            all_ub_ys.append(ub_ys)
            all_lb_ys.append(lb_ys)

        #print("By size (log):")
        #for size in sorted(matches_by_size.keys()):
        #    print("- {}. N: {}. Mean log count: {:.4f}. Baseline: {:.4f}. "
        #        "Different with p={:.4f}".format(size, len(matches_by_size[size]),
        #            np.mean(np.log10(matches_by_size[size])),
        #            np.mean(np.log10(matches_by_size_bl[size])),
        #            ttest_ind(np.log10(matches_by_size[size]),
        #                np.log10(matches_by_size_bl[size])).pvalue))

        for i in range(len(all_xs)):
            plt.style.use("default")
            plt.plot(all_xs[i], np.power(10, all_ys[i]), label=all_labels[i],
                marker="o")
            plt.fill_between(all_xs[i], np.power(10, all_lb_ys[i]),
                np.power(10, all_ub_ys[i]), alpha=0.3)
        plt.xlabel("Graph size")
        plt.ylabel("Frequency")
        plt.yscale("log")
        plt.legend()
        plt.savefig(plot_path)
        plt.close()
        info("Pattern counts plot saved → {}".format(plot_path))

        out_dir = os.path.dirname(args.out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "graph_size",
                    "n_patterns",
                    "median_frequency",
                    "q1_frequency",
                    "q3_frequency",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        info("Analysis CSV saved → {}".format(args.out_path))
        write_manifest(
            artifact_dir / "manifest.json",
            args,
            outputs={
                "analysis_csv": args.out_path,
                "plot": str(plot_path),
                "counts_path": args.counts_path,
            },
            task="analyze",
        )
