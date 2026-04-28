from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.compare.analysis import add_runtime_metrics, build_accuracy_table
from src.compare.benchmarking import (
    build_gspan_db_from_edge_list,
    case_output_paths,
    prepare_spminer_dataset_from_gspan_db,
    resolve_path,
    run_gspan,
    run_spminer,
    size_tag,
    trim_gspan_top_k,
    trim_spminer_top_k,
)
from src.compare.plotting import plot_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPMiner vs gSpan 对比脚本")
    parser.add_argument("--dataset", type=str, default="facebook", help="数据集名称")
    parser.add_argument(
        "--edge-list",
        type=str,
        default="data/facebook_combined.txt",
        help="构建多规模图时使用的原始边列表文件",
    )
    parser.add_argument(
        "--graph-sizes",
        type=int,
        nargs="+",
        default=[],
        help="批量图规模（例如 40 60 80）。为空时使用 --gspan-db-file 单图模式。",
    )
    parser.add_argument(
        "--gspan-db-file",
        type=str,
        default="",
        help="gSpan 图数据库文件路径（留空则不传该变量）",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=list(range(5, 16)),
        help="要测试的子图大小列表",
    )
    parser.add_argument("--min-sup", type=float, default=0.1, help="gSpan 最小支持度")
    parser.add_argument("--timeout-sec", type=int, default=900, help="每次算法运行超时时间（秒）")
    parser.add_argument("--poll-interval", type=float, default=0.1, help="轮询进程状态的时间间隔（秒）")
    parser.add_argument("--python-bin", type=str, default="python", help="运行 SPMiner 时使用的 Python 可执行文件")
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="仓库根目录（默认自动定位）",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="results/facebook_train_big.pt",
        help="SPMiner 使用的模型路径（相对 repo-root 或绝对路径）",
    )
    parser.add_argument("--spminer-trials", type=int, default=5, help="SPMiner n_trials 参数")
    parser.add_argument("--spminer-neighborhoods", type=int, default=50, help="SPMiner n_neighborhoods 参数")
    parser.add_argument("--spminer-batch-size", type=int, default=50, help="SPMiner batch_size 参数")
    parser.add_argument(
        "--gspan-cmd-template",
        type=str,
        default="",
        help=(
            "gSpan 命令模板。支持变量：{dataset} {k} {min_sup} {out_file} {gspan_db}。"
            "示例：\"python -m gspan_mining -s {min_sup} -l {k} -u {k} {gspan_db}\""
        ),
    )
    parser.add_argument("--use-gspan-mining", action="store_true", help="使用内置 gspan_mining 命令构造（推荐 Windows 下开启）")
    parser.add_argument("--out-dir", type=str, default="compare/out", help="对比结果输出目录（相对 repo-root 或绝对路径）")
    parser.add_argument("--fair-shared-input", action="store_true", help="开启严格公平模式：SPMiner 与 gSpan 使用同一份 gSpan 子图输入")
    parser.add_argument("--top-k-patterns", type=int, default=3, help="每次挖掘后保留的高频子图数量")
    parser.add_argument("--frequency-workers", type=int, default=4, help="真实频率验证时使用的 worker 数量")
    parser.add_argument(
        "--node-anchored",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用节点锚定模式",
    )
    parser.add_argument(
        "--evaluate-frequency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在输出中计算真实频率/支持度对齐指标",
    )
    return parser.parse_args()


def _safe_float(value):
    return value if pd.notna(value) and np.isfinite(value) else np.nan


def _file_contains_memory_error(path: Path | str) -> bool:
    try:
        p = Path(path)
        if not p.exists():
            return False
        txt = p.read_text(encoding="utf-8", errors="ignore")
        low = txt.lower()
        return "memoryerror" in low or "outofmemory" in low or "oom" in low or "memory error" in low
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    repo_root = resolve_path(Path.cwd(), args.repo_root)
    out_dir = resolve_path(repo_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    size_contexts: list[tuple[int | None, Path | None]] = []
    if args.graph_sizes:
        edge_list_path = resolve_path(repo_root, args.edge_list)
        data_dir = repo_root / "src" / "compare" / "data"
        for graph_size in args.graph_sizes:
            gspan_db = data_dir / f"{args.dataset}_gspan_{graph_size}.txt"
            n_nodes, n_edges = build_gspan_db_from_edge_list(edge_list_path, gspan_db, graph_size)
            print(f"[multi] 生成规模图: {gspan_db} (nodes={n_nodes}, edges={n_edges})")
            size_contexts.append((graph_size, gspan_db))
    else:
        gspan_db = resolve_path(repo_root, args.gspan_db_file) if args.gspan_db_file else None
        size_contexts.append((None, gspan_db))

    records: list[dict] = []
    run_id = 0
    for graph_size, gspan_db in size_contexts:
        local_args = argparse.Namespace(**vars(args))
        if gspan_db:
            local_args.gspan_db_file = str(gspan_db)

        benchmark_dataset = args.dataset
        if args.fair_shared_input:
            benchmark_dataset = prepare_spminer_dataset_from_gspan_db(local_args, repo_root)
            print(f"[fair] 使用共享输入运行：gSpan + SPMiner 均基于 {local_args.gspan_db_file}")

        for k in args.ks:
            print(f"===== 正在测试 {size_tag(graph_size)}, k={k} =====")
            paths = case_output_paths(out_dir, args.dataset, graph_size, k)
            record = {
                "run_id": run_id,
                "dataset": args.dataset,
                "frequency_dataset": benchmark_dataset,
                "graph_size": graph_size if graph_size is not None else np.nan,
                "k": k,
                "gspan_result_file": str(paths["gspan_out"]),
                "spminer_result_file": str(paths["spminer_out"]),
                "spminer_log_file": str(paths["spminer_log"]),
            }

            try:
                g_time, g_mem = run_gspan(local_args, repo_root, paths["gspan_out"], k)
                g_kept = trim_gspan_top_k(paths["gspan_out"], local_args.top_k_patterns)
                record.update({
                    "gspan_time": g_time,
                    "gspan_mem": g_mem,
                    "gspan_status": "ok",
                    "gspan_topk_kept": g_kept,
                })
                print(f"gSpan 完成：时间={g_time:.2f}s, 内存={g_mem:.2f}MB, top-{local_args.top_k_patterns}保留={g_kept}")
            except Exception as exc:
                # 检测是否为子进程内存错误（通过输出文件内容判断），如果是则标记为无法计算
                mem_err = _file_contains_memory_error(paths["gspan_out"]) if paths.get("gspan_out") else False
                if mem_err:
                    record.update({
                        "gspan_time": np.nan,
                        "gspan_mem": np.nan,
                        "gspan_status": "oom",
                        "gspan_topk_kept": 0,
                    })
                    print(f"gSpan OOM（内存不足），已标记为 oom: {paths.get('gspan_out')}")
                else:
                    record.update({
                        "gspan_time": np.nan,
                        "gspan_mem": np.nan,
                        "gspan_status": str(exc),
                        "gspan_topk_kept": 0,
                    })
                    print(f"gSpan 失败：{exc}")

            try:
                s_time, s_mem = run_spminer(
                    local_args,
                    repo_root,
                    paths["spminer_out"],
                    paths["spminer_log"],
                    k,
                    benchmark_dataset,
                )
                s_kept = trim_spminer_top_k(paths["spminer_out"], local_args.top_k_patterns)
                record.update({
                    "spminer_time": s_time,
                    "spminer_mem": s_mem,
                    "spminer_status": "ok",
                    "spminer_topk_kept": s_kept,
                })
                print(f"SPMiner 完成：时间={s_time:.2f}s, 内存={s_mem:.2f}MB, top-{local_args.top_k_patterns}保留={s_kept}")
            except Exception as exc:
                mem_err = _file_contains_memory_error(paths.get("spminer_log"))
                if mem_err:
                    record.update({
                        "spminer_time": np.nan,
                        "spminer_mem": np.nan,
                        "spminer_status": "oom",
                        "spminer_topk_kept": 0,
                    })
                    print(f"SPMiner OOM（内存不足），已标记为 oom: {paths.get('spminer_log')}")
                else:
                    record.update({
                        "spminer_time": np.nan,
                        "spminer_mem": np.nan,
                        "spminer_status": str(exc),
                        "spminer_topk_kept": 0,
                    })
                    print(f"SPMiner 失败：{exc}")

            records.append(record)
            run_id += 1

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No benchmark records were generated")

    df = add_runtime_metrics(df)
    df["gspan_time"] = df["gspan_time"].map(_safe_float)
    df["spminer_time"] = df["spminer_time"].map(_safe_float)
    df["gspan_mem"] = df["gspan_mem"].map(_safe_float)
    df["spminer_mem"] = df["spminer_mem"].map(_safe_float)

    accuracy_df = build_accuracy_table(
        df,
        dataset_name=None,
        top_k=args.top_k_patterns,
        node_anchored=args.node_anchored,
        exact_frequency=args.evaluate_frequency,
        frequency_workers=args.frequency_workers,
    )
    df = df.merge(accuracy_df, on="run_id", how="left", suffixes=("", "_analysis"))
    drop_analysis_cols = [
        col for col in ["graph_size_analysis", "k_analysis", "spminer_result_file_analysis", "gspan_result_file_analysis"]
        if col in df.columns
    ]
    if drop_analysis_cols:
        df = df.drop(columns=drop_analysis_cols)

    csv_path = out_dir / f"experiment_result_{args.dataset}.csv"
    df.to_csv(csv_path, index=False)

    if args.graph_sizes:
        summary_path = out_dir / f"summary_{args.dataset}_by_size_k.csv"
        df.to_csv(summary_path, index=False)
        print(f"多规模汇总已保存：{summary_path}")

    accuracy_path = out_dir / f"accuracy_summary_{args.dataset}.csv"
    accuracy_df.to_csv(accuracy_path, index=False)
    print(f"准确率/频率汇总已保存：{accuracy_path}")

    plot_results(df, args.dataset, out_dir)

    print(f"实验结果已保存：{csv_path}")
    print(f"对比图已生成到目录：{out_dir}")


if __name__ == "__main__":
    main()