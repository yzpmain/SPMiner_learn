import argparse
import math
import os
import shlex
import subprocess
import time
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil


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
        default=[5, 8, 10, 12],
        help="要测试的子图大小列表",
    )
    parser.add_argument("--min-sup", type=float, default=0.1, help="gSpan 最小支持度")
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=900,
        help="每次算法运行超时时间（秒）",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="轮询进程状态的时间间隔（秒）",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default="python",
        help="运行 SPMiner 时使用的 Python 可执行文件",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="仓库根目录（默认自动定位）",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="results/facebook_train_big.pt",
        help="SPMiner 使用的模型路径（相对 repo-root 或绝对路径）",
    )
    parser.add_argument(
        "--spminer-trials",
        type=int,
        default=5,
        help="SPMiner n_trials 参数",
    )
    parser.add_argument(
        "--spminer-neighborhoods",
        type=int,
        default=50,
        help="SPMiner n_neighborhoods 参数",
    )
    parser.add_argument(
        "--spminer-batch-size",
        type=int,
        default=50,
        help="SPMiner batch_size 参数",
    )
    parser.add_argument(
        "--gspan-cmd-template",
        type=str,
        default="",
        help=(
            "gSpan 命令模板。支持变量：{dataset} {k} {min_sup} {out_file} {gspan_db}。"
            "示例：\"python -m gspan_mining -s {min_sup} -l {k} -u {k} {gspan_db}\""
        ),
    )
    parser.add_argument(
        "--use-gspan-mining",
        action="store_true",
        help="使用内置 gspan_mining 命令构造（推荐 Windows 下开启）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="compare/out",
        help="对比结果输出目录（相对 repo-root 或绝对路径）",
    )
    parser.add_argument(
        "--fair-shared-input",
        action="store_true",
        help="开启严格公平模式：SPMiner 与 gSpan 使用同一份 gSpan 子图输入",
    )
    parser.add_argument(
        "--top-k-patterns",
        type=int,
        default=3,
        help="每次挖掘后保留的高频子图数量",
    )
    return parser.parse_args()


def resolve_path(base: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base / path)


def build_gspan_db_from_edge_list(edge_list_path: Path, out_path: Path, max_nodes: int) -> tuple[int, int]:
    """从边列表构建 gSpan 数据文件。"""
    graph = {}
    with edge_list_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            a, b = map(int, s.split()[:2])
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)

    nodes = sorted(graph.keys())
    if max_nodes > 0:
        nodes = nodes[:max_nodes]
    node_set = set(nodes)

    edges = []
    for u in nodes:
        for v in graph.get(u, ()):
            if v in node_set and u < v:
                edges.append((u, v))

    idx = {n: i for i, n in enumerate(nodes)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("t # 0\n")
        for n in nodes:
            f.write(f"v {idx[n]} 0\n")
        for u, v in edges:
            f.write(f"e {idx[u]} {idx[v]} 0\n")
        f.write("t # -1\n")

    return len(nodes), len(edges)


def prepare_spminer_dataset_from_gspan_db(args: argparse.Namespace, repo_root: Path) -> str:
    """将 gSpan DB 的第一张图导出为 SPMiner 可读的 roadnet-* 边列表。"""
    if not args.gspan_db_file.strip():
        raise RuntimeError("--fair-shared-input 需要同时提供 --gspan-db-file")

    gspan_db = resolve_path(repo_root, args.gspan_db_file)
    if not gspan_db.exists():
        raise RuntimeError(f"gSpan DB 文件不存在: {gspan_db}")

    edges = []
    in_first_graph = False
    seen_graph = False
    with gspan_db.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("t "):
                toks = line.split()
                graph_id = toks[-1] if toks else ""
                if graph_id == "0" and not seen_graph:
                    in_first_graph = True
                    seen_graph = True
                    continue
                if in_first_graph:
                    break
                continue
            if not in_first_graph:
                continue
            if line.startswith("e "):
                toks = line.split()
                if len(toks) >= 3:
                    edges.append((int(toks[1]), int(toks[2])))

    if not edges:
        raise RuntimeError("未在 gSpan DB 中解析到可用边，请检查输入格式")

    stem = Path(args.gspan_db_file).stem.replace("_", "-")
    dataset_name = f"roadnet-{stem}-fair"
    out_edge_file = repo_root / "data" / f"{dataset_name}.txt"
    with out_edge_file.open("w", encoding="utf-8") as f:
        for u, v in edges:
            f.write(f"{u}\t{v}\n")

    print(f"[fair] SPMiner 数据集已生成: data/{dataset_name}.txt (edges={len(edges)})")
    return dataset_name


def run_and_monitor(
    cmd,
    cwd: Path,
    timeout_sec: int,
    poll_interval: float,
    ok_exit_codes=(0,),
    stdout_path: Path | None = None,
):
    start = time.time()
    stdout_handle = None
    stderr_target = None
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_handle = stdout_path.open("w", encoding="utf-8")
        stderr_target = subprocess.STDOUT

    proc = psutil.Popen(
        cmd,
        cwd=str(cwd),
        stdout=stdout_handle if stdout_handle else open(os.devnull, "w"),
        stderr=stderr_target if stdout_handle else open(os.devnull, "w"),
    )
    max_mem_mb = 0.0

    try:
        while proc.poll() is None:
            elapsed = time.time() - start
            if elapsed > timeout_sec:
                proc.kill()
                raise TimeoutError(f"timeout after {timeout_sec}s")
            try:
                rss = proc.memory_info().rss / 1024 / 1024
                max_mem_mb = max(max_mem_mb, rss)
            except Exception:
                pass
            time.sleep(poll_interval)

        ret_code = proc.returncode
        elapsed = time.time() - start
        if ret_code not in ok_exit_codes:
            raise RuntimeError(f"process exited with code {ret_code}")
        return elapsed, max_mem_mb
    finally:
        if stdout_handle:
            stdout_handle.close()


def run_spminer(
    args: argparse.Namespace,
    repo_root: Path,
    out_file: Path,
    log_file: Path,
    k: int,
    spminer_dataset: str,
):
    model_path = resolve_path(repo_root, args.model_path)
    n_trials = max(args.spminer_trials, args.top_k_patterns)
    cmd = [
        args.python_bin,
        "-u",
        "-m",
        "subgraph_mining.decoder",
        f"--dataset={spminer_dataset}",
        "--node_anchored",
        f"--model_path={model_path}",
        f"--n_neighborhoods={args.spminer_neighborhoods}",
        f"--batch_size={args.spminer_batch_size}",
        f"--n_trials={n_trials}",
        f"--min_pattern_size={k}",
        f"--max_pattern_size={k}",
        f"--out_path={out_file}",
    ]
    return run_and_monitor(
        cmd,
        repo_root,
        args.timeout_sec,
        args.poll_interval,
        ok_exit_codes=(0,),
        stdout_path=log_file,
    )


def run_gspan(args: argparse.Namespace, repo_root: Path, out_file: Path, k: int):
    gspan_db = ""
    if args.gspan_db_file.strip():
        gspan_db = str(resolve_path(repo_root, args.gspan_db_file))

    if args.use_gspan_mining:
        if not gspan_db:
            raise RuntimeError("missing --gspan-db-file for --use-gspan-mining")
        cmd = [
            args.python_bin,
            "-m",
            "gspan_mining",
            "-s",
            str(int(args.min_sup) if float(args.min_sup).is_integer() else args.min_sup),
            "-l",
            str(k),
            "-u",
            str(k),
            gspan_db,
        ]
        return run_and_monitor(
            cmd,
            repo_root,
            args.timeout_sec,
            args.poll_interval,
            ok_exit_codes=(0, 1),
            stdout_path=out_file,
        )

    if not args.gspan_cmd_template.strip():
        raise RuntimeError("missing --gspan-cmd-template")

    min_sup = int(args.min_sup) if float(args.min_sup).is_integer() else args.min_sup

    cmd_str = args.gspan_cmd_template.format(
        dataset=args.dataset,
        k=k,
        min_sup=min_sup,
        out_file=str(out_file),
        gspan_db=gspan_db,
    )
    cmd = shlex.split(cmd_str, posix=(os.name != "nt"))
    return run_and_monitor(
        cmd,
        repo_root,
        args.timeout_sec,
        args.poll_interval,
        ok_exit_codes=(0, 1),
        stdout_path=out_file,
    )


def trim_spminer_top_k(out_file: Path, top_k: int) -> int:
    if top_k <= 0 or (not out_file.exists()):
        return 0

    with out_file.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list):
        return 0

    trimmed = obj[:top_k]
    with out_file.open("wb") as f:
        pickle.dump(trimmed, f)
    return len(trimmed)


def _extract_gspan_blocks(lines: list[str]) -> list[dict]:
    blocks = []
    current_lines = []
    current_support = None
    order_idx = 0

    for raw in lines:
        line = raw.rstrip("\n")
        s = line.strip()
        if s.startswith("t "):
            if current_lines:
                blocks.append(
                    {
                        "lines": current_lines,
                        "support": current_support,
                        "order": order_idx,
                    }
                )
                order_idx += 1
            current_lines = [line]
            current_support = None
            continue

        if not current_lines:
            continue
        current_lines.append(line)
        if s.lower().startswith("support"):
            try:
                current_support = float(s.split(":", 1)[1].strip())
            except Exception:
                pass

    if current_lines:
        blocks.append(
            {
                "lines": current_lines,
                "support": current_support,
                "order": order_idx,
            }
        )

    return blocks


def trim_gspan_top_k(out_file: Path, top_k: int) -> int:
    if top_k <= 0 or (not out_file.exists()):
        return 0

    lines = out_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    blocks = _extract_gspan_blocks(lines)

    pattern_blocks = []
    for block in blocks:
        first = block["lines"][0].strip() if block["lines"] else ""
        if first.startswith("t # -1"):
            continue
        has_node = any(x.strip().startswith("v ") for x in block["lines"])
        if has_node:
            pattern_blocks.append(block)

    if not pattern_blocks:
        out_file.write_text("", encoding="utf-8")
        return 0

    sorted_blocks = sorted(
        pattern_blocks,
        key=lambda b: (
            -(b["support"] if b["support"] is not None else -1),
            b["order"],
        ),
    )
    top_blocks = sorted_blocks[:top_k]

    out_lines = []
    for block in top_blocks:
        out_lines.extend(block["lines"])
    out_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return len(top_blocks)


def finite_or_nan(value: float) -> float:
    return value if math.isfinite(value) else np.nan


def plot_results(df: pd.DataFrame, dataset: str, out_dir: Path):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    if "graph_size" in df.columns and df["graph_size"].nunique() > 1:
        plt.figure(figsize=(10, 6))
        for size in sorted(df["graph_size"].dropna().unique()):
            sub = df[df["graph_size"] == size].sort_values("k")
            plt.plot(sub["k"], sub["gspan_time"], marker="o", linewidth=2,
                label=f"gSpan | {size}节点")
            plt.plot(sub["k"], sub["spminer_time"], marker="s", linewidth=2,
                label=f"SPMiner | {size}节点")
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(df["k"], df["gspan_time"], marker="o", label="gSpan", linewidth=2)
        plt.plot(df["k"], df["spminer_time"], marker="s", label="SPMiner", linewidth=2)
    plt.xlabel("子图大小 k (节点数)", fontsize=12)
    plt.ylabel("运行时间 (秒)", fontsize=12)
    plt.title(f"SPMiner vs gSpan 运行时间对比 ({dataset})", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"time_comparison_{dataset}.png", dpi=300)

    if "graph_size" in df.columns and df["graph_size"].nunique() > 1:
        plt.figure(figsize=(10, 6))
        for size in sorted(df["graph_size"].dropna().unique()):
            sub = df[df["graph_size"] == size].sort_values("k")
            plt.plot(sub["k"], sub["gspan_mem"], marker="o", linewidth=2,
                label=f"gSpan | {size}节点")
            plt.plot(sub["k"], sub["spminer_mem"], marker="s", linewidth=2,
                label=f"SPMiner | {size}节点")
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(df["k"], df["gspan_mem"], marker="o", label="gSpan", linewidth=2)
        plt.plot(df["k"], df["spminer_mem"], marker="s", label="SPMiner", linewidth=2)
    plt.xlabel("子图大小 k (节点数)", fontsize=12)
    plt.ylabel("最大内存占用 (MB)", fontsize=12)
    plt.title(f"SPMiner vs gSpan 内存占用对比 ({dataset})", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"mem_comparison_{dataset}.png", dpi=300)


def main():
    args = parse_args()
    repo_root = resolve_path(Path.cwd(), args.repo_root)
    out_dir = resolve_path(repo_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    size_contexts = []
    if args.graph_sizes:
        edge_list_path = resolve_path(repo_root, args.edge_list)
        data_dir = repo_root / "compare" / "data"
        for size in args.graph_sizes:
            gspan_db = data_dir / f"facebook_gspan_{size}.txt"
            n_nodes, n_edges = build_gspan_db_from_edge_list(edge_list_path, gspan_db, size)
            print(f"[multi] 生成规模图: {gspan_db} (nodes={n_nodes}, edges={n_edges})")
            size_contexts.append((size, gspan_db))
    else:
        gspan_db = resolve_path(repo_root, args.gspan_db_file) if args.gspan_db_file else None
        size_contexts.append((None, gspan_db))

    results = {
        "graph_size": [],
        "k": [],
        "gspan_time": [],
        "spminer_time": [],
        "gspan_mem": [],
        "spminer_mem": [],
        "gspan_status": [],
        "spminer_status": [],
        "gspan_result_file": [],
        "spminer_result_file": [],
        "spminer_log_file": [],
    }

    for graph_size, gspan_db in size_contexts:
        local_args = argparse.Namespace(**vars(args))
        if gspan_db:
            local_args.gspan_db_file = str(gspan_db)

        spminer_dataset = args.dataset
        if args.fair_shared_input:
            spminer_dataset = prepare_spminer_dataset_from_gspan_db(local_args, repo_root)
            print(f"[fair] 使用共享输入运行：gSpan + SPMiner 均基于 {local_args.gspan_db_file}")

        size_tag = f"n{graph_size}" if graph_size is not None else "nbase"
        for k in args.ks:
            print(f"===== 正在测试 {size_tag}, k={k} =====")
            results["graph_size"].append(graph_size if graph_size is not None else -1)
            results["k"].append(k)

            gspan_out = out_dir / f"gspan_out_{args.dataset}_{size_tag}_k{k}.txt"
            spminer_out = out_dir / f"spminer_out_{args.dataset}_{size_tag}_k{k}.p"
            spminer_log = out_dir / f"spminer_log_{args.dataset}_{size_tag}_k{k}.txt"

            results["gspan_result_file"].append(str(gspan_out))
            results["spminer_result_file"].append(str(spminer_out))
            results["spminer_log_file"].append(str(spminer_log))

            try:
                g_time, g_mem = run_gspan(local_args, repo_root, gspan_out, k)
                g_kept = trim_gspan_top_k(gspan_out, local_args.top_k_patterns)
                results["gspan_time"].append(g_time)
                results["gspan_mem"].append(g_mem)
                results["gspan_status"].append("ok")
                print(
                    f"gSpan 完成：时间={g_time:.2f}s, 内存={g_mem:.2f}MB, "
                    f"top-{local_args.top_k_patterns}保留={g_kept}"
                )
            except Exception as exc:
                results["gspan_time"].append(np.inf)
                results["gspan_mem"].append(np.inf)
                results["gspan_status"].append(str(exc))
                print(f"gSpan 失败：{exc}")

            try:
                s_time, s_mem = run_spminer(
                    local_args,
                    repo_root,
                    spminer_out,
                    spminer_log,
                    k,
                    spminer_dataset,
                )
                s_kept = trim_spminer_top_k(spminer_out, local_args.top_k_patterns)
                results["spminer_time"].append(s_time)
                results["spminer_mem"].append(s_mem)
                results["spminer_status"].append("ok")
                print(
                    f"SPMiner 完成：时间={s_time:.2f}s, 内存={s_mem:.2f}MB, "
                    f"top-{local_args.top_k_patterns}保留={s_kept}"
                )
            except Exception as exc:
                results["spminer_time"].append(np.inf)
                results["spminer_mem"].append(np.inf)
                results["spminer_status"].append(str(exc))
                print(f"SPMiner 失败：{exc}")

    df = pd.DataFrame(results)
    csv_path = out_dir / f"experiment_result_{args.dataset}.csv"
    df.to_csv(csv_path, index=False)

    plot_df = df.copy()
    plot_df["gspan_time"] = plot_df["gspan_time"].map(finite_or_nan)
    plot_df["spminer_time"] = plot_df["spminer_time"].map(finite_or_nan)
    plot_df["gspan_mem"] = plot_df["gspan_mem"].map(finite_or_nan)
    plot_df["spminer_mem"] = plot_df["spminer_mem"].map(finite_or_nan)
    plot_results(plot_df, args.dataset, out_dir)

    if "graph_size" in df.columns and df["graph_size"].nunique() > 1:
        summary_path = out_dir / f"summary_{args.dataset}_by_size_k.csv"
        df.to_csv(summary_path, index=False)
        print(f"多规模汇总已保存：{summary_path}")

    print(f"实验结果已保存：{csv_path}")
    print(f"对比图已生成到目录：{out_dir}")


if __name__ == "__main__":
    main()