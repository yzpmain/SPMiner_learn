from __future__ import annotations

import argparse
import os
import pickle
import shlex
import subprocess
import time
from pathlib import Path

import psutil

__all__ = [
    "resolve_path",
    "build_gspan_db_from_edge_list",
    "prepare_spminer_dataset_from_gspan_db",
    "run_and_monitor",
    "run_spminer",
    "run_gspan",
    "trim_spminer_top_k",
    "trim_gspan_top_k",
    "size_tag",
    "case_output_paths",
]


def resolve_path(base: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else base / path


def build_gspan_db_from_edge_list(edge_list_path: Path, out_path: Path, max_nodes: int) -> tuple[int, int]:
    """从边列表构建 gSpan 数据文件。"""
    if not edge_list_path.exists():
        raise FileNotFoundError(f"edge list file not found: {edge_list_path}")

    graph: dict[int, set[int]] = {}
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
    """将 gSpan DB 的第一张图导出为 SPMiner 可读的边列表。"""
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
    out_edge_file.parent.mkdir(parents=True, exist_ok=True)
    with out_edge_file.open("w", encoding="utf-8") as f:
        for u, v in edges:
            f.write(f"{u}\t{v}\n")

    print(f"[fair] SPMiner 数据集已生成: data/{dataset_name}.txt (edges={len(edges)})")
    return dataset_name


def _process_rss_mb(proc: psutil.Process) -> float:
    total_rss = 0.0
    try:
        total_rss += proc.memory_info().rss
    except Exception:
        pass
    try:
        for child in proc.children(recursive=True):
            try:
                total_rss += child.memory_info().rss
            except Exception:
                continue
    except Exception:
        pass
    return total_rss / 1024 / 1024


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
    stderr_target = subprocess.DEVNULL
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_handle = stdout_path.open("w", encoding="utf-8")
        stderr_target = subprocess.STDOUT

    proc = psutil.Popen(
        cmd,
        cwd=str(cwd),
        stdout=stdout_handle if stdout_handle else subprocess.DEVNULL,
        stderr=stderr_target,
    )
    max_mem_mb = 0.0

    try:
        while proc.poll() is None:
            elapsed = time.time() - start
            if elapsed > timeout_sec:
                proc.kill()
                raise TimeoutError(f"timeout after {timeout_sec}s")
            max_mem_mb = max(max_mem_mb, _process_rss_mb(proc))
            time.sleep(poll_interval)

        ret_code = proc.returncode
        elapsed = time.time() - start
        if ret_code not in ok_exit_codes:
            raise RuntimeError(f"process exited with code {ret_code}")
        return elapsed, max_mem_mb
    finally:
        if stdout_handle:
            stdout_handle.close()


def _format_min_sup(min_sup: float):
    return int(min_sup) if float(min_sup).is_integer() else min_sup


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
        "src.subgraph_mining.decoder",
        f"--dataset={spminer_dataset}",
        f"--model_path={model_path}",
        f"--n_neighborhoods={args.spminer_neighborhoods}",
        f"--batch_size={args.spminer_batch_size}",
        f"--n_trials={n_trials}",
        f"--min_pattern_size={k}",
        f"--max_pattern_size={k}",
        f"--out_path={out_file}",
    ]
    if args.node_anchored:
        cmd.append("--node_anchored")
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
            str(_format_min_sup(args.min_sup)),
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

    cmd_str = args.gspan_cmd_template.format(
        dataset=args.dataset,
        k=k,
        min_sup=_format_min_sup(args.min_sup),
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


def size_tag(graph_size: int | None) -> str:
    return f"n{graph_size}" if graph_size is not None else "nbase"


def case_output_paths(out_dir: Path, dataset: str, graph_size: int | None, k: int) -> dict[str, Path]:
    tag = size_tag(graph_size)
    return {
        "gspan_out": out_dir / f"gspan_out_{dataset}_{tag}_k{k}.txt",
        "spminer_out": out_dir / f"spminer_out_{dataset}_{tag}_k{k}.p",
        "spminer_log": out_dir / f"spminer_log_{dataset}_{tag}_k{k}.txt",
    }