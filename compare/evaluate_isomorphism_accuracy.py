import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
import pickle


PAIR_RE = re.compile(r"n(?P<n>\d+)_k(?P<k>\d+)")


def parse_gspan_output(file_path: Path) -> List[nx.Graph]:
    graphs: List[nx.Graph] = []
    current = None

    for raw in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.startswith("t "):
            if current is not None and current.number_of_nodes() > 0:
                graphs.append(current)
            current = nx.Graph()
            continue

        if line.startswith("v ") and current is not None:
            toks = line.split()
            if len(toks) >= 3:
                node_id = int(toks[1])
                label = toks[2]
                current.add_node(node_id, label=label)
            continue

        if line.startswith("e ") and current is not None:
            toks = line.split()
            if len(toks) >= 4:
                u = int(toks[1])
                v = int(toks[2])
                label = toks[3]
                current.add_edge(u, v, label=label)
            continue

    if current is not None and current.number_of_nodes() > 0:
        graphs.append(current)
    return graphs


def load_spminer_pickle(file_path: Path) -> List[nx.Graph]:
    with file_path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported SPMiner result format in {file_path}")


def key_of_file(file_path: Path) -> Tuple[int, int] | None:
    m = PAIR_RE.search(file_path.stem)
    if not m:
        return None
    return int(m.group("n")), int(m.group("k"))


def quick_sig(g: nx.Graph) -> Tuple[int, int, Tuple[int, ...]]:
    deg = tuple(sorted((d for _, d in g.degree()), reverse=True))
    return g.number_of_nodes(), g.number_of_edges(), deg


def count_isomorphic_matches(spminer_graphs: List[nx.Graph], gspan_graphs: List[nx.Graph]) -> int:
    if not spminer_graphs or not gspan_graphs:
        return 0

    adjacency: Dict[int, List[int]] = {}
    gspan_sigs = [quick_sig(g) for g in gspan_graphs]
    spminer_sigs = [quick_sig(g) for g in spminer_graphs]

    for i, sg in enumerate(spminer_graphs):
        for j, gg in enumerate(gspan_graphs):
            if spminer_sigs[i] != gspan_sigs[j]:
                continue
            if nx.is_isomorphic(sg, gg):
                adjacency.setdefault(i, []).append(j)

    # Left partition nodes: ('s', i), right partition nodes: ('g', j)
    bip = nx.Graph()
    left_nodes = [("s", i) for i in range(len(spminer_graphs))]
    right_nodes = [("g", j) for j in range(len(gspan_graphs))]
    bip.add_nodes_from(left_nodes, bipartite=0)
    bip.add_nodes_from(right_nodes, bipartite=1)

    for i, js in adjacency.items():
        for j in js:
            bip.add_edge(("s", i), ("g", j))

    matching = nx.algorithms.bipartite.maximum_matching(bip, top_nodes=set(left_nodes))
    # maximum_matching returns both directions in dict.
    matched = sum(1 for n in left_nodes if n in matching)
    return matched


def evaluate_pair(spminer_graphs: List[nx.Graph], gspan_graphs: List[nx.Graph], top_k: int) -> dict:
    if top_k > 0:
        spminer_graphs = spminer_graphs[:top_k]
        gspan_graphs = gspan_graphs[:top_k]

    n_sp = len(spminer_graphs)
    n_gs = len(gspan_graphs)
    matched = count_isomorphic_matches(spminer_graphs, gspan_graphs)

    precision = matched / n_sp if n_sp > 0 else 0.0
    recall = matched / n_gs if n_gs > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    # 以 gSpan 结果作为参考集合时的同构准确率。
    accuracy = recall

    union = n_sp + n_gs - matched
    jaccard = matched / union if union > 0 else 1.0

    return {
        "spminer_count": n_sp,
        "gspan_count": n_gs,
        "isomorphic_matches": matched,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "jaccard": round(jaccard, 6),
    }


def collect_files_from_summary(summary_csv: Path) -> Tuple[Dict[Tuple[int, int], Path], Dict[Tuple[int, int], Path]]:
    df = pd.read_csv(summary_csv)
    sp_files: Dict[Tuple[int, int], Path] = {}
    gs_files: Dict[Tuple[int, int], Path] = {}

    for _, row in df.iterrows():
        src = str(row["source"]).strip().lower()
        file_path = Path(str(row["file"]))
        key = key_of_file(file_path)
        if key is None:
            continue
        if src == "spminer":
            sp_files[key] = file_path
        elif src == "gspan":
            gs_files[key] = file_path

    return sp_files, gs_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate isomorphism-based accuracy between SPMiner and gSpan")
    parser.add_argument(
        "--summary-csv",
        default="compare/visuals/batch_n40_60_80_k5_8_10_12/visualization_summary.csv",
        help="Summary CSV generated by visualize_mined_subgraphs.py",
    )
    parser.add_argument(
        "--out-csv",
        default="compare/visuals/batch_n40_60_80_k5_8_10_12/isomorphism_accuracy.csv",
        help="Output CSV path for per-(n,k) metrics",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="每组仅使用前 top-k 个模式进行同构准确率计算",
    )
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    out_csv = Path(args.out_csv)

    sp_files, gs_files = collect_files_from_summary(summary_csv)
    keys = sorted(set(sp_files) & set(gs_files))
    if not keys:
        raise RuntimeError("No matched (n, k) file pairs found in summary CSV")

    rows = []
    for n, k in keys:
        sp_path = sp_files[(n, k)]
        gs_path = gs_files[(n, k)]

        sp_graphs = load_spminer_pickle(sp_path)
        gs_graphs = parse_gspan_output(gs_path)

        metrics = evaluate_pair(sp_graphs, gs_graphs, args.top_k)
        row = {
            "graph_size": n,
            "k": k,
            "spminer_file": str(sp_path),
            "gspan_file": str(gs_path),
        }
        row.update(metrics)
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["graph_size", "k"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    macro_acc = float(df["accuracy"].mean())
    weighted_acc = float((df["accuracy"] * df["gspan_count"]).sum() / max(df["gspan_count"].sum(), 1))
    print(f"Saved: {out_csv}")
    print(f"Pairs: {len(df)}")
    print(f"Macro accuracy: {macro_acc:.4f}")
    print(f"Weighted accuracy (by gspan_count): {weighted_acc:.4f}")


if __name__ == "__main__":
    main()