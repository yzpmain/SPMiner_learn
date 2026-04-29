import argparse
import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from src.core.utils import parse_gspan_output, load_spminer_pickle

__all__ = [
    "draw_graph",
    "save_single_graphs",
    "save_montage",
    "summarize_records",
]



def draw_graph(ax, graph: nx.Graph, title: str) -> None:
    pos = nx.spring_layout(graph, seed=42)
    has_anchor = any("anchor" in graph.nodes[n] for n in graph.nodes)

    if has_anchor:
        colors = ["#e74c3c" if graph.nodes[n].get("anchor", 0) else "#4ea8de" for n in graph.nodes]
    else:
        colors = ["#4ea8de"] * graph.number_of_nodes()

    nx.draw_networkx(
        graph,
        pos=pos,
        ax=ax,
        node_color=colors,
        edge_color="#5c677d",
        with_labels=True,
        font_size=8,
        node_size=380,
    )
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def save_single_graphs(graphs: List[nx.Graph], out_dir: Path, prefix: str, max_graphs: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = min(len(graphs), max_graphs)
    for i in range(limit):
        g = graphs[i]
        fig, ax = plt.subplots(figsize=(4, 4))
        draw_graph(ax, g, f"{prefix} #{i} | V={g.number_of_nodes()} E={g.number_of_edges()}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_{i:03d}.png", dpi=220)
        plt.close(fig)


def save_montage(graphs: List[nx.Graph], out_path: Path, prefix: str, max_graphs: int) -> None:
    if not graphs:
        return

    limit = min(len(graphs), max_graphs)
    cols = 4
    rows = math.ceil(limit / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.0))
    if rows == 1 and cols == 1:
        axes_list = [axes]
    elif rows == 1:
        axes_list = list(axes)
    else:
        axes_list = [ax for row in axes for ax in row]

    for i, ax in enumerate(axes_list):
        if i < limit:
            g = graphs[i]
            draw_graph(ax, g, f"{prefix} #{i}")
        else:
            ax.axis("off")

    fig.suptitle(f"{prefix} mined frequent subgraphs (top {limit})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def summarize_records(records: List[dict], out_csv: Path) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize mined frequent subgraph results")
    parser.add_argument("--spminer", nargs="*", default=[], help="SPMiner pickle result files")
    parser.add_argument("--gspan", nargs="*", default=[], help="gSpan text result files")
    parser.add_argument("--out-dir", default="compare/visuals", help="Output directory")
    parser.add_argument("--max-graphs", type=int, default=24, help="Max graphs to draw for each source")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []

    for file_name in args.spminer:
        src = Path(file_name)
        graphs = load_spminer_pickle(src)
        tag = src.stem
        single_dir = out_dir / f"{tag}_single"
        save_single_graphs(graphs, single_dir, tag, args.max_graphs)
        save_montage(graphs, out_dir / f"{tag}_montage.png", tag, args.max_graphs)
        summary_records.append(
            {
                "source": "spminer",
                "file": str(src),
                "num_graphs": len(graphs),
                "avg_nodes": round(sum(g.number_of_nodes() for g in graphs) / max(len(graphs), 1), 4),
                "avg_edges": round(sum(g.number_of_edges() for g in graphs) / max(len(graphs), 1), 4),
                "single_dir": str(single_dir),
                "montage": str(out_dir / f"{tag}_montage.png"),
            }
        )

    for file_name in args.gspan:
        src = Path(file_name)
        graphs = parse_gspan_output(src)
        tag = src.stem
        single_dir = out_dir / f"{tag}_single"
        save_single_graphs(graphs, single_dir, tag, args.max_graphs)
        save_montage(graphs, out_dir / f"{tag}_montage.png", tag, args.max_graphs)
        summary_records.append(
            {
                "source": "gspan",
                "file": str(src),
                "num_graphs": len(graphs),
                "avg_nodes": round(sum(g.number_of_nodes() for g in graphs) / max(len(graphs), 1), 4),
                "avg_edges": round(sum(g.number_of_edges() for g in graphs) / max(len(graphs), 1), 4),
                "single_dir": str(single_dir),
                "montage": str(out_dir / f"{tag}_montage.png"),
            }
        )

    summarize_records(summary_records, out_dir / "visualization_summary.csv")
    print(f"Visualization finished. Output directory: {out_dir}")


if __name__ == "__main__":
    main()
