from pathlib import Path
import argparse
import networkx as nx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build gSpan database from edge list")
    parser.add_argument("--edge-list", type=str, required=True, help="Input edge list path")
    parser.add_argument("--out", type=str, required=True, help="Output gSpan DB file")
    parser.add_argument("--max-nodes", type=int, default=0, help="Keep first N sorted nodes (0 means all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    edge_path = Path(args.edge_list)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    graph = nx.Graph()
    with edge_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            a, b = map(int, s.split()[:2])
            graph.add_edge(a, b)

    if args.max_nodes > 0:
        nodes = sorted(graph.nodes())[: args.max_nodes]
        graph = graph.subgraph(nodes).copy()

    nodes = sorted(graph.nodes())
    idx = {n: i for i, n in enumerate(nodes)}

    with out_path.open("w", encoding="utf-8") as f:
        f.write("t # 0\n")
        for n in nodes:
            f.write(f"v {idx[n]} 0\n")
        for u, v in graph.edges():
            f.write(f"e {idx[u]} {idx[v]} 0\n")
        f.write("t # -1\n")

    print(f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")
    print(out_path)


if __name__ == "__main__":
    main()
