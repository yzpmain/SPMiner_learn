from pathlib import Path
import argparse

from src.compare.benchmarking import build_gspan_db_from_edge_list


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

    n_nodes, n_edges = build_gspan_db_from_edge_list(edge_path, out_path, args.max_nodes)
    print(f"nodes={n_nodes} edges={n_edges}")
    print(out_path)


if __name__ == "__main__":
    main()
