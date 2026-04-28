from pathlib import Path

import networkx as nx

from src.compare.analysis import evaluate_pair, match_isomorphic_patterns
from src.compare.benchmarking import trim_gspan_top_k


def _make_path_graph(n: int) -> nx.Graph:
    graph = nx.path_graph(n)
    return nx.convert_node_labels_to_integers(graph)


def test_match_isomorphic_patterns_finds_pairs():
    spminer_graphs = [_make_path_graph(3), _make_path_graph(4)]
    gspan_graphs = [_make_path_graph(4), _make_path_graph(3)]

    pairs = match_isomorphic_patterns(spminer_graphs, gspan_graphs)

    assert sorted(pairs) == [(0, 1), (1, 0)]


def test_evaluate_pair_reports_perfect_overlap():
    spminer_graphs = [_make_path_graph(3), _make_path_graph(4)]
    gspan_graphs = [_make_path_graph(3), _make_path_graph(4)]

    metrics = evaluate_pair(spminer_graphs, gspan_graphs, top_k=2)

    assert metrics["isomorphic_matches"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_trim_gspan_top_k_keeps_high_support_blocks(tmp_path):
    out_file = tmp_path / "gspan_out.txt"
    out_file.write_text(
        "\n".join(
            [
                "t # 0",
                "v 0 0",
                "v 1 0",
                "e 0 1 0",
                "Support: 5",
                "t # 1",
                "v 0 0",
                "v 1 0",
                "e 0 1 0",
                "Support: 2",
                "t # -1",
            ]
        ),
        encoding="utf-8",
    )

    kept = trim_gspan_top_k(out_file, 1)

    contents = out_file.read_text(encoding="utf-8")
    assert kept == 1
    assert "Support: 5" in contents
    assert "Support: 2" not in contents