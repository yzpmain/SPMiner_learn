"""Comparison and benchmarking tools package."""

__all__ = [
    "add_runtime_metrics",
    "build_accuracy_table",
    "build_gspan_db_from_edge_list",
    "case_output_paths",
    "compare_main",
    "evaluate_pair",
    "exact_support_counts",
    "match_isomorphic_patterns",
    "plot_results",
    "prepare_spminer_dataset_from_gspan_db",
    "run_gspan",
    "run_spminer",
    "trim_gspan_top_k",
    "trim_spminer_top_k",
]

from src.compare.analysis import (
    add_runtime_metrics,
    build_accuracy_table,
    evaluate_pair,
    exact_support_counts,
    match_isomorphic_patterns,
)
from src.compare.benchmarking import (
    build_gspan_db_from_edge_list,
    case_output_paths,
    prepare_spminer_dataset_from_gspan_db,
    run_gspan,
    run_spminer,
    trim_gspan_top_k,
    trim_spminer_top_k,
)
from src.compare.compare import main as compare_main
from src.compare.plotting import plot_results

