import argparse
import os

import torch

from src.core import CoreFacade
from src.core import utils
from src.subgraph_mining.config import parse_decoder
from src.subgraph_matching.config import parse_encoder
from src.subgraph_mining.pipeline import PatternGrowthPipeline
from src.logger import RunLogger, info, section
from src.core.artifacts import (
    choose_cli_output_path,
    task_output_dir,
)

import networkx as nx

from src.core.cli import setup_runtime

__all__ = ["pattern_growth", "main"]


def pattern_growth(dataset, task, args):
    """SPMiner 主流程（薄包装，实际逻辑在 PatternGrowthPipeline 中）。"""
    model = CoreFacade.build_model(args, for_inference=True, load_weights=True)
    pipeline = PatternGrowthPipeline(args, model, dataset, task)
    return pipeline.run()

def main():
    """解码器 CLI 入口。

    负责：
    - 组合 encoder/decoder 参数；
    - 读取数据集；
    - 调用 pattern_growth 执行完整挖掘。
    """
    parser = argparse.ArgumentParser(description='解码器参数')
    parse_encoder(parser)
    parse_decoder(parser)
    args = parser.parse_args()

    setup_runtime(args)

    with RunLogger(args):
        section("数据加载")
        # 统一命名与入口校验：所有挖掘数据集都从注册中心加载。
        normalized_dataset, dataset, task = CoreFacade.load_stage_dataset(args.dataset, "mining")
        args.dataset = normalized_dataset
        info("Using dataset {}".format(args.dataset))

        artifact_dir = task_output_dir(args, "mining", args.dataset)
        default_out_path = "results/out-patterns.p"
        args.out_path = str(choose_cli_output_path(
            args,
            args.out_path,
            default_cli_path=default_out_path,
            suggested_default_path=artifact_dir / "patterns.p",
        ))
        args.artifact_dir = str(artifact_dir)
        args.pattern_plot_dir = str(artifact_dir / "plots")
        args.analysis_out_path = str(artifact_dir / "analyze.p")
        args.analysis_plot_path = str(artifact_dir / "analyze.png")
        os.makedirs(args.pattern_plot_dir, exist_ok=True)

        pattern_growth(dataset, task, args)

if __name__ == '__main__':
    main()

