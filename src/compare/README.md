# compare 目录

本目录用于存放 SPMiner 与其他子图挖掘方法的对比分析。

当前报告：

- [SPMiner 与 gSpan 对比分析](spminer_vs_gspan.md)

## 快速运行对比

1. 先生成 gSpan 输入文件（示例：从 Facebook 边列表生成一个小规模数据库）：

```bash
python -m src.compare.build_gspan_db --edge-list data/facebook_combined.txt --out src/compare/data/facebook_gspan_small.txt --max-nodes 200
```

2. 运行对比脚本（Windows 推荐用内置 gspan_mining 模式，避免命令模板转义问题）：

```bash
python -m src.compare.compare --dataset facebook --ks 5 6 7 8 9 10 11 12 13 14 15 --min-sup 1 --timeout-sec 120 --spminer-trials 1 --spminer-neighborhoods 10 --spminer-batch-size 10 --gspan-db-file src/compare/data/facebook_gspan_small.txt --use-gspan-mining --evaluate-frequency
```

结果会输出到 `compare/out/`，包括：

- `experiment_result_*.csv`
- `time_comparison_*.png`
- `mem_comparison_*.png`
- `speedup_comparison_*.png`
- `frequency_comparison_*.png`
- `accuracy_summary_*.csv`

如果后续要继续扩展，可以在这里补充：

- 实验对比表
- 不同数据集上的结果汇总
- 运行时间与内存占用统计
- 可视化图表和结论摘要

当前 compare 流程已经拆成运行层、分析层和绘图层，支持批量 k=5~15 的时间/内存/加速比对比，以及基于 gSpan support 和精确计数的频率对齐分析。