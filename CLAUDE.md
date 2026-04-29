# CLAUDE.md

此文件为 Claude Code 在此仓库中的操作提供指导。完整功能说明和参数表见 README.md。

## 常用命令

```bash
# 快速测试训练（完整命令见 README）
python -m src.subgraph_matching.train --dataset=facebook --node_anchored --n_batches 20 --eval_interval 10 --batch_size 32 --n_workers 1 --model_path results/test.pt --output_strategy version

# 快速测试挖掘
python -u -m src.subgraph_mining.decoder --dataset=facebook --node_anchored --model_path results/test.pt --n_neighborhoods 50 --batch_size 50 --n_trials 5 --out_path results/test_patterns.p --output_strategy version

# 统计计数（示例）
python -m src.analyze.count_patterns --dataset=facebook --queries_path results/test_patterns.p --out_path results/test_counts.json --output_strategy version

# 测试
python -m pytest tests/
python -m pytest tests/unit/
python -m pytest tests/integration/
```

## 架构与设计

两阶段工作流：先训练子图匹配编码器，再复用编码器作为评分器进行频繁子图挖掘。

```
src/
  core/                          # 基础模块
    config.py                    # 配置 dataclass: MatchingConfig / MiningConfig / RuntimeConfig / AugmentConfig
    models.py                    # GNN: OrderEmbedder(序嵌入), SkipLastGNN, SAGEConv, GINConv
    data.py                      # 数据源: OTFSynDataSource(在线合成), DiskDataSource(真实数据集)
    dataset_registry.py          # 数据集注册表（名称校验 + 加载器映射）
    facade.py                    # CoreFacade 统一入口
    feature_preprocess.py        # 特征增强管线（FeatureAugment / Preprocess）
    device.py                    # 设备管理（DeviceManager）
    hashing.py                   # WL 哈希与向量哈希
    optimizer.py                 # 优化器构建
    batch.py                     # 批处理（batch_nx_graphs）
    combined_syn.py              # 合成图生成器 (ER/WS等)
    utils.py                     # 向后兼容 shim，委托到各子模块
    cli.py                       # 运行时参数设置
    artifacts.py                 # 产物路径与 manifest 管理
    io/                          # 图 I/O（gspan_parser / pickle_io / graph_io）
    sampling/                    # 邻域采样（neighborhood / enumeration / baseline_queries）

  subgraph_matching/             # 阶段1: 训练子图关系编码器
    train.py                     # train / train_step / train_loop 三层
    test.py                      # validation / compute_metrics / plot_pr_curve
    config.py / alignment.py

  subgraph_mining/               # 阶段2: 频繁子图挖掘
    decoder.py                   # SPMiner 入口（薄包装，委托给 pipeline）
    pipeline.py                  # PatternGrowthPipeline 流水线类
    search_agents.py             # GreedySearchAgent / MCTSSearchAgent
    config.py

  analyze/                       # 挖掘后分析
  compare/                       # 基线比较 (gSpan等)
```

### 关键设计点

- **Order embedding 损失** — 训练目标：正例最小化违反量（emb_子图 ≤ emb_超图），负例推过 margin
- **SkipLastGNN** — 多层 MPNN，支持 GCN/GIN/SAGE/GAT/PNA 卷积，可配置 skip 连接
- **数据源** — 支持在线合成（OTFSynDataSource）和真实数据集（DiskDataSource 加载 TUDataset/SNAP），均支持 balanced/imbalanced 采样
- **节点锚定** — query/target 共享指定锚点，子图映射必须保留该锚点

## 产物输出约定

- 默认使用统一输出根目录：`results/`
- 推荐开启 `--output_strategy version` 防覆盖（默认已为 version）
- 可用 `--output_tag <tag>` 将同批实验归档到可读目录
- 主要任务默认归档到：`results/<task>/<dataset>/<run_name>/`
- 每次运行的参数与输入/输出摘要会写入 `manifest.json`
