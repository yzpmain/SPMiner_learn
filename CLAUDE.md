# CLAUDE.md

此文件为 Claude Code 在此仓库中的操作提供指导。完整功能说明和参数表见 README.md。

## 常用命令

```bash
# 快速测试训练（完整命令见 README）
python -m src.subgraph_matching.train --dataset=facebook --node_anchored --n_batches 20 --eval_interval 10 --batch_size 32 --n_workers 1 --model_path results/test.pt

# 快速测试挖掘
python -u -m src.subgraph_mining.decoder --dataset=facebook --node_anchored --model_path results/test.pt --n_neighborhoods 50 --batch_size 50 --n_trials 5 --out_path results/test_patterns.p

# GUI
python -m src.gui.main_window

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
    models.py                    # GNN: OrderEmbedder(序嵌入), SkipLastGNN, SAGEConv, GINConv
    data.py                      # 数据源: OTFSynDataSource(在线合成), DiskDataSource(真实数据集)
    utils.py                     # 邻域采样、WL哈希、批处理、优化器构建
    combined_syn.py              # 合成图生成器 (ER/WS等)
    feature_preprocess.py        # 特征增强管线

  subgraph_matching/             # 阶段1: 训练子图关系编码器
    train.py / test.py / config.py / alignment.py

  subgraph_mining/               # 阶段2: 频繁子图挖掘
    decoder.py                   # SPMiner 主流程
    search_agents.py             # GreedySearchAgent / MCTSSearchAgent
    config.py

  analyze/                       # 挖掘后分析
  compare/                       # 基线比较 (gSpan等)
  gui/                           # Tkinter 图形界面
```

### 关键设计点

- **Order embedding 损失** — 训练目标：正例最小化违反量（emb_子图 ≤ emb_超图），负例推过 margin
- **SkipLastGNN** — 多层 MPNN，支持 GCN/GIN/SAGE/GAT/PNA 卷积，可配置 skip 连接
- **数据源** — 支持在线合成（OTFSynDataSource）和真实数据集（DiskDataSource 加载 TUDataset/SNAP），均支持 balanced/imbalanced 采样
- **节点锚定** — query/target 共享指定锚点，子图映射必须保留该锚点
