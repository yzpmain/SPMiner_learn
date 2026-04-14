# Neural Subgraph Learning Library

Neural Subgraph Learning Library，简称 NSL，是一个面向子图关系学习的图神经网络项目。它把“子图匹配”和“频繁子图挖掘”放在同一个框架中，先用 GNN 学习图表示，再用任务头完成判别或搜索。

这个仓库主要包含两条主线：

1. 神经子图匹配（Neural Subgraph Matching / NeuroMatch）。
2. 频繁子图挖掘（Frequent Subgraph Mining / SPMiner）。

如果你只想先跑通项目，建议直接看“快速开始”。如果你想系统了解项目结构和参数，再从“项目概览”和“参数对照表”开始。

## 项目概览

NSL 关注的问题不是普通图分类，而是更细粒度的子图关系判断与模式挖掘：

- 判断一个 query 图是否可以作为 target 图的子图。
- 在图数据集或大图局部邻域中搜索高频结构模式。
- 把图的结构信息编码到 embedding 空间中，再进行匹配和搜索。

仓库提供了训练、推理、挖掘、统计分析和可视化的一整套入口，适合做方法复现、基线比较和模式分析。

## 功能一览

| 功能 | 模块 | 说明 |
| --- | --- | --- |
| 子图匹配 | `subgraph_matching/` | 训练图编码器并判断两图是否存在子图关系 |
| 频繁子图挖掘 | `subgraph_mining/` | 复用匹配模型，对候选 pattern 做搜索与筛选 |
| 结果分析 | `analyze/` | 分析 embedding、统计 pattern 频次、可视化结果 |

## 项目结构

```text
common/
   combined_syn.py       合成数据生成逻辑
   data.py               真实数据集与数据源封装
   feature_preprocess.py  特征预处理
   models.py             GNN 编码器与任务头
   utils.py              通用工具函数

subgraph_matching/
   config.py             子图匹配训练参数
   train.py              训练入口
   test.py               评估与验证入口
   alignment.py          图对齐与匹配分数工具
   hyp_search.py         超参数搜索入口

subgraph_mining/
   config.py             挖掘阶段参数
   decoder.py            SPMiner 挖掘入口
   search_agents.py      Greedy / MCTS 搜索代理

analyze/
   Analyze Embeddings.ipynb
   Visualize Graph Statistics.ipynb
   count_patterns.py
   analyze_pattern_counts.py

data/                   本地数据文件
ckpt/                   预训练 checkpoint
results/                挖掘输出与统计结果
plots/                  图像输出
```

## 工作流程

典型工作流分成两步：

1. 先训练子图匹配编码器。
2. 再用这个编码器作为打分器做频繁子图挖掘。

仓库中提供了一个可用的示例 checkpoint：

- [ckpt/model.pt](ckpt/model.pt)

## 环境与安装

建议使用 conda 环境，尤其是在 Windows 上。项目的实际可运行环境需要至少满足以下条件：

| 项目 | 建议 |
| --- | --- |
| 操作系统 | Windows / Linux |
| Python | 3.10.x |
| 深度学习框架 | PyTorch、PyTorch Geometric |
| 图数据处理 | DeepSNAP、NetworkX |
| 统计与绘图 | NumPy、SciPy、scikit-learn、Matplotlib、Seaborn |
| 训练日志 | TensorBoard |

### 推荐安装顺序

| 步骤 | 命令 | 说明 |
| --- | --- | --- |
| 1 | `conda create -n neural-subgraph-learning-GNN python=3.10` | 创建独立环境 |
| 2 | `conda activate neural-subgraph-learning-GNN` | 激活环境 |
| 3 | `pip install -r requirements.txt` | 安装基础依赖 |
| 4 | `pip install tensorboard` | 补齐训练日志依赖 |
| 5 | 按 PyG 官方方式安装 torch-geometric 相关包 | 不同平台可能需要单独处理 |

仓库中的 `requirements.txt` 和 `environment.yml` 可以作为参考，但不同平台上 PyTorch Geometric 相关包常常需要按当前 CUDA / CPU 版本调整。

### 完整安装示例

如果你想从零开始搭建环境，可以按下面的顺序执行：

```powershell
conda create -n neural-subgraph-learning-GNN python=3.10
conda activate neural-subgraph-learning-GNN
pip install -r requirements.txt
pip install tensorboard
```

如果你的机器上已经有 PyTorch，但还没有 PyTorch Geometric 相关包，请按照 [PyG 官方安装说明](https://pytorch-geometric.readthedocs.io/) 选择与当前 torch 和 CUDA 版本匹配的 wheel。

### 常见安装检查

安装完成后，建议先验证下面这些包是否可导入：

```bash
python -c "import torch, torch_geometric, deepsnap, networkx, numpy; print('imports_ok')"
```

如果这一步通过，说明基础运行环境基本就绪。

### Windows 环境激活示例

```powershell
(D:\conda\shell\condabin\conda-hook.ps1) ; conda activate neural-subgraph-learning-GNN
```

## 快速开始

### 1. 训练子图匹配模型

使用合成数据训练：

```bash
python -m subgraph_matching.train --node_anchored
```

使用真实数据训练，以 Facebook 为例：

```bash
python -m subgraph_matching.train --dataset=facebook --node_anchored
```

### 2. 评估模型

```bash
python -m subgraph_matching.test --node_anchored
```

### 3. 运行频繁子图挖掘

以 ENZYMES 为例：

```bash
python -m subgraph_mining.decoder --dataset=enzymes --node_anchored
```

以 Facebook 为例：

```bash
python -m subgraph_mining.decoder --dataset=facebook --node_anchored
```

## 子图匹配任务

### 任务定义

给定 query 图 Q 和 target 图 T，如果 T 中存在一个子图与 Q 同构，并且锚定节点的映射关系满足要求，则预测为正样本；否则为负样本。

仓库支持 node anchored 设定，这在很多图匹配任务中很重要，因为它让模型不仅关注整体结构，还关注指定锚点附近的局部上下文。

### 训练入口

训练脚本位于 [subgraph_matching/train.py](subgraph_matching/train.py)。最常用的参数如下：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset` | `syn` | 训练数据集 |
| `--node_anchored` | 开启 | 是否使用节点锚定 |
| `--method_type` | `order` | 匹配方式 |
| `--conv_type` | `SAGE` | GNN 卷积类型 |
| `--hidden_dim` | `64` | 隐藏维度 |
| `--n_layers` | `8` | 图卷积层数 |
| `--batch_size` | `64` | 批大小 |
| `--n_batches` | `1000000` | 训练步数 |
| `--eval_interval` | `1000` | 每隔多少步评估一次 |
| `--model_path` | `ckpt/model.pt` | checkpoint 保存路径 |

### 常见数据集

仓库支持两类数据源：

#### 合成数据

| 数据集 | 说明 |
| --- | --- |
| `syn` | 默认合成数据 |
| `syn-balanced` | 平衡合成数据 |
| `syn-imbalanced` | 不平衡合成数据 |

合成数据适合先验证模型和代码是否正常工作，也适合做预训练。

#### 真实数据

| 数据集 | 说明 |
| --- | --- |
| `enzymes` | TUDataset 基准 |
| `proteins` | TUDataset 基准 |
| `cox2` | TUDataset 基准 |
| `aids` | TUDataset 基准 |
| `reddit-binary` | TUDataset 基准 |
| `imdb-binary` | TUDataset 基准 |
| `dblp` | TUDataset 基准 |
| `firstmm_db` | TUDataset 基准 |
| `facebook` | SNAP ego-Facebook 社交网络 |

其中 `facebook` 依赖本地 [data/facebook_combined.txt](data/facebook_combined.txt)。

### 输出

训练过程会：

- 在 TensorBoard 中记录 loss 和 accuracy。
- 周期性评估验证集。
- 将 checkpoint 保存到 `args.model_path`。

## 频繁子图挖掘

### 任务定义

SPMiner 会在目标图数据集上寻找常见子图模式。它不是简单计数，而是结合训练好的图匹配模型对候选 pattern 打分，再通过搜索策略筛选高质量模式。

### 挖掘入口

挖掘脚本位于 [subgraph_mining/decoder.py](subgraph_mining/decoder.py)。它会：

1. 读取训练好的 checkpoint。
2. 从目标数据集中采样邻域或整图。
3. 对候选子图做 batch embedding。
4. 使用 Greedy 或 MCTS 搜索模式。
5. 将结果保存到 `results/`，并把图像输出到 `plots/cluster/`。

### 挖掘参数对照表

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset` | `enzymes` | 目标数据集 |
| `--model_path` | `ckpt/model.pt` | 训练好的匹配模型 checkpoint |
| `--n_neighborhoods` | `10000` | 采样邻域数量 |
| `--batch_size` | `1000` | 挖掘批大小 |
| `--n_trials` | `1000` | 搜索试验次数 |
| `--out_path` | `results/out-patterns.p` | 输出结果文件 |
| `--search_strategy` | `greedy` | 搜索策略 |
| `--frontier_top_k` | `5` | 前沿扩展时保留的候选数量 |
| `--radius` | `3` | 邻域半径 |
| `--sample_method` | `tree` | 采样方式 |
| `--min_pattern_size` | `5` | 最小 pattern 尺寸 |
| `--max_pattern_size` | `20` | 最大 pattern 尺寸 |

### 输出

挖掘完成后会生成：

- `.p` 格式的 pattern 结果文件。
- `plots/cluster/` 下的 PNG 和 PDF 图像。

## 示例输出截图

仓库会把挖掘出的 pattern 自动导出到 `plots/cluster/`。下面是一个已经生成的示例结果，可直接点击查看：

| 类型 | 示例 |
| --- | --- |
| PNG 图像 | [plots/cluster/5-0.png](plots/cluster/5-0.png) |
| PDF 图像 | [plots/cluster/5-0.pdf](plots/cluster/5-0.pdf) |

如果你希望 README 中展示更完整的可视化效果，可以再补一张“pattern 总览图”或“TensorBoard 曲线截图”。目前仓库里已有的是单个 pattern 导出图，适合作为输出示例。

## 结果分析

仓库还提供了若干分析脚本和 Notebook，方便查看 embedding、统计 pattern 频次，或者对模式分布做可视化。

### 分析 embedding

```bash
python -m analyze.analyze_embeddings --node_anchored
```

### 统计 pattern 频次

以 ENZYMES 为例：

```bash
python -m analyze.count_patterns --dataset=enzymes --out_path=results/counts.json --node_anchored
```

以 Facebook 为例：

```bash
python -m analyze.count_patterns --dataset=facebook --out_path=results/counts.json --node_anchored
```

### 分析统计结果

```bash
python -m analyze.analyze_pattern_counts --counts_path=results/
```

## 数据文件说明

`data/` 目录下放的是本地可直接读取的数据文件，例如：

- `facebook_combined.txt`
- `as20000102.txt`
- `roadnet-er.txt`

如果你要跑 Facebook 或 AS-733 相关任务，通常需要保证对应边列表文件已经放在 `data/` 下。

## 已验证的本地运行方式

下面这套命令已经在当前 Windows 环境中验证通过，适合作为真实任务的参考：

### 训练示例

```bash
python -m subgraph_matching.train --dataset=facebook --node_anchored --n_batches 20 --eval_interval 10 --batch_size 32 --n_workers 1 --model_path results/facebook_train_big.pt
```

### 挖掘示例

```bash
python -u -m subgraph_mining.decoder --dataset=facebook --node_anchored --model_path results/facebook_train_big.pt --n_neighborhoods 50 --batch_size 50 --n_trials 5 --out_path results/facebook_patterns_big.p
```

对应输出文件为：

- [results/facebook_train_big.pt](results/facebook_train_big.pt)
- [results/facebook_patterns_big.p](results/facebook_patterns_big.p)

## 常见问题

### 1. 提示找不到 torch、deepsnap 或 torch_geometric

通常是没有进入正确的 conda 环境，或者环境里缺少对应依赖。请先确认 Python 版本和环境路径，再检查包是否安装完整。

### 2. 提示找不到 tensorboard

训练脚本会导入 `torch.utils.tensorboard.SummaryWriter`，如果环境中没有 tensorboard，就会在导入阶段失败。补装 tensorboard 后再运行即可。

### 3. Facebook 数据集报文件不存在

请确认 [data/facebook_combined.txt](data/facebook_combined.txt) 已经存在。这个文件来自 SNAP 的 ego-Facebook 数据集。

### 4. 挖掘太慢

先减小这些参数：

- `n_neighborhoods`
- `n_trials`
- `batch_size`

如果只是验证流程，先用小参数跑通，再逐步放大。

### 5. 输出很多图像文件

这是正常的。decoder 会把不同大小的 pattern 导出到 `plots/cluster/`，方便人工查看。

## 参考与来源

- NeuroMatch：子图匹配模块的实现参考。
- SPMiner：频繁子图挖掘模块的实现参考。
- Facebook 数据集：SNAP ego-Facebook。
- AS-733 数据集：SNAP Autonomous Systems 数据。

## 贡献说明

欢迎基于这个仓库继续扩展新的数据集、搜索策略或分析脚本。比较合适的贡献方式包括：

1. 新增或完善数据加载器。
2. 为训练和挖掘补充更多参数说明。
3. 增加新的搜索代理或更稳定的 pattern 评估策略。
4. 补充分析脚本、Notebook 或可视化输出。

如果你要提交改动，建议保持以下风格：

- 优先修改现有模块，而不是复制出新的重复实现。
- 新增命令行参数时同步更新 README 中的参数表。
- 如果输出格式发生变化，记得同步更新结果分析脚本。

## 引用格式

如果你在论文、报告或项目说明中使用了本仓库，可以按下面的简化格式引用：

```bibtex
@misc{neural-subgraph-learning,
   title={Neural Subgraph Learning Library},
   author={SPMiner Learn Contributors},
   year={2026},
   note={Neural subgraph matching and frequent subgraph mining toolbox}
}
```

如果你需要更正式的学术引用，建议结合原始 NeuroMatch 和 SPMiner 论文进行双重引用。

## 许可证

仓库当前未包含单独的 LICENSE 文件。若你计划对外发布、二次分发或做团队协作，建议先补充明确的许可证说明。

## 小结

这个项目的核心目标，是把“图匹配”作为一个可学习的子图关系判别器，再进一步用它去做频繁子图挖掘。如果你想从零开始理解它，建议按下面顺序看：

1. 先看 [common/models.py](common/models.py) 和 [common/data.py](common/data.py)，理解模型与数据流。
2. 再看 [subgraph_matching/train.py](subgraph_matching/train.py) 和 [subgraph_matching/test.py](subgraph_matching/test.py)，理解匹配任务。
3. 最后看 [subgraph_mining/decoder.py](subgraph_mining/decoder.py) 和 [subgraph_mining/search_agents.py](subgraph_mining/search_agents.py)，理解挖掘流程。

如果你愿意，我下一步可以继续把 README 再补成更完整的发布版样式，比如增加“安装命令块”“完整参数清单”“贡献说明”和“引用格式”。

