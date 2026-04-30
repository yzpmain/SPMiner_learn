# SPMiner 实验命令参数说明

## 主入口

```bash
python -m main.run_all --model-path <path> [options]
```

## 参数一览

### 必需参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--model-path` | str | 预训练编码器 .pt 文件路径。模型须用 `--node_anchored` 训练，架构为 SAGE+order。 |

### 数据集选择

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--datasets` | list | `sbm facebook ppi as733` | 要运行的数据集子集，用空格分隔。可选: `sbm`, `facebook`, `ppi`, `as733`, `all` |

可用数据集:

| 名称 | 类型 | 规模 |
|------|------|------|
| `sbm` | 人工 SBM 社区图 | 50 个图（自动生成） |
| `facebook` | Stanford 社交网络 | 1 个大图（4039 节点） |
| `ppi` | PyG 蛋白质网络 | 24 个小图 |
| `as733` | AS 互联网拓扑 | 733 个快照（自动下载） |

### 挖掘参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--min-size` | int | 4 | 最小模式尺寸（节点数） |
| `--max-size` | int | 10 | 最大模式尺寸（节点数） |
| `--n-trials` | int | 500 | 贪心搜索试验次数。每次从一个随机种子节点开始搜索。值越大覆盖面越广，但线性增长耗时。 |
| `--n-neighborhoods` | int | 2000 | 邻域采样数量。对每个邻域编码嵌入，用于搜索阶段的模式匹配打分。多图数据集中每个图至少采样一次。 |

### 计数参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--count-method` | str | `bin` | 模式频次计算方法: `bin` 存在性检测(快), `freq` 枚举所有同构嵌入(极慢), `sample` 对目标图随机采样后计数(折衷) |
| `--count-sample-size` | int | 100 | `sample` 模式下采样的目标图数量 |

### 运行控制

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--dry-run` | flag | False | 快速测试模式: n_neighborhoods=50, n_trials=5, batch_size=50。约 3 分钟跑完单数据集 |
| `--skip-baseline` | flag | False | 跳过 ER 随机图基线对比，节省一半时间 |

---

## 默认挖掘配置

以下参数通过 `main/config.py` 中的 `MINE_CONFIG` 设置，当前不可通过 CLI 覆盖:

| 参数 | 默认 | 说明 |
|------|------|------|
| `search_strategy` | `greedy` | 搜索策略: greedy(贪心) / mcts(蒙特卡洛树搜索) |
| `global_top_k` | 30 | 全局输出模式数量（所有尺寸统排取前 K） |
| `node_anchored` | True | 启用节点锚定搜索 |
| `sample_method` | `tree` | 邻域采样方法: tree(树形) / radial(辐射形) |
| `frontier_top_k` | 5 | 每步 frontier 候选数上限 |
| `batch_size` | 1000 | 嵌入编码批大小 |
| `n_workers` | 4 | 计数并行进程数 |

---

## 关键命令示例

### 快速验证（单个数据集 ~3 分钟）

```bash
# SBM 合成网络 dry-run
python -m main.run_all --model-path ckpt/model_converted.pt --datasets sbm --dry-run

# 跳过基线对比
python -m main.run_all --model-path ckpt/model_converted.pt --datasets sbm --dry-run --skip-baseline
```

### 单个数据集正式运行（~10-30 分钟）

```bash
# Facebook 社交网络，模式尺寸 4-8
python -m main.run_all --model-path ckpt/model_converted.pt \
    --datasets facebook --min-size 4 --max-size 8 --skip-baseline

# PPI 蛋白质网络，采样计数 200 图
python -m main.run_all --model-path ckpt/model_converted.pt \
    --datasets ppi --count-method sample --count-sample-size 200 -skip-baseline
```

### 完整实验（4 个数据集，含基线 ~2-6 小时）

```bash
# 全部数据集，含 ER 基线
python -m main.run_all --model-path ckpt/model_converted.pt

# 自定义搜索强度
python -m main.run_all --model-path ckpt/model_converted.pt \
    --n-trials 1000 --n-neighborhoods 5000 --min-size 3 --max-size 8
```

### 单数据集分步测试

```bash
# step 1: Facebook + 基线
python -m main.run_all --model-path ckpt/model_converted.pt --datasets facebook

# step 2: PPI 
python -m main.run_all --model-path ckpt/model_converted.pt --datasets ppi --skip-baseline

# step 3: AS733 (快照多，建议 sample 计数)
python -m main.run_all --model-path ckpt/model_converted.pt \
    --datasets as733 --count-method sample --count-sample-size 300 --skip-baseline
```

---

## 生成分析报告

```bash
# 为所有有结果的数据集生成 MD 报告 + 图表
python -m main.analyze

# 查看报告
cat expdata/outputs/sbm/report.md
```

---

## 输出结构

```
expdata/
  datasets/           # 原始数据缓存
    sbm/graphs.p      # 生成的 SBM 图
    as733/*.txt       # 下载的 AS733 快照
  outputs/
    <dataset>/
      patterns.p           # 挖掘出的模式 (pickle[nx.Graph])
      counts.json          # 频次数据
      pattern_meta.json    # 全局排名元数据 (供论文用)
      summary.json         # 统计摘要
      report.md            # 分析报告
      report_plots/        # 报告图表
      baseline/            # ER 基线结果 (如启用)
        patterns.p
        summary.json
        counts.json
      plots/               # 模式可视化 PNG
  summary.csv              # 跨数据集汇总
  plots/                   # 汇总图表
```
