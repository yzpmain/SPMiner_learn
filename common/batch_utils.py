"""批处理工具：将 networkx 图列表转换为 DeepSNAP Batch。"""
import torch
from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch

from common.train_utils import get_device


def batch_nx_graphs(graphs, anchors=None):
    """将 networkx 图列表打包为带特征增强的 DeepSNAP Batch。

    参数：
        graphs: networkx.Graph 列表。
        anchors: 可选的锚点节点列表，与 graphs 一一对应。
    返回：
        已迁移到当前设备的 DeepSNAP Batch。
    """
    # 懒导入以避免 batch_utils → feature_preprocess → utils → batch_utils 循环
    from common import feature_preprocess  # noqa: PLC0415

    augmenter = feature_preprocess.FeatureAugment()

    if anchors is not None:
        for anchor, g in zip(anchors, graphs):
            for v in g.nodes:
                g.nodes[v]["node_feature"] = torch.tensor([float(v == anchor)])

    batch = Batch.from_data_list([DSGraph(g) for g in graphs])
    batch = augmenter.augment(batch)
    batch = batch.to(get_device())
    return batch
