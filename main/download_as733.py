"""AS733 数据集下载与加载。

从 SNAP (https://snap.stanford.edu/data/as-733.html) 下载 AS 自治系统
733 个每日通信快照，解析为 nx.Graph 列表并缓存。
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import networkx as nx

from main.config import DSET_DIR

__all__ = ["ensure_as733"]

AS733_URL = "https://snap.stanford.edu/data/as-733.tar.gz"
CACHE_DIR = DSET_DIR / "as733"
TAR_PATH = DSET_DIR / "as-733.tar.gz"


def ensure_as733(cache_dir: Path = CACHE_DIR) -> list[nx.Graph]:
    """下载/加载 AS733 数据集。

    策略:
        1. 若 cache_dir 下有已解析的缓存 → 直接加载
        2. 否则若 tar.gz 存在 → 解压并解析
        3. 否则从 SNAP 下载 → 解压 → 解析

    返回:
        list[nx.Graph]: 733 个快照图，按文件名排序
    """
    # 检查是否有预解析缓存
    cache_pkl = cache_dir.with_suffix(".p")
    if cache_pkl.exists():
        import pickle
        with open(cache_pkl, "rb") as f:
            return pickle.load(f)

    # 下载
    if not TAR_PATH.exists():
        TAR_PATH.parent.mkdir(parents=True, exist_ok=True)
        print("下载 AS733 数据集 from {} ...".format(AS733_URL))
        urllib.request.urlretrieve(AS733_URL, TAR_PATH)
        print("下载完成: {}".format(TAR_PATH))

    # 解压
    extract_dir = cache_dir
    if not extract_dir.exists():
        print("解压 {} ...".format(TAR_PATH))
        with tarfile.open(TAR_PATH, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print("解压到 {}".format(extract_dir))

    # 解析所有 .txt 边列表文件
    graphs = []
    txt_files = sorted(extract_dir.glob("*.txt"))
    if not txt_files:
        # 可能解压后有一个子目录
        subdirs = list(extract_dir.iterdir())
        if subdirs:
            txt_files = sorted(subdirs[0].glob("*.txt"))
    if not txt_files:
        txt_files = sorted(extract_dir.rglob("*.txt"))

    print("解析 {} 个快照文件 ...".format(len(txt_files)))
    for fpath in txt_files:
        g = nx.Graph()
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("%"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        g.add_edge(u, v)
                    except ValueError:
                        continue
        if g.number_of_nodes() > 0:
            graphs.append(g)

    # 缓存解析结果
    with open(cache_pkl, "wb") as f:
        import pickle
        pickle.dump(graphs, f)
    print("AS733 加载完成: {} 个快照".format(len(graphs)))
    return graphs


if __name__ == "__main__":
    gs = ensure_as733()
    print("共 {} 个快照".format(len(gs)))
    sizes = sorted([len(g) for g in gs])
    print("节点数范围: {} ~ {} (中位数 {})".format(sizes[0], sizes[-1], sizes[len(sizes)//2]))
    edges = sorted([g.number_of_edges() for g in gs])
    print("边数范围: {} ~ {} (中位数 {})".format(edges[0], edges[-1], edges[len(edges)//2]))
