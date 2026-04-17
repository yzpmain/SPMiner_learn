# ---------------------------------------------------------------------------
# 向后兼容重导出层
# 原来分散在本文件中的函数已移入三个子模块：
#   common/graph_utils.py  —— 图采样、WL 哈希、ESU 枚举、SNAP 加载
#   common/train_utils.py  —— 设备管理、优化器构建与参数注册
#   common/batch_utils.py  —— networkx 图 → DeepSNAP Batch
# 本文件仅做 re-export，确保现有 `from common import utils` 调用不变。
# ---------------------------------------------------------------------------

from common.graph_utils import (  # noqa: F401
    sample_neigh,
    vec_hash,
    wl_hash,
    enumerate_subgraph,
    extend_subgraph,
    gen_baseline_queries_rand_esu,
    gen_baseline_queries_mfinder,
    load_snap_edgelist,
)

from common.train_utils import (  # noqa: F401
    get_device,
    parse_optimizer,
    build_optimizer,
)

from common.batch_utils import batch_nx_graphs  # noqa: F401
