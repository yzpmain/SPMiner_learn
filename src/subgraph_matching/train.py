"""训练子图匹配模型（编码器）的入口模块。

该模块负责：
1. 构造数据源；
2. 构建图嵌入模型；
3. 在多进程训练循环中优化子图匹配损失；
4. 周期性调用验证函数并保存 checkpoint。
"""


import argparse
import os
import queue as queue_mod
import threading
import time

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
from deepsnap.batch import Batch
from torch.utils.tensorboard import SummaryWriter

from src.core import data
from src.core import dataset_registry
from src.core import models
from src.core import utils
from src.subgraph_matching.config import parse_encoder
from src.subgraph_matching.test import validation
from src.logger import RunLogger, info, section, progress

def build_model(args):
    """根据命令行参数构建编码器模型，并在测试模式下加载权重。"""
    if args.method_type == "order":
        model = models.OrderEmbedder(1, args.hidden_dim, args)
    elif args.method_type == "mlp":
        model = models.BaselineMLP(1, args.hidden_dim, args)
    model.to(utils.get_device())
    if args.test and args.model_path:
        model.load_state_dict(torch.load(args.model_path,
            map_location=utils.get_device()))
    return model

def make_data_source(args):
    """按数据集名称创建对应的数据源对象。

    这里将数据集分为两类：
    - syn：在线生成的合成数据；
    - disk：磁盘上已有图数据集。

    另外支持 balanced / imbalanced 两种采样方式。
    """
    raw_dataset = args.dataset.strip().lower()

    # 仅识别尾缀采样模式，避免 reddit-binary / as-733 这类合法名称被误拆分。
    mode = "balanced"
    base_dataset = raw_dataset
    if raw_dataset.endswith("-balanced"):
        base_dataset = raw_dataset[: -len("-balanced")]
        mode = "balanced"
    elif raw_dataset.endswith("-imbalanced"):
        base_dataset = raw_dataset[: -len("-imbalanced")]
        mode = "imbalanced"

    if base_dataset.startswith("syn"):
        if mode == "balanced":
            data_source = data.OTFSynDataSource(node_anchored=args.node_anchored)
        else:
            data_source = data.OTFSynImbalancedDataSource(node_anchored=args.node_anchored)
    else:
        normalized_dataset = dataset_registry.validate_dataset_name(base_dataset, "train-disk")
        if mode == "balanced":
            data_source = data.DiskDataSource(normalized_dataset,
                node_anchored=args.node_anchored)
        else:
            data_source = data.DiskImbalancedDataSource(normalized_dataset,
                node_anchored=args.node_anchored)
    return data_source

def _prefetch_worker(data_source, loader_iter, prefetch_queue, cancel_event, max_batches):
    """后台线程：在 GPU 计算的同时预取下一批数据。"""
    for _ in range(max_batches):
        if cancel_event.is_set():
            return
        try:
            batch = next(loader_iter)
        except StopIteration:
            return
        if cancel_event.is_set():
            return
        result = data_source.gen_batch(*batch, True)
        prefetch_queue.put(result)


def train(args, model, logger, in_queue, out_queue):
    """训练序嵌入模型。

    args: 命令行参数
    logger: 用于记录训练进度的日志器
    in_queue: 交叉点计算工作进程的输入队列
    out_queue: 交叉点计算工作进程的输出队列
    """
    # 主优化器负责图嵌入器的参数更新。
    scheduler, opt = utils.build_optimizer(args, model.parameters())
    if args.method_type == "order":
        # order 模型额外有一个二分类头，需要单独训练。
        clf_opt = optim.Adam(model.clf_model.parameters(), lr=args.lr)

    done = False
    while not done:
        data_source = make_data_source(args)
        loaders = data_source.gen_data_loaders(args.eval_interval *
            args.batch_size, args.batch_size, train=True)
        loader_iter = iter(zip(*loaders))

        # 启动预取线程：后台 CPU 采样，与 GPU 计算重叠
        prefetch_queue: queue_mod.Queue = queue_mod.Queue()
        cancel_event = threading.Event()
        prefetcher = threading.Thread(
            target=_prefetch_worker,
            args=(data_source, loader_iter, prefetch_queue,
                  cancel_event, args.eval_interval),
            daemon=True,
        )
        prefetcher.start()

        for _ in range(args.eval_interval):
            msg, _ = in_queue.get()
            if msg == "done":
                cancel_event.set()
                done = True
                break

            # 从预取队列获取（通常已有数据，无需等待）
            pos_a, pos_b, neg_a, neg_b = prefetch_queue.get()

            # 训练单个 batch：正样本对子图关系应成立，负样本对不成立。
            model.train()
            model.zero_grad()

            # 分别计算正负样本的嵌入
            emb_as = model.emb_model(pos_a)
            emb_bs = model.emb_model(pos_b)
            neg_as = model.emb_model(neg_a)
            neg_bs = model.emb_model(neg_b)
            emb_as = torch.cat([emb_as, neg_as])
            emb_bs = torch.cat([emb_bs, neg_bs])
            n_pos = pos_a.num_graphs

            labels = torch.tensor([1]*n_pos + [0]*neg_a.num_graphs).to(
                utils.get_device())
            intersect_embs = None
            pred = model(emb_as, emb_bs)
            loss = model.criterion(pred, intersect_embs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if scheduler:
                scheduler.step()

            if args.method_type == "order":
                # order 模型用额外的分类器把"违反量"映射为二分类概率。
                with torch.no_grad():
                    pred = model.predict(pred)
                model.clf_model.zero_grad()
                pred = model.clf_model(pred.unsqueeze(1))
                criterion = nn.NLLLoss()
                clf_loss = criterion(pred, labels)
                clf_loss.backward()
                clf_opt.step()
            pred = pred.argmax(dim=-1)
            acc = torch.mean((pred == labels).type(torch.float))

            out_queue.put(("step", (loss.item(), acc)))

        cancel_event.set()

def train_loop(args):
    """训练主循环：启动 worker、准备验证集并周期性评估。"""
    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    if not os.path.exists("plots/"):
        os.makedirs("plots/")

    info("Starting {} workers".format(args.n_workers))
    in_queue, out_queue = mp.Queue(), mp.Queue()

    section("数据加载")
    info("Using dataset {}".format(args.dataset))

    record_keys = ["conv_type", "n_layers", "hidden_dim",
        "margin", "dataset", "max_graph_size", "skip"]
    args_str = ".".join(["{}={}".format(k, v)
        for k, v in sorted(vars(args).items()) if k in record_keys])
    logger = SummaryWriter(comment=args_str)

    model = build_model(args)
    model.share_memory()

    if args.method_type == "order":
        clf_opt = optim.Adam(model.clf_model.parameters(), lr=args.lr)
    else:
        clf_opt = None

    # 先准备一批固定测试点，训练过程中每个 eval_interval 做一次评估。
    data_source = make_data_source(args)
    loaders = data_source.gen_data_loaders(args.val_size, args.batch_size,
        train=False, use_distributed_sampling=False)
    test_pts = []
    for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
        pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
            batch_neg_target, batch_neg_query, False)
        if pos_a:
            pos_a = pos_a.to(torch.device("cpu"))
            pos_b = pos_b.to(torch.device("cpu"))
        neg_a = neg_a.to(torch.device("cpu"))
        neg_b = neg_b.to(torch.device("cpu"))
        test_pts.append((pos_a, pos_b, neg_a, neg_b))

    workers = []
    # 多进程 worker 并行生产训练 step。
    for i in range(args.n_workers):
        worker = mp.Process(target=train, args=(args, model, data_source,
            in_queue, out_queue))
        worker.start()
        workers.append(worker)

    if args.test:
        validation(args, model, test_pts, logger, 0, 0, verbose=True)
    else:
        batch_n = 0
        section("训练")
        for epoch in range(args.n_batches // args.eval_interval):
            for i in range(args.eval_interval):
                in_queue.put(("step", None))
            for i in range(args.eval_interval):
                msg, params = out_queue.get()
                train_loss, train_acc = params
                progress(batch_n, args.n_batches, Loss=train_loss, Acc=train_acc)
                logger.add_scalar("Loss/train", train_loss, batch_n)
                logger.add_scalar("Accuracy/train", train_acc, batch_n)
                batch_n += 1
            section("验证")
            validation(args, model, test_pts, logger, batch_n, epoch)

    for i in range(args.n_workers):
        in_queue.put(("done", None))
    for worker in workers:
        worker.join()

def main(force_test=False):
    """命令行入口。

    该入口兼容两种模式：
    - 正常训练：执行 train_loop；
    - 测试模式：只跑验证逻辑，不更新参数。
    """
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description='序嵌入参数')

    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    if force_test:
        args.test = True

    with RunLogger(args):
        train_loop(args)

if __name__ == '__main__':
    main()
