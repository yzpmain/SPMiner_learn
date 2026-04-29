"""Optimizer and learning rate scheduler factory.

Previously in utils.py.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

__all__ = [
    "parse_optimizer",
    "build_optimizer",
]


def parse_optimizer(parser: argparse.ArgumentParser) -> None:
    """Register optimizer CLI arguments on a parser."""
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument("--opt", dest="opt", type=str, help="优化器类型")
    opt_parser.add_argument("--opt-scheduler", dest="opt_scheduler", type=str, help="优化器调度器类型")
    opt_parser.add_argument("--opt-restart", dest="opt_restart", type=int, help="重启前的训练轮数")
    opt_parser.add_argument("--opt-decay-step", dest="opt_decay_step", type=int, help="衰减前的训练轮数")
    opt_parser.add_argument("--opt-decay-rate", dest="opt_decay_rate", type=float, help="学习率衰减比率")
    opt_parser.add_argument("--lr", dest="lr", type=float, help="学习率")
    opt_parser.add_argument("--clip", dest="clip", type=float, help="梯度裁剪")
    opt_parser.add_argument("--weight_decay", type=float, help="优化器权重衰减")


def build_optimizer(args: argparse.Namespace, params: nn.parameter.Parameter):
    """Create optimizer and scheduler from CLI args."""
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == "adam":
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == "sgd":
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == "rmsprop":
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == "adagrad":
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)

    if args.opt_scheduler == "none":
        return None, optimizer
    if args.opt_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    else:
        scheduler = None
    return scheduler, optimizer
