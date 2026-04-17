"""训练相关工具函数：设备管理、优化器构建与参数注册。"""
import torch
import torch.optim as optim


device_cache = None


def get_device():
    """懒加载运行设备（优先 CUDA）。"""
    global device_cache
    if device_cache is None:
        device_cache = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
    return device_cache


def parse_optimizer(parser):
    """向解析器注册优化器相关参数。"""
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str,
            help='优化器类型')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
            help='优化器调度器类型，默认为无')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
            help='重启前的训练轮数（默认为 0，即不重启）')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
            help='衰减前的训练轮数')
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
            help='学习率衰减比率')
    opt_parser.add_argument('--lr', dest='lr', type=float,
            help='学习率')
    opt_parser.add_argument('--clip', dest='clip', type=float,
            help='梯度裁剪')
    opt_parser.add_argument('--weight_decay', type=float,
            help='优化器权重衰减')


def build_optimizer(args, params):
    """按配置创建优化器与学习率调度器。"""
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95,
            weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr,
            weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr,
            weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
            step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
            T_max=args.opt_restart)
    return scheduler, optimizer
