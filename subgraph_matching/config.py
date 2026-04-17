"""子图匹配（编码器训练）阶段参数注册。

编码器结构参数（两阶段共用）通过 common.config_base.add_encoder_args 注入，
本文件只保留训练阶段独有的参数：批大小、训练轮数、margin、数据集、
评估频率、验证集大小、优化器调度器、worker 数量等。
"""
import argparse

from common.config_base import add_encoder_args
from common import utils


def parse_encoder(parser, arg_str=None):
    """注册子图匹配（编码器）阶段参数。

    该函数先注入编码器结构参数（共用），再添加训练阶段独有参数，
    并提供一组与原论文实现一致的默认值。

    参数：
        parser: argparse.ArgumentParser 实例。
        arg_str: 预留参数，当前实现未使用。
    """
    # 注入两阶段共用的编码器结构参数
    add_encoder_args(parser)

    # 训练阶段独有参数
    enc_parser = parser.add_argument_group("训练阶段独有参数")
    enc_parser.add_argument('--batch_size', type=int,
                        help='训练批大小')
    enc_parser.add_argument('--n_batches', type=int,
                        help='训练小批次数量')
    enc_parser.add_argument('--margin', type=float,
                        help='损失函数的 margin')
    enc_parser.add_argument('--dataset', type=str,
                        help='数据集')
    enc_parser.add_argument('--test_set', type=str,
                        help='测试集文件名')
    enc_parser.add_argument('--eval_interval', type=int,
                        help='训练中评估频率')
    enc_parser.add_argument('--val_size', type=int,
                        help='验证集大小')
    enc_parser.add_argument('--opt_scheduler', type=str,
                        help='调度器名称')
    enc_parser.add_argument('--test', action="store_true")
    enc_parser.add_argument('--n_workers', type=int)
    enc_parser.add_argument('--tag', type=str,
        help='用于标识本次运行的标签')

    # 默认配置偏向稳定训练：SAGE 卷积 + order embedding。
    enc_parser.set_defaults(conv_type='SAGE',
                        method_type='order',
                        dataset='syn',
                        n_layers=8,
                        batch_size=64,
                        hidden_dim=64,
                        skip="learnable",
                        dropout=0.0,
                        n_batches=1000000,
                        opt='adam',
                        opt_scheduler='none',
                        opt_restart=100,
                        weight_decay=0.0,
                        lr=1e-4,
                        margin=0.1,
                        test_set='',
                        eval_interval=1000,
                        n_workers=4,
                        model_path="ckpt/model.pt",
                        tag='',
                        val_size=4096,
                        node_anchored=True)
