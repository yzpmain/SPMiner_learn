import argparse
from common import utils

def parse_encoder(parser, arg_str=None):
    """注册子图匹配（编码器）阶段参数。

    该函数把训练/测试子图匹配模型所需参数添加到解析器，
    并提供一组与原论文实现一致的默认值。SPMiner 也会复用这部分参数
    （例如 method_type、model_path、hidden_dim）。

    参数：
        parser: argparse.ArgumentParser 实例。
        arg_str: 预留参数，当前实现未使用。
    """
    enc_parser = parser.add_argument_group()
    #utils.parse_optimizer(parser)

    enc_parser.add_argument('--conv_type', type=str,
                        help='卷积类型')
    enc_parser.add_argument('--method_type', type=str,
                        help='嵌入类型')
    enc_parser.add_argument('--batch_size', type=int,
                        help='训练批大小')
    enc_parser.add_argument('--n_layers', type=int,
                        help='图卷积层数')
    enc_parser.add_argument('--hidden_dim', type=int,
                        help='训练隐层维度')
    enc_parser.add_argument('--skip', type=str,
                        help='"all" 或 "last"')
    enc_parser.add_argument('--dropout', type=float,
                        help='Dropout 比率')
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
    enc_parser.add_argument('--model_path', type=str,
                        help='模型保存/加载路径')
    enc_parser.add_argument('--opt_scheduler', type=str,
                        help='调度器名称')
    enc_parser.add_argument('--node_anchored', action="store_true",
                        help='训练时是否使用节点锚定')
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
                        opt='adam',   # opt_enc_parser
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

    # 注意：这里不直接 parse，统一由外部主程序解析。
    #return enc_parser.parse_args(arg_str)

