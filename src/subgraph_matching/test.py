from src.core import utils
from src.logger import info, warning
from collections import defaultdict
from datetime import datetime
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch

__all__ = [
    "validation",
    "compute_metrics",
    "plot_pr_curve",
    "log_checkpoint_metrics",
]

def compute_metrics(pred, labels, raw_pred):
    """计算分类指标，返回 (metrics_dict, labels_np, raw_pred_np, pred_np)。"""
    acc = torch.mean((pred == labels).type(torch.float))
    prec = (torch.sum(pred * labels).item() / torch.sum(pred).item() if
        torch.sum(pred) > 0 else float("NaN"))
    recall = (torch.sum(pred * labels).item() /
        torch.sum(labels).item() if torch.sum(labels) > 0 else
        float("NaN"))
    labels_np = labels.detach().cpu().numpy()
    raw_pred_np = raw_pred.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    auroc = roc_auc_score(labels_np, raw_pred_np)
    avg_prec = average_precision_score(labels_np, raw_pred_np)
    tn, fp, fn, tp = confusion_matrix(labels_np, pred_np).ravel()
    metrics = {
        "acc": acc.item(), "prec": prec, "recall": recall,
        "auroc": auroc, "avg_prec": avg_prec,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return metrics, labels_np, raw_pred_np, pred_np


def plot_pr_curve(labels_np, raw_pred_np, path):
    """绘制 PR 曲线并保存。"""
    import matplotlib.pyplot as plt
    precs, recalls, _ = precision_recall_curve(labels_np, raw_pred_np)
    plt.plot(recalls, precs)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(path)
    plt.close()
    info("PR curve saved → {}".format(path))


def log_checkpoint_metrics(logger, metrics, batch_n, args, model, epoch):
    """记指标到 TensorBoard 并保存模型权重。"""
    logger.add_scalar("Accuracy/test", metrics["acc"], batch_n)
    logger.add_scalar("Precision/test", metrics["prec"], batch_n)
    logger.add_scalar("Recall/test", metrics["recall"], batch_n)
    logger.add_scalar("AUROC/test", metrics["auroc"], batch_n)
    logger.add_scalar("AvgPrec/test", metrics["avg_prec"], batch_n)
    logger.add_scalar("TP/test", metrics["tp"], batch_n)
    logger.add_scalar("TN/test", metrics["tn"], batch_n)
    logger.add_scalar("FP/test", metrics["fp"], batch_n)
    logger.add_scalar("FN/test", metrics["fn"], batch_n)
    info("Checkpoint saved → {}".format(args.model_path))
    torch.save(model.state_dict(), args.model_path)


def validation(args, model, test_pts, logger, batch_n, epoch, verbose=False, use_orca_feats=False):
    """在固定测试点上评估子图匹配模型。

    评估流程：
    1. 对测试集中每个正/负样本对做前向推理；
    2. 计算分类结果和原始分数；
    3. 汇总 Accuracy / Precision / Recall / AUROC / AP 等指标；
    4. 在需要时保存 PR 曲线与模型权重。
    """
    model.eval()
    all_raw_preds, all_preds, all_labels = [], [], []
    for pos_a, pos_b, neg_a, neg_b in test_pts:
        if pos_a:
            pos_a = pos_a.to(utils.get_device())
            pos_b = pos_b.to(utils.get_device())
        neg_a = neg_a.to(utils.get_device())
        neg_b = neg_b.to(utils.get_device())
        labels = torch.tensor([1]*(pos_a.num_graphs if pos_a else 0) +
            [0]*neg_a.num_graphs).to(utils.get_device())
        with torch.no_grad():
            emb_neg_a, emb_neg_b = (model.emb_model(neg_a),
                model.emb_model(neg_b))
            if pos_a:
                emb_pos_a, emb_pos_b = (model.emb_model(pos_a),
                    model.emb_model(pos_b))
                emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
                emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
            else:
                emb_as, emb_bs = emb_neg_a, emb_neg_b
            pred = model(emb_as, emb_bs)
            raw_pred = model.predict(pred)
            if use_orca_feats:
                import orca
                import matplotlib.pyplot as plt
                def make_feats(g):
                    counts5 = np.array(orca.orbit_counts("node", 5, g))
                    for v, n in zip(counts5, g.nodes):
                        if g.nodes[n]["node_feature"][0] > 0:
                            anchor_v = v
                            break
                    v5 = np.sum(counts5, axis=0)
                    return v5, anchor_v
                MAX_MARGIN_SCORE = 1e9  # orca 约束的极大 margin 分数
                for i, (ga, gb) in enumerate(zip(neg_a.G, neg_b.G)):
                    (va, na), (vb, nb) = make_feats(ga), make_feats(gb)
                    if (va < vb).any() or (na < nb).any():
                        raw_pred[pos_a.num_graphs + i] = MAX_MARGIN_SCORE

            if args.method_type == "order":
                # order 模型把违反量再送入分类器得到最终判定。
                pred = model.clf_model(raw_pred.unsqueeze(1)).argmax(dim=-1)
                raw_pred *= -1
            elif args.method_type == "ensemble":
                pred = torch.stack([m.clf_model(
                    raw_pred.unsqueeze(1)).argmax(dim=-1) for m in model.models])
                for i in range(pred.shape[1]):
                    pass  # ensemble debug: print(pred[:,i])
                pred = torch.min(pred, dim=0)[0]
                raw_pred *= -1
            elif args.method_type == "mlp":
                raw_pred = raw_pred[:,1]
                pred = pred.argmax(dim=-1)
        all_raw_preds.append(raw_pred)
        all_preds.append(pred)
        all_labels.append(labels)
    pred = torch.cat(all_preds, dim=-1)
    labels = torch.cat(all_labels, dim=-1)
    raw_pred = torch.cat(all_raw_preds, dim=-1)

    metrics, labels_np, raw_pred_np, pred_np = compute_metrics(pred, labels, raw_pred)

    if verbose:
        pr_curve_path = getattr(args, "pr_curve_path", "plots/precision-recall-curve.png")
        plot_pr_curve(labels_np, raw_pred_np, pr_curve_path)

    info("Epoch {} | Acc: {:.4f} | P: {:.4f} | R: {:.4f} | AUROC: {:.4f} | "
        "AP: {:.4f} | TN: {} | FP: {} | FN: {} | TP: {}".format(
            epoch, metrics["acc"], metrics["prec"], metrics["recall"],
            metrics["auroc"], metrics["avg_prec"],
            metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]))

    if not args.test:
        log_checkpoint_metrics(logger, metrics, batch_n, args, model, epoch)

    if verbose:
        conf_mat_examples = defaultdict(list)
        idx = 0
        for pos_a, pos_b, neg_a, neg_b in test_pts:
            if pos_a:
                pos_a = pos_a.to(utils.get_device())
                pos_b = pos_b.to(utils.get_device())
            neg_a = neg_a.to(utils.get_device())
            neg_b = neg_b.to(utils.get_device())
            for list_a, list_b in [(pos_a, pos_b), (neg_a, neg_b)]:
                if not list_a: continue
                for a, b in zip(list_a.G, list_b.G):
                    correct = pred_np[idx] == labels_np[idx]
                    conf_mat_examples[correct, pred_np[idx]].append((a, b))
                    idx += 1

if __name__ == "__main__":
    # 复用 train.py 的 main(force_test=True) 只做评估。
    from src.subgraph_matching.train import main
    main(force_test=True)
