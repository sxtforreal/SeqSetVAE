from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, f1_score


def _best_f1(y_true: np.ndarray, prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    # thresholds has length-1 of precision/recall; align by skipping first precision=1.0
    f1s = 2 * precision * recall / (precision + recall + 1e-12)
    return float(np.nanmax(f1s))


def recall_at_precision(y_true: np.ndarray, prob: np.ndarray, target_p: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    mask = precision >= target_p
    if not np.any(mask):
        return 0.0
    return float(np.max(recall[mask]))


def precision_at_topk(y_true: np.ndarray, prob: np.ndarray, k: int) -> float:
    k = int(min(k, len(prob)))
    idx = np.argpartition(-prob, k-1)[:k]
    return float(np.mean(y_true[idx]))


def ece_score(y_true: np.ndarray, prob: np.ndarray, num_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    inds = np.digitize(prob, bins) - 1
    ece = 0.0
    n = len(prob)
    for b in range(num_bins):
        mask = inds == b
        if not np.any(mask):
            continue
        conf = float(np.mean(prob[mask]))
        acc = float(np.mean(y_true[mask])) if np.any(mask) else 0.0
        ece += abs(acc - conf) * (np.sum(mask) / n)
    return float(ece)


def evaluate_all_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    recall_at_ps: Iterable[float] = (0.6, 0.7),
    topk_list: Iterable[int] = (10, 50),
    with_ece: bool = True,
) -> Dict[str, float]:
    prob = 1.0 / (1.0 + np.exp(-logits))
    metrics: Dict[str, float] = {}
    try:
        metrics["auprc"] = float(average_precision_score(y_true, prob))
    except Exception:
        metrics["auprc"] = float("nan")
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, prob))
    except Exception:
        metrics["auroc"] = float("nan")
    metrics["best_f1"] = _best_f1(y_true, prob)
    for p in recall_at_ps:
        metrics[f"recall_at_p>={p}"] = recall_at_precision(y_true, prob, p)
    for k in topk_list:
        metrics[f"precision_at_top{k}"] = precision_at_topk(y_true, prob, int(k))
    if with_ece:
        metrics["ece"] = ece_score(y_true, prob)
    return metrics

