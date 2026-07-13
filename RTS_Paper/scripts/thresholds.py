"""Phase E: threshold-selection rules. Pure arithmetic over probabilities —
all rules are fit on training-fold probabilities only, then applied
unchanged to the held-out test fold. Zero retraining.

Matches the draft's threshold table (4 rows): Fixed 0.5, Max F1,
Max Youden J, Constrained FPR. Constrained-FPR target is not specified in
the draft text; documented assumption: target FPR <= 0.10.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_curve

CONSTRAINED_FPR_TARGET = 0.10  # documented assumption; draft does not specify a value


def threshold_fixed(y_train, prob_train) -> float:
    return 0.5


def threshold_max_f1(y_train, prob_train) -> float:
    """O(n log n) threshold sweep (not O(n^2) re-scoring): sort descending by
    score, incrementally track TP/FP as the threshold drops past each point.
    """
    y_train = np.asarray(y_train)
    order = np.argsort(-prob_train)
    y_sorted = y_train[order]
    p_sorted = prob_train[order]
    n_pos = int(y_train.sum())

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(n_pos > 0, tp / n_pos, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)

    # collapse to the last index at each distinct score (ties must share a threshold)
    last_at_score = np.r_[np.diff(p_sorted) != 0, True]
    f1_at_boundaries = np.where(last_at_score, f1, -1.0)
    best_idx = int(np.argmax(f1_at_boundaries))
    return float(p_sorted[best_idx])


def threshold_max_youden_j(y_train, prob_train) -> float:
    fpr, tpr, thr = roc_curve(y_train, prob_train)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def threshold_constrained_fpr(y_train, prob_train, target_fpr: float = CONSTRAINED_FPR_TARGET) -> float:
    fpr, tpr, thr = roc_curve(y_train, prob_train)
    valid = fpr <= target_fpr
    if not valid.any():
        return float(thr[np.argmin(fpr)])
    idx = np.where(valid)[0]
    best_idx = idx[np.argmax(tpr[idx])]
    return float(thr[best_idx])


THRESHOLD_RULES = {
    "fixed_0.5": threshold_fixed,
    "max_f1": threshold_max_f1,
    "max_youden_j": threshold_max_youden_j,
    "constrained_fpr_10pct": threshold_constrained_fpr,
}
