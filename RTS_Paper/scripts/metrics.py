from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    out = {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "fpr": fpr,
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "n_test": int(len(y_true)),
        "n_attack": int(y_true.sum()),
        "n_normal": int((y_true == 0).sum()),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, prob)
    except ValueError:
        out["roc_auc"] = np.nan
    try:
        out["pr_auc"] = average_precision_score(y_true, prob)
    except ValueError:
        out["pr_auc"] = np.nan
    return out
