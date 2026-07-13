"""Phase F (P1): Wilcoxon signed-rank over fold-level pairs, Holm correction.

The 15 held-out runs are the paired experimental units (not pooled rows).
Compares every pair of the 4 models on F1 and ROC-AUC from the clean
central-128 LORO fold results.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "RTS_Paper" / "outputs" / "metrics"


def holm_correction(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted


def main() -> None:
    fold_df = pd.read_csv(METRICS_DIR / "phase_b_loro_clean_fold_metrics.csv")
    models = sorted(fold_df.model.unique())

    rows = []
    for metric in ["f1", "roc_auc"]:
        pvals = []
        pairs = []
        for m1, m2 in combinations(models, 2):
            s1 = fold_df[fold_df.model == m1].sort_values("fold_id")[metric].to_numpy()
            s2 = fold_df[fold_df.model == m2].sort_values("fold_id")[metric].to_numpy()
            diff = s1 - s2
            try:
                stat, p = wilcoxon(diff)
            except ValueError:
                stat, p = float("nan"), 1.0
            median_diff = float(pd.Series(diff).median())
            n_pos = int((diff > 0).sum())
            n_neg = int((diff < 0).sum())
            effect_size = (n_pos - n_neg) / len(diff)  # matched-pairs rank-biserial-style sign effect
            pairs.append((m1, m2, metric, stat, p, median_diff, effect_size))
            pvals.append(p)
        adjusted = holm_correction(pvals)
        for (m1, m2, metric_, stat, p, median_diff, effect_size), p_adj in zip(pairs, adjusted):
            rows.append({
                "metric": metric_, "model_a": m1, "model_b": m2,
                "wilcoxon_stat": stat, "p_value": p, "p_value_holm": p_adj,
                "median_diff_a_minus_b": median_diff, "effect_size_sign": effect_size,
                "n_folds": 15, "significant_holm_0.05": bool(p_adj < 0.05),
            })

    out = pd.DataFrame(rows)
    out.to_csv(METRICS_DIR / "wilcoxon_model_comparisons.csv", index=False)
    print(out.to_string(index=False))
    print(f"wrote {METRICS_DIR}/wilcoxon_model_comparisons.csv")


if __name__ == "__main__":
    main()
