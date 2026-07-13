"""Phase F (P1): ranked feature-budget experiment (64/32/16).

For each fold, ranks features by RandomForest importance fit on that fold's
training data ONLY (rank_features_train_only in data.py — never touches
test data), takes the top-k, retrains each of the 4 models on that subset,
and evaluates on the held-out run. Structural budgets (128/PMU-only/
cyber-log-only/edge_r1) are already covered by run_loro.py's
--feature-budget flag; this script covers only the ranked top-k budgets.

Persists per-fold models like run_loro.py (same hard requirement).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from RTS_Paper.scripts.data import load_dataset, rank_features_train_only
from RTS_Paper.scripts.metrics import compute_metrics
from RTS_Paper.scripts.models import FITTERS
from RTS_Paper.scripts.run_loro import slug

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "RTS_Paper" / "outputs" / "models"
METRICS_DIR = ROOT / "RTS_Paper" / "outputs" / "metrics"
PRED_DIR = ROOT / "RTS_Paper" / "outputs" / "predictions"

BUDGETS = [64, 32, 16]
SEED = 42


def main() -> None:
    for d in [MODELS_DIR, METRICS_DIR, PRED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    data, all_features, meta = load_dataset()
    X_full = data[all_features]
    y_full = data["target"].to_numpy()
    groups = data["run_id"].to_numpy()
    splits = list(GroupKFold(n_splits=15).split(X_full, y_full, groups))

    rows = []
    pred_rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        held_out_run = int(np.unique(groups[test_idx])[0])
        X_train, y_train = X_full.iloc[train_idx], y_full[train_idx]
        X_test, y_test = X_full.iloc[test_idx], y_full[test_idx]

        ranked = rank_features_train_only(X_train, y_train, all_features, seed=SEED)

        for k in BUDGETS:
            subset = ranked[:k]
            for model_name, fitter in FITTERS.items():
                fitted, train_time = fitter(X_train, y_train, subset)
                model_path = MODELS_DIR / f"{slug(model_name)}_top{k}_fold{fold_idx:02d}.joblib"
                fitted.save(model_path)

                prob = fitted.predict_proba(X_test)
                m = compute_metrics(y_test, prob, threshold=0.5)
                m.update({
                    "fold_id": fold_idx, "held_out_run": held_out_run, "model": model_name,
                    "condition": "clean", "feature_budget": str(k), "placement": "central",
                    "threshold_rule": "fixed_0.5", "seed": SEED, "train_time_sec": train_time,
                })
                rows.append(m)
                for yt, p, idx in zip(y_test, prob, test_idx):
                    pred_rows.append({
                        "fold_id": fold_idx, "held_out_run": held_out_run, "model": model_name,
                        "feature_budget": str(k), "row_index": int(idx), "y_true": int(yt), "y_prob": float(p),
                    })
            print(f"[{time.strftime('%H:%M:%S')}] fold {fold_idx}/15 budget=top{k} done")

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(METRICS_DIR / "feature_budget_ranked_fold_metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(PRED_DIR / "feature_budget_ranked_predictions.csv", index=False)

    macro = (
        fold_df.groupby(["model", "feature_budget"])[["precision", "recall", "f1", "fpr", "roc_auc", "pr_auc"]]
        .agg(["mean", "std"])
    )
    macro.columns = [f"{c}_{s}" for c, s in macro.columns]
    macro.reset_index().to_csv(METRICS_DIR / "feature_budget_ranked_macro.csv", index=False)
    print(f"wrote feature_budget_ranked_fold_metrics.csv ({len(fold_df)} rows), feature_budget_ranked_macro.csv")


if __name__ == "__main__":
    main()
