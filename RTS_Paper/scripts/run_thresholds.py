"""Phase E: threshold stability across the 4 rules, over cached fold models.

For each fold/model: recompute train-fold probabilities (inference only, the
model was already fit in Phase B — no retraining), select each of the 4
threshold rules on those training probabilities, then apply the resulting
fixed threshold unchanged to the held-out test fold's already-computed
probabilities. Reports per-fold thresholds, alert/false-alarm/miss rates per
1,000 records, and cross-run threshold variability.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from RTS_Paper.scripts.data import load_dataset
from RTS_Paper.scripts.metrics import compute_metrics
from RTS_Paper.scripts.models import FittedRTSModel
from RTS_Paper.scripts.run_loro import slug
from RTS_Paper.scripts.thresholds import THRESHOLD_RULES

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "RTS_Paper" / "outputs" / "models"
METRICS_DIR = ROOT / "RTS_Paper" / "outputs" / "metrics"

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "Compact MLP"]


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    data, features, meta = load_dataset()
    X_full = data[features]
    y_full = data["target"].to_numpy()
    groups = data["run_id"].to_numpy()
    splits = list(GroupKFold(n_splits=15).split(X_full, y_full, groups))

    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        held_out_run = int(np.unique(groups[test_idx])[0])
        X_train, y_train = X_full.iloc[train_idx], y_full[train_idx]
        X_test, y_test = X_full.iloc[test_idx], y_full[test_idx]

        for model_name in MODEL_NAMES:
            model_path = MODELS_DIR / f"{slug(model_name)}_fold{fold_idx:02d}.joblib"
            if not model_path.exists():
                continue
            fitted = FittedRTSModel.load(model_path)
            train_prob = fitted.predict_proba(X_train)
            test_prob = fitted.predict_proba(X_test)

            for rule_name, rule_fn in THRESHOLD_RULES.items():
                t = rule_fn(y_train, train_prob)
                m = compute_metrics(y_test, test_prob, threshold=t)
                pred = (test_prob >= t).astype(int)
                n = len(y_test)
                false_alarms = int(((pred == 1) & (y_test == 0)).sum())
                misses = int(((pred == 0) & (y_test == 1)).sum())
                alerts = int((pred == 1).sum())
                n_attack = int((y_test == 1).sum())
                m.update({
                    "fold_id": fold_idx, "held_out_run": held_out_run, "model": model_name,
                    "threshold_rule": rule_name, "threshold_value": t,
                    "alerts_per_1000": alerts / n * 1000,
                    "false_alarms_per_1000": false_alarms / n * 1000,
                    "misses_per_1000_attacks": misses / n_attack * 1000 if n_attack else np.nan,
                })
                rows.append(m)
        print(f"[{time.strftime('%H:%M:%S')}] fold {fold_idx}/15 (run {held_out_run}) thresholds done")

    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / "threshold_stability_fold_results.csv", index=False)

    variability = (
        df.groupby(["model", "threshold_rule"])["threshold_value"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    variability.to_csv(METRICS_DIR / "threshold_variability_summary.csv", index=False)

    summary = (
        df.groupby(["model", "threshold_rule"])[["precision", "recall", "f1", "fpr", "alerts_per_1000", "false_alarms_per_1000", "misses_per_1000_attacks"]]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary.reset_index().to_csv(METRICS_DIR / "threshold_rule_macro_summary.csv", index=False)
    print(f"[{time.strftime('%H:%M:%S')}] wrote threshold_stability_fold_results.csv ({len(df)} rows), threshold_variability_summary.csv, threshold_rule_macro_summary.csv")


if __name__ == "__main__":
    main()
