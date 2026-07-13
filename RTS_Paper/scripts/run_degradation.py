"""Phase D: availability/robustness suite over cached fold models.

Reuses the central-128 fold models persisted by run_loro.py (feature_budget
"128") — zero retraining. Applies missingness, feature-group unavailability,
and relay loss to each held-out test fold only, then predicts with the
already-fitted (train-only) preprocessor + estimator for that fold.

Usage:
    python3 -m RTS_Paper.scripts.run_degradation
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
from RTS_Paper.scripts.degradations import FEATURE_GROUPS, RELAY_PREFIXES, group_unavailable, random_missingness, relay_loss
from RTS_Paper.scripts.metrics import compute_metrics
from RTS_Paper.scripts.models import FittedRTSModel
from RTS_Paper.scripts.run_loro import slug

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "RTS_Paper" / "outputs" / "models"
ROBUST_DIR = ROOT / "RTS_Paper" / "outputs" / "robustness"

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "Compact MLP"]
MISSING_RATES = [0.05, 0.10, 0.20, 0.30]
MISSING_SEEDS = [42, 43, 44, 45, 46]


def main() -> None:
    ROBUST_DIR.mkdir(parents=True, exist_ok=True)
    data, features, meta = load_dataset()
    X_full = data[features]
    y_full = data["target"].to_numpy()
    groups = data["run_id"].to_numpy()
    splits = list(GroupKFold(n_splits=15).split(X_full, y_full, groups))

    missing_rows = []
    group_rows = []
    relay_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        held_out_run = int(np.unique(groups[test_idx])[0])
        X_test, y_test = X_full.iloc[test_idx], y_full[test_idx]

        for model_name in MODEL_NAMES:
            model_path = MODELS_DIR / f"{slug(model_name)}_fold{fold_idx:02d}.joblib"
            if not model_path.exists():
                print(f"MISSING cached model, skipping: {model_path}")
                continue
            fitted = FittedRTSModel.load(model_path)

            # clean baseline (already computed in Phase B, recomputed here for a same-table anchor row)
            clean_prob = fitted.predict_proba(X_test)
            clean_m = compute_metrics(y_test, clean_prob, threshold=0.5)

            # -- missingness --
            for rate in MISSING_RATES:
                for seed in MISSING_SEEDS:
                    Xd = random_missingness(X_test, features, rate, seed)
                    prob = fitted.predict_proba(Xd)
                    m = compute_metrics(y_test, prob, threshold=0.5)
                    m.update({
                        "fold_id": fold_idx, "held_out_run": held_out_run, "model": model_name,
                        "condition": f"missing_{int(rate*100)}pct", "rate": rate, "seed": seed,
                        "clean_f1": clean_m["f1"], "clean_roc_auc": clean_m["roc_auc"],
                        "abs_f1_drop": clean_m["f1"] - m["f1"], "abs_auc_drop": clean_m["roc_auc"] - m["roc_auc"],
                    })
                    missing_rows.append(m)

            # -- feature-group unavailability at decision time --
            for group_name in FEATURE_GROUPS:
                Xd = group_unavailable(X_test, group_name)
                prob = fitted.predict_proba(Xd)
                m = compute_metrics(y_test, prob, threshold=0.5)
                m.update({
                    "fold_id": fold_idx, "held_out_run": held_out_run, "model": model_name,
                    "condition": f"group_unavailable_{group_name}",
                    "clean_f1": clean_m["f1"], "clean_roc_auc": clean_m["roc_auc"],
                    "abs_f1_drop": clean_m["f1"] - m["f1"], "abs_auc_drop": clean_m["roc_auc"] - m["roc_auc"],
                })
                group_rows.append(m)

            # -- complete relay loss R1-R4 --
            for relay in RELAY_PREFIXES:
                Xd = relay_loss(X_test, relay)
                prob = fitted.predict_proba(Xd)
                m = compute_metrics(y_test, prob, threshold=0.5)
                m.update({
                    "fold_id": fold_idx, "held_out_run": held_out_run, "model": model_name,
                    "condition": f"relay_loss_{relay}",
                    "clean_f1": clean_m["f1"], "clean_roc_auc": clean_m["roc_auc"],
                    "abs_f1_drop": clean_m["f1"] - m["f1"], "abs_auc_drop": clean_m["roc_auc"] - m["roc_auc"],
                })
                relay_rows.append(m)

        print(f"[{time.strftime('%H:%M:%S')}] fold {fold_idx}/15 (run {held_out_run}) degradation conditions done")

    pd.DataFrame(missing_rows).to_csv(ROBUST_DIR / "missingness_results.csv", index=False)
    pd.DataFrame(group_rows).to_csv(ROBUST_DIR / "group_unavailability_results.csv", index=False)
    pd.DataFrame(relay_rows).to_csv(ROBUST_DIR / "relay_loss_results.csv", index=False)
    print(f"[{time.strftime('%H:%M:%S')}] wrote missingness_results.csv ({len(missing_rows)} rows), "
          f"group_unavailability_results.csv ({len(group_rows)} rows), relay_loss_results.csv ({len(relay_rows)} rows)")


if __name__ == "__main__":
    main()
