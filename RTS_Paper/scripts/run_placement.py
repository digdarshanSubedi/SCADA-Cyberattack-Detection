"""Phase D (task D): placement comparison table.

Edge = model trained + evaluated on R1's 29 features only (feature_budget
edge_r1 from run_loro.py), d_agg = 0.
Central = model trained on all 128 features (feature_budget 128), evaluated
under: clean, 10% random missingness, all-logs-unavailable, and the single
worst relay lost (max AUC drop among R1-R4 per model) — each + d_agg from
run_latency.py's envelope verdicts.

Reuses: phase_b_loro_clean_fold_metrics.csv (central clean),
phase_b_loro_clean_fold_metrics_edge_r1.csv (edge clean),
missingness_results.csv (central 10% missing),
relay_loss_results.csv (worst relay), and a fresh cached-model pass for
all-logs-unavailable (not computed elsewhere). No retraining anywhere.
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
from RTS_Paper.scripts.degradations import logs_unavailable
from RTS_Paper.scripts.metrics import compute_metrics
from RTS_Paper.scripts.models import FittedRTSModel
from RTS_Paper.scripts.run_loro import slug

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "RTS_Paper" / "outputs" / "models"
METRICS_DIR = ROOT / "RTS_Paper" / "outputs" / "metrics"
ROBUST_DIR = ROOT / "RTS_Paper" / "outputs" / "robustness"

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "Compact MLP"]


def logs_unavailable_metrics() -> pd.DataFrame:
    data, features, meta = load_dataset()
    X_full = data[features]
    y_full = data["target"].to_numpy()
    groups = data["run_id"].to_numpy()
    splits = list(GroupKFold(n_splits=15).split(X_full, y_full, groups))

    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        held_out_run = int(np.unique(groups[test_idx])[0])
        X_test, y_test = X_full.iloc[test_idx], y_full[test_idx]
        Xd = logs_unavailable(X_test)
        for model_name in MODEL_NAMES:
            path = MODELS_DIR / f"{slug(model_name)}_fold{fold_idx:02d}.joblib"
            if not path.exists():
                continue
            fitted = FittedRTSModel.load(path)
            prob = fitted.predict_proba(Xd)
            m = compute_metrics(y_test, prob, threshold=0.5)
            m.update({"fold_id": fold_idx, "held_out_run": held_out_run, "model": model_name, "condition": "logs_unavailable"})
            rows.append(m)
        print(f"[{time.strftime('%H:%M:%S')}] logs-unavailable fold {fold_idx}/15 done")
    return pd.DataFrame(rows)


def main() -> None:
    clean_central = pd.read_csv(METRICS_DIR / "phase_b_loro_clean_fold_metrics.csv")
    clean_edge = pd.read_csv(METRICS_DIR / "phase_b_loro_clean_fold_metrics_edge_r1.csv")
    missing10 = pd.read_csv(ROBUST_DIR / "missingness_results.csv")
    missing10 = missing10[missing10.condition == "missing_10pct"]
    relay = pd.read_csv(ROBUST_DIR / "relay_loss_results.csv")
    logs = logs_unavailable_metrics()
    logs.to_csv(ROBUST_DIR / "logs_unavailable_results.csv", index=False)

    metric_cols = ["precision", "recall", "f1", "fpr", "roc_auc", "pr_auc"]
    rows = []
    for model_name in MODEL_NAMES:
        e = clean_edge[clean_edge.model == model_name]
        rows.append({"model": model_name, "placement_condition": "Edge (edge-visible features)", **{c: e[c].mean() for c in metric_cols}, **{f"{c}_std": e[c].std() for c in metric_cols}})

        c = clean_central[clean_central.model == model_name]
        rows.append({"model": model_name, "placement_condition": "Central, clean aggregation", **{c2: c[c2].mean() for c2 in metric_cols}, **{f"{c2}_std": c[c2].std() for c2 in metric_cols}})

        m10 = missing10[missing10.model == model_name]
        rows.append({"model": model_name, "placement_condition": "Central, 10% missing", **{c2: m10[c2].mean() for c2 in metric_cols}, **{f"{c2}_std": m10[c2].std() for c2 in metric_cols}})

        lg = logs[logs.model == model_name]
        rows.append({"model": model_name, "placement_condition": "Central, logs unavailable", **{c2: lg[c2].mean() for c2 in metric_cols}, **{f"{c2}_std": lg[c2].std() for c2 in metric_cols}})

        r = relay[relay.model == model_name]
        worst_relay_per_fold = r.loc[r.groupby("fold_id")["roc_auc"].idxmin()]
        rows.append({"model": model_name, "placement_condition": "Central, one relay lost (worst)", **{c2: worst_relay_per_fold[c2].mean() for c2 in metric_cols}, **{f"{c2}_std": worst_relay_per_fold[c2].std() for c2 in metric_cols}})

    out = pd.DataFrame(rows)
    out.to_csv(METRICS_DIR / "placement_comparison_table.csv", index=False)
    print(out.to_string(index=False))
    print(f"wrote {METRICS_DIR}/placement_comparison_table.csv")


if __name__ == "__main__":
    main()
