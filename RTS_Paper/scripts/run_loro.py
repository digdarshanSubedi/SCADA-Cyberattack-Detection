"""Phase B: config-driven leave-one-run-out harness.

Trains LR / RF / XGBoost / Compact MLP on 14 runs, tests on the held-out
15th run, for all 15 folds. Persists every fitted fold model under
RTS_Paper/outputs/models/ (hard requirement: Phase D/E reuse these caches
instead of retraining per degradation condition).

Usage:
    python3 -m RTS_Paper.scripts.run_loro                 # full run, all models
    python3 -m RTS_Paper.scripts.run_loro --smoke-test     # 2 folds, 1 model
    python3 -m RTS_Paper.scripts.run_loro --models "Logistic Regression,Random Forest"
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from RTS_Paper.scripts.data import feature_budget_subset, load_dataset
from RTS_Paper.scripts.metrics import compute_metrics
from RTS_Paper.scripts.models import FITTERS, FITTERS_P1

ALL_FITTERS = {**FITTERS, **FITTERS_P1}

ROOT = Path(__file__).resolve().parents[2]
RTS_OUTPUTS = ROOT / "RTS_Paper" / "outputs"
MODELS_DIR = RTS_OUTPUTS / "models"
METRICS_DIR = RTS_OUTPUTS / "metrics"
PRED_DIR = RTS_OUTPUTS / "predictions"

SEED = 42


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def run(models: list[str], smoke_test: bool = False, feature_budget: str = "128") -> None:
    for d in [MODELS_DIR, METRICS_DIR, PRED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    data, all_features, meta = load_dataset()
    assert "run_id" not in all_features, "GROUND RULE VIOLATION: run_id in feature matrix"

    if feature_budget == "128":
        features = all_features
    else:
        features = feature_budget_subset(all_features, meta, feature_budget)
    print(f"[{time.strftime('%H:%M:%S')}] loaded {meta['n_rows']} rows, budget={feature_budget} -> {len(features)} features, {meta['n_runs']} runs")

    X_full = data[features]
    y_full = data["target"].to_numpy()
    groups = data["run_id"].to_numpy()

    splits = list(GroupKFold(n_splits=15).split(X_full, y_full, groups))
    assert len(splits) == 15

    # Structural leakage check: no run appears in both train and test of any fold.
    for fold_idx, (tr, te) in enumerate(splits, start=1):
        train_runs = set(groups[tr])
        test_runs = set(groups[te])
        assert train_runs.isdisjoint(test_runs), f"fold {fold_idx}: run overlap between train/test"
        assert len(test_runs) == 1, f"fold {fold_idx}: expected exactly 1 held-out run, got {test_runs}"

    if smoke_test:
        splits = splits[:2]
        models = models[:1]
        print(f"[SMOKE TEST] {len(splits)} folds x {models}")

    rows = []
    pred_rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        held_out_run = int(np.unique(groups[test_idx])[0])
        X_train, y_train = X_full.iloc[train_idx], y_full[train_idx]
        X_test, y_test = X_full.iloc[test_idx], y_full[test_idx]
        for model_name in models:
            fitter = ALL_FITTERS[model_name]
            fitted, train_time = fitter(X_train, y_train, features)
            budget_tag = "" if feature_budget == "128" else f"_{feature_budget}"
            model_path = MODELS_DIR / f"{slug(model_name)}{budget_tag}_fold{fold_idx:02d}.joblib"
            fitted.save(model_path)

            placement = "edge" if feature_budget == "edge_r1" else "central"
            prob = fitted.predict_proba(X_test)
            m = compute_metrics(y_test, prob, threshold=0.5)
            m.update({
                "fold_id": fold_idx,
                "held_out_run": held_out_run,
                "model": model_name,
                "condition": "clean",
                "feature_budget": feature_budget,
                "placement": placement,
                "threshold_rule": "fixed_0.5",
                "seed": SEED,
                "train_time_sec": train_time,
                "model_path": str(model_path.relative_to(ROOT)),
            })
            rows.append(m)

            for yt, p, idx in zip(y_test, prob, test_idx):
                pred_rows.append({
                    "fold_id": fold_idx, "held_out_run": held_out_run, "model": model_name,
                    "condition": "clean", "feature_budget": feature_budget, "row_index": int(idx),
                    "y_true": int(yt), "y_prob": float(p),
                })
            print(f"[{time.strftime('%H:%M:%S')}] fold {fold_idx}/15 run={held_out_run} {model_name}: f1={m['f1']:.3f} auc={m['roc_auc']:.3f} train_s={train_time:.1f}")

    fold_df = pd.DataFrame(rows)
    pred_df = pd.DataFrame(pred_rows)
    suffix = "_smoke" if smoke_test else ""
    budget_suffix = "" if feature_budget == "128" else f"_{feature_budget}"
    fold_df.to_csv(METRICS_DIR / f"phase_b_loro_clean_fold_metrics{budget_suffix}{suffix}.csv", index=False)
    pred_df.to_csv(PRED_DIR / f"phase_b_loro_clean_predictions{budget_suffix}{suffix}.csv", index=False)

    if not smoke_test:
        macro = (
            fold_df.groupby("model")[["precision", "recall", "f1", "fpr", "roc_auc", "pr_auc", "balanced_accuracy"]]
            .agg(["mean", "std"])
        )
        macro.columns = [f"{c}_{stat}" for c, stat in macro.columns]
        macro = macro.reset_index()
        macro.to_csv(METRICS_DIR / f"phase_b_loro_clean_macro{budget_suffix}.csv", index=False)

        pooled_rows = []
        for model_name, g in pred_df.groupby("model"):
            pm = compute_metrics(g["y_true"], g["y_prob"], threshold=0.5)
            pm["model"] = model_name
            pooled_rows.append(pm)
        pd.DataFrame(pooled_rows).to_csv(METRICS_DIR / f"phase_b_loro_clean_pooled{budget_suffix}.csv", index=False)
        print(f"[{time.strftime('%H:%M:%S')}] wrote macro (mean/std across folds) and pooled (out-of-fold) metrics separately")

    print(f"[{time.strftime('%H:%M:%S')}] done. {len(fold_df)} fold rows written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=",".join(FITTERS.keys()))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--feature-budget", type=str, default="128", choices=["128", "pmu_only", "cyber_log_only", "edge_r1"])
    args = parser.parse_args()
    model_list = [m.strip() for m in args.models.split(",")]
    for m in model_list:
        assert m in ALL_FITTERS, f"unknown model '{m}', choices: {list(ALL_FITTERS)}"
    run(model_list, smoke_test=args.smoke_test, feature_budget=args.feature_budget)
