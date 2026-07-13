"""Phase D (task, saved after initial inline run): complete PMU-group unavailability.

Masks all 116 PMU features (pmu_voltage + pmu_current + freq_impedance +
relay_status) on each held-out test fold, reusing the cached central-128
fold models from run_loro.py — zero retraining. Complements
logs_unavailable (the 12 non-PMU features) in run_placement.py.

This script formalizes an ablation that was first run as an inline
interactive snippet during drafting (see VERIFICATION_PACKET.md Section 1.7
and 3 for the reproducibility gap this closes) and reruns it from scratch to
confirm the saved script reproduces the original inline result exactly.
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
from RTS_Paper.scripts.degradations import all_pmu_unavailable
from RTS_Paper.scripts.metrics import compute_metrics
from RTS_Paper.scripts.models import FittedRTSModel
from RTS_Paper.scripts.run_loro import slug

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "RTS_Paper" / "outputs" / "models"
ROBUST_DIR = ROOT / "RTS_Paper" / "outputs" / "robustness"

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "Compact MLP"]


def main() -> None:
    ROBUST_DIR.mkdir(parents=True, exist_ok=True)
    data, features, meta = load_dataset()
    assert len(meta["pmu_features"]) == 116, f"expected 116 PMU features, got {len(meta['pmu_features'])}"

    X_full = data[features]
    y_full = data["target"].to_numpy()
    groups = data["run_id"].to_numpy()
    splits = list(GroupKFold(n_splits=15).split(X_full, y_full, groups))

    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        held_out_run = int(np.unique(groups[test_idx])[0])
        X_test, y_test = X_full.iloc[test_idx], y_full[test_idx]
        Xd = all_pmu_unavailable(X_test)
        masked_cols = [c for c in Xd.columns if Xd[c].isna().all()]
        assert len(masked_cols) == 116, f"fold {fold_idx}: expected 116 masked columns, got {len(masked_cols)}"

        for model_name in MODEL_NAMES:
            path = MODELS_DIR / f"{slug(model_name)}_fold{fold_idx:02d}.joblib"
            if not path.exists():
                print(f"MISSING cached model, skipping: {path}")
                continue
            fitted = FittedRTSModel.load(path)
            prob = fitted.predict_proba(Xd)
            m = compute_metrics(y_test, prob, threshold=0.5)
            m.update({"fold_id": fold_idx, "held_out_run": held_out_run, "model": model_name, "condition": "pmu_group_unavailable"})
            rows.append(m)
        print(f"[{time.strftime('%H:%M:%S')}] fold {fold_idx}/15 (run {held_out_run}) PMU-group-unavailable done")

    out = pd.DataFrame(rows)
    out.to_csv(ROBUST_DIR / "pmu_group_unavailable_results.csv", index=False)
    print(out.groupby("model")["roc_auc"].mean().sort_values())
    print(f"[{time.strftime('%H:%M:%S')}] wrote pmu_group_unavailable_results.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
