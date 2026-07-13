"""Side-effect-free data loading for the RTC extension harness.

Reuses only pure helper functions from src.run_all (column parsing, label
mapping, feature ordering) — never calls audit_and_load(), which writes into
the original paper's outputs/ directory. That directory is frozen.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.run_all import (
    feature_groups,
    ordered_features,
    sorted_data_files,
    target_from_marker,
)

ROOT = Path(__file__).resolve().parents[2]
RTS_OUTPUTS = ROOT / "RTS_Paper" / "outputs"


def load_dataset() -> tuple[pd.DataFrame, list[str], dict]:
    """Load all 15 runs, tag run_id, map labels. No files written."""
    frames = []
    schema = None
    for run_id, path in enumerate(sorted_data_files(), start=1):
        df = pd.read_csv(path)
        if schema is None:
            schema = list(df.columns)
        df = df.replace([np.inf, -np.inf], np.nan)
        df["target"] = df["marker"].map(target_from_marker).astype(int)
        df["run_id"] = run_id
        df["source_file"] = path.name
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    features = ordered_features(list(schema))
    groups = feature_groups(list(schema))
    assert "run_id" not in features, "run_id leaked into feature list"
    assert "source_file" not in features, "source_file leaked into feature list"
    assert "marker" not in features, "marker leaked into feature list"
    meta = {
        "n_rows": len(data),
        "n_features": len(features),
        "pmu_features": groups["pmu"],
        "cyber_log_features": groups["control_relay_snort"],
        "n_runs": data["run_id"].nunique(),
    }
    return data, features, meta


# --- Feature-budget subset definitions -------------------------------------

def feature_budget_subset(all_features: list[str], groups: dict, budget: str) -> list[str]:
    """budget in {'128','64','32','16','pmu_only','cyber_log_only','edge_r1'}.

    64/32/16 are defined at harness call time by the caller using a
    training-fold-only ranking (see rank_features_train_only); this function
    only resolves the structural (non-ranked) subsets.
    """
    pmu = groups["pmu_features"]
    cyber = groups["cyber_log_features"]
    if budget == "128":
        return list(all_features)
    if budget == "pmu_only":
        return [f for f in all_features if f in pmu]
    if budget == "cyber_log_only":
        return [f for f in all_features if f in cyber]
    if budget == "edge_r1":
        return [f for f in all_features if f.startswith("R1-") or f.startswith("R1:")]
    raise ValueError(f"budget '{budget}' requires a training-fold ranking; use rank_features_train_only")


def rank_features_train_only(X_train: pd.DataFrame, y_train: np.ndarray, all_features: list[str], seed: int = 42) -> list[str]:
    """Rank features by RandomForest importance fit on the training fold only.

    Used to build the 64/32/16 top-k budgets. Never touches test data.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    Xt = imputer.fit_transform(X_train[all_features])
    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    clf.fit(Xt, y_train)
    order = np.argsort(clf.feature_importances_)[::-1]
    return [all_features[i] for i in order]
