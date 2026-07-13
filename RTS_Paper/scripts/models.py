"""Model fitters for the RTC extension harness.

Each fitter returns a FittedRTSModel exposing predict_proba(X) and a
.save(path)/.load(path) pair so Phase B can persist all 15 fold models
(hard requirement — Phase D/E reuse these without retraining).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42


@dataclass
class FittedRTSModel:
    name: str
    preprocessor: Pipeline
    estimator: object
    feature_order: list[str]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.preprocessor.transform(X[self.feature_order])
        return self.estimator.predict_proba(Xt)[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "FittedRTSModel":
        return joblib.load(path)


def _tree_preprocessor() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median"))])


def _scaled_preprocessor() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])


def fit_logistic_regression(X_train: pd.DataFrame, y_train, features: list[str]) -> tuple[FittedRTSModel, float]:
    pre = _scaled_preprocessor()
    start = time.perf_counter()
    Xt = pre.fit_transform(X_train[features])
    clf = LogisticRegression(max_iter=2000, random_state=SEED, class_weight=None)
    clf.fit(Xt, y_train)
    return FittedRTSModel("Logistic Regression", pre, clf, list(features)), time.perf_counter() - start


def fit_random_forest(X_train: pd.DataFrame, y_train, features: list[str]) -> tuple[FittedRTSModel, float]:
    pre = _tree_preprocessor()
    start = time.perf_counter()
    Xt = pre.fit_transform(X_train[features])
    clf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    clf.fit(Xt, y_train)
    return FittedRTSModel("Random Forest", pre, clf, list(features)), time.perf_counter() - start


def fit_xgboost(X_train: pd.DataFrame, y_train, features: list[str]) -> tuple[FittedRTSModel, float]:
    pre = _tree_preprocessor()
    start = time.perf_counter()
    Xt = pre.fit_transform(X_train[features])
    clf = XGBClassifier(
        n_estimators=200,
        random_state=SEED,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        tree_method="hist",
    )
    clf.fit(Xt, y_train)
    return FittedRTSModel("XGBoost", pre, clf, list(features)), time.perf_counter() - start


def fit_compact_mlp(X_train: pd.DataFrame, y_train, features: list[str]) -> tuple[FittedRTSModel, float]:
    pre = _scaled_preprocessor()
    start = time.perf_counter()
    Xt = pre.fit_transform(X_train[features])
    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        alpha=1e-4,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=SEED,
    )
    clf.fit(Xt, y_train)
    return FittedRTSModel("Compact MLP", pre, clf, list(features)), time.perf_counter() - start


class _TorchProbaWrapper:
    """Adapts a torch FeatureCNN to the estimator.predict_proba(Xt)[:, 1]
    interface FittedRTSModel expects, so it plugs in without special-casing.
    """

    def __init__(self, module):
        self.module = module

    def predict_proba(self, Xt: np.ndarray) -> np.ndarray:
        import torch

        self.module.eval()
        with torch.no_grad():
            tensor = torch.tensor(Xt, dtype=torch.float32).unsqueeze(1)
            logits = self.module(tensor).squeeze(1)
            p1 = torch.sigmoid(logits).cpu().numpy()
        return np.stack([1 - p1, p1], axis=1)


def fit_feature_cnn(X_train: pd.DataFrame, y_train, features: list[str], max_epochs: int = 50, patience: int = 10) -> tuple[FittedRTSModel, float]:
    """P1 stretch model. Reuses the original paper's FeatureCNN architecture
    (src.run_all.FeatureCNN) for a fair cross-check, forced onto CPU for
    reproducibility across machines (no MPS/CUDA nondeterminism).
    """
    import torch
    import torch.nn as nn
    from sklearn.model_selection import StratifiedShuffleSplit
    from torch.utils.data import DataLoader, TensorDataset

    from src.run_all import FeatureCNN

    pre = _scaled_preprocessor()
    start = time.perf_counter()
    Xt = pre.fit_transform(X_train[features]).astype("float32")
    y = np.asarray(y_train).astype("float32")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tr_idx, val_idx = next(sss.split(Xt, y))
    device = torch.device("cpu")
    train_ds = TensorDataset(torch.tensor(Xt[tr_idx]).unsqueeze(1), torch.tensor(y[tr_idx]))
    val_x = torch.tensor(Xt[val_idx]).unsqueeze(1).to(device)
    val_y = torch.tensor(y[val_idx]).to(device)
    loader = DataLoader(train_ds, batch_size=min(8192, len(train_ds)), shuffle=True, generator=torch.Generator().manual_seed(SEED))
    model = FeatureCNN(Xt.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state, best_val, bad = None, float("inf"), 0
    for epoch in range(max_epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = loss_fn(model(bx).squeeze(1), by)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_x).squeeze(1), val_y).item()
        if val_loss < best_val - 1e-5:
            best_val, best_state, bad = val_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return FittedRTSModel("Feature-space CNN", pre, _TorchProbaWrapper(model), list(features)), time.perf_counter() - start


FITTERS = {
    "Logistic Regression": fit_logistic_regression,
    "Random Forest": fit_random_forest,
    "XGBoost": fit_xgboost,
    "Compact MLP": fit_compact_mlp,
}

FITTERS_P1 = {"Feature-space CNN": fit_feature_cnn}
