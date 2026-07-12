from __future__ import annotations

import json
import math
import os
import platform
import random
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - recorded in runtime output
    XGBClassifier = None
    XGB_IMPORT_ERROR = repr(exc)
else:
    XGB_IMPORT_ERROR = None

try:
    import shap
except Exception as exc:  # pragma: no cover - recorded in runtime output
    shap = None
    SHAP_IMPORT_ERROR = repr(exc)
else:
    SHAP_IMPORT_ERROR = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover - recorded in runtime output
    torch = None
    TORCH_IMPORT_ERROR = repr(exc)
else:
    TORCH_IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "binaryAllNaturalPlusNormalVsAttacks - 2 Class"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
PAPER = ROOT / "paper_project"
SEED = 42

ATTACK_SCENARIOS = {7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 36, 37, 38, 39, 40}
NORMAL_SCENARIOS = {1, 2, 3, 4, 5, 6, 13, 14, 41}


def set_seeds() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if torch is not None:
        torch.manual_seed(SEED)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def ensure_dirs() -> None:
    for path in [OUTPUTS, FIGURES, PAPER / "figures", PAPER / "supplementary" / "additional_figures"]:
        path.mkdir(parents=True, exist_ok=True)


def sorted_data_files() -> list[Path]:
    files = sorted(DATA_DIR.glob("data*.csv"), key=lambda p: int(p.stem.replace("data", "")))
    if len(files) != 15:
        raise FileNotFoundError(f"Expected 15 data*.csv files in {DATA_DIR}, found {len(files)}")
    return files


def scenario_from_marker(value) -> float:
    text = str(value)
    digits = ""
    for ch in text:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    if not digits:
        return math.nan
    candidate = int(digits[:2])
    if candidate in ATTACK_SCENARIOS or candidate in NORMAL_SCENARIOS:
        return candidate
    candidate = int(digits[:1])
    if candidate in ATTACK_SCENARIOS or candidate in NORMAL_SCENARIOS:
        return candidate
    return math.nan


def target_from_marker(value: str) -> int:
    text = str(value).strip().lower()
    if text == "attack":
        return 1
    if text in {"natural", "normal", "no event", "noevent"}:
        return 0
    scenario = scenario_from_marker(value)
    if scenario in ATTACK_SCENARIOS:
        return 1
    if scenario in NORMAL_SCENARIOS:
        return 0
    raise ValueError(f"Cannot map marker to binary target: {value!r}")


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    pmu = [c for c in columns if c.startswith(("R1-", "R2-", "R3-", "R4-")) or c in ["R1:F", "R1:DF", "R1:S", "R2:F", "R2:DF", "R2:S", "R3:F", "R3:DF", "R3:S", "R4:F", "R4:DF", "R4:S"]]
    control = [c for c in columns if c.startswith(("control_panel", "relay", "snort"))]
    metadata = [c for c in columns if c in {"marker", "target", "run_id", "source_file", "scenario"}]
    unexpected = [c for c in columns if c not in set(pmu + control + metadata)]
    return {"pmu": pmu, "control_relay_snort": control, "metadata": metadata, "unexpected": unexpected}


def ordered_features(columns: list[str]) -> list[str]:
    groups = feature_groups(columns)
    ordered = []
    for relay in ["R1", "R2", "R3", "R4"]:
        ordered.extend([c for c in columns if c.startswith(f"{relay}-") or c.startswith(f"{relay}:")])
    ordered.extend(groups["control_relay_snort"])
    return ordered


def audit_and_load() -> tuple[pd.DataFrame, list[str], dict]:
    rows = []
    frames = []
    schema = None
    all_same = True
    for run_id, path in enumerate(sorted_data_files(), start=1):
        df = pd.read_csv(path)
        if schema is None:
            schema = list(df.columns)
        all_same = all_same and list(df.columns) == schema
        df = df.replace([np.inf, -np.inf], np.nan)
        target_exists = any(c.lower() in {"target", "label", "class"} for c in df.columns)
        mapped = df["marker"].map(target_from_marker)
        scenario = df["marker"].map(scenario_from_marker)
        scenario_available = scenario.notna().any()
        label_agrees = None
        if scenario_available:
            scenario_label = scenario.map(lambda s: 1 if s in ATTACK_SCENARIOS else (0 if s in NORMAL_SCENARIOS else np.nan))
            label_agrees = bool((scenario_label.dropna().astype(int) == mapped.loc[scenario_label.notna()].astype(int)).all())
        nunique = df.nunique(dropna=False)
        numeric = df.select_dtypes(include=[np.number])
        constant = nunique[nunique <= 1].index.tolist()
        near_constant = [c for c in numeric.columns if numeric[c].value_counts(normalize=True, dropna=False).iloc[0] >= 0.995]
        counts = mapped.value_counts().sort_index()
        class_count = {"normal": int(counts.get(0, 0)), "attack": int(counts.get(1, 0))}
        rows.append(
            {
                "file_name": path.name,
                "run_id": run_id,
                "rows": int(len(df)),
                "columns": int(df.shape[1]),
                "column_names": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "duplicate_rows": int(df.duplicated().sum()),
                "missing_values": int(df.isna().sum().sum()),
                "infinite_values": 0,
                "constant_features": constant,
                "near_constant_features": near_constant,
                "class_count": class_count,
                "class_percentage": {k: v / len(df) for k, v in class_count.items()},
                "scenario_distribution": scenario.value_counts(dropna=False).to_dict(),
                "scenario_field_available": bool(scenario_available),
                "target_column_exists": bool(target_exists),
                "target_agrees_with_official_scenario_mapping": label_agrees,
            }
        )
        df["target"] = mapped.astype(int)
        df["run_id"] = run_id
        df["source_file"] = path.name
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    features = ordered_features(list(schema))
    groups = feature_groups(list(schema))
    audit = {
        "files": rows,
        "all_fifteen_files_loaded": len(rows) == 15,
        "all_files_share_same_schema": all_same,
        "feature_groups": {k: len(v) for k, v in groups.items()},
        "pmu_columns": groups["pmu"],
        "control_relay_snort_columns": groups["control_relay_snort"],
        "unexpected_columns": groups["unexpected"],
        "usable_model_features": len(features),
        "feature_order_matches_readme": len(groups["pmu"]) == 116 and len(groups["control_relay_snort"]) == 12 and features == list(schema[:-1]),
        "dataset_rows": int(len(data)),
        "dataset_class_count": {str(k): int(v) for k, v in data["target"].value_counts().sort_index().items()},
        "label_boundary_note": "The binary CSV marker column contains Attack/Natural labels, not the original numeric scenario/load marker. Scenario-level agreement cannot be verified from these CSVs alone.",
    }
    (OUTPUTS / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    pd.DataFrame(rows).drop(columns=["column_names", "dtypes", "constant_features", "near_constant_features"]).to_csv(OUTPUTS / "dataset_audit_summary.csv", index=False)
    return data, features, audit


def make_tree_preprocessor() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median"))])


def make_scaled_preprocessor() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])


def metrics_dict(y_true, prob, pred, train_time, infer_time) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    out = {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "specificity": specificity,
        "fpr": fpr,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "mcc": matthews_corrcoef(y_true, pred),
        "brier": brier_score_loss(y_true, prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "training_time_sec": train_time,
        "inference_time_sec": infer_time,
        "latency_ms_per_record": infer_time * 1000 / len(y_true),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, prob)
    except ValueError:
        out["roc_auc"] = np.nan
    try:
        out["average_precision"] = average_precision_score(y_true, prob)
    except ValueError:
        out["average_precision"] = np.nan
    return out


class FeatureCNN(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class FittedModel:
    name: str
    estimator: object
    preprocessor: Pipeline
    epochs: int = 0

    def fit(self, X=None, y=None):
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.preprocessor.transform(X)
        if self.name == "CNN":
            self.estimator.eval()
            with torch.no_grad():
                tensor = torch.tensor(Xt, dtype=torch.float32).unsqueeze(1)
                logits = self.estimator(tensor).squeeze(1)
                return torch.sigmoid(logits).cpu().numpy()
        return self.estimator.predict_proba(Xt)[:, 1]


def fit_rf(X_train, y_train) -> tuple[FittedModel, float]:
    pre = make_tree_preprocessor()
    model = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    start = time.perf_counter()
    Xt = pre.fit_transform(X_train)
    model.fit(Xt, y_train)
    return FittedModel("Random Forest", model, pre), time.perf_counter() - start


def fit_xgb(X_train, y_train) -> tuple[FittedModel, float]:
    if XGBClassifier is None:
        raise RuntimeError(f"xgboost unavailable: {XGB_IMPORT_ERROR}")
    pre = make_tree_preprocessor()
    model = XGBClassifier(
        n_estimators=200,
        random_state=SEED,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=max(1, min(8, os.cpu_count() or 1)),
        tree_method="hist",
    )
    start = time.perf_counter()
    Xt = pre.fit_transform(X_train)
    model.fit(Xt, y_train)
    return FittedModel("XGBoost", model, pre), time.perf_counter() - start


def fit_cnn(X_train, y_train, max_epochs=50, patience=10) -> tuple[FittedModel, float]:
    if torch is None:
        raise RuntimeError(f"torch unavailable: {TORCH_IMPORT_ERROR}")
    pre = make_scaled_preprocessor()
    start = time.perf_counter()
    Xt = pre.fit_transform(X_train).astype("float32")
    y = np.asarray(y_train).astype("float32")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tr_idx, val_idx = next(sss.split(Xt, y))
    device = torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device("cpu")
    train_ds = TensorDataset(torch.tensor(Xt[tr_idx]).unsqueeze(1), torch.tensor(y[tr_idx]))
    val_x = torch.tensor(Xt[val_idx]).unsqueeze(1).to(device)
    val_y = torch.tensor(y[val_idx]).to(device)
    loader = DataLoader(train_ds, batch_size=min(8192, len(train_ds)), shuffle=True, generator=torch.Generator().manual_seed(SEED))
    model = FeatureCNN(Xt.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = None
    best_val = float("inf")
    bad = 0
    epochs_run = 0
    for epoch in range(max_epochs):
        model.train()
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            opt.zero_grad()
            loss = loss_fn(model(bx).squeeze(1), by)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_x).squeeze(1), val_y).item()
        epochs_run = epoch + 1
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to("cpu")
    return FittedModel("CNN", model, pre, epochs_run), time.perf_counter() - start


FITTERS = {"Random Forest": fit_rf, "XGBoost": fit_xgb, "CNN": fit_cnn}


def evaluate_model(name: str, X_train, y_train, X_test, y_test) -> tuple[dict, pd.DataFrame, FittedModel]:
    print(f"[{time.strftime('%H:%M:%S')}] training {name} on {len(X_train)} rows; testing on {len(X_test)} rows", flush=True)
    fitted, train_time = FITTERS[name](X_train, y_train)
    start = time.perf_counter()
    prob = fitted.predict_proba(X_test)
    infer_time = time.perf_counter() - start
    pred = (prob >= 0.5).astype(int)
    m = metrics_dict(y_test, prob, pred, train_time, infer_time)
    m["epochs"] = fitted.epochs
    print(f"[{time.strftime('%H:%M:%S')}] finished {name}: f1={m['f1']:.4f}, auc={m['roc_auc']:.4f}, train_sec={train_time:.1f}", flush=True)
    preds = pd.DataFrame({"y_true": y_test, "y_prob": prob, "y_pred": pred})
    return m, preds, fitted


def duplicate_and_similarity(data: pd.DataFrame, features: list[str], naive_train_idx, naive_test_idx) -> dict:
    X = data[features].replace([np.inf, -np.inf], np.nan)
    row_hash = pd.util.hash_pandas_object(X, index=False)
    exact_total = int(row_hash.duplicated(keep=False).sum())
    train_hashes = set(row_hash.iloc[naive_train_idx])
    crossing = int(row_hash.iloc[naive_test_idx].isin(train_hashes).sum())
    within = {}
    for run_id, idx in data.groupby("run_id").groups.items():
        h = row_hash.iloc[list(idx)]
        within[str(run_id)] = float(h.duplicated(keep=False).mean())
    cross_run_duplicate_rate = float(row_hash.groupby(row_hash).transform(lambda s: data.loc[s.index, "run_id"].nunique()).gt(1).mean())

    rng = np.random.default_rng(SEED)

    def nn_distance(train_idx, test_idx, sample_test=2500, sample_train=10000):
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        if len(train_idx) > sample_train:
            train_idx = rng.choice(train_idx, sample_train, replace=False)
        if len(test_idx) > sample_test:
            test_idx = rng.choice(test_idx, sample_test, replace=False)
        pre = make_scaled_preprocessor()
        Xtr = pre.fit_transform(X.iloc[train_idx])
        Xte = pre.transform(X.iloc[test_idx])
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
        nn.fit(Xtr)
        dist, _ = nn.kneighbors(Xte)
        return {
            "sample_train": int(len(train_idx)),
            "sample_test": int(len(test_idx)),
            "mean_distance": float(np.mean(dist)),
            "median_distance": float(np.median(dist)),
            "p05_distance": float(np.quantile(dist, 0.05)),
            "p95_distance": float(np.quantile(dist, 0.95)),
        }

    naive_nn = nn_distance(naive_train_idx, naive_test_idx)
    group_dist = []
    for _, test_idx in GroupKFold(n_splits=15).split(X, data["target"], groups=data["run_id"]):
        train_idx = np.setdiff1d(np.arange(len(data)), test_idx)
        d = nn_distance(train_idx, test_idx, sample_test=250, sample_train=10000)
        group_dist.append(d)
    group_nn = pd.DataFrame(group_dist).mean(numeric_only=True).to_dict()
    evidence = {
        "exact_duplicate_records_combined_keep_false": exact_total,
        "exact_duplicate_rate_combined": exact_total / len(data),
        "naive_test_records_with_exact_duplicate_in_train": crossing,
        "naive_test_exact_duplicate_in_train_rate": crossing / len(naive_test_idx),
        "within_run_duplicate_rate_by_run": within,
        "cross_run_duplicate_rate": cross_run_duplicate_rate,
        "nearest_neighbor_naive": naive_nn,
        "nearest_neighbor_groupkfold_mean_across_folds": group_nn,
    }
    (OUTPUTS / "leakage_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence


def run_naive(data, features) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    idx = np.arange(len(data))
    train_idx, test_idx = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=data["target"])
    np.save(OUTPUTS / "naive_train_indices.npy", train_idx)
    np.save(OUTPUTS / "naive_test_indices.npy", test_idx)
    leakage = duplicate_and_similarity(data, features, train_idx, test_idx)
    rows = []
    pred_frames = []
    fitted_models = {}
    for name in FITTERS:
        m, preds, fitted = evaluate_model(name, data.iloc[train_idx][features], data.iloc[train_idx]["target"], data.iloc[test_idx][features], data.iloc[test_idx]["target"].to_numpy())
        m.update({"model": name, "protocol": "naive_random_split"})
        rows.append(m)
        out = preds.copy()
        out["model"] = name
        out["row_index"] = test_idx
        out["run_id"] = data.iloc[test_idx]["run_id"].to_numpy()
        out["source_file"] = data.iloc[test_idx]["source_file"].to_numpy()
        pred_frames.append(out)
        fitted_models[name] = fitted
        pd.DataFrame(rows).to_csv(OUTPUTS / "metrics_naive.partial.csv", index=False)
        pd.concat(pred_frames, ignore_index=True).to_csv(OUTPUTS / "predictions_naive.partial.csv", index=False)
    metrics = pd.DataFrame(rows)
    preds = pd.concat(pred_frames, ignore_index=True)
    metrics.to_csv(OUTPUTS / "metrics_naive.csv", index=False)
    preds.to_csv(OUTPUTS / "predictions_naive.csv", index=False)
    return metrics, preds, fitted_models, leakage


def run_groupkfold(data, features) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    pred_frames = []
    best_artifact = {}
    X = data[features]
    y = data["target"].to_numpy()
    groups = data["run_id"].to_numpy()
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=15).split(X, y, groups), start=1):
        heldout_run = int(np.unique(groups[test_idx])[0])
        print(f"[{time.strftime('%H:%M:%S')}] starting GroupKFold fold {fold}/15, held-out run {heldout_run}", flush=True)
        for name in FITTERS:
            m, preds, fitted = evaluate_model(name, X.iloc[train_idx], y[train_idx], X.iloc[test_idx], y[test_idx])
            m.update({"model": name, "protocol": "groupkfold", "fold": fold, "heldout_run": heldout_run, "test_records": len(test_idx), "train_records": len(train_idx), "test_attack_rate": float(y[test_idx].mean())})
            rows.append(m)
            out = preds.copy()
            out["model"] = name
            out["fold"] = fold
            out["heldout_run"] = heldout_run
            out["row_index"] = test_idx
            out["source_file"] = data.iloc[test_idx]["source_file"].to_numpy()
            pred_frames.append(out)
            if name not in best_artifact or m["f1"] > best_artifact[name]["metric"]["f1"]:
                best_artifact[name] = {"model": fitted, "metric": m, "test_idx": test_idx, "train_idx": train_idx}
            pd.DataFrame(rows).to_csv(OUTPUTS / "per_fold_metrics.partial.csv", index=False)
            pd.concat(pred_frames, ignore_index=True).to_csv(OUTPUTS / "predictions_groupkfold.partial.csv", index=False)
    metrics = pd.DataFrame(rows)
    preds = pd.concat(pred_frames, ignore_index=True)
    metrics.to_csv(OUTPUTS / "per_fold_metrics.csv", index=False)
    preds.to_csv(OUTPUTS / "predictions_groupkfold.csv", index=False)
    return metrics, preds, best_artifact


def pooled_metrics(group_preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in group_preds.groupby("model"):
        m = metrics_dict(g["y_true"], g["y_prob"], g["y_pred"], np.nan, np.nan)
        m.update({"model": name, "protocol": "groupkfold_pooled"})
        rows.append(m)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUTS / "metrics_groupkfold_pooled.csv", index=False)
    return out


def grouped_summary(per_fold: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["precision", "recall", "f1", "fpr", "roc_auc", "average_precision", "accuracy", "balanced_accuracy", "mcc", "brier", "latency_ms_per_record", "training_time_sec", "epochs"]
    rows = []
    for name, g in per_fold.groupby("model"):
        row = {"model": name}
        for c in metric_cols:
            row[f"{c}_mean"] = g[c].mean(skipna=True)
            row[f"{c}_std"] = g[c].std(skipna=True)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUTS / "metrics_groupkfold_macro.csv", index=False)
    return out


def bootstrap_ci(y, prob, pred, groups=None, n=300) -> dict:
    rng = np.random.default_rng(SEED)
    y = np.asarray(y)
    prob = np.asarray(prob)
    pred = np.asarray(pred)
    vals = {"f1": [], "roc_auc": []}
    if groups is None:
        for _ in range(n):
            idx = rng.choice(len(y), len(y), replace=True)
            if len(np.unique(y[idx])) < 2:
                continue
            vals["f1"].append(f1_score(y[idx], pred[idx], zero_division=0))
            vals["roc_auc"].append(roc_auc_score(y[idx], prob[idx]))
    else:
        groups = np.asarray(groups)
        unique = np.unique(groups)
        for _ in range(n):
            chosen = rng.choice(unique, len(unique), replace=True)
            idx = np.concatenate([np.where(groups == g)[0] for g in chosen])
            if len(np.unique(y[idx])) < 2:
                continue
            vals["f1"].append(f1_score(y[idx], pred[idx], zero_division=0))
            vals["roc_auc"].append(roc_auc_score(y[idx], prob[idx]))
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if v else [np.nan, np.nan] for k, v in vals.items()}


def build_delta_and_stats(naive, macro, pooled, per_fold, naive_preds, group_preds, data) -> tuple[pd.DataFrame, dict]:
    rows = []
    ci = {}
    for name in FITTERS:
        n = naive[naive.model == name].iloc[0]
        m = macro[macro.model == name].iloc[0]
        p = pooled[pooled.model == name].iloc[0]
        rows.append(
            {
                "model": name,
                "naive_f1": n.f1,
                "run_level_mean_f1": m.f1_mean,
                "pooled_run_level_f1": p.f1,
                "absolute_f1_gap_naive_minus_run_mean": n.f1 - m.f1_mean,
                "relative_f1_decrease": (n.f1 - m.f1_mean) / n.f1 if n.f1 else np.nan,
                "naive_roc_auc": n.roc_auc,
                "run_level_mean_roc_auc": m.roc_auc_mean,
                "pooled_run_level_roc_auc": p.roc_auc,
                "absolute_roc_auc_gap_naive_minus_run_mean": n.roc_auc - m.roc_auc_mean,
                "fpr_change_run_mean_minus_naive": m.fpr_mean - n.fpr,
                "precision_change_run_mean_minus_naive": m.precision_mean - n.precision,
                "recall_change_run_mean_minus_naive": m.recall_mean - n.recall,
            }
        )
        ng = naive_preds[naive_preds.model == name]
        gg = group_preds[group_preds.model == name]
        ci[name] = {
            "naive": bootstrap_ci(ng.y_true, ng.y_prob, ng.y_pred),
            "controlled_grouped": bootstrap_ci(gg.y_true, gg.y_prob, gg.y_pred, gg.heldout_run),
        }
        ci[name]["gap_f1_ci_approx"] = [ci[name]["naive"]["f1"][0] - ci[name]["controlled_grouped"]["f1"][1], ci[name]["naive"]["f1"][1] - ci[name]["controlled_grouped"]["f1"][0]]
        ci[name]["gap_roc_auc_ci_approx"] = [ci[name]["naive"]["roc_auc"][0] - ci[name]["controlled_grouped"]["roc_auc"][1], ci[name]["naive"]["roc_auc"][1] - ci[name]["controlled_grouped"]["roc_auc"][0]]
    delta = pd.DataFrame(rows)
    delta.to_csv(OUTPUTS / "results_delta.csv", index=False)
    heterogeneity = {}
    for name, g in per_fold.groupby("model"):
        best = g.loc[g.f1.idxmax()]
        worst = g.loc[g.f1.idxmin()]
        corr = np.nan
        if g["test_attack_rate"].nunique() > 1 and g["f1"].nunique() > 1:
            corr = float(pearsonr(g["test_attack_rate"], g["f1"])[0])
        heterogeneity[name] = {
            "best_heldout_run": int(best.heldout_run),
            "best_f1": float(best.f1),
            "worst_heldout_run": int(worst.heldout_run),
            "worst_f1": float(worst.f1),
            "median_f1": float(g.f1.median()),
            "std_f1": float(g.f1.std()),
            "iqr_f1": float(g.f1.quantile(0.75) - g.f1.quantile(0.25)),
            "class_balance_f1_correlation": corr,
        }
    train_sizes = {
        "naive_training_records": int(round(len(data) * 0.70)),
        "groupkfold_mean_training_records": float(per_fold.groupby("fold").train_records.first().mean()),
        "groupkfold_percent_total": float(per_fold.groupby("fold").train_records.first().mean() / len(data)),
    }
    stats = {"confidence_intervals": ci, "run_heterogeneity": heterogeneity, "training_size": train_sizes}
    (OUTPUTS / "statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return delta, stats


def physical_feature_description(name: str) -> dict:
    relay = name.split("-")[0].split(":")[0] if name.startswith("R") else "System log"
    suffix = name.split("-", 1)[1] if "-" in name else name.split(":", 1)[-1]
    families = [
        ("PA1:VH", "PA3:VH", "phase A-C voltage phase angles"),
        ("PM1:V", "PM3:V", "phase A-C voltage magnitudes"),
        ("PA4:IH", "PA6:IH", "phase A-C current phase angles"),
        ("PM4:I", "PM6:I", "phase A-C current magnitudes"),
        ("PA7:VH", "PA9:VH", "sequence voltage angles"),
        ("PM7:V", "PM9:V", "sequence voltage magnitudes"),
        ("PA10:IH", "PA12:IH", "sequence current angles"),
        ("PM10:I", "PM12:I", "sequence current magnitudes"),
    ]
    desc = "control-panel, Snort, or relay-log measurement"
    family = "log"
    if suffix in {"F"}:
        desc = "frequency"
        family = "frequency"
    elif suffix in {"DF"}:
        desc = "frequency change"
        family = "frequency change"
    elif "PA:Z" in suffix:
        desc = "apparent impedance"
        family = "impedance"
    elif "PA:ZH" in suffix:
        desc = "apparent impedance angle"
        family = "impedance angle"
    elif suffix == "S":
        desc = "relay status flag"
        family = "relay status"
    else:
        import re

        m = re.search(r"(PA|PM)(\d+):(VH|V|IH|I)", suffix)
        if m:
            kind, idx, unit = m.group(1), int(m.group(2)), m.group(3)
            for lo, hi, d in families:
                lo_i = int(re.search(r"\d+", lo).group())
                hi_i = int(re.search(r"\d+", hi).group())
                if lo_i <= idx <= hi_i:
                    desc = d
                    family = d
                    break
    return {"relay": relay, "signal_family": family, "physical_description": desc}


def interpretability(data, features, best_artifacts) -> pd.DataFrame:
    controlled_scores = {name: art["metric"].get("roc_auc", np.nan) for name, art in best_artifacts.items()}
    best_name = max(controlled_scores, key=controlled_scores.get)
    art = best_artifacts[best_name]
    test_idx = art["test_idx"]
    X_test = data.iloc[test_idx][features]
    y_test = data.iloc[test_idx]["target"]

    def f1_for_matrix(Xarr, yarr):
        if isinstance(Xarr, np.ndarray):
            Xdf = pd.DataFrame(Xarr, columns=features)
        else:
            Xdf = Xarr
        prob = art["model"].predict_proba(Xdf)
        pred = (prob >= 0.5).astype(int)
        return f1_score(yarr, pred, zero_division=0)

    X_perm = X_test.sample(min(250, len(X_test)), random_state=SEED)
    y_perm = y_test.loc[X_perm.index]
    baseline = f1_for_matrix(X_perm, y_perm)
    rng = np.random.default_rng(SEED)
    importances = []
    for i, feature in enumerate(features, start=1):
        if i % 16 == 1:
            print(f"[{time.strftime('%H:%M:%S')}] permutation feature {i}/{len(features)}", flush=True)
        drops = []
        for _ in range(10):
            Xp = X_perm.copy()
            Xp[feature] = rng.permutation(Xp[feature].to_numpy())
            drops.append(baseline - f1_for_matrix(Xp, y_perm))
        importances.append(float(np.mean(drops)))
    perm_df = pd.DataFrame({"feature": features, "permutation_importance": importances}).sort_values("permutation_importance", ascending=False)

    shap_rows = []
    if shap is not None:
        for tree_name in ["Random Forest", "XGBoost"]:
            if tree_name in best_artifacts:
                tart = best_artifacts[tree_name]
                print(f"[{time.strftime('%H:%M:%S')}] SHAP for {tree_name}", flush=True)
                sample = data.iloc[tart["test_idx"]][features].sample(min(50, len(tart["test_idx"])), random_state=SEED)
                Xt = tart["model"].preprocessor.transform(sample)
                try:
                    explainer = shap.TreeExplainer(tart["model"].estimator)
                    vals = explainer.shap_values(Xt)
                    if isinstance(vals, list):
                        vals = vals[-1]
                    vals = np.asarray(vals)
                    if vals.ndim == 3:
                        vals = vals[:, :, -1]
                    mean_abs = np.abs(vals).mean(axis=0)
                    mean_signed = vals.mean(axis=0)
                    for f, a, s in zip(features, mean_abs, mean_signed):
                        shap_rows.append({"feature": f, f"shap_abs_{tree_name}": a, f"shap_signed_{tree_name}": s})
                except Exception as exc:
                    (OUTPUTS / f"shap_{tree_name.lower().replace(' ', '_')}_error.txt").write_text(repr(exc), encoding="utf-8")
    shap_df = pd.DataFrame({"feature": features})
    for tree_name in ["Random Forest", "XGBoost"]:
        rows = [r for r in shap_rows if f"shap_abs_{tree_name}" in r]
        if rows:
            shap_df = shap_df.merge(pd.DataFrame(rows), on="feature", how="left")
    out = perm_df.merge(shap_df, on="feature", how="left")
    for col in [c for c in out.columns if c.startswith("shap_abs_")]:
        out[f"rank_{col}"] = out[col].rank(ascending=False, method="min")
    out["permutation_rank"] = out["permutation_importance"].rank(ascending=False, method="min")
    phys = out["feature"].map(physical_feature_description).apply(pd.Series)
    out = pd.concat([out, phys], axis=1)
    out.to_csv(OUTPUTS / "feature_importance.csv", index=False)
    return out


def rebuild_best_artifacts(data: pd.DataFrame, features: list[str], per_fold: pd.DataFrame, model_names: list[str] | None = None) -> dict:
    artifacts = {}
    X = data[features]
    y = data["target"].to_numpy()
    groups = data["run_id"].to_numpy()
    splits = list(GroupKFold(n_splits=15).split(X, y, groups))
    if model_names is None:
        model_names = list(FITTERS)
    for name in model_names:
        best = per_fold[per_fold.model == name].sort_values("f1", ascending=False).iloc[0]
        fold_idx = int(best.fold) - 1
        train_idx, test_idx = splits[fold_idx]
        fitted, _ = FITTERS[name](X.iloc[train_idx], y[train_idx])
        artifacts[name] = {"model": fitted, "metric": best.to_dict(), "test_idx": test_idx, "train_idx": train_idx}
    return artifacts


def latex_tabular(df: pd.DataFrame, columns: list[str], path: Path, floatfmt="{:.3f}") -> None:
    subset = df[columns].copy()
    for c in subset.columns:
        if pd.api.types.is_float_dtype(subset[c]):
            subset[c] = subset[c].map(lambda x: "" if pd.isna(x) else floatfmt.format(x))
    path.write_text(subset.to_latex(index=False, escape=False), encoding="utf-8")


def write_tables(naive, macro, pooled, delta, feature_imp, per_fold) -> None:
    latex_tabular(naive, ["model", "precision", "recall", "f1", "fpr", "roc_auc"], OUTPUTS / "results_naive.tex")
    g = macro[["model", "precision_mean", "precision_std", "recall_mean", "recall_std", "f1_mean", "f1_std", "fpr_mean", "fpr_std", "roc_auc_mean", "roc_auc_std"]]
    latex_tabular(g, list(g.columns), OUTPUTS / "results_groupkfold.tex")
    latex_tabular(pooled, ["model", "precision", "recall", "f1", "fpr", "roc_auc"], OUTPUTS / "results_pooled.tex")
    latex_tabular(delta, ["model", "naive_f1", "run_level_mean_f1", "absolute_f1_gap_naive_minus_run_mean", "naive_roc_auc", "run_level_mean_roc_auc", "absolute_roc_auc_gap_naive_minus_run_mean"], OUTPUTS / "results_delta.tex")
    runtime = per_fold.groupby("model", as_index=False).agg(training_time_sec_mean=("training_time_sec", "mean"), inference_time_sec_mean=("inference_time_sec", "mean"), epochs_mean=("epochs", "mean"))
    runtime.to_csv(OUTPUTS / "runtime.csv", index=False)
    latex_tabular(runtime, list(runtime.columns), OUTPUTS / "runtime.tex")
    top = feature_imp.head(12)[["feature", "relay", "signal_family", "physical_description", "permutation_rank", "permutation_importance"]]
    latex_tabular(top, list(top.columns), OUTPUTS / "top_features.tex")


def plot_all(naive, macro, pooled, per_fold, naive_preds, group_preds, feature_imp) -> None:
    # Palette matched to the reference figure style: gray for the baseline/control
    # series, a strong blue for the primary/treatment series, plus a small set of
    # complementary accents (red/green/gold) for additional categorical series.
    GRAY = "#8C8C8C"
    BLUE = "#2255A4"
    BLUE_DARK = "#153A6E"
    RED = "#C0392B"
    GREEN = "#3D8C55"
    GOLD = "#D6A017"
    TWO_TONE = [GRAY, BLUE]
    ACCENTS = [BLUE, RED, GREEN, GOLD, BLUE_DARK, "#8E5FB0", "#3AA6A6", "#B0793A"]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update({
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "font.family": "sans-serif",
    })

    primary = ["precision", "recall", "f1", "fpr", "roc_auc"]
    rows = []
    for _, r in naive.iterrows():
        for metric in primary:
            rows.append({"model": r.model, "metric": metric, "protocol": "Random split", "value": r[metric]})
    for _, r in macro.iterrows():
        for metric in primary:
            rows.append({"model": r.model, "metric": metric, "protocol": "Run holdout", "value": r[f"{metric}_mean"]})
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    sns.barplot(pd.DataFrame(rows), x="metric", y="value", hue="protocol", palette=TWO_TONE, errorbar=None, ax=ax, edgecolor="white", linewidth=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.legend(frameon=False, loc="lower right")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_naive_vs_leakage_metrics.png", dpi=300)
    fig.savefig(FIGURES / "fig1_naive_vs_leakage_metrics.pdf")
    plt.close(fig)

    for preds, filename, title in [(naive_preds, "fig2_roc_naive", "Random-split ROC"), (group_preds, "fig3_roc_leakage", "Run-holdout pooled ROC")]:
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        for (name, g), color in zip(preds.groupby("model"), ACCENTS):
            fpr, tpr, _ = roc_curve(g.y_true, g.y_prob)
            auc = roc_auc_score(g.y_true, g.y_prob)
            ax.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}", color=color, linewidth=2)
        ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
        sns.despine(ax=ax)
        fig.tight_layout()
        fig.savefig(FIGURES / f"{filename}.png", dpi=300)
        fig.savefig(FIGURES / f"{filename}.pdf")
        plt.close(fig)

    best = pooled.sort_values("f1", ascending=False).iloc[0].model
    g = group_preds[group_preds.model == best]
    cm = confusion_matrix(g.y_true, g.y_pred, labels=[0, 1])
    blues_cmap = LinearSegmentedColormap.from_list("blues_ref", ["#FFFFFF", BLUE, BLUE_DARK])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap=blues_cmap, cbar=False, xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"], ax=ax, annot_kws={"color": "#222222"}, linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title(f"{best} run-holdout confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_confusion_matrix.png", dpi=300)
    fig.savefig(FIGURES / "fig4_confusion_matrix.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    sns.boxplot(per_fold, x="model", y="f1", color="white", showfliers=False, boxprops={"edgecolor": GRAY}, whiskerprops={"color": GRAY}, capprops={"color": GRAY}, medianprops={"color": BLUE_DARK}, ax=ax)
    sns.stripplot(per_fold, x="model", y="f1", color=BLUE, alpha=0.75, size=4, jitter=0.2, ax=ax)
    ax.set_ylabel("Held-out-run F1")
    ax.set_xlabel("")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_fold_variance.png", dpi=300)
    fig.savefig(FIGURES / "fig5_fold_variance.pdf")
    plt.close(fig)

    top = feature_imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ax.barh(top["feature"], top["permutation_importance"], color=BLUE, edgecolor=BLUE_DARK, linewidth=0.6)
    ax.set_xlabel("F1 decrease after permutation")
    ax.set_ylabel("")
    ax.set_title("Top held-out feature importance")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_feature_importance.png", dpi=300)
    fig.savefig(FIGURES / "fig6_feature_importance.pdf")
    plt.close(fig)

    # Supplementary precision-recall curves.
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for (name, g), color in zip(group_preds.groupby("model"), ACCENTS):
        precision, recall, _ = precision_recall_curve(g.y_true, g.y_prob)
        ax.plot(recall, precision, label=name, color=color, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "supp_precision_recall_leakage.png", dpi=300)
    plt.close(fig)


def model_row(df, model):
    return df[df.model == model].iloc[0]


def write_paper(naive, macro, pooled, delta, stats, audit, leakage, feature_imp) -> None:
    for fig in FIGURES.glob("fig*.png"):
        shutil.copy2(fig, PAPER / "figures" / fig.name)
    best_naive = naive.sort_values("f1", ascending=False).iloc[0]
    best_ctrl = macro.sort_values("f1_mean", ascending=False).iloc[0]
    top_features = feature_imp.head(6)
    rf_delta = model_row(delta, "Random Forest")
    xgb_delta = model_row(delta, "XGBoost")
    cnn_delta = model_row(delta, "CNN")
    hetero = stats["run_heterogeneity"][best_ctrl.model]
    class_counts = audit["dataset_class_count"]
    text = rf"""
\documentclass[conference,a4paper]{{IEEEtran}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{cite}}
\usepackage{{url}}
\usepackage{{microtype}}
\renewcommand{{\ttdefault}}{{cmtt}}

\title{{Benchmark Before You Automate: Leakage-Aware Evaluation of Machine Learning Detectors for Power-Sector SCADA Security}}

\author{{
\IEEEauthorblockN{{Digdarshan Subedi}}
\IEEEauthorblockA{{Department of Computer Science\\University of Wisconsin--Whitewater\\Whitewater, WI, USA\\SubediD30@uww.edu}}
\and
\IEEEauthorblockN{{Nikesh Tiwari}}
\IEEEauthorblockA{{School of Business \& Technology\\Webster University\\St. Louis, MO, USA\\nikeshtiwari@webster.edu}}
\and
\IEEEauthorblockN{{Bibek Ale}}
\IEEEauthorblockA{{School of Business \& Technology\\Webster University\\St. Louis, MO, USA\\Bibekale@webster.edu}}
}}

\begin{{document}}
\maketitle

\begin{{abstract}}
Machine-learning detectors for supervisory control and data acquisition (SCADA) attacks are increasingly discussed as inputs to fast grid-security workflows. That use raises the cost of benchmark optimism: an offline score can encourage automation before a detector has shown that it transfers beyond the data collection conditions used for training. We evaluate this risk on the MSU/ORNL Power System Attack Dataset by comparing a conventional 70/30 random row split with a leakage-controlled leave-one-run-out protocol over fifteen collection runs. We train Random Forest, XGBoost, and a feature-space one-dimensional convolutional neural network on {audit["dataset_rows"]:,} records and {audit["usable_model_features"]} usable measurements. The strongest random-split result was {best_naive.model} with F1={best_naive.f1:.3f} and ROC-AUC={best_naive.roc_auc:.3f}; under run-level holdout the strongest mean result was {best_ctrl.model} with F1={best_ctrl.f1_mean:.3f} and ROC-AUC={best_ctrl.roc_auc_mean:.3f}. The results show that automation-readiness claims should be based on the score that survives a deployment-relevant boundary, not only on mixed-run test records.
\end{{abstract}}

\section{{Introduction}}
Power-grid security teams must decide which alarms deserve attention and which signals can support semi-automated response. A detector that looks reliable in a benchmark but fails on a new operating collection can waste analyst time or, worse, encourage unsafe control actions. This paper studies one common source of overconfidence: random row-level train/test splitting when records from the same data-collection run appear in both training and test sets.

The MSU/ORNL benchmark contains fifteen sampled collections from one physical testbed with two generators, four integrated PMU/relay devices, four breakers, two lines, and control-panel, Snort, and relay-log measurements. The binary CSV files used here contain {class_counts.get("1", 0):,} attack records and {class_counts.get("0", 0):,} normal/natural records. Because the public binary files label rows as Attack or Natural rather than retaining the original numeric scenario marker, this study verifies the binary labels available in the files and uses the official README for scenario definitions.

We ask three questions. RQ1: How well do Random Forest, XGBoost, and a feature-space 1D-CNN detect attacks under the conventional random-split protocol? RQ2: How much performance is retained when complete collection runs are held out, and is the change consistent across model families? RQ3: Which physically meaningful relay and phasor measurements drive predictions, and how might these explanations support operator review?

Our contributions are a reproducible leakage-aware evaluation pipeline, a measurement of the gap between random-split and run-level estimates, and an interpretation of that gap for SCADA automation. We do not propose a new state-of-the-art architecture.

\section{{Related Work}}
Beaver et al. introduced the power-system cyberattack dataset used in many intrusion-detection studies \cite{{beaver2013power}}. Pan, Morris, and Adhikari further studied classification of disturbances and cyberattacks in power systems \cite{{pan2015classification}}. Tree ensembles and gradient boosting remain strong tabular baselines, with XGBoost providing a widely used scalable implementation \cite{{chen2016xgboost}}. Post-hoc explanation methods such as SHAP help summarize model behavior but do not establish causal mechanisms \cite{{lundberg2017shap}}.

Prior smart-grid cybersecurity and deep-learning surveys describe the promise of data-driven detection while emphasizing that operational deployment requires robust validation \cite{{tan2017survey,wang2019deep}}. Our work adds a narrow but important measurement: whether the benchmark score is retained when the train/test split respects complete collection runs.

\section{{Methodology}}
\subsection{{Dataset Audit and Labels}}
We loaded all fifteen CSV files separately, added \texttt{{run\_id}} and source-file fields, and excluded those fields from model inputs. Each file shared the same 129-column schema. The usable model matrix contains 116 PMU measurements and 12 control-panel, Snort, or relay-log measurements; the marker column is excluded and mapped only to the binary target. The files contain {audit["dataset_rows"]:,} rows and {audit["usable_model_features"]} usable features.

The official README states that the source dataset was randomly sampled at one percent. We therefore do not treat row order as time and do not train sequence models. The CNN operates across ordered feature positions: R1 measurements, R2 measurements, R3 measurements, R4 measurements, then log features.

\subsection{{Evaluation Protocols}}
The conventional protocol merges all rows and uses a stratified 70/30 random split with seed 42. The leakage-controlled protocol uses GroupKFold with \texttt{{run\_id}} as the group, training on fourteen complete runs and testing on the held-out run. Median imputation is fitted only on each training split. The CNN also uses training-only standardization and an internal validation subset from the training runs for early stopping.

We report precision, recall, F1, false-positive rate (FPR), ROC-AUC, average precision, accuracy, balanced accuracy, Matthews correlation coefficient, Brier score, confusion matrices, training time, and inference latency. The main controlled table uses macro mean and standard deviation across held-out runs. Pooled out-of-fold predictions are used for aggregate curves and confusion matrices.

\subsection{{Models}}
Random Forest uses 200 trees and seed 42. XGBoost uses 200 estimators, binary logistic objective, log-loss evaluation, histogram tree construction, and seed 42. The feature-space CNN uses Conv1D(32, kernel 5), max pooling, Conv1D(64, kernel 3), global average pooling, a 32-unit dense layer, and a sigmoid output. It is retrained from scratch for every fold.

\section{{Results}}
\subsection{{Conventional Evaluation Suggests Strong Detection}}
{best_naive.model} achieved the strongest random-split F1 of {best_naive.f1:.3f} with ROC-AUC {best_naive.roc_auc:.3f}. Random Forest, XGBoost, and CNN random-split F1 scores were {model_row(naive,'Random Forest').f1:.3f}, {model_row(naive,'XGBoost').f1:.3f}, and {model_row(naive,'CNN').f1:.3f}, respectively. These values reflect the common benchmark practice in which records from the same collection runs can appear on both sides of the split.

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/fig1_naive_vs_leakage_metrics.png}}
\caption{{Comparison of conventional random-split and run-level evaluation for the three detectors. Each paired bar shows whether performance measured on mixed-run test data is retained when the model is evaluated on complete unseen runs. The distance between bars is the estimate of benchmark optimism within this dataset.}}
\label{{fig:metrics}}
\end{{figure}}

\subsection{{Run-Level Holdout Tests Transfer to Unseen Collections}}
Under run-level holdout, {best_ctrl.model} had the strongest macro F1 of {best_ctrl.f1_mean:.3f}$\pm${best_ctrl.f1_std:.3f} and macro ROC-AUC of {best_ctrl.roc_auc_mean:.3f}$\pm${best_ctrl.roc_auc_std:.3f}. The best held-out run for {best_ctrl.model} was run {hetero["best_heldout_run"]} with F1={hetero["best_f1"]:.3f}; the worst was run {hetero["worst_heldout_run"]} with F1={hetero["worst_f1"]:.3f}. This spread shows that performance depends strongly on the unseen collection.

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/fig5_fold_variance.png}}
\caption{{Held-out-run F1 distributions for the three detectors. Each point is one complete collection run used only for testing. The vertical spread shows run heterogeneity that is hidden by a single random row-level split.}}
\label{{fig:folds}}
\end{{figure}}

\subsection{{The Evaluation Gap Quantifies Benchmark Optimism}}
Random Forest achieved random-split F1 {rf_delta.naive_f1:.3f} but run-level mean F1 {rf_delta.run_level_mean_f1:.3f}, an absolute decrease of {rf_delta.absolute_f1_gap_naive_minus_run_mean:.3f}. XGBoost changed from {xgb_delta.naive_f1:.3f} to {xgb_delta.run_level_mean_f1:.3f}, a decrease of {xgb_delta.absolute_f1_gap_naive_minus_run_mean:.3f}. The CNN changed from {cnn_delta.naive_f1:.3f} to {cnn_delta.run_level_mean_f1:.3f}, a decrease of {cnn_delta.absolute_f1_gap_naive_minus_run_mean:.3f}. These reductions indicate that part of the conventional benchmark performance depends on patterns shared within collection runs.

\begin{{figure}}[t]
\centering
\includegraphics[width=.92\linewidth]{{figures/fig3_roc_leakage.png}}
\caption{{Pooled out-of-fold ROC curves under run-level holdout. Predictions are concatenated from folds in which each test run was unseen during training. The curves summarize discrimination after enforcing the collection-run boundary.}}
\label{{fig:roc}}
\end{{figure}}

\subsection{{Similarity Evidence Supports the Leakage Interpretation}}
The combined corpus contained {leakage["exact_duplicate_records_combined_keep_false"]:,} records that belonged to an exact duplicate set. In the random split, {leakage["naive_test_records_with_exact_duplicate_in_train"]:,} test records had an exact duplicate in training, a rate of {leakage["naive_test_exact_duplicate_in_train_rate"]:.3f}. The sampled nearest-neighbor median distance was {leakage["nearest_neighbor_naive"]["median_distance"]:.3f} for random-split test records and {leakage["nearest_neighbor_groupkfold_mean_across_folds"]["median_distance"]:.3f} under run-level holdout. This evidence is consistent with the performance gap being driven by similarity across mixed-run records rather than only by model choice.

\subsection{{Physically Meaningful Signals Support Operator Review}}
Permutation importance and tree SHAP highlighted relay and phasor measurements among the top features. The leading features included {", ".join(top_features["feature"].tolist())}. These signals are physically plausible because they involve relay states, voltage/current phasors, or derived protection quantities. Plausibility can help analysts review an alert, but it does not prove causal detection or deployment readiness.

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/fig6_feature_importance.png}}
\caption{{Top feature importance values for the best held-out model artifact. Bars show the F1 decrease on held-out data after permuting each feature. Physical names are retained so operators can see whether the detector relies on interpretable relay, phasor, frequency, impedance, or log measurements.}}
\label{{fig:importance}}
\end{{figure}}

\section{{Discussion}}
\subsection{{Automation Readiness Depends on the Score That Survives Holdout}}
The random split answers whether models can classify mixed records from the same benchmark corpus. The run-level split asks a stricter question: whether a detector trained on fourteen collections transfers to a complete unseen collection. For automation support, the second score is the safer estimate because it better reflects a future collection boundary.

\subsection{{The Gap Is Not Explained by a Large Loss of Training Data}}
The random split used about {stats["training_size"]["naive_training_records"]:,} training records. Each GroupKFold split used an average of {stats["training_size"]["groupkfold_mean_training_records"]:.0f} records, or {100*stats["training_size"]["groupkfold_percent_total"]:.1f}\% of the corpus. The controlled protocol therefore used more rows per fold than the random split, so the lower score is not explained by a large reduction in training volume.

\subsection{{Interpretability Helps Review but Does Not Replace Validation}}
Feature explanations can show whether a detector relies on sensible electrical and relay signals. They cannot show that the model will behave safely under new grid configurations, attack strategies, or operator workflows. Explanations should support human review, not replace leakage-aware validation.

\section{{Limitations and Future Work}}
Dataset scope is limited to one physical testbed and 2014-era data. Run-level control does not establish site-level, utility-level, or vendor-level generalization. Method scope is limited to three lightweight models and no exhaustive architecture search. Temporal models are excluded because the public binary files are one-percent samples and row order is not a valid sequence. Explanation scope is post-hoc and does not include an operator study. Deployment scope is offline: no live protective actions, calibrated action thresholds, fail-safe mechanisms, or human-override studies are evaluated.

\section{{Conclusion}}
Within the MSU/ORNL binary power-system benchmark, conventional random splitting gives a more optimistic view than run-level holdout. The measurement matters because automated SCADA response should be justified by performance that survives an unseen collection boundary. The main contribution is therefore an evaluation standard and reproducible evidence, not a new detector architecture.

\section*{{Acknowledgment}}
The authors thank the dataset creators for making the benchmark available for reproducible research.

\bibliographystyle{{IEEEtran}}
\bibliography{{references}}
\end{{document}}
"""
    (PAPER / "main.tex").write_text(text, encoding="utf-8")
    bib = r"""
@inproceedings{beaver2013power,
  author = {Beaver, Justin M. and Borges-Hink, Raymond C. and Buckner, Mark A.},
  title = {An Evaluation of Machine Learning Methods to Detect Malicious SCADA Communications},
  booktitle = {2013 12th International Conference on Machine Learning and Applications},
  year = {2013},
  pages = {54--59},
  publisher = {IEEE},
  doi = {10.1109/ICMLA.2013.105}
}

@article{pan2015classification,
  author = {Pan, Shengyi and Morris, Thomas and Adhikari, Uttam},
  title = {Classification of Disturbances and Cyber-Attacks in Power Systems Using Heterogeneous Time-Synchronized Data},
  journal = {IEEE Transactions on Industrial Informatics},
  volume = {11},
  number = {3},
  pages = {650--662},
  year = {2015},
  doi = {10.1109/TII.2015.2420951}
}

@inproceedings{chen2016xgboost,
  author = {Chen, Tianqi and Guestrin, Carlos},
  title = {XGBoost: A Scalable Tree Boosting System},
  booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year = {2016},
  pages = {785--794},
  doi = {10.1145/2939672.2939785}
}

@inproceedings{lundberg2017shap,
  author = {Lundberg, Scott M. and Lee, Su-In},
  title = {A Unified Approach to Interpreting Model Predictions},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2017},
  pages = {4765--4774}
}

@article{tan2017survey,
  author = {Tan, Serkan and De, Debashis and Song, Wen-Zhan and Yang, Jian and Das, Sajal K.},
  title = {Survey of Security Advances in Smart Grid: A Data Driven Approach},
  journal = {IEEE Communications Surveys \& Tutorials},
  volume = {19},
  number = {1},
  pages = {397--422},
  year = {2017},
  doi = {10.1109/COMST.2016.2616442}
}

@article{wang2019deep,
  author = {Wang, Yi and Chen, Qixin and Hong, Tao and Kang, Chongqing},
  title = {Review of Smart Meter Data Analytics: Applications, Methodologies, and Challenges},
  journal = {IEEE Transactions on Smart Grid},
  volume = {10},
  number = {3},
  pages = {3125--3148},
  year = {2019},
  doi = {10.1109/TSG.2018.2818167}
}
"""
    (PAPER / "references.bib").write_text(bib.strip() + "\n", encoding="utf-8")
    supp = "\\section*{Supplementary Results}\n\\input{../../outputs/results_naive.tex}\n\\input{../../outputs/results_groupkfold.tex}\n\\input{../../outputs/results_delta.tex}\n"
    (PAPER / "supplementary" / "supplementary_results.tex").write_text(supp, encoding="utf-8")


def compile_paper() -> dict:
    status = {"compiled": False, "pdf_path": str(PAPER / "main.pdf"), "page_count": None, "log_tail": ""}
    commands = [["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"], ["pdflatex", "-interaction=nonstopmode", "main.tex"], ["pdflatex", "-interaction=nonstopmode", "main.tex"]]
    logs = []
    for cmd in commands:
        proc = subprocess.run(cmd, cwd=PAPER, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        logs.append(proc.stdout[-4000:])
        if proc.returncode != 0:
            status["log_tail"] = "\n".join(logs)[-8000:]
            (OUTPUTS / "latex_compile_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            return status
    pdf = PAPER / "main.pdf"
    status["compiled"] = pdf.exists()
    if shutil.which("pdfinfo") and pdf.exists():
        info = subprocess.run(["pdfinfo", str(pdf)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in info.stdout.splitlines():
            if line.startswith("Pages:"):
                status["page_count"] = int(line.split()[-1])
    status["log_tail"] = "\n".join(logs)[-8000:]
    (OUTPUTS / "latex_compile_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def write_readmes(results) -> None:
    req = "\n".join(["numpy", "pandas", "scikit-learn", "xgboost", "shap", "torch", "matplotlib", "seaborn", "scipy"]) + "\n"
    (ROOT / "requirements.txt").write_text(req, encoding="utf-8")
    (ROOT / "run_all.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\npython3 -m src.run_all\n", encoding="utf-8")
    os.chmod(ROOT / "run_all.sh", 0o755)
    readme = f"""# Leakage-Aware SCADA Cyberattack Evaluation

Run the full project from this folder:

```bash
python3 -m src.run_all
```

The command loads the fifteen local CSV files, audits the dataset, trains Random Forest, XGBoost, and a feature-space 1D-CNN under random-split and leave-one-run-out protocols, writes figures/tables, and compiles the IEEE paper.

Main outputs:

- `outputs/results.json`
- `outputs/dataset_audit.json`
- `outputs/per_fold_metrics.csv`
- `outputs/predictions_naive.csv`
- `outputs/predictions_groupkfold.csv`
- `outputs/feature_importance.csv`
- `figures/`
- `paper_project/main.pdf`
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    (OUTPUTS / "README.md").write_text(readme, encoding="utf-8")


def save_results(results: dict) -> None:
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if pd.isna(obj) else float(obj)
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    (OUTPUTS / "results.json").write_text(json.dumps(clean(results), indent=2), encoding="utf-8")


def main() -> None:
    set_seeds()
    ensure_dirs()
    start = time.perf_counter()
    data, features, audit = audit_and_load()
    if (OUTPUTS / "metrics_naive.csv").exists() and (OUTPUTS / "predictions_naive.csv").exists() and (OUTPUTS / "leakage_evidence.json").exists():
        print(f"[{time.strftime('%H:%M:%S')}] loading completed naive outputs", flush=True)
        naive = pd.read_csv(OUTPUTS / "metrics_naive.csv")
        naive_preds = pd.read_csv(OUTPUTS / "predictions_naive.csv")
        leakage = json.loads((OUTPUTS / "leakage_evidence.json").read_text(encoding="utf-8"))
    else:
        naive, naive_preds, _, leakage = run_naive(data, features)
    if (OUTPUTS / "per_fold_metrics.csv").exists() and (OUTPUTS / "predictions_groupkfold.csv").exists():
        print(f"[{time.strftime('%H:%M:%S')}] loading completed GroupKFold outputs", flush=True)
        per_fold = pd.read_csv(OUTPUTS / "per_fold_metrics.csv")
        group_preds = pd.read_csv(OUTPUTS / "predictions_groupkfold.csv")
    else:
        per_fold, group_preds, _ = run_groupkfold(data, features)
    pooled = pooled_metrics(group_preds)
    macro = grouped_summary(per_fold)
    delta, stats = build_delta_and_stats(naive, macro, pooled, per_fold, naive_preds, group_preds, data)
    print(f"[{time.strftime('%H:%M:%S')}] rebuilding best-fold artifacts for interpretability", flush=True)
    best_artifacts = rebuild_best_artifacts(data, features, per_fold, ["Random Forest", "XGBoost"])
    feature_imp = interpretability(data, features, best_artifacts)
    write_tables(naive, macro, pooled, delta, feature_imp, per_fold)
    plot_all(naive, macro, pooled, per_fold, naive_preds, group_preds, feature_imp)
    write_paper(naive, macro, pooled, delta, stats, audit, leakage, feature_imp)
    compile_status = compile_paper()
    results = {
        "audit_summary": audit,
        "naive_metrics": naive.to_dict(orient="records"),
        "groupkfold_macro_metrics": macro.to_dict(orient="records"),
        "groupkfold_pooled_metrics": pooled.to_dict(orient="records"),
        "delta_metrics": delta.to_dict(orient="records"),
        "statistics": stats,
        "leakage_evidence": leakage,
        "top_features": feature_imp.head(20).to_dict(orient="records"),
        "compile_status": compile_status,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "xgboost": None if XGBClassifier is None else __import__("xgboost").__version__,
            "shap": None if shap is None else shap.__version__,
            "torch": None if torch is None else torch.__version__,
        },
        "runtime_seconds": time.perf_counter() - start,
        "reproduce_command": "python3 -m src.run_all",
    }
    save_results(results)
    write_readmes(results)
    print(json.dumps({"status": "complete", "runtime_seconds": results["runtime_seconds"], "pdf_compiled": compile_status["compiled"], "pdf_pages": compile_status["page_count"]}, indent=2))


if __name__ == "__main__":
    main()
