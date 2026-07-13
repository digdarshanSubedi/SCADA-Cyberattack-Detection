"""Phase D: test-time-only degradation transforms.

Every function takes a test-fold DataFrame and returns a degraded copy.
None of these touch training data or refit any preprocessing — the cached
fold model's already-fitted imputer (median learned on that fold's training
runs) absorbs the induced NaNs at prediction time, which is exactly the
"feature unavailable at decision time" scenario. Never phrase results as
packet delay; use "feature-group unavailability at decision time."
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RELAY_PREFIXES = {"R1": ("R1-", "R1:"), "R2": ("R2-", "R2:"), "R3": ("R3-", "R3:"), "R4": ("R4-", "R4:")}

FEATURE_GROUPS = {
    "snort": lambda cols: [c for c in cols if c.startswith("snort_log")],
    "control_panel": lambda cols: [c for c in cols if c.startswith("control_panel_log")],
    "relay_log": lambda cols: [c for c in cols if c.startswith("relay") and c.endswith("_log")],
    "relay_status": lambda cols: [c for c in cols if c.endswith(":S")],
    "pmu_voltage": lambda cols: [c for c in cols if ":V" in c or "VH" in c],
    "pmu_current": lambda cols: [c for c in cols if ":I" in c or "IH" in c],
    "freq_impedance": lambda cols: [c for c in cols if c.endswith(":F") or c.endswith(":DF") or "PA:Z" in c],
}


def random_missingness(X: pd.DataFrame, features: list[str], rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Xd = X.copy()
    mask = rng.random((len(X), len(features))) < rate
    arr = Xd[features].to_numpy(dtype=float, copy=True)
    arr[mask] = np.nan
    Xd[features] = arr
    return Xd


def group_unavailable(X: pd.DataFrame, group_name: str) -> pd.DataFrame:
    cols = FEATURE_GROUPS[group_name](list(X.columns))
    Xd = X.copy()
    if cols:
        Xd[cols] = np.nan
    return Xd


def logs_unavailable(X: pd.DataFrame) -> pd.DataFrame:
    """All cyber/log groups unavailable together (draft's 'logs delayed' placement row):
    control-panel + relay-log + Snort, i.e. the 12 non-PMU features."""
    cols = []
    for g in ("control_panel", "relay_log", "snort"):
        cols.extend(FEATURE_GROUPS[g](list(X.columns)))
    Xd = X.copy()
    if cols:
        Xd[cols] = np.nan
    return Xd


def all_pmu_unavailable(X: pd.DataFrame) -> pd.DataFrame:
    """All 116 PMU features unavailable together (complements logs_unavailable's
    12 cyber/log features): pmu_voltage (48) + pmu_current (48) + freq_impedance (16)
    + relay_status (4) = 116, verified against feature_source_map.csv's pmu count."""
    cols = []
    for g in ("pmu_voltage", "pmu_current", "freq_impedance", "relay_status"):
        cols.extend(FEATURE_GROUPS[g](list(X.columns)))
    Xd = X.copy()
    if cols:
        Xd[cols] = np.nan
    return Xd


def relay_loss(X: pd.DataFrame, relay: str) -> pd.DataFrame:
    prefixes = RELAY_PREFIXES[relay]
    cols = [c for c in X.columns if c.startswith(prefixes)]
    Xd = X.copy()
    if cols:
        Xd[cols] = np.nan
    return Xd


def gaussian_noise(X: pd.DataFrame, features: list[str], sigma_frac: float, train_std: pd.Series, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Xd = X.copy()
    noise = rng.normal(0.0, sigma_frac, size=(len(X), len(features))) * train_std[features].to_numpy()
    Xd[features] = Xd[features].to_numpy(dtype=float) + noise
    return Xd


def quantize(X: pd.DataFrame, features: list[str], bits: int, train_min: pd.Series, train_max: pd.Series) -> pd.DataFrame:
    Xd = X.copy()
    levels = 2**bits - 1
    lo = train_min[features].to_numpy()
    hi = train_max[features].to_numpy()
    span = np.where(hi > lo, hi - lo, 1.0)
    vals = Xd[features].to_numpy(dtype=float)
    q = np.round((vals - lo) / span * levels)
    q = np.clip(q, 0, levels)
    Xd[features] = lo + (q / levels) * span
    return Xd
