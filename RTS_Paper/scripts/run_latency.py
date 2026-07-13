"""Phase C: batch-size-1 latency benchmarking + envelope verdicts.

Uses the fold-01 persisted model per model family (representative of that
architecture's per-record cost; hyperparameters are identical across folds).
Measures preprocessing, inference, and end-to-end latency separately at
batch size 1, with 500 warm-up + 5000 timed iterations, matching the draft's
stated protocol (line 210).
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import sklearn
import xgboost

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from RTS_Paper.scripts.data import load_dataset
from RTS_Paper.scripts.models import FittedRTSModel

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "RTS_Paper" / "outputs" / "models"
TIMINGS_DIR = ROOT / "RTS_Paper" / "outputs" / "timings"
METRICS_DIR = ROOT / "RTS_Paper" / "outputs" / "metrics"

WARMUP = 500
TIMED = 5000

# Envelope classes stated in the draft (E1/E2/E3), seconds.
ENVELOPES_MS = {"E1": 10.0, "E2": 16.7, "E3": 100.0}
D_AGG_MS = [2, 5, 10]  # stated modeling assumption for central aggregation delay, not measured

MODEL_SLUGS = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "Compact MLP": "compact_mlp",
}


def time_batch1(fn, n_warmup: int, n_timed: int) -> np.ndarray:
    for _ in range(n_warmup):
        fn()
    samples = np.empty(n_timed, dtype=np.int64)
    for i in range(n_timed):
        start = time.perf_counter_ns()
        fn()
        samples[i] = time.perf_counter_ns() - start
    return samples


def summarize(samples_ns: np.ndarray) -> dict:
    ms = samples_ns / 1e6
    return {
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "mean_ms": float(np.mean(ms)),
        "throughput_records_per_sec": float(1000.0 / np.median(ms)),
    }


def main() -> None:
    for d in [TIMINGS_DIR, METRICS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    data, features, meta = load_dataset()
    # one held-out record, never used for fitting any fold-01 model's training run
    x_row = data[features].iloc[[0]]

    rows = []
    raw_samples = []
    for model_name, slug in MODEL_SLUGS.items():
        path = MODELS_DIR / f"{slug}_fold01.joblib"
        if not path.exists():
            print(f"skip {model_name}: {path} not found yet")
            continue
        fitted: FittedRTSModel = FittedRTSModel.load(path)
        model_size_bytes = path.stat().st_size

        pre_samples = time_batch1(lambda: fitted.preprocessor.transform(x_row[fitted.feature_order]), WARMUP, TIMED)
        Xt = fitted.preprocessor.transform(x_row[fitted.feature_order])
        infer_samples = time_batch1(lambda: fitted.estimator.predict_proba(Xt), WARMUP, TIMED)
        e2e_samples = time_batch1(lambda: fitted.predict_proba(x_row), WARMUP, TIMED)

        proc = psutil.Process()
        mem_rss_mb = proc.memory_info().rss / (1024 * 1024)

        for stage, samples in [("preprocessing", pre_samples), ("inference", infer_samples), ("end_to_end", e2e_samples)]:
            s = summarize(samples)
            s.update({"model": model_name, "stage": stage, "model_size_bytes": model_size_bytes, "memory_rss_mb": mem_rss_mb})
            rows.append(s)
            for v in samples:
                raw_samples.append({"model": model_name, "stage": stage, "latency_ns": int(v)})

        e2e = summarize(e2e_samples)
        for env_name, env_ms in ENVELOPES_MS.items():
            rows.append({
                "model": model_name, "stage": "envelope_verdict", "envelope": env_name,
                "envelope_ms": env_ms, "d_agg_ms": 0, "placement": "edge",
                "p95_plus_dagg_ms": e2e["p95_ms"],
                "verdict_pass": bool(e2e["p95_ms"] <= env_ms),
            })
            for d_agg in D_AGG_MS:
                rows.append({
                    "model": model_name, "stage": "envelope_verdict", "envelope": env_name,
                    "envelope_ms": env_ms, "d_agg_ms": d_agg, "placement": "central",
                    "p95_plus_dagg_ms": e2e["p95_ms"] + d_agg,
                    "verdict_pass": bool(e2e["p95_ms"] + d_agg <= env_ms),
                })
        print(f"model={model_name} e2e_p95_ms={e2e['p95_ms']:.4f} model_size_MB={model_size_bytes/1e6:.2f}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(METRICS_DIR / "latency_and_envelope_verdicts.csv", index=False)
    envelope_only = out_df[out_df.stage == "envelope_verdict"].drop(columns=["stage"])
    envelope_only.to_csv(METRICS_DIR / "envelope_verdicts.csv", index=False)

    raw_df = pd.DataFrame(raw_samples)
    raw_df.to_parquet(TIMINGS_DIR / "raw_latency_samples.parquet", index=False)

    env_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "warmup_iterations": WARMUP,
        "timed_iterations": TIMED,
        "batch_size": 1,
    }
    (TIMINGS_DIR / "environment.json").write_text(json.dumps(env_info, indent=2))
    print(f"wrote {METRICS_DIR}/latency_and_envelope_verdicts.csv, envelope_verdicts.csv, {TIMINGS_DIR}/raw_latency_samples.parquet")


if __name__ == "__main__":
    main()
