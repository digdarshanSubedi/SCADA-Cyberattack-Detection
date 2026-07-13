"""Phase E: SHAP explanation overhead for Random Forest and XGBoost.

Uses TreeSHAP (exact, model-specific) on the fold-01 cached models. Measures
prediction-only latency vs. prediction+explanation latency at batch size 1,
and reports explanations/sec against envelope E3 (100ms). Also computes the
arithmetic cost of 4 selective-explanation strategies from Phase B's already
saved predictions (no new explanation calls beyond what each strategy would
actually invoke).

SHAP values here are post-hoc attribution, not a causal claim (ground rule).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from RTS_Paper.scripts.data import load_dataset
from RTS_Paper.scripts.models import FittedRTSModel
from RTS_Paper.scripts.run_loro import slug

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "RTS_Paper" / "outputs" / "models"
METRICS_DIR = ROOT / "RTS_Paper" / "outputs" / "metrics"
PRED_DIR = ROOT / "RTS_Paper" / "outputs" / "predictions"

MODEL_NAMES = ["Random Forest", "XGBoost"]
N_TIMED = 200  # SHAP is expensive; smaller sample than latency's 5000, still enough for stable median/P95
ENVELOPE_E3_MS = 100.0


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    data, features, meta = load_dataset()
    x_rows = data[features].iloc[:N_TIMED]

    rows = []
    for model_name in MODEL_NAMES:
        path = MODELS_DIR / f"{slug(model_name)}_fold01.joblib"
        if not path.exists():
            print(f"skip {model_name}: {path} not found")
            continue
        fitted = FittedRTSModel.load(path)
        Xt = fitted.preprocessor.transform(x_rows[fitted.feature_order])
        explainer = shap.TreeExplainer(fitted.estimator)

        pred_only_ns = []
        pred_plus_explain_ns = []
        explain_only_ns = []
        for i in range(N_TIMED):
            row = Xt[i : i + 1]
            t0 = time.perf_counter_ns()
            fitted.estimator.predict_proba(row)
            t1 = time.perf_counter_ns()
            pred_only_ns.append(t1 - t0)

            t2 = time.perf_counter_ns()
            explainer.shap_values(row)
            t3 = time.perf_counter_ns()
            explain_only_ns.append(t3 - t2)
            pred_plus_explain_ns.append((t1 - t0) + (t3 - t2))

        pred_ms = np.array(pred_only_ns) / 1e6
        explain_ms = np.array(explain_only_ns) / 1e6
        combined_ms = np.array(pred_plus_explain_ns) / 1e6

        row = {
            "model": model_name,
            "pred_only_median_ms": float(np.median(pred_ms)),
            "pred_only_p95_ms": float(np.percentile(pred_ms, 95)),
            "explain_only_median_ms": float(np.median(explain_ms)),
            "explain_only_p95_ms": float(np.percentile(explain_ms, 95)),
            "pred_plus_explain_median_ms": float(np.median(combined_ms)),
            "pred_plus_explain_p95_ms": float(np.percentile(combined_ms, 95)),
            "explanations_per_sec": float(1000.0 / np.median(explain_ms)),
            "fits_e3_every_record": bool(np.percentile(combined_ms, 95) <= ENVELOPE_E3_MS),
        }
        rows.append(row)
        print(f"{model_name}: pred={row['pred_only_median_ms']:.4f}ms explain={row['explain_only_median_ms']:.4f}ms combined_p95={row['pred_plus_explain_p95_ms']:.4f}ms")

    shap_timing_df = pd.DataFrame(rows)
    shap_timing_df.to_csv(METRICS_DIR / "shap_timing.csv", index=False)

    # Selective-explanation strategies: arithmetic over Phase B's saved pooled predictions.
    pooled_path = PRED_DIR / "phase_b_loro_clean_predictions.csv"
    strategy_rows = []
    if pooled_path.exists():
        pooled = pd.read_csv(pooled_path)
        for model_name in MODEL_NAMES:
            g = pooled[pooled.model == model_name]
            n_total = len(g)
            n_predicted_attack = int((g.y_prob >= 0.5).sum())
            uncertain_band = g[(g.y_prob >= 0.4) & (g.y_prob <= 0.6)]
            n_uncertain = len(uncertain_band)
            high_risk = g[g.y_prob >= 0.9]
            n_high_risk = len(high_risk)
            timing_row = shap_timing_df[shap_timing_df.model == model_name].iloc[0]
            per_explanation_ms = timing_row["explain_only_median_ms"]
            for strategy, n_explained in [
                ("explain_every_record", n_total),
                ("explain_every_predicted_attack", n_predicted_attack),
                ("explain_uncertain_only_0.4_0.6", n_uncertain),
                ("explain_high_risk_only_ge_0.9", n_high_risk),
            ]:
                strategy_rows.append({
                    "model": model_name, "strategy": strategy,
                    "n_test_records_pooled": n_total, "n_explained": n_explained,
                    "fraction_explained": n_explained / n_total if n_total else np.nan,
                    "estimated_total_explain_time_sec": n_explained * per_explanation_ms / 1000.0,
                })
        pd.DataFrame(strategy_rows).to_csv(METRICS_DIR / "shap_selective_strategies.csv", index=False)
        print(f"wrote shap_selective_strategies.csv ({len(strategy_rows)} rows)")
    else:
        print(f"WARNING: {pooled_path} not found; selective-strategy arithmetic skipped")

    print("wrote shap_timing.csv")


if __name__ == "__main__":
    main()
