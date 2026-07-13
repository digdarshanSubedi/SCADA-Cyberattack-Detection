"""Phase C (task B): payload model P = n_features * bytes_per_field.

Pure arithmetic over feature-budget definitions; no training required.
Matches the draft's stated model (line 185): 4 bytes at 32-bit float,
2 bytes at 16-bit, 1 byte at 8-bit quantized representation. Protocol
framing overhead is deliberately excluded (deployment-specific), per draft.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from RTS_Paper.scripts.data import load_dataset

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "RTS_Paper" / "outputs" / "metrics" / "payload_model.csv"

BYTES_PER_FIELD = {"32-bit float": 4, "16-bit quantized": 2, "8-bit quantized": 1}


def main() -> None:
    _, features, meta = load_dataset()
    n_pmu = len(meta["pmu_features"])
    n_cyber = len(meta["cyber_log_features"])
    n_edge = len([f for f in features if f.startswith("R1-") or f.startswith("R1:")])

    budgets = {
        "128": len(features),
        "64": 64,
        "32": 32,
        "16": 16,
        "PMU only": n_pmu,
        "Cyber/log only": n_cyber,
        "Edge (R1 only, 29 features)": n_edge,
    }

    rows = []
    for budget_name, n_f in budgets.items():
        for repr_name, b in BYTES_PER_FIELD.items():
            rows.append({
                "feature_budget": budget_name,
                "n_features": n_f,
                "representation": repr_name,
                "bytes_per_field": b,
                "payload_bytes_per_record": n_f * b,
            })
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
