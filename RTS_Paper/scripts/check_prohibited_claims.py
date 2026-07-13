"""Phase F: prohibited-claim grep over the draft .tex.

Flags temporal/causal/production-readiness language the ground rules forbid.
Run before every submission-candidate compile.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "RTS_Paper" / "paper" / "ieee_rtc_telemetry_budgets_draft.tex"

PROHIBITED = [
    r"\bonset\b", r"\bforecast(ing)?\b", r"time[- ]to[- ]detection", r"\bearly detection\b",
    r"\bearly[- ]warning\b", r"production[- ]ready", r"\bcausal(ly)?\b", r"built a digital twin",
    r"packet[- ]delay", r"streaming prediction", r"attack forecasting",
]


def main() -> int:
    text = DRAFT.read_text(encoding="utf-8")
    hits = []
    for i, line in enumerate(text.splitlines(), start=1):
        if line.strip().startswith("%"):
            continue
        for pat in PROHIBITED:
            if re.search(pat, line, re.IGNORECASE):
                hits.append((i, pat, line.strip()[:120]))
    if hits:
        print(f"FOUND {len(hits)} PROHIBITED-TERM MATCHES:")
        for line_no, pat, snippet in hits:
            print(f"  line {line_no} [{pat}]: {snippet}")
        return 1
    print("Prohibited-claim grep: CLEAN (no matches).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
