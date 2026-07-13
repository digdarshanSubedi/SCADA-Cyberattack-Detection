"""Phase F: headline and supporting figures. Palette matches the original
paper's recolored figures (gray=baseline/control, blue=primary) for a
consistent visual identity across the whole research program.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "RTS_Paper" / "outputs" / "metrics"
ROBUST_DIR = ROOT / "RTS_Paper" / "outputs" / "robustness"
FIG_DIR = ROOT / "RTS_Paper" / "figures"

GRAY = "#8C8C8C"
BLUE = "#2255A4"
BLUE_DARK = "#153A6E"
RED = "#C0392B"
GREEN = "#3D8C55"
GOLD = "#D6A017"
MODEL_COLORS = {"Logistic Regression": GOLD, "Random Forest": RED, "XGBoost": BLUE, "Compact MLP": GREEN}

ENVELOPES_MS = {"E1": 10.0, "E2": 16.7, "E3": 100.0}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({"axes.edgecolor": "#333333", "axes.labelcolor": "#222222", "text.color": "#222222",
                      "xtick.color": "#333333", "ytick.color": "#333333", "font.family": "sans-serif"})


def fig_tradespace() -> None:
    lat = pd.read_csv(METRICS_DIR / "latency_and_envelope_verdicts.csv")
    e2e = lat[lat.stage == "end_to_end"][["model", "p95_ms"]]
    payload = pd.read_csv(METRICS_DIR / "payload_model.csv")
    payload_128 = payload[(payload.feature_budget == "128") & (payload.representation == "32-bit float")][["payload_bytes_per_record"]].iloc[0, 0]
    payload_edge = payload[(payload.feature_budget.str.contains("Edge")) & (payload.representation == "32-bit float")][["payload_bytes_per_record"]].iloc[0, 0]

    macro_central = pd.read_csv(METRICS_DIR / "phase_b_loro_clean_macro.csv")
    macro_edge = pd.read_csv(METRICS_DIR / "phase_b_loro_clean_macro_edge_r1.csv")

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for _, row in e2e.iterrows():
        model = row["model"]
        color = MODEL_COLORS[model]
        auc_c = macro_central[macro_central.model == model]["roc_auc_mean"].iloc[0]
        auc_e = macro_edge[macro_edge.model == model]["roc_auc_mean"].iloc[0]
        ax.scatter(payload_128, row["p95_ms"], s=80 + 400 * auc_c, color=color, alpha=0.85, edgecolor=BLUE_DARK, linewidth=0.6, marker="o", label=f"{model} (central, 128f)")
        ax.scatter(payload_edge, row["p95_ms"], s=80 + 400 * auc_e, color=color, alpha=0.55, edgecolor=GRAY, linewidth=0.6, marker="^", label=f"{model} (edge, 29f)")

    for env_name, env_ms in ENVELOPES_MS.items():
        ax.axhline(env_ms, color=GRAY, linestyle="--", linewidth=1)
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 600, env_ms, f" {env_name}={env_ms:.1f}ms", va="bottom", fontsize=8, color=GRAY)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Payload bytes/record (32-bit float, log scale)")
    ax.set_ylabel("End-to-end P95 latency, ms (log scale)")
    ax.set_title("Trade-space: payload, latency, and run-aware ROC-AUC (marker size)")
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), frameon=False, fontsize=6.5, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
    sns.despine(ax=ax)
    fig.savefig(FIG_DIR / "fig_tradespace.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_tradespace.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_tradespace.{pdf,png}")


def fig_missingness() -> None:
    df = pd.read_csv(ROBUST_DIR / "missingness_results.csv")
    agg = df.groupby(["model", "rate"])["f1"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for model, color in MODEL_COLORS.items():
        g = agg[agg.model == model]
        ax.errorbar(g.rate * 100, g["mean"], yerr=g["std"], label=model, color=color, marker="o", capsize=3)
    ax.set_xlabel("Missing telemetry (%)")
    ax.set_ylabel("Run-aware F1")
    ax.legend(frameon=False, fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_missingness.pdf")
    fig.savefig(FIG_DIR / "fig_missingness.png", dpi=300)
    plt.close(fig)
    print("wrote fig_missingness.{pdf,png}")


def fig_relay_loss() -> None:
    df = pd.read_csv(ROBUST_DIR / "relay_loss_results.csv")
    agg = df.groupby(["model", "condition"])["f1"].mean().reset_index()
    pivot = agg.pivot(index="model", columns="condition", values="f1")
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    pivot.plot(kind="bar", ax=ax, color=[BLUE, RED, GREEN, GOLD])
    ax.set_ylabel("Run-aware F1")
    ax.set_xlabel("")
    ax.legend(frameon=False, fontsize=7, title="Relay lost")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_relay_loss.pdf")
    fig.savefig(FIG_DIR / "fig_relay_loss.png", dpi=300)
    plt.close(fig)
    print("wrote fig_relay_loss.{pdf,png}")


def fig_random_vs_run_aware() -> None:
    # Reuses the same clean central LORO fold results; random-split comparison
    # requires a random 70/30 split run, out of scope for this figure pass —
    # deferred; placeholder note left in checklist.
    pass


def fig_threshold_tradeoffs() -> None:
    df = pd.read_csv(METRICS_DIR / "threshold_rule_macro_summary.csv")
    rules = ["fixed_0.5", "max_f1", "max_youden_j", "constrained_fpr_10pct"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for model, color in MODEL_COLORS.items():
        g = df[df.model == model].set_index("threshold_rule").reindex(rules)
        axes[0].plot(rules, g["f1_mean"], marker="o", color=color, label=model)
        axes[1].plot(rules, g["false_alarms_per_1000_mean"], marker="o", color=color, label=model)
    axes[0].set_ylabel("Run-aware F1")
    axes[1].set_ylabel("False alarms / 1,000 records")
    for ax in axes:
        ax.set_xticklabels(rules, rotation=20, ha="right", fontsize=8)
        sns.despine(ax=ax)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_threshold_tradeoffs.pdf")
    fig.savefig(FIG_DIR / "fig_threshold_tradeoffs.png", dpi=300)
    plt.close(fig)
    print("wrote fig_threshold_tradeoffs.{pdf,png}")


def fig_placement_comparison() -> None:
    df = pd.read_csv(METRICS_DIR / "placement_comparison_table.csv")
    order = ["Edge (edge-visible features)", "Central, clean aggregation", "Central, 10% missing",
              "Central, logs unavailable", "Central, one relay lost (worst)"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.2
    x = np.arange(len(order))
    for i, (model, color) in enumerate(MODEL_COLORS.items()):
        g = df[df.model == model].set_index("placement_condition").reindex(order)
        ax.bar(x + i * width, g["roc_auc"], width=width, color=color, label=model,
               yerr=g["roc_auc_std"], capsize=2)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(order, rotation=20, ha="right", fontsize=7.5)
    ax.set_ylabel("Run-aware ROC-AUC")
    ax.legend(frameon=False, fontsize=7)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_placement_comparison.pdf")
    fig.savefig(FIG_DIR / "fig_placement_comparison.png", dpi=300)
    plt.close(fig)
    print("wrote fig_placement_comparison.{pdf,png}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_tradespace()
    if (ROBUST_DIR / "missingness_results.csv").exists():
        fig_missingness()
    if (ROBUST_DIR / "relay_loss_results.csv").exists():
        fig_relay_loss()
    if (METRICS_DIR / "threshold_rule_macro_summary.csv").exists():
        fig_threshold_tradeoffs()
    if (METRICS_DIR / "placement_comparison_table.csv").exists():
        fig_placement_comparison()


if __name__ == "__main__":
    main()
