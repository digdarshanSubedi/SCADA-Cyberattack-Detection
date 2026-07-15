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
    """Paper Fig. 4 (fig04_payload_latency_quality)."""
    payload = pd.read_csv(METRICS_DIR / "payload_model.csv")
    payload = payload[(payload.representation == "32-bit float") &
                      (payload.feature_budget.astype(str).isin(["128", "64", "32", "16"]))]
    payload = payload.assign(feature_budget=payload.feature_budget.astype(int))

    lat_128 = pd.read_csv(METRICS_DIR / "latency_and_envelope_verdicts.csv")
    lat_128 = lat_128[lat_128.stage == "end_to_end"][["model", "p95_ms"]]
    latency_budget = pd.read_csv(METRICS_DIR / "latency_by_budget.csv")
    latency_budget = latency_budget[latency_budget.feature_budget.astype(str).isin(["64", "32", "16"])]
    latency_budget = latency_budget.assign(feature_budget=latency_budget.feature_budget.astype(int))

    macro_128 = pd.read_csv(METRICS_DIR / "phase_b_loro_clean_macro.csv")
    macro_budget = pd.read_csv(METRICS_DIR / "feature_budget_ranked_macro.csv")

    rows = []
    for model in MODEL_COLORS:
        payload_128 = payload[payload.feature_budget == 128]["payload_bytes_per_record"].iloc[0]
        p95_128 = lat_128[lat_128.model == model]["p95_ms"].iloc[0]
        auc_128 = macro_128[macro_128.model == model]["roc_auc_mean"].iloc[0]
        f1_128 = macro_128[macro_128.model == model]["f1_mean"].iloc[0]
        rows.append({"model": model, "feature_budget": 128, "payload_bytes_per_record": payload_128,
                     "p95_ms": p95_128, "roc_auc_mean": auc_128, "f1_mean": f1_128})
        for budget in [64, 32, 16]:
            payload_b = payload[payload.feature_budget == budget]["payload_bytes_per_record"].iloc[0]
            p95_b = latency_budget[(latency_budget.model == model) &
                                   (latency_budget.feature_budget == budget)]["p95_ms"].iloc[0]
            macro_b = macro_budget[(macro_budget.model == model) &
                                   (macro_budget.feature_budget == budget)].iloc[0]
            rows.append({"model": model, "feature_budget": budget, "payload_bytes_per_record": payload_b,
                         "p95_ms": p95_b, "roc_auc_mean": macro_b["roc_auc_mean"],
                         "f1_mean": macro_b["f1_mean"]})

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharex=True)
    marker_map = {128: "o", 64: "s", 32: "D", 16: "^"}

    for model, color in MODEL_COLORS.items():
        g = df[df.model == model].sort_values("payload_bytes_per_record", ascending=False)
        axes[0].plot(g["payload_bytes_per_record"], g["p95_ms"], color=color, linewidth=1.5,
                     marker="o", markersize=4.8, label=model)
        axes[1].plot(g["payload_bytes_per_record"], g["roc_auc_mean"], color=color, linewidth=1.5,
                     marker="o", markersize=4.8, label=model)
        for _, point in g.iterrows():
            for ax, y_col in [(axes[0], "p95_ms"), (axes[1], "roc_auc_mean")]:
                ax.scatter(point["payload_bytes_per_record"], point[y_col],
                           s=62, color=color, marker=marker_map[point["feature_budget"]],
                           edgecolor="#222222", linewidth=0.45, zorder=3)

    for env_name, env_ms in ENVELOPES_MS.items():
        axes[0].axhline(env_ms, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.75)
        axes[0].text(540, env_ms, f"{env_name}", va="bottom", ha="right", fontsize=6.6, color=GRAY)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlim(56, 580)
    axes[0].set_ylim(0.65, 120)
    axes[0].set_xticks([64, 128, 256, 512])
    axes[0].set_xticklabels(["64\n16f", "128\n32f", "256\n64f", "512\n128f"])
    axes[0].set_xlabel("Payload bytes/record")
    axes[0].set_ylabel("P95 latency (ms)")
    axes[0].set_title("(a) Latency vs. payload")

    axes[1].set_xscale("log")
    axes[1].set_xlim(56, 580)
    axes[1].set_xticks([64, 128, 256, 512])
    axes[1].set_xticklabels(["64\n16f", "128\n32f", "256\n64f", "512\n128f"])
    axes[1].set_ylim(0.60, 0.74)
    axes[1].set_xlabel("Payload bytes/record")
    axes[1].set_ylabel("Run-aware ROC-AUC")
    axes[1].set_title("(b) Quality vs. payload")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7.2, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Feature-budget trade-off from 128 to 16 transmitted fields", y=1.02, fontsize=10)
    for ax in axes:
        sns.despine(ax=ax)
        ax.grid(True, which="major", linewidth=0.5, alpha=0.45)
        ax.grid(True, which="minor", linewidth=0.25, alpha=0.2)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(FIG_DIR / "fig04_payload_latency_quality.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig04_payload_latency_quality.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig04_payload_latency_quality.{pdf,png}")


def fig_missingness() -> None:
    """Paper Fig. 5 (fig05_missing_telemetry_robustness)."""
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
    fig.savefig(FIG_DIR / "fig05_missing_telemetry_robustness.pdf")
    fig.savefig(FIG_DIR / "fig05_missing_telemetry_robustness.png", dpi=300)
    plt.close(fig)
    print("wrote fig05_missing_telemetry_robustness.{pdf,png}")


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
    """Paper Fig. 6 (fig06_edge_vs_central). Already includes the
    worst-relay-loss placement condition, so fig_relay_loss.png stays a
    separate supporting figure rather than becoming Fig. 6."""
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
    fig.savefig(FIG_DIR / "fig06_edge_vs_central.pdf")
    fig.savefig(FIG_DIR / "fig06_edge_vs_central.png", dpi=300)
    plt.close(fig)
    print("wrote fig06_edge_vs_central.{pdf,png}")


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
