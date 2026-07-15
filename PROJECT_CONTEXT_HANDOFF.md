# SCADA / RTS Paper Project Handoff

Last updated: 2026-07-15

Repository root:

`/Users/digdarshansubedi/Documents/Paper Publications/Saudi Arabia Sustainable Energy - Cuber Security Paper`

GitHub remote:

`https://github.com/digdarshanSubedi/SCADA-Cyberattack-Detection.git`

Current branch at the time of this handoff:

`rts-paper-extension`

Latest pushed commit at the time of this handoff:

`738f3ce Update RTC paper figures and Overleaf export`

Pull request URL for the pushed branch:

`https://github.com/digdarshanSubedi/SCADA-Cyberattack-Detection/pull/new/rts-paper-extension`

## 1. What This Repository Contains

This repository contains two related research-paper efforts built on the MSU/ORNL power-system cyberattack dataset.

1. Original leakage-aware SCADA detector paper.
2. RTS/RTC extension paper about real-time SCADA intrusion detection under communication constraints.

The original paper focuses on leakage-aware evaluation: random row-level splits versus leave-one-run-out evaluation. The RTS/RTC extension builds on that result and studies payload, latency, feature budgets, telemetry degradation, and edge-visible versus centrally aggregated detection.

## 2. Authors And Paper Identity

The current author list used in both paper versions is:

- Digdarshan Subedi, Department of Computer Science, University of Wisconsin--Whitewater, Whitewater, WI, USA, `SubediD30@uww.edu`
- Nikesh Tiwari, School of Business & Technology, Webster University, St. Louis, MO, USA, `nikeshtiwari@webster.edu`
- Bibek Ale, School of Business & Technology, Webster University, St. Louis, MO, USA, `Bibekale@webster.edu`

Original paper title:

`Benchmark Before You Automate: Leakage-Aware Evaluation of Machine Learning Detectors for Power-Sector SCADA Security`

RTS/RTC extension title:

`Real-Time SCADA Intrusion Detection Under Communication Constraints: Latency, Payload, and Generalization in Industrial Edge Environments`

## 3. Dataset Context

Dataset folder:

`binaryAllNaturalPlusNormalVsAttacks - 2 Class/`

The dataset contains fifteen local CSV files:

- `data1.csv` through `data15.csv`

The research treats these files as independently collected runs. The key methodological concern is that random row-level splitting can place very similar records from the same run in both training and test sets. The leakage-aware protocol instead holds out complete runs.

Dataset documentation:

`binaryAllNaturalPlusNormalVsAttacks - 2 Class/PowerSystem_Dataset_README.pdf`

Main dataset citation context used in the RTS handoff:

- Official UAH ICS dataset page: `https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets`
- Do not invent a DOI for the dataset webpage.
- Existing references are in `paper_project/references.bib`.

## 4. Original SCADA Paper

Main TeX:

`paper_project/main.tex`

Compiled PDF:

`paper_project/main.pdf`

Original pipeline command:

```bash
python3 -m src.run_all
```

Primary source code:

- `src/audit_data.py`
- `src/evaluate_naive.py`
- `src/evaluate_groupkfold.py`
- `src/models.py`
- `src/preprocessing.py`
- `src/plotting.py`
- `src/statistics.py`
- `src/build_paper.py`
- `src/run_all.py`

Original paper figures:

- `figures/fig1_naive_vs_leakage_metrics.png`
- `figures/fig2_roc_naive.png`
- `figures/fig3_roc_leakage.png`
- `figures/fig4_confusion_matrix.png`
- `figures/fig5_fold_variance.png`
- `figures/fig6_feature_importance.png`
- `figures/supp_precision_recall_leakage.png`

Original output files:

- `outputs/results.json`
- `outputs/dataset_audit.json`
- `outputs/metrics_naive.csv`
- `outputs/metrics_groupkfold_macro.csv`
- `outputs/metrics_groupkfold_pooled.csv`
- `outputs/per_fold_metrics.csv`
- `outputs/predictions_naive.csv`
- `outputs/predictions_groupkfold.csv`
- `outputs/feature_importance.csv`
- `outputs/statistics.json`

Original paper core result:

Random row-level splitting overstated detector performance compared with run-level evaluation. Random Forest had strong random-split performance, but run-aware mean performance dropped substantially.

Important original run-aware metrics from `outputs/metrics_groupkfold_macro.csv`:

| Model | F1 mean | ROC-AUC mean | FPR mean |
| --- | ---: | ---: | ---: |
| CNN | 0.829 | 0.527 | 0.998 |
| Random Forest | 0.807 | 0.677 | 0.667 |
| XGBoost | 0.805 | 0.704 | 0.650 |

Important interpretation boundary:

The CNN has stable/high F1 but near-chance ROC-AUC and extremely high false-positive rate, so it should not be framed as operationally strong.

## 5. RTS/RTC Extension Paper

RTS folder:

`RTS_Paper/`

Primary local Overleaf-ready TeX:

`RTS_Paper/main-local.tex`

Primary local Overleaf-ready compiled PDF:

`RTS_Paper/main-local.pdf`

Paper-folder TeX:

`RTS_Paper/paper/ieee_rtc_telemetry_budgets_draft.tex`

Paper-folder compiled PDF:

`RTS_Paper/paper/ieee_rtc_telemetry_budgets_draft.pdf`

Build output PDF from the latest local compile:

`RTS_Paper/build/main-local.pdf`

The Overleaf-ready file uses:

```latex
\graphicspath{{figures/}}
```

The paper-folder TeX uses:

```latex
\graphicspath{{../figures/}}
```

Do not casually mix these two paths. The user wants the Overleaf copy to match a root-level `figures/` folder.

## 6. RTS Research Questions And Story

The RTS/RTC paper asks how SCADA intrusion detectors behave when they are evaluated as real-time communication-workflow components rather than only offline classifiers.

The key dimensions are:

- Leave-one-run-out generalization.
- Batch-size-one detector latency.
- Modeled payload per record.
- Feature-budget reduction.
- Missing telemetry robustness.
- Relay/log/PMU availability loss.
- Edge-visible R1-only telemetry versus centrally aggregated telemetry.
- Threshold transfer and false-alarm burden.

Safe central message:

Low inference latency alone does not establish real-time usefulness. A credible detector must also be evaluated for unseen-run transfer, payload efficiency, placement-dependent visibility, degradation resilience, and false-alarm burden.

## 7. RTS Scripts

Primary RTS scripts:

- `RTS_Paper/scripts/data.py`
- `RTS_Paper/scripts/models.py`
- `RTS_Paper/scripts/metrics.py`
- `RTS_Paper/scripts/run_loro.py`
- `RTS_Paper/scripts/run_latency.py`
- `RTS_Paper/scripts/payload_model.py`
- `RTS_Paper/scripts/run_feature_budgets.py`
- `RTS_Paper/scripts/run_degradation.py`
- `RTS_Paper/scripts/run_placement.py`
- `RTS_Paper/scripts/run_pmu_ablation.py`
- `RTS_Paper/scripts/run_thresholds.py`
- `RTS_Paper/scripts/run_shap_timing.py`
- `RTS_Paper/scripts/run_stats.py`
- `RTS_Paper/scripts/build_macros.py`
- `RTS_Paper/scripts/build_figures.py`

Important model definitions are in:

`RTS_Paper/scripts/models.py`

Primary detector set:

- Logistic Regression
- Random Forest
- XGBoost
- Compact MLP

The older feature-space CNN is diagnostic background only in the RTS extension and should not be presented as part of the new real-time evaluation suite.

## 8. RTS Model Configurations

Configuration summary:

| Model | Implementation | Main settings |
| --- | --- | --- |
| Logistic Regression | scikit-learn `LogisticRegression` | median imputation, standardization, `max_iter=2000`, `random_state=42` |
| Random Forest | scikit-learn `RandomForestClassifier` | median imputation, `n_estimators=200`, `random_state=42`, `n_jobs=-1` |
| XGBoost | `xgboost.XGBClassifier` | median imputation, `n_estimators=200`, `random_state=42`, `objective=binary:logistic`, `eval_metric=logloss`, `tree_method=hist`, `n_jobs=4` |
| Compact MLP | scikit-learn `MLPClassifier` | median imputation, standardization, hidden layers `(32,16)`, ReLU, `alpha=1e-4`, `max_iter=200`, early stopping |

Use the code as the source of truth before changing text.

## 9. RTS Key Metrics

Primary run-aware central 128-feature metrics from:

`RTS_Paper/outputs/metrics/phase_b_loro_clean_macro.csv`

| Model | F1 mean | ROC-AUC mean | PR-AUC mean | FPR mean |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.828 | 0.687 | 0.854 | 0.934 |
| Random Forest | 0.807 | 0.677 | 0.826 | 0.667 |
| XGBoost | 0.804 | 0.702 | 0.863 | 0.652 |
| Compact MLP | 0.815 | 0.683 | 0.850 | 0.767 |

Latency source:

`RTS_Paper/outputs/metrics/latency_and_envelope_verdicts.csv`

Main end-to-end detector-pipeline P95 values:

| Model | P95 latency ms |
| --- | ---: |
| Logistic Regression | 1.112 |
| Random Forest | 51.711 |
| XGBoost | 4.864 |
| Compact MLP | 1.690 |

Interpretation:

- Logistic Regression, XGBoost, and Compact MLP meet the 10 ms and 16.7 ms study references.
- Random Forest misses 10 ms and 16.7 ms, but meets 100 ms.
- Latency excludes file I/O and network transport.

Feature-budget source:

`RTS_Paper/outputs/metrics/feature_budget_ranked_macro.csv`

Payload source:

`RTS_Paper/outputs/metrics/payload_model.csv`

Key payload arithmetic:

| Features | 32-bit payload bytes/record |
| ---: | ---: |
| 128 | 512 |
| 64 | 256 |
| 32 | 128 |
| 16 | 64 |
| Edge R1 only, 29 features | 116 |
| PMU only, 116 features | 464 |
| Cyber/log only, 12 features | 48 |

Important feature-budget result:

The 16-feature condition cuts modeled 32-bit payload from 512 to 64 bytes. It preserves comparable or higher mean ROC-AUC for Random Forest, XGBoost, and Compact MLP in the saved benchmark, but not for Logistic Regression.

Avoid overclaiming statistical significance unless an inferential test is added and supports it.

## 10. RTS Reproducibility And Validation Artifacts

Important handoff and audit files already created:

- `RTS_Paper/rtc_reproducibility_handoff.md`
- `RTS_Paper/rtc_reproducibility_values.json`
- `RTS_Paper/rtc_number_validation.csv`
- `RTS_Paper/rtc_claim_audit.csv`
- `RTS_Paper/rtc_feature_budget_folds.csv`
- `RTS_Paper/paper/float_sequence_validation_report.md`
- `RTS_Paper/outputs/audit/EXECUTION_CHECKLIST.md`
- `RTS_Paper/outputs/audit/VERIFICATION_PACKET.md`
- `RTS_Paper/outputs/audit/project_continuity_audit.md`

`RTS_Paper/rtc_number_validation.csv` reported 116 numeric checks matching at 3-decimal precision at the time it was created.

Known unavailable information from the RTS reproducibility handoff:

- Exact machine model name and CPU marketing name for the saved latency run.
- Exact power mode or CPU affinity control.
- Original BLAS/OpenMP thread environment variables at latency-run time.
- Per-experiment full package snapshots for every library.
- Dataset webpage DOI.

## 11. Figures And Overleaf Export

Canonical figure folder for local/Overleaf copying:

`RTS_Paper/figures/`

Export folder requested by the user:

`RTS_Paper/latex-export-figures/`

The user specifically requested all paper figures to be inside a folder named `figures` for Overleaf path compatibility. Keep the Overleaf version using paths like:

```latex
\includegraphics{figures/fig04_payload_latency_quality.png}
```

Current Figure 1 through Figure 6:

| Figure | File | Meaning |
| --- | --- | --- |
| Figure 1 | `fig01_system_architecture.png` | System architecture |
| Figure 2 | `fig02_random_vs_runaware.png` | Random split vs run-aware protocol |
| Figure 3 | `fig03_latency_envelopes.png` | Latency envelopes |
| Figure 4 | `fig04_payload_latency_quality.png` | Payload-latency-quality feature-budget trade-off |
| Figure 5 | `fig05_missing_telemetry_robustness.png` | Missing-telemetry robustness |
| Figure 6 | `fig06_edge_vs_central.png` | Edge-visible vs central telemetry comparison |

Supporting figures:

- `fig_relay_loss.png`
- `fig_threshold_tradeoffs.png`

Backups that should usually remain local and not be committed:

- `RTS_Paper/figures/backup_pre_redesign_20260713_113300/`
- `RTS_Paper/figures/backup_pre_fig04_budget_sweep_20260714_105512/`
- `RTS_Paper/latex-export-figures/backup_pre_fig04_budget_sweep_20260714_105512/`

## 12. Figure 4 Details

Figure 4 was replaced with a more relevant budget-sweep graph on 2026-07-14.

Current file:

`RTS_Paper/figures/fig04_payload_latency_quality.png`

Also copied to:

`RTS_Paper/latex-export-figures/fig04_payload_latency_quality.png`

Generator:

`RTS_Paper/scripts/build_figures.py`, function `fig_tradespace()`

Current Figure 4 shows:

- Payload reduction from 512 bytes / 128 features to 64 bytes / 16 features.
- P95 detector-pipeline latency.
- Run-aware ROC-AUC.
- Four model trajectories: Logistic Regression, Random Forest, XGBoost, Compact MLP.

Data files used for Figure 4:

- `RTS_Paper/outputs/metrics/payload_model.csv`
- `RTS_Paper/outputs/metrics/latency_and_envelope_verdicts.csv`
- `RTS_Paper/outputs/metrics/latency_by_budget.csv`
- `RTS_Paper/outputs/metrics/phase_b_loro_clean_macro.csv`
- `RTS_Paper/outputs/metrics/feature_budget_ranked_macro.csv`

Current Figure 4 PNG dimensions after the replacement:

`2120 x 1090`

Important: Earlier Figure 4 was a central-vs-edge scatter using 128-feature central payload and 29-feature R1 edge payload. The current Figure 4 is now aligned with the manuscript caption and Table IV feature-budget story.

## 13. Float Order And LaTeX Flow

User-requested flow:

1. Title and authors.
2. Abstract.
3. Keywords.
4. Introduction, no floats.
5. Related Work, no floats.
6. Communication-budgeted detection framework, then Figure 1.
7. Dataset and visibility conditions.
8. Method, including Figure 2 and Table I.
9. Processing and payload measurements.
10. Telemetry degradation and threshold transfer.
11. Results with Tables II-VI and Figures 3-6 in sequence.
12. Discussion.
13. Threats to Validity and Limitations.
14. Conclusion.
15. References.

`RTS_Paper/paper/float_sequence_validation_report.md` states that the final checked paper had:

- Figure order: Figure 1 through Figure 6.
- Table order: Table I through Table VI.
- No floats in front matter, Introduction, or Related Work.
- No Results floats leaking into Discussion, Threats, Conclusion, or References.
- Final PDF page count: 6.

## 14. Compile Commands

Compile Overleaf-ready local root file:

```bash
python3 /Users/digdarshansubedi/.codex/plugins/cache/openai-bundled/latex/0.2.4/scripts/compile_latex.py "$PWD/RTS_Paper/main-local.tex" --output-directory "$PWD/RTS_Paper/build"
```

The last known compile passed and produced:

`RTS_Paper/build/main-local.pdf`

Regenerate all RTS figures:

```bash
python RTS_Paper/scripts/build_figures.py
```

After regenerating, copy required files into the Overleaf export folder if needed:

```bash
cp RTS_Paper/figures/fig04_payload_latency_quality.* RTS_Paper/latex-export-figures/
```

Run original paper pipeline:

```bash
python3 -m src.run_all
```

## 15. Git And Push Status

Latest pushed branch:

`rts-paper-extension`

Latest pushed commit:

`738f3ce Update RTC paper figures and Overleaf export`

That commit added or updated:

- RTS figures 1-6.
- `RTS_Paper/latex-export-figures/`.
- `RTS_Paper/main-local.tex`.
- `RTS_Paper/main-local.pdf`.
- `RTS_Paper/paper/ieee_rtc_telemetry_budgets_draft.tex`.
- `RTS_Paper/paper/ieee_rtc_telemetry_budgets_draft.pdf`.
- RTS audit/reproducibility artifacts.
- Updated `RTS_Paper/scripts/build_figures.py`.

At the time this handoff file was created, there were still local generated/scratch files not intended for commit:

- `.claude/`
- `RTS_Paper/build/`
- `RTS_Paper/main-local.aux`
- `RTS_Paper/main-local.log`
- `RTS_Paper/paper/ieee_rtc_telemetry_budgets_draft.aux`
- `RTS_Paper/paper/ieee_rtc_telemetry_budgets_draft.log`
- `RTS_Paper/paper/ieee_rtc_telemetry_budgets_draft.bbl`
- `RTS_Paper/paper/ieee_rtc_telemetry_budgets_draft.blg`
- timestamped backup figure folders

Do not push large cached model files. `.gitignore` already excludes:

```gitignore
RTS_Paper/outputs/models/*.joblib
RTS_Paper/outputs/models/*.pt
```

## 16. Known User Preferences From This Work

- The user wants execution first, not long planning.
- When asked for paths, provide exact paths.
- The user wants figures in `RTS_Paper/figures/` so Overleaf paths do not need changing.
- The user also requested `RTS_Paper/latex-export-figures/` as a collected export folder.
- Do not fabricate metrics, figures, citations, or claims.
- Use saved CSVs and scripts as source of truth.
- If the user provides an actual PNG and says to use it, copy the PNG directly instead of regenerating it.
- Do not casually redesign diagrams after the user says to use the previous one.
- Keep `main-local.tex` Overleaf-friendly.
- For final paper editing, preserve manuscript text unless explicitly asked to rewrite content.

## 17. Next Recommended Work

Good next steps:

1. Open the GitHub PR for `rts-paper-extension`.
2. Copy `RTS_Paper/main-local.tex` and `RTS_Paper/figures/` into Overleaf.
3. Confirm Overleaf compiles with the same figure paths.
4. Review Figure 1 architecture image. If the user supplies a PNG, replace the file directly at `RTS_Paper/figures/fig01_system_architecture.png` and copy it to `RTS_Paper/latex-export-figures/`.
5. Re-run LaTeX compile and inspect PDF page count and figure/table order.
6. If any numbers change, update `rtc_number_validation.csv` and this handoff.

## 18. Important Safety Notes

- The 16-feature result is strong for the paper story, but phrase it carefully: it preserves comparable or higher mean discrimination for some models in this benchmark; it is not universal proof that fewer features are always better.
- Random Forest is accurate enough to discuss but slow under the P95 latency references.
- High F1 can be misleading because several models have high false-positive rates.
- Payload values are modeled numeric-field payloads only and exclude protocol overhead, encryption, headers, retransmissions, buffering, and transport delays.
- Edge-visible R1-only is an analytical proxy, not a deployed edge-device benchmark.

