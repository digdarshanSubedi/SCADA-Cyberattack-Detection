# RTC SCADA Reproducibility Handoff

Generated from saved repository artifacts on 2026-07-13. No completed model experiments were rerun. Derived tables were computed from saved CSV/joblib artifacts only.

## A. Executive summary

Recovered: pipeline file map, model configurations, saved latency protocol/environment, feature-ranking procedure, fold-level 128-vs-16 AUC comparisons, payload arithmetic, missingness/relay-loss definitions, edge-visible definition, threshold provenance, SHAP timing provenance, figure audit, dataset citation details, and manuscript-number validation. Remaining unavailable: exact machine model name, OS marketing version beyond `macOS-26.5.1-arm64-arm-64bit`, power mode/CPU affinity, original OMP/MKL/OPENBLAS env vars, per-experiment full package snapshots for NumPy/pandas/SHAP/Torch/joblib, and any DOI for the dataset webpage itself.

## B. Exact file map

| Purpose | File path | Relevant function/class/section |
| --- | --- | --- |
| Repository root | /Users/digdarshansubedi/Documents/Paper Publications/Saudi Arabia Sustainable Energy - Cuber Security Paper | All paths below relative to this root |
| Main clean LORO harness | RTS_Paper/scripts/run_loro.py | run(); GroupKFold(n_splits=15); writes phase_b_loro_clean_* |
| Model definitions | RTS_Paper/scripts/models.py | fit_logistic_regression; fit_random_forest; fit_xgboost; fit_compact_mlp; FittedRTSModel |
| Data/preprocessing helpers | RTS_Paper/scripts/data.py | load_dataset; feature_budget_subset; rank_features_train_only |
| Feature ranking | RTS_Paper/scripts/data.py | rank_features_train_only |
| Latency benchmark | RTS_Paper/scripts/run_latency.py | time_batch1; summarize; main |
| Missingness/relay loss | RTS_Paper/scripts/run_degradation.py; RTS_Paper/scripts/degradations.py | random_missingness; group_unavailable; relay_loss |
| PMU unavailable | RTS_Paper/scripts/run_pmu_ablation.py | main masks all_pmu_unavailable |
| Placement comparison | RTS_Paper/scripts/run_placement.py | logs_unavailable_metrics; main |
| Threshold analysis | RTS_Paper/scripts/run_thresholds.py; RTS_Paper/scripts/thresholds.py | THRESHOLD_RULES; threshold_max_f1; constrained_fpr_10pct |
| SHAP timing | RTS_Paper/scripts/run_shap_timing.py | N_TIMED=200; TreeExplainer fold01 RF/XGBoost |
| Payload model | RTS_Paper/scripts/payload_model.py | BYTES_PER_FIELD; main |
| Figures | RTS_Paper/scripts/build_figures.py | fig_system_architecture through fig06_edge_vs_central |
| Saved fold metrics | RTS_Paper/outputs/metrics/*.csv | clean, feature budget, placement, threshold, SHAP |
| Saved robustness outputs | RTS_Paper/outputs/robustness/*.csv | missingness, group unavailable, relay loss, logs/PMU unavailable |
| Saved model cache | RTS_Paper/outputs/models/*.joblib | per-fold FittedRTSModel caches |
| Environment record | RTS_Paper/outputs/timings/environment.json; RTS_Paper/outputs/audit/repository_inventory.json | latency saved env; broader end-of-session package inventory |

Final validated result directory: `/Users/digdarshansubedi/Documents/Paper Publications/Saudi Arabia Sustainable Energy - Cuber Security Paper/RTS_Paper/outputs`. Original baseline RF/XGBoost/CNN rows also use root-level `outputs/metrics_naive.csv` and `outputs/metrics_groupkfold_macro.csv`.

## C. Model configuration table

| Model | Implementation | Preprocessing | Explicit settings | Important defaults / recovered fitted settings | Source |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | scikit-learn `LogisticRegression` | `SimpleImputer(strategy="median")` then `StandardScaler()` | `max_iter=2000`, `random_state=42`, `class_weight=None` | solver `lbfgs`, penalty `l2`, `C=1.0`, `tol=1e-4`, `fit_intercept=True`, `n_jobs=None` | `RTS_Paper/scripts/models.py`; fold01 joblib params |
| Random Forest | scikit-learn `RandomForestClassifier` | `SimpleImputer(strategy="median")` | `n_estimators=200`, `random_state=42`, `n_jobs=-1` | criterion `gini`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `max_features=sqrt`, `bootstrap=True`, `class_weight=None` | `RTS_Paper/scripts/models.py`; fold01 joblib params |
| XGBoost | `xgboost.XGBClassifier` | `SimpleImputer(strategy="median")` | `n_estimators=200`, `random_state=42`, `objective="binary:logistic"`, `eval_metric="logloss"`, `n_jobs=4`, `tree_method="hist"` | booster config recovers `max_depth=6`, `learning_rate=0.300000012`, `subsample=1`, `colsample_bytree=1`, `gamma=0`, `min_child_weight=1`, `reg_alpha=0`, `reg_lambda=1`, `device=cpu` | `RTS_Paper/scripts/models.py`; fold01 booster config |
| Compact MLP | scikit-learn `MLPClassifier` | `SimpleImputer(strategy="median")` then `StandardScaler()` | `hidden_layer_sizes=(32,16)`, `activation="relu"`, `alpha=1e-4`, `max_iter=200`, `early_stopping=True`, `n_iter_no_change=10`, `random_state=42` | solver `adam`, learning rate `constant`, `learning_rate_init=0.001`, batch size `auto`, validation fraction `0.1`, tol `1e-4`, shuffle `True`, beta1/beta2 `0.9/0.999`, CPU scikit-learn implementation | `RTS_Paper/scripts/models.py`; fold01 joblib params |

Method text: Four primary detectors were fitted inside each leave-one-run-out training partition: Logistic Regression, Random Forest, XGBoost, and a compact MLP. Median imputation was fitted on training runs only for all models. Logistic Regression and compact MLP additionally used training-fitted standardization. The feature-space CNN is diagnostic only and is excluded from new latency, feature-budget, availability, placement, threshold, and SHAP experiments.

## D. Hardware/software environment table

| Item | Value | Evidence |
| --- | --- | --- |
| Platform | macOS-26.5.1-arm64-arm-64bit | `RTS_Paper/outputs/timings/environment.json` |
| Processor string | arm | `environment.json` |
| Physical cores | 10 | `environment.json` |
| Logical cores | 10 | `environment.json` |
| Memory | 32.0 GB | `environment.json` |
| Python for latency run | 3.9.6 | `environment.json` |
| scikit-learn | 1.6.1 | `environment.json` |
| XGBoost for latency run | 2.1.4 | `environment.json` |
| NumPy | 2.0.2 | `repository_inventory.json`, end-of-session capture, not per-experiment snapshot |
| pandas | 2.2.3 | `repository_inventory.json`, end-of-session capture |
| SHAP | 0.49.1 | `repository_inventory.json`, end-of-session capture |
| PyTorch | 2.8.0 | `repository_inventory.json`, end-of-session capture |
| SciPy | 1.13.1 | `repository_inventory.json`, end-of-session capture |
| Warm-up iterations | 500 | `environment.json`; `run_latency.py::WARMUP` |
| Timed iterations | 5000 | `environment.json`; `run_latency.py::TIMED` |
| Batch size | 1 | `environment.json`; `run_latency.py` |
| Timing function | `time.perf_counter_ns()` | `RTS_Paper/scripts/run_latency.py::time_batch1` |
| CPU/GPU | CPU only for saved scikit-learn/XGBoost timings; no GPU setting recorded | model code and XGBoost booster config; no GPU metadata in `environment.json` |
| Thread env vars | not captured in saved latency environment | unavailable |

Latency method text: Batch-size-one detector-pipeline latency was measured on the fold-01 cached model for each primary model. The timing harness used 500 warm-up calls and 5,000 timed calls with `time.perf_counter_ns()`, measuring preprocessing, estimator inference, and complete `FittedRTSModel.predict_proba()` separately. The manuscript latency table uses the `end_to_end` stage, which includes preprocessing and prediction but excludes file I/O and network transport. Thresholding is not included in the `predict_proba()` timing despite earlier prose mentioning thresholding; revise text accordingly or time thresholding explicitly.

Latency table source:

| model | median_ms | p95_ms | throughput_records_per_sec |
| --- | --- | --- | --- |
| Logistic Regression | 1.080 | 1.112 | 925.926 |
| Random Forest | 15.669 | 51.711 | 63.821 |
| XGBoost | 1.224 | 4.864 | 817.313 |
| Compact MLP | 1.160 | 1.690 | 862.007 |

Throughput check: `throughput_records_per_sec = 1000 / median_ms` exactly in `run_latency.py::summarize`. Recommended label: **Derived median-equivalent records/s**.

## E. Feature-ranking procedure

`RTS_Paper/scripts/data.py::rank_features_train_only` fits `SimpleImputer(strategy="median")` on the fourteen training runs, trains `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)`, sorts `feature_importances_` descending, and returns one shared ranking for that fold. `RTS_Paper/scripts/run_feature_budgets.py` uses the same fold-specific top-k list for all four primary models at k=64/32/16. The held-out run is not used for ranking. Selected-feature lists are not saved as standalone CSV, but are recoverable from cached top-k joblib `feature_order` values; those recovered lists are included in `rtc_reproducibility_values.json`.

Top feature-selection frequencies are in `rtc_reproducibility_values.json` under `feature_selection_frequency`.

## F. Feature-budget uncertainty

Machine-readable fold table: `rtc_feature_budget_folds.csv`.

| model | n_folds | auc_128_mean | auc_128_std | auc_16_mean | auc_16_std | mean_paired_diff_auc_16_minus_128 | std_paired_diff | median_paired_diff | folds_16_better | bootstrap95_ci_low | bootstrap95_ci_high | ci_method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 15 | 0.6870 | 0.0305 | 0.6294 | 0.0311 | -0.0577 | 0.0285 | -0.0601 | 1 | -0.0707 | -0.0430 | paired nonparametric bootstrap over 15 folds; 50000 resamples; seed=20260713 |
| Random Forest | 15 | 0.6768 | 0.0288 | 0.7214 | 0.0290 | 0.0446 | 0.0211 | 0.0456 | 15 | 0.0342 | 0.0548 | paired nonparametric bootstrap over 15 folds; 50000 resamples; seed=20260713 |
| XGBoost | 15 | 0.7015 | 0.0311 | 0.7078 | 0.0282 | 0.0063 | 0.0183 | 0.0021 | 9 | -0.0022 | 0.0157 | paired nonparametric bootstrap over 15 folds; 50000 resamples; seed=20260713 |
| Compact MLP | 15 | 0.6827 | 0.0223 | 0.7064 | 0.0272 | 0.0237 | 0.0189 | 0.0185 | 14 | 0.0148 | 0.0333 | paired nonparametric bootstrap over 15 folds; 50000 resamples; seed=20260713 |

Cautious interpretation: The 16-feature budget has positive mean paired AUC differences for Random Forest, XGBoost, and Compact MLP, but not Logistic Regression. Bootstrap confidence intervals are wide with only fifteen folds, so the safest claim is that the 16-feature configuration retained comparable mean discrimination for XGBoost/MLP and improved the saved mean for Random Forest in this benchmark; do not frame the increase as statistically established unless an inferential test is added and supports it.

## G. Missingness and relay-loss definitions

Random missingness: `run_degradation.py` uses rates 0.05, 0.10, 0.20, 0.30 and seeds 42,43,44,45,46. `degradations.py::random_missingness` applies an independent per-cell Bernoulli mask across all 128 model features in each held-out test fold. Labels and metadata are excluded because `X_test` contains only feature columns. Missingness is applied after model fitting and before the fitted preprocessor transforms test records; imputation medians come only from the training runs.

Aggregation: `missingness_results.csv` stores one row per fold/model/rate/seed. Fig. 5 groups by model/rate and plots mean ± standard deviation of **F1**, not ROC-AUC, across the combined fold-seed rows. Correct caption: "Run-aware F1 under test-time random missingness at 5%, 10%, 20%, and 30%; markers show the mean over fifteen held-out runs and five masking seeds, with error bars showing standard deviation across the saved fold-seed results."

Relay/group loss: unavailable values are represented as `NaN`; the cached model's train-fitted median imputer processes them. Relay loss is applied only to held-out test data. `logs_unavailable` masks control-panel logs, relay logs, and Snort logs. `all_pmu_unavailable` masks voltage, current, frequency/impedance, and relay-status columns. "Worst relay" in the placement table is the per-fold minimum ROC-AUC among R1-R4 followed by averaging those fold-wise minima, not one fixed relay selected globally.

Worst relay recovery:

| model | lowest_average_relay | lowest_average_auc | per_fold_worst_relay_counts | per_fold_worst_mean_auc |
| --- | --- | --- | --- | --- |
| Logistic Regression | R3 | 0.5065 | {'R3': 11, 'R2': 2, 'R4': 2} | 0.4993 |
| Random Forest | R4 | 0.5803 | {'R4': 15} | 0.5803 |
| XGBoost | R4 | 0.6260 | {'R4': 10, 'R3': 5} | 0.6117 |
| Compact MLP | R4 | 0.5714 | {'R4': 11, 'R2': 3, 'R1': 1} | 0.5644 |

Feature groups are listed in `rtc_reproducibility_values.json` under `feature_groups`.

## H. Edge-visible condition

Edge-visible means the R1-only feature set: 29 features whose names begin `R1-` or `R1:`. The same R1 feature set was used for all four primary models. Cyber/log variables are excluded. Edge-visible models were retrained using only these 29 features via `run_loro.py --feature-budget edge_r1`; edge-visible feature ranking was not performed. The reported edge-visible result is one predefined analytical feature-visibility proxy (R1), not an average/best/worst across all relay positions and not a deployed edge device.

## I. Threshold-analysis provenance

Source: `RTS_Paper/scripts/run_thresholds.py` and `thresholds.py`; outputs `threshold_stability_fold_results.csv`, `threshold_variability_summary.csv`, and `threshold_rule_macro_summary.csv`. For each fold and model, the cached central-128 model predicts on the fourteen training runs, a threshold is selected from those training predictions and labels, and the fixed threshold is applied to the held-out run. No held-out labels are used for threshold selection. Rules are fixed 0.5, maximum training F1, maximum Youden J, and constrained FPR with target FPR <= 0.10. Alerts per 1,000 are predicted positives divided by held-out records times 1000. The manuscript's four-row threshold table is the four-model mean across `threshold_rule_macro_summary.csv`, not a single model.

Four-model threshold means:

| threshold_rule | fpr_mean | recall_mean | alerts_per_1000_mean |
| --- | --- | --- | --- |
| fixed_0.5 | 0.755 | 0.898 | 856.228 |
| max_f1 | 0.724 | 0.868 | 826.205 |
| max_youden_j | 0.410 | 0.647 | 578.150 |
| constrained_fpr_10pct | 0.357 | 0.592 | 523.834 |

Caption: "Held-out-run alert behavior under thresholds selected from training-fold predictions; values are four-model means across leave-one-run-out folds for the central 128-feature condition."

## J. SHAP timing provenance

Source: `RTS_Paper/scripts/run_shap_timing.py`; output `shap_timing.csv`. Uses fold-01 cached Random Forest and XGBoost central-128 models, first 200 dataset rows, batch size one, `shap.TreeExplainer(fitted.estimator)`, and `explainer.shap_values(row)`. Explainer initialization is excluded from timing. Prediction and explanation are independently timed; combined median is calculated from per-row prediction time plus explanation time. Preprocessing and thresholding are excluded because rows are transformed once before timing (`Xt = fitted.preprocessor.transform(...)`). This explains why SHAP-table prediction latency differs from main detector-pipeline latency: the SHAP table times estimator-only prediction, while the main latency table times the full preprocessing-plus-prediction pipeline.

| model | pred_only_median_ms | pred_only_p95_ms | explain_only_median_ms | explain_only_p95_ms | pred_plus_explain_median_ms | pred_plus_explain_p95_ms | explanations_per_sec | fits_e3_every_record |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 14.459 | 28.675 | 881.275 | 972.549 | 897.631 | 996.312 | 1.135 | False |
| XGBoost | 0.125 | 0.221 | 1.126 | 1.325 | 1.263 | 1.516 | 888.050 | True |

Caption: "Estimator-only prediction and TreeSHAP timing on fold-01 central-128 cached models, measured over 200 single-record calls after pre-transforming the input rows; explainer construction and preprocessing are excluded."

## K. Figure audit

| file | width_px | height_px | dpi | single_column_legibility |
| --- | --- | --- | --- | --- |
| /Users/digdarshansubedi/Documents/Paper Publications/Saudi Arabia Sustainable Energy - Cuber Security Paper/RTS_Paper/figures/fig01_system_architecture.png | 947 | 753 | (299.9994, 299.9994) | likely OK |
| /Users/digdarshansubedi/Documents/Paper Publications/Saudi Arabia Sustainable Energy - Cuber Security Paper/RTS_Paper/figures/fig02_random_vs_runaware.png | 1065 | 795 | (299.9994, 299.9994) | likely OK |
| /Users/digdarshansubedi/Documents/Paper Publications/Saudi Arabia Sustainable Energy - Cuber Security Paper/RTS_Paper/figures/fig03_latency_envelopes.png | 1145 | 720 | (299.9994, 299.9994) | likely OK |
| /Users/digdarshansubedi/Documents/Paper Publications/Saudi Arabia Sustainable Energy - Cuber Security Paper/RTS_Paper/figures/fig04_payload_latency_quality.png | 2254 | 1317 | (299.9994, 299.9994) | dense at one-column; consider figure* or simplified labels if reviewer readability is a concern |
| /Users/digdarshansubedi/Documents/Paper Publications/Saudi Arabia Sustainable Energy - Cuber Security Paper/RTS_Paper/figures/fig05_missing_telemetry_robustness.png | 1680 | 1140 | (299.9994, 299.9994) | likely OK |
| /Users/digdarshansubedi/Documents/Paper Publications/Saudi Arabia Sustainable Energy - Cuber Security Paper/RTS_Paper/figures/fig06_edge_vs_central.png | 2160 | 1200 | (299.9994, 299.9994) | dense at one-column; consider figure* or simplified labels if reviewer readability is a concern |

LaTeX path: because the main TeX file is `RTS_Paper/paper/ieee_rtc_telemetry_budgets_draft.tex` and figures are in `RTS_Paper/figures/`, the correct local setting is `\graphicspath{{../figures/}}`. Fig. 5 plots F1. Fig. 4 currently labels the y-axis "End-to-end P95 latency" in the PNG; this should be interpreted as detector-pipeline/application-processing P95, not network end-to-end delay. Prefer changing that label in a future figure rebuild.

## L. Dataset references

Official dataset page: "Industrial Control System (ICS) Cyber Attack Datasets", Tommy Morris UAH page, Dataset 1: Power System Datasets, URL `https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets`, accessed 2026-07-13. The page names Uttam Adhikari, Shengyi Pan, and Tommy Morris, with Raymond Borges and Justin Beaver of ORNL, and describes synchrophasor measurements plus Snort/control-panel/relay logs.

IEEE-style candidate bibitems:

```latex
\bibitem{ics-datasets}
T. Morris, ``Industrial Control System (ICS) Cyber Attack Datasets,'' Dataset 1: Power System Datasets, Univ. of Alabama in Huntsville Google Sites. Accessed: Jul. 13, 2026. [Online]. Available: https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets

\bibitem{borges2014machine}
R. C. Borges Hink, J. M. Beaver, M. A. Buckner, T. Morris, U. Adhikari, and S. Pan, ``Machine learning for power system disturbance and cyber-attack discrimination,'' in Proc. 7th Int. Symp. Resilient Control Systems (ISRCS), Denver, CO, USA, 2014, Art. no. 6900095, doi: 10.1109/ISRCS.2014.6900095.

\bibitem{pan2015hybrid}
S. Pan, T. Morris, and U. Adhikari, ``Developing a hybrid intrusion detection system using data mining for power systems,'' IEEE Trans. Smart Grid, vol. 6, no. 6, pp. 3104--3113, 2015, doi: 10.1109/TSG.2015.2409775.

\bibitem{pan2015disturbances}
S. Pan, T. Morris, and U. Adhikari, ``Classification of disturbances and cyber-attacks in power systems using heterogeneous time-synchronized data,'' IEEE Trans. Ind. Informat., 2015, doi: 10.1109/TII.2015.2420951.
```

Do not invent a DOI for the dataset webpage.

## M. Number-validation table

Full CSV: `rtc_number_validation.csv`. Status counts: {'match': 116}.

Mismatches/unavailable rows:

No mismatches at 3-decimal precision in the requested validation set.

## N. Remaining unavailable information

- Exact machine model name and CPU marketing name for the saved latency run.
- OS version beyond `macOS-26.5.1-arm64-arm-64bit`.
- Power mode and CPU affinity control; no evidence they were controlled.
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` at latency-run time.
- Per-experiment full package snapshots for NumPy, pandas, SHAP, Torch, joblib, BLAS; repository inventory is an end-of-session capture, not per-run proof.
- Standalone saved feature-ranking CSVs; feature selections are recovered from cached model `feature_order`.
- Dataset webpage DOI; none found/claimed.

## O. Publication-ready text blocks

### Dataset subsection
We use the binary-labeled MSU/ORNL power-system attack dataset from the public ICS cyber-attack dataset collection. The audited local corpus contains fifteen source runs, 78,377 records, and 128 usable model features: 116 PMU/relay measurements and 12 control-panel, Snort, or relay-log variables. Binary labels contain 55,663 attack records and 22,714 normal or natural-event records. Each source file is assigned a run identifier used only for grouped evaluation and never included as a model feature.

### Models subsection
The primary evaluation uses four lightweight tabular detectors: Logistic Regression, Random Forest, XGBoost, and a compact MLP. All preprocessing is fitted inside the training partition of each leave-one-run-out fold. Tree models use median imputation; Logistic Regression and the compact MLP use median imputation followed by standardization. A feature-space CNN is reported only as a diagnostic prior-work baseline and is excluded from the new communication-budget experiments.

### Processing-budget methodology
Batch-size-one detector-pipeline latency is measured on cached fold-01 models using 500 warm-up calls and 5,000 timed calls with `time.perf_counter_ns()`. The reported latency table uses the full `predict_proba` path, including preprocessing and estimator prediction, and excludes file I/O, network transport, and operator-interface delay. Throughput is derived as `1000 / median latency in ms` and should be labeled median-equivalent records per second.

### Feature-budget methodology
For ranked 64-, 32-, and 16-feature budgets, a Random Forest with 100 trees is fitted only on each fold's fourteen training runs after training-fitted median imputation. Features are sorted by impurity importance, and the same fold-specific ranking is shared by all four primary models. Models are then retrained on the selected top-k features and evaluated on the held-out run.

### Availability methodology
Availability degradations are applied only to the held-out test fold and never refit preprocessing. Random missingness masks individual feature cells with independent Bernoulli draws at 5%, 10%, 20%, and 30% using seeds 42--46. Group and relay unavailability set the corresponding test-time features to NaN, which are then handled by the train-fitted median imputer.

### Threshold methodology
Thresholds are selected from cached-model predictions on the training runs and then applied unchanged to the held-out run. The evaluated rules are fixed 0.5, maximum training F1, maximum Youden J, and a training-FPR-constrained rule with target FPR <= 0.10. Alert volume is reported as held-out predicted positives per 1,000 records.

### SHAP methodology
TreeSHAP timing is measured for fold-01 Random Forest and XGBoost central-128 models using `shap.TreeExplainer`. The timing uses 200 single-record calls on pre-transformed rows and excludes explainer construction, preprocessing, thresholding, and file I/O. Therefore, SHAP prediction latencies are estimator-only and are not directly comparable to the main detector-pipeline latency table.

### Corrected figure captions
- Fig. 1: Communication-aware evaluation framework for record-level SCADA intrusion detection. Dashed paths denote missing telemetry, unavailable feature groups, and relay-channel loss; the diagram is conceptual and not a deployed architecture.
- Fig. 2: Random row-level splitting versus leave-one-run-out evaluation for previously validated baseline models. Bars report ROC-AUC.
- Fig. 3: Batch-size-one detector-pipeline P95 latency for the four primary models, with 10 ms, 16.7 ms, and 100 ms study reference budgets.
- Fig. 4: Modeled numeric payload bytes per record versus detector-pipeline P95 latency; marker size encodes run-aware ROC-AUC.
- Fig. 5: Run-aware F1 under test-time random missingness at 5%, 10%, 20%, and 30%; error bars show standard deviation across saved fold-seed results.
- Fig. 6: Model-specific ROC-AUC for edge-visible R1 telemetry and central telemetry under clean, 10% missing, logs-unavailable, and per-fold worst-relay-loss conditions.

## P. Machine-readable handoff

Created alongside this file:
- `rtc_reproducibility_values.json`
- `rtc_feature_budget_folds.csv`
- `rtc_claim_audit.csv`
- `rtc_number_validation.csv`
