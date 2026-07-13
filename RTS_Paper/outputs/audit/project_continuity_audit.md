# Phase A Continuity Audit — IEEE RTC 2026 Extension

Branch: `rts-paper-extension`. Generated during Phase A only; no experiments were run beyond
reading existing files and recomputing metrics already present in `outputs/`.

## 1. Dataset verification

`outputs/dataset_audit.json` claims and this audit's independent recount both agree:

| Check | Expected (plan) | Found | Match |
|---|---|---|---|
| Collection runs | 15 | 15 | yes |
| Total rows | 78,377 | 78,377 | yes |
| Attack rows | 55,663 | 55,663 | yes |
| Normal rows | 22,714 | 22,714 | yes |
| Usable features | 128 | 128 | yes |
| PMU features | 116 | 116 | yes |
| Cyber/log features | 12 | 12 | yes |
| Schema identical across all 15 files | yes | yes | yes |

See `RTS_Paper/outputs/audit/baseline_verification.json` for the full programmatic diff.

## 2. Baseline metric verification

Both tables in the planning documents were recomputed by loading the *existing* saved CSVs
(`outputs/metrics_naive.csv`, `outputs/metrics_groupkfold_macro.csv`) — no retraining — and
diffed against the claimed values. **Every value matches to 3 decimal places, zero discrepancies.**

**Verdict: baseline reproducible: YES**, directly from saved artifacts, no rerun needed.

## 3. Leakage / methodology check

- `run_id` is excluded from the feature matrix in `ordered_features()`/`feature_groups()`
  (`src/run_all.py:137-151`) and used only as the `GroupKFold` grouping key
  (`src/run_all.py:439,493,751`). No leakage path found.
- Imputer/scaler are fit only on `X_train` inside `fit_rf`/`fit_xgb`/`fit_cnn`
  (`src/run_all.py:308-380`) — training-only preprocessing confirmed for the existing 3 models.
- Threshold is currently hard-coded at 0.5 everywhere (`pred = prob >= 0.5`,
  `src/run_all.py:392`). No threshold-selection logic exists yet — this is new work, not a fix.

## 4. Feature-source map (new artifact this audit)

Built directly from the raw CSV header (`RTS_Paper/outputs/audit/feature_source_map.csv`) by
parsing the `R1-`/`R2-`/`R3-`/`R4-` prefixes vs. the 12 `control_panel_log*`, `relay*_log`,
`snort_log*` columns. Confirms 116 PMU + 12 cyber/log = 128.

**Proposed edge-visible definition (needs author sign-off before Phase D placement work):**
a single relay's own 29 measurements (e.g., R1 only — voltage/current magnitude+angle,
frequency, frequency-delta, impedance, impedance-angle, status) represent what one IED/relay
at a single substation bay position can observe locally without aggregation. The 12 cyber/log
features (control-panel, Snort, relay-log) are treated as substation-central because they are
not tied to a single bay in this dataset's schema. **This is a proposal, not a confirmed
definition** — per the ground rules, the placement analysis must not proceed until you confirm
or adjust it.

## 5. RTC draft LaTeX file — NOT FOUND

Neither `ieee_rtc_scada_low_latency_draft.tex` nor `ieee_rtc_telemetry_budgets_draft.tex`
exists anywhere in this repository (`find . -iname "*.tex"` returns only `paper_project/main.tex`
— the finished original paper — plus the auto-generated `outputs/*.tex` table snippets it
includes). The four documents pasted into this conversation are all **planning markdown**, not
the LaTeX draft itself.

**Consequence:** `resulttodo_map.csv` could not be built by extracting real `\resulttodo{}`
placeholders — I only inferred a checklist from the planning docs' prose. I cannot confirm exact
figure/table counts, exact placeholder wording, or the exact page budget without the actual
`.tex` file. **This blocks Phase F (fill placeholders) but not Phases B–E (the underlying
experiments are well-specified without the file).**

## 6. Original paper status

`paper_project/main.tex` is complete, compiles (`outputs/latex_compile_status.json` present),
and contains zero placeholders — confirmed frozen and untouched by this audit. Per ground
rules, it will not be modified.

## 7. Environment gap

Current Python environment has `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`,
`scipy` installed, but **`xgboost`, `shap`, and `torch` are NOT installed**
(`requirements.txt` declares all three but they're absent — see
`RTS_Paper/outputs/audit/repository_inventory.json`). This means:
- XGBoost baseline can be recomputed from saved CSVs (no retraining needed) but cannot be
  *retrained* in this environment until installed.
- CNN (torch) cannot be retrained here at all right now.
- SHAP timing (P0 requirement) cannot run at all until `shap` is installed.

This is a same-day fix (`pip install xgboost shap torch`) but must happen before Phase B/E
can execute new runs.

## 8. Pipeline architecture gap (relevant to Phase B design)

`src/run_all.py` is a single monolithic script hard-coded to exactly 3 models
(`FITTERS = {"Random Forest": fit_rf, "XGBoost": fit_xgb, "CNN": fit_cnn}`) and does **not**
persist per-fold fitted models — `rebuild_best_artifacts()` only refits the single
best-F1 fold per model, discarding the other 14. The plan's Phase B requires a config-driven
harness that (a) adds LR and MLP, (b) caches **all 15** fold models so degradation/threshold/SHAP
phases can reuse them without retraining. This is new harness code, not a fix to existing code —
existing `run_all.py` will be left untouched under `src/`, and the new harness will live under
`RTS_Paper/scripts/` importing shared pieces (`audit_and_load`, `ordered_features`,
`feature_groups`) from `src/run_all.py` where safe to reuse.

## 9. Reuse / derive / rerun summary

**Reuse as-is (no rerun):**
- Dataset audit and counts
- Random-split (naive) metrics + pooled predictions for RF/XGBoost/CNN
- Run-aware (LORO) fold metrics + pooled out-of-fold predictions for RF/XGBoost/CNN
- Leakage evidence, original figures, original paper LaTeX

**Derive from saved predictions (no retraining):**
- Threshold-stability rules 1–5, *if* per-fold (not just pooled) probability arrays in
  `predictions_groupkfold.csv` are confirmed complete for all 15 folds × 3 models (spot-checked:
  yes, `fold` and `heldout_run` columns present per row)

**Must run (P0):**
- LR + MLP fitters added to a new config-driven harness with per-fold model persistence
- Batch-1 latency benchmarking (new harness, cheap, do early — Phase C)
- Payload model + envelope verdicts (arithmetic, cheap, do early)
- Missingness / group-unavailability / relay-loss test-time transforms
- SHAP timing for RF+XGB (after `pip install shap`)
- Edge-visible feature mapping — author confirmation, then placement table
- Headline `fig_tradespace`

**Must run (P1, cut first if behind schedule):**
- Feature-budget experiments (128/64/32/16/PMU-only/cyber-log-only)
- Wilcoxon + effect sizes + Holm correction
- CNN reproduction (needs torch install; documented as one-paragraph limitation if unstable)

**P2 (only if time remains):** Gaussian noise, quantization.

## 10. Risks

1. **RTC draft `.tex` missing** — blocks final placeholder-fill (Phase F) until provided. Does
   not block Phases B–E.
2. **xgboost/shap/torch not installed** — blocks any new XGBoost/CNN training and all SHAP
   timing until `pip install -r requirements.txt` succeeds in this environment.
3. **No per-fold model persistence in existing code** — Phase B harness must be built fresh
   (config-driven, caches all 15 folds), not adapted from `run_all.py`'s single-best-fold pattern.
4. **Edge-visible feature definition is unconfirmed** — placement analysis (Phase D task D)
   cannot start until authors approve or revise the proposal in §4.
5. **Threshold logic doesn't exist yet** — currently only 0.5 is used anywhere; the 5 threshold
   rules are new logic, though can run purely against saved probabilities.
6. Large dataset directory (~78 MB of CSVs) is already committed to `main` — fine for GitHub,
   just noting it inflates every future push slightly.

## 11. Exact commands

**Existing pipeline (reproduces original paper end-to-end, idempotent — skips already-completed
naive/GroupKFold stages by checking for existing output files):**
```bash
cd "/Users/digdarshansubedi/Documents/Paper Publications/Saudi Arabia Sustainable Energy - Cuber Security Paper"
pip install -r requirements.txt   # needed once for xgboost/shap/torch
bash run_all.sh
# equivalent to: python3 -m src.run_all
```

**Proposed RTC extension pipeline (Phase B onward, not yet built — proposed entry point):**
```bash
cd "/Users/digdarshansubedi/Documents/Paper Publications/Saudi Arabia Sustainable Energy - Cuber Security Paper"
git checkout rts-paper-extension
pip install -r requirements.txt
python3 -m RTS_Paper.scripts.run_loro_harness --config RTS_Paper/configs/base_loro.yaml   # Phase B
python3 -m RTS_Paper.scripts.run_latency --config RTS_Paper/configs/latency.yaml          # Phase C
python3 -m RTS_Paper.scripts.run_degradation --config RTS_Paper/configs/degradation.yaml  # Phase D
python3 -m RTS_Paper.scripts.run_thresholds_shap --config RTS_Paper/configs/post.yaml     # Phase E
python3 -m RTS_Paper.scripts.build_figures_tables                                          # Phase F
```
(Scripts/configs referenced above do not exist yet — this is the proposed Phase B+ layout,
pending your approval.)

## 12. Open questions requiring your confirmation before Phase B starts

1. Confirm or revise the proposed edge-visible feature definition (§4).
2. Provide the actual `ieee_rtc_*.tex` draft file so `resulttodo_map.csv` can be built from
   real placeholders instead of inferred from planning-doc prose.
3. Confirm it's acceptable to `pip install xgboost shap torch` in this environment now.
4. Confirm the new harness should live under `RTS_Paper/scripts/` importing read-only from
   `src/run_all.py` (reusing `audit_and_load`, `feature_groups`, `ordered_features` only),
   rather than modifying `src/run_all.py` itself.
