# Human Verification Packet — IEEE RTC 2026 Draft

Generated 2026-07-13 for `RTS_Paper/paper/ieee_rtc_telemetry_budgets_draft.tex`.
This is a reporting task only — no further edits were made to the `.tex` as part of assembling this packet.
All commands below were actually run against the repository state at the time of writing; outputs shown are real, not illustrative.

---

## 1. Numeric spot-check table

Repo root for all relative paths: `/Users/digdarshansubedi/Documents/Paper Publications/Saudi Arabia Sustainable Energy - Cuber Security Paper/`

### 1.1 Random Forest P95 end-to-end latency (51.711 ms)
- **As written:** `.tex` line 291: `Random Forest & 15.669 & 51.711 & 64 & -- & -- & \checkmark \\`; also restated in prose line 299 as "P95~=~51.7~ms"
- **Source:** `RTS_Paper/outputs/metrics/latency_and_envelope_verdicts.csv`, row where `model == "Random Forest"` and `stage == "end_to_end"`, column `p95_ms`
- **Reproduce:**
  ```
  python3 -c "
  import pandas as pd
  df = pd.read_csv('RTS_Paper/outputs/metrics/latency_and_envelope_verdicts.csv')
  print(df[(df.model=='Random Forest') & (df.stage=='end_to_end')][['median_ms','p95_ms','p99_ms']])
  "
  ```
  Output: `median_ms=15.668855 p95_ms=51.711376 p99_ms=79.846388`

### 1.2 Top-16 ranked feature budget F1 (0.821) and AUC (0.691)
- **As written:** `.tex` line 322: `16 (ranked) & 0.821 & 0.691 & 4.5 & 64 \\`; also line 330 prose
- **Source:** `RTS_Paper/outputs/metrics/feature_budget_ranked_macro.csv`, rows where `feature_budget == 16`, columns `f1_mean`, `roc_auc_mean`, averaged across the 4 rows (one per model)
- **Reproduce:**
  ```
  python3 -c "
  import pandas as pd
  r = pd.read_csv('RTS_Paper/outputs/metrics/feature_budget_ranked_macro.csv')
  b16 = r[r.feature_budget==16]
  print(b16[['model','f1_mean','roc_auc_mean']])
  print('mean F1:', b16.f1_mean.mean(), 'mean AUC:', b16.roc_auc_mean.mean())
  "
  ```
  Output: mean F1 = `0.8206818980259257` → 0.821; mean AUC = `0.6912399918888184` → 0.691
  Per-model: Compact MLP 0.830/0.706, Logistic Regression 0.829/0.629, Random Forest 0.817/0.721, XGBoost 0.807/0.708

### 1.3 Full-128-feature budget F1 (0.813) and AUC (0.687)
- **As written:** `.tex` line 319: `128 & 0.813 & 0.687 & 14.8 & 512 \\`
- **Source:** `RTS_Paper/outputs/metrics/phase_b_loro_clean_macro.csv`, all 4 model rows, columns `f1_mean`, `roc_auc_mean`
- **Reproduce:**
  ```
  python3 -c "
  import pandas as pd
  m = pd.read_csv('RTS_Paper/outputs/metrics/phase_b_loro_clean_macro.csv')
  print(m[['model','f1_mean','roc_auc_mean']])
  print('mean F1:', m.f1_mean.mean(), 'mean AUC:', m.roc_auc_mean.mean())
  "
  ```
  Output: mean F1 = `0.8133431914139999` → 0.813; mean AUC = `0.6870100908488843` → 0.687

### 1.4 Logistic Regression worst-relay-loss central AUC (0.499) and edge-only AUC (0.531)
- **As written:** `.tex` line 343 (table row) and line 382 (prose): "central AUC falls to 0.499...edge-only AUC (0.531)"
- **Source:** `RTS_Paper/outputs/metrics/placement_comparison_table.csv`, rows where `model == "Logistic Regression"`, column `roc_auc`, for `placement_condition` in `{"Edge (edge-visible features)", "Central, one relay lost (worst)"}`
- **Reproduce:**
  ```
  python3 -c "
  import pandas as pd
  p = pd.read_csv('RTS_Paper/outputs/metrics/placement_comparison_table.csv')
  print(p[p.model=='Logistic Regression'][['placement_condition','roc_auc']])
  "
  ```
  Output: Edge = `0.530577` → 0.531; Central worst-relay-lost = `0.499256` → 0.499
- **Upstream of this file:** the "worst relay lost" value is itself derived from `RTS_Paper/outputs/robustness/relay_loss_results.csv` by, for each fold, taking the relay condition with the minimum `roc_auc` in that fold, then averaging that per-fold minimum across the 15 folds (built by `RTS_Paper/scripts/run_placement.py`, function `main()`, lines computing `worst_relay_per_fold`). This is a per-fold-worst definition, not a single fixed "worst relay on average" — reproducible via:
  ```
  python3 -c "
  import pandas as pd
  r = pd.read_csv('RTS_Paper/outputs/robustness/relay_loss_results.csv')
  lr = r[r.model=='Logistic Regression']
  worst = lr.loc[lr.groupby('fold_id')['roc_auc'].idxmin()]
  print(worst.roc_auc.mean())
  "
  ```

### 1.5 Random Forest TreeSHAP median explanation time (881.275 ms) and XGBoost's (1.126 ms)
- **As written:** `.tex` line 412: `Random Forest & 14.459 & 881.275 & 897.631 & -- \\`; line 413: `XGBoost & 0.125 & 1.126 & 1.263 & \checkmark \\`
- **Source:** `RTS_Paper/outputs/metrics/shap_timing.csv`, column `explain_only_median_ms`
- **Reproduce:**
  ```
  python3 -c "
  import pandas as pd
  s = pd.read_csv('RTS_Paper/outputs/metrics/shap_timing.csv')
  print(s[['model','pred_only_median_ms','explain_only_median_ms','pred_plus_explain_median_ms']])
  "
  ```
  Output: RF `explain_only_median_ms=881.275355`; XGBoost `explain_only_median_ms=1.126063`
- **Sample size caveat:** these medians are computed over `N_TIMED = 200` explanation calls on the first 200 rows of the dataset (fold-01 models only), not all 15 folds — see `RTS_Paper/scripts/run_shap_timing.py`, `N_TIMED` constant. This is a smaller sample than the 5,000-iteration latency benchmark and was not cross-validated across folds.

### 1.6 Logistic Regression max-F1 threshold std (0.213) and min/max (0.0007/0.492)
- **As written:** `.tex` line 418: "swings from 0.0007 to 0.492 across the fifteen held-out runs (std 0.213)"
- **Source:** `RTS_Paper/outputs/metrics/threshold_variability_summary.csv`, row where `model == "Logistic Regression"` and `threshold_rule == "max_f1"`, columns `std`, `min`, `max`
- **Reproduce:**
  ```
  python3 -c "
  import pandas as pd
  v = pd.read_csv('RTS_Paper/outputs/metrics/threshold_variability_summary.csv')
  print(v[(v.model=='Logistic Regression') & (v.threshold_rule=='max_f1')])
  "
  ```
  Output: `mean=0.34157 std=0.212851 min=0.000654 max=0.492357`
- **Note:** this table's numbers were produced by the O(n log n) threshold sweep in `RTS_Paper/scripts/thresholds.py::threshold_max_f1` — see Section 3.1 below for the bug history on this exact function.

### 1.7 PMU-group-unavailable AUC range across models (0.506–0.531)
- **As written:** `.tex` line 359: "collapses every model to near-chance discrimination (ROC-AUC 0.51--0.53)"
- **Source:** `RTS_Paper/outputs/robustness/pmu_group_unavailable_results.csv`, column `roc_auc`, grouped by `model`
- **Reproduce:**
  ```
  python3 -c "
  import pandas as pd
  p = pd.read_csv('RTS_Paper/outputs/robustness/pmu_group_unavailable_results.csv')
  print(p.groupby('model')['roc_auc'].mean().sort_values())
  "
  ```
  Output: RF 0.506340, XGBoost 0.508532, Compact MLP 0.517681, Logistic Regression 0.531202 — range 0.506–0.531, textually rounded to "0.51--0.53"
- **UPDATE (resolved):** the transform is now saved as `RTS_Paper/scripts/degradations.py::all_pmu_unavailable` (masks pmu_voltage + pmu_current + freq_impedance + relay_status = 116 columns, asserted in code) and a standalone runner `RTS_Paper/scripts/run_pmu_ablation.py`. Rerun from scratch after saving; the original inline-generated file was backed up first and diffed against the rerun output. **Result: bit-for-bit identical** on all 6 numeric columns (precision/recall/f1/fpr/roc_auc/pr_auc) across all 60 rows (15 folds × 4 models) — max absolute difference `0.0`. This is no longer a reproducibility gap; run with `python3 -m RTS_Paper.scripts.run_pmu_ablation`.

### 1.8 Random Forest model size in MB (178.45 MB)
- **As written:** **NOT CURRENTLY STATED IN THE DRAFT.** The draft's Methodology section (lines 212, 249) lists "model size" as a metric category that is measured and reported, but no table or sentence in the current `.tex` states the actual 178.45 MB value anywhere. This number exists only in the source data, not yet in the paper text.
- **Source:** `RTS_Paper/outputs/metrics/latency_and_envelope_verdicts.csv`, row where `model == "Random Forest"` and `stage == "end_to_end"`, column `model_size_bytes`
- **Reproduce:**
  ```
  python3 -c "
  import pandas as pd
  df = pd.read_csv('RTS_Paper/outputs/metrics/latency_and_envelope_verdicts.csv')
  row = df[(df.model=='Random Forest') & (df.stage=='end_to_end')]
  print(row.model_size_bytes.iloc[0], row.model_size_bytes.iloc[0]/1e6, 'MB')
  "
  ```
  Output: `178447071.0 178.447071 MB`

### 1.9 4-model mean P95 latency used in the placement table (14.844 ms)
- **As written:** `.tex` line 372: `Edge (edge-visible features) & 0.556 & 0.805 & 14.8 \\` and line 373 uses `14.8 + 5 = 19.8`
- **Source:** `RTS_Paper/outputs/metrics/latency_and_envelope_verdicts.csv`, `stage == "end_to_end"`, mean of `p95_ms` across all 4 models
- **Reproduce:**
  ```
  python3 -c "
  import pandas as pd
  lat = pd.read_csv('RTS_Paper/outputs/metrics/latency_and_envelope_verdicts.csv')
  e2e = lat[lat.stage=='end_to_end']
  print(e2e[['model','p95_ms']]); print('mean:', e2e.p95_ms.mean())
  "
  ```
  Output: mean = `14.844330437500007`
- **Caveat already flagged in the draft's own italic footnote (line 379):** this mean is dominated by Random Forest's 51.7 ms — averaging is stated explicitly as an approximation, not a claim that all 4 models share this latency.

### 1.10 XGBoost run-aware ROC-AUC (0.702)
- **As written:** `.tex` line 299: "XGBoost has the highest run-aware ROC-AUC (0.702)"
- **Source:** `RTS_Paper/outputs/metrics/phase_b_loro_clean_macro.csv`, row `model == "XGBoost"`, column `roc_auc_mean`
- **Reproduce:**
  ```
  python3 -c "
  import pandas as pd
  m = pd.read_csv('RTS_Paper/outputs/metrics/phase_b_loro_clean_macro.csv')
  print(m[m.model=='XGBoost'][['roc_auc_mean','roc_auc_std']])
  "
  ```
  Output: `roc_auc_mean=0.70153 roc_auc_std=0.03111`
- **Cross-check:** this is the NEW harness's independently-recomputed XGBoost number, not the original paper's. The original paper's XGBoost run-aware AUC (`outputs/metrics_groupkfold_macro.csv`, which feeds Table IV's fixed XGBoost row) is `0.7044786863477569` — close but not identical (0.702 vs 0.704), because it is a genuinely separate training run with a different codebase, not a copy. See Section 6 for which table uses which source.

### 1.11 Cyber/log-only F1 (0.828) vs AUC (0.538)
- **As written:** `.tex` line 324: `Cyber/log only (12) & 0.828 & 0.538 & 4.5 & 48 \\`; also line 330 prose calls this "deceptively high F1"
- **Source:** `RTS_Paper/outputs/metrics/phase_b_loro_clean_macro_cyber_log_only.csv`, columns `f1_mean`, `roc_auc_mean`, averaged across the 4 model rows
- **Reproduce:**
  ```
  python3 -c "
  import pandas as pd
  m = pd.read_csv('RTS_Paper/outputs/metrics/phase_b_loro_clean_macro_cyber_log_only.csv')
  print(m[['model','f1_mean','roc_auc_mean']])
  print('mean F1:', m.f1_mean.mean(), 'mean AUC:', m.roc_auc_mean.mean())
  "
  ```
  Output: mean F1 = `0.8280424975819658` → 0.828; mean AUC = `0.5375835695000606` → 0.538 (all 4 models individually cluster tightly around F1≈0.827-0.828, AUC≈0.531-0.540)

### 1.12 Edge-visible feature count (29) and cyber/log exclusion
- **As written:** `.tex` line 158: "the 29 measurements local to a single relay/PMU position...verified programmatically to contain none of the 12 control-panel/Snort/relay-log columns"
- **Source:** `RTS_Paper/outputs/audit/feature_source_map.csv`, filtered to `relay_or_group == "R1"`
- **Reproduce:**
  ```
  python3 -c "
  import pandas as pd
  fmap = pd.read_csv('RTS_Paper/outputs/audit/feature_source_map.csv')
  r1 = fmap[fmap.relay_or_group=='R1']
  print('count:', len(r1))
  print('any cyber/log in R1:', (r1.source_category=='cyber/log').any())
  "
  ```
  Output: `count: 29`; `any cyber/log in R1: False`

---

## 2. Full environment and versioning record

### 2.1 `RTS_Paper/outputs/timings/environment.json` (full contents)
```json
{
  "platform": "macOS-26.5.1-arm64-arm-64bit",
  "processor": "arm",
  "python_version": "3.9.6",
  "cpu_count_logical": 10,
  "cpu_count_physical": 10,
  "total_memory_gb": 32.0,
  "sklearn_version": "1.6.1",
  "xgboost_version": "2.1.4",
  "warmup_iterations": 500,
  "timed_iterations": 5000,
  "batch_size": 1
}
```
Note: this file is written by `run_latency.py` and only captures the packages it directly imports (sklearn, xgboost); it does not enumerate the full environment. Section 2.2 below is the complete `pip freeze` for the packages actually in use.

### 2.2 Exact package versions (`pip freeze`, filtered to relevant packages, captured at packet-assembly time — not captured per-experiment)
```
joblib==1.4.2
matplotlib==3.9.4
matplotlib-inline==0.1.7
numpy==2.0.2
pandas==2.2.3
psutil==6.1.1
pyarrow==21.0.0
PyYAML==6.0.3
scikit-learn==1.6.1
scipy==1.13.1
seaborn==0.13.2
shap==0.49.1
torch==2.8.0
xgboost==2.1.4
```
**Caveat:** no per-experiment package-version snapshot was taken. xgboost, shap, and torch were installed mid-session (see Section 3 bug log); everything after that install used these versions consistently, but there is no automated proof each individual script run used exactly this version set beyond this single end-of-session capture and the fact that no packages were reinstalled or changed afterward.

### 2.3 Git commit hash at each phase boundary
**There is only one commit on `rts-paper-extension`.** Requested per-phase-boundary commits (B, C, D, E, F) **do not exist** — this was not caught or flagged during the session and is being surfaced now.

```
$ git log --oneline --all
1c470ef Phase A audit for IEEE RTC 2026 extension: verify baseline, map artifacts, scaffold RTS_Paper/
2d591e7 Initial commit: SCADA leakage-aware evaluation paper, pipeline, and figures

$ git rev-parse HEAD
1c470efc274472be81c1302d19b1d2340ccf4d1b

$ git branch --show-current
rts-paper-extension
```

All of Phases B through F — every script under `RTS_Paper/scripts/`, every output under `RTS_Paper/outputs/{metrics,predictions,robustness,timings}/`, every figure, and every edit to the `.tex` — exist **only in the uncommitted working tree** on top of commit `1c470ef`. `git status --short` currently shows dozens of untracked/modified files (full list reproducible with `git status --short` from repo root). **This means there is no git-level provenance separating Phase B output from Phase F output** — only filesystem timestamps (Section 2.4) and the narrative order in this conversation.

### 2.4 Script-to-output correspondence via filesystem timestamps
Since git commits don't separate phases, timestamps are the only available reconciliation evidence. Key script/output timestamp pairs relevant to the two known bugs (Section 3):

```
$ stat -f "%Sm %N" -t "%Y-%m-%d %H:%M:%S" RTS_Paper/scripts/thresholds.py RTS_Paper/outputs/metrics/threshold_stability_fold_results.csv
2026-07-13 06:23:37 RTS_Paper/scripts/thresholds.py
2026-07-13 06:24:16 RTS_Paper/outputs/metrics/threshold_stability_fold_results.csv

$ stat -f "%Sm %N" -t "%Y-%m-%d %H:%M:%S" RTS_Paper/scripts/run_latency.py RTS_Paper/outputs/metrics/latency_and_envelope_verdicts.csv RTS_Paper/outputs/metrics/envelope_verdicts.csv
2026-07-13 06:02:17 RTS_Paper/scripts/run_latency.py
2026-07-13 06:16:44 RTS_Paper/outputs/metrics/latency_and_envelope_verdicts.csv
2026-07-13 06:16:44 RTS_Paper/outputs/metrics/envelope_verdicts.csv
```
The output timestamps postdate the final script edits in both cases, consistent with (but not cryptographic proof of) the fixed code having produced the final files. There is no checksum or run-log tying a specific script hash to a specific output file beyond this.

---

## 3. Every bug/mistake caught during the session

### 3.1 Threshold selection O(n²) performance bug (caught before any output was produced — no data was ever wrong)
- **What was wrong:** the original `threshold_max_f1` implementation in `RTS_Paper/scripts/thresholds.py` iterated over every unique training-fold probability value (up to ~66,000 for continuous-output models) and recomputed `f1_score` from scratch each time — O(n²) in the worst case.
- **What it could have affected:** `threshold_stability_fold_results.csv`, `threshold_variability_summary.csv`, `threshold_rule_macro_summary.csv`, and every downstream number in the draft's Table VII (thresholds) and Table VIII (SHAP, indirectly via the same script family) and the threshold-related prose (line 418).
- **How/when caught:** the job was launched, observed still running after ~4 minutes with memory climbing, and was killed manually before it produced any output file. It never completed and never wrote a CSV.
- **Fix:** replaced with an O(n log n) cumulative-sum sweep (sort by score descending, incrementally track TP/FP).
- **Verification before trusting the fix:** ran a 500-sample synthetic test comparing the new function's output against the original brute-force reference implementation; confirmed bit-identical threshold and F1 score before rerunning on real data (this comparison was not saved to a file — it was a one-off interactive check, reproducible by re-implementing the brute-force loop and diffing against `thresholds.py::threshold_max_f1`).
- **Regeneration confirmation:** ALL threshold output files were generated fresh, exactly once, by the fixed function — because the buggy version never completed a run, there is no "old" data to have missed regenerating. Confirmed no threshold CSV predates the fix via the timestamps in Section 2.4.

### 3.2 PMU-only / cyber-log-only checklist mislabel (caught before any number reached the paper — process error, not a data error)
- **What was wrong:** `RTS_Paper/outputs/audit/EXECUTION_CHECKLIST.md` was written at one point stating "PMU-only/cyber-log-only structural budgets ARE done via Phase B/D" when in fact only the `edge_r1` feature-budget LORO run had actually been executed; `run_loro.py --feature-budget pmu_only` and `--feature-budget cyber_log_only` had never been run.
- **What it could have affected:** the feature-budget table (Table VI) rows for "PMU only" and "Cyber/log only" — if the `.tex` had been filled from the (nonexistent) files the checklist claimed existed, those cells would have been fabricated or left blank.
- **How/when caught:** while assembling the feature-budget table content, a directory listing (`ls RTS_Paper/outputs/metrics/ | grep -E "pmu_only|cyber_log_only"`) returned nothing, contradicting the checklist claim.
- **Fix:** ran both experiments (`run_loro.py --feature-budget pmu_only` then `--feature-budget cyber_log_only`) before writing anything into the `.tex`.
- **Regeneration confirmation:** no incorrect numbers were ever written to the `.tex` for this — the mislabel was caught and corrected during table-assembly, before any text was produced from it. The checklist file itself was corrected in the same edit.

### 3.3 Envelope-verdict row duplication bug in `run_latency.py` (caught by manual inspection, fixed, rerun before any use in the paper)
- **What was wrong:** the original loop structure in `run_latency.py` nested the `d_agg` loop outside the edge-vs-central branching, causing the "edge" placement row (which doesn't vary with `d_agg`) to be written 3 times per envelope instead of once — `envelope_verdicts.csv` had 3 identical duplicate rows for every edge/envelope combination.
- **What it could have affected:** `envelope_verdicts.csv` row counts and any aggregate statistics computed naively over that file (e.g., a `groupby().mean()` would have been unaffected since duplicate identical rows don't change a mean, but a `.count()` or unweighted concatenation into another table could have been wrong).
- **How/when caught:** manual inspection of the output (`df[df.model=='Random Forest']` printed to check shape) showed 12 rows for Random Forest where 4 were expected (3 envelopes × edge/central mix) — visually obvious duplication.
- **Fix:** moved the edge-row append outside the `d_agg` loop so it's written once per envelope.
- **Regeneration confirmation:** reran `run_latency.py` completely after the fix; verified the corrected file has exactly 48 rows (4 models × 3 envelopes × 4 placement rows [1 edge + 3 central d_agg values]) before any number from this file was used anywhere in the `.tex`. No prose or table in the draft was written from the buggy version — the fix happened before Phase F table-filling began.

### 3.4 CNN segfault, workaround, and time-based cut (not a data-correctness bug, but a scope decision worth full disclosure)
- **What happened:** `fit_feature_cnn` (added to `RTS_Paper/scripts/models.py`, reusing `src.run_all.FeatureCNN`) crashed with exit code 139 (SIGSEGV) on first invocation — a known macOS conflict between torch's and xgboost's bundled OpenMP runtimes when both are loaded in the same process.
- **Fix attempted:** setting `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` stopped the crash (confirmed: process ran without crashing for 9:45 minutes on a 2-fold smoke test, versus instant crash before).
- **Why cut instead of completed:** the workaround forces single-threaded CPU training; even so, the 2-fold smoke test had not converged after 9:45 elapsed, versus 2-20 seconds/fold for the other 4 models. Extrapolated full-15-fold cost was well over an hour for a P1-tier stretch model. Killed and cut per the plan's own stated cut-line ("Drop CNN entirely").
- **What it affected:** no CNN numbers appear anywhere in the new experiments (Tables V, VI robustness cells all say "Not evaluated (excluded; see Limitations)"). Table IV's CNN row uses the **original paper's** pre-existing validated CNN numbers (see Section 6), not anything from this session's harness — those were never at risk from the crash since they were never recomputed.
- **Outstanding gap (see Section 6):** the Limitations section of the `.tex` does **not** currently contain a sentence explaining the CNN exclusion, despite two table cells (lines 294, 347) saying "see Limitations." A limitation paragraph was drafted into `EXECUTION_CHECKLIST.md` but was never actually inserted into the `.tex`. This is a live inconsistency in the current draft.

### 3.5 LaTeX syntax slip: unclosed `\emph{}` brace (caught immediately, fixed same turn)
- **What was wrong:** while writing the placement-table footnote, an `\emph{...}` was left unclosed across a paragraph break, which would have broken compilation (everything after it would have rendered in italics or thrown an error).
- **How/when caught:** re-reading the edit immediately after making it, before compiling.
- **Fix:** closed the brace.
- **Regeneration confirmation:** not applicable (this was caught before any compile attempt, no data affected).

### 3.6 `\captionof{figure}` used without the `caption` package (caught by the first compile attempt)
- **What was wrong:** the missingness figure was initially inserted using `\captionof{figure}{...}`, which requires the `caption` package — not loaded in this document's preamble (adapted from a different template).
- **How/when caught:** would have surfaced as a compile error; caught by inspection before compiling and replaced proactively.
- **Fix:** wrapped in a standard `\begin{figure}...\end{figure}` environment instead.

### 3.7 `pcrr7t` font-cache compile failure (caught by first actual pdflatex run)
- **What was wrong:** `\texttt{}` usage triggered pdfTeX to try to generate a bitmap Courier font (`pcrr7t`) via `mktextfm`, which failed on this machine's TeX Live install (`! I can't find file 'pcrr7t'`), producing a fatal compile error with **zero PDF output**.
- **How/when caught:** first `pdflatex` compile attempt on the completed draft failed outright.
- **Fix:** added `\renewcommand{\ttdefault}{cmtt}` to the preamble — the exact same fix already present in the original paper's auto-generated LaTeX (`src/run_all.py::write_paper`), so this was a known, previously-solved issue in this repository that the new draft's preamble simply hadn't inherited.
- **Regeneration confirmation:** recompiled twice after the fix (two-pass pdflatex for cross-references); final PDF is 7 pages, confirmed via `pdftoppm` visual rendering of pages 4-5 that tables and the two included figures render without errors or overlap.

### 3.8 `fig04_payload_latency_quality` legend overlapping data points (cosmetic, caught by visual inspection, fixed twice)
- **What was wrong:** the headline trade-space figure's legend, first placed inside the plot area then via `bbox_to_anchor` outside it, visually overlapped the rightmost cluster of data points in both attempts because `fig.tight_layout()` doesn't reserve canvas space for a legend placed outside the axes.
- **How/when caught:** visual inspection of the rendered PNG after each build.
- **Fix:** used `bbox_inches="tight"` on `fig.savefig()` instead of `fig.tight_layout()`, which expands the canvas to include the external legend.
- **Regeneration confirmation:** rebuilt via `build_figures.py`, re-inspected visually, confirmed no overlap in the final version embedded in the compiled PDF.

### 3.9 No other bugs identified
No other data-correctness, script, or LaTeX issues were identified or fixed during this session beyond the 8 items above. This is a self-report by the same session that did the work, not an independent audit — treat it as a starting checklist for your own review, not a substitute for it.

---

## 4. Citation-critical claims needing standards verification

Every sentence making a specific numeric or factual claim attributed to IEC 61850-5, IEC TR 61850-90-4, or IEEE C37.118, quoted exactly as written, with line number. **Not independently verified against the standards texts by this session, per your instruction.**

**Line 68:** "Phasor measurement units (PMUs), protective relays, merging units, and station gateways exchange periodic and event-driven telemetry whose timing is bounded by standardized transfer-time requirements \cite{iec61850-5,iec61850-90-4}."

**Line 87 (RQ1):** "Which detectors satisfy IEC~61850-derived and synchrophasor-derived per-record timing envelopes at batch size one, and at what accuracy cost?"

**Line 112:** "IEC~61850 defines message performance classes with transfer-time requirements ranging from trip-class messages at a few milliseconds to operational and reporting traffic at tens to hundreds of milliseconds \cite{iec61850-5,iec61850-90-4}. Synchrophasor streams governed by IEEE~C37.118 report at rates up to the nominal system frequency, bounding the inter-record interval available to any per-record analytic \cite{ieee-c37118}."

**Line 165 (the most load-bearing sentence — this is where the exact envelope numbers come from):** "IEC~61850-5 groups substation messages into performance classes whose transfer-time requirements span roughly 3~ms for trip-class messages, 10--20~ms for fast automation interactions, and 100~ms or more for operational and reporting traffic \cite{iec61850-5}. A per-record analytic that consumes synchrophasor input is additionally bounded by the reporting interval: at C37.118 reporting rates of 30 and 60 frames per second, a new record arrives every 33.3~ms and 16.7~ms respectively, so sustained per-record processing must complete within that interval to avoid unbounded queuing \cite{ieee-c37118}."

**Lines 178–180 (Table II, the envelope definitions used for every fits/fails verdict in the paper):**
```
E1 (strict) & 10~ms & Fast-message class, IEC~61850-5 \\
E2 (streaming) & 16.7~ms & 60~fps C37.118 inter-record interval \\
E3 (operational) & 100~ms & Operational-message class, IEC~61850-5 \\
```

**Bibliography entries citing the standards directly (lines 494–503):**
```
Line 494-496: \bibitem{iec61850-5}
IEC, ``Communication networks and systems for power utility automation --- Part 5: Communication requirements for functions and device models,'' IEC 61850-5, 2013.
% AUTHORS: verify edition/year and the transfer-time class values you cite in Sec. IV-A against the actual standard text.

Line 498-499: \bibitem{iec61850-90-4}
IEC, ``Communication networks and systems for power utility automation --- Part 90-4: Network engineering guidelines,'' IEC TR 61850-90-4, 2020.
% AUTHORS: verify edition/year.

Line 502-503: \bibitem{ieee-c37118}
IEEE, ``IEEE standard for synchrophasor data transfer for power systems,'' IEEE Std C37.118.2-2011, 2011.
% AUTHORS: verify part (measurement .1 vs data transfer .2) and reporting-rate statement.
```

The draft itself already flags all three standards citations with `% AUTHORS: verify...` comments — these were present in the draft as uploaded, not added by this session. They are reproduced here for completeness, not as a new finding.

---

## 5. Bibliography placeholder candidates

**Not selected, not inserted — candidates only, for you to evaluate and verify.** Found via web search during this session; full bibliographic details (exact author lists, page numbers) were not independently confirmed beyond what the search results returned, and you should pull the canonical citation from the DOI/publisher page before using any of these.

### Slot 1 — `substation-comms-survey` (line 507): digital-substation communication engineering / latency verification
1. **"Direct Evaluation of IEC 61850-9-2 Process Bus Network Performance"** — IEEE Journals & Magazine. Presents a technique to assess sampled-value process bus network performance from single-point measurements using externally time-synchronized Ethernet capture; directly measures switch-induced latency. Strong topical fit for the "latency verification" framing. https://ieeexplore.ieee.org/document/6244823/
2. **"Performance Analysis of IEC 61850 Sampled Value Process Bus Networks"** — evaluates a 132kV digital protection scheme with IEC 61850-9-2 IEDs using OPNET simulation. https://www.researchgate.net/publication/260512686_Performance_Analysis_of_IEC_61850_Sampled_Value_Process_Bus_Networks
3. **"A methodology for the evaluation of the message transmission delay over IEC 61850 communication network — a real-time HV/MV substation case study"** — ScienceDirect. Directly evaluates message transmission delay in a real substation case study, close fit to the "latency verification" angle. https://www.sciencedirect.com/science/article/abs/pii/S2352467721001260

### Slot 2 — `edge-ids-survey` (line 510): edge/lightweight IDS survey or representative paper
1. **"Lightweight Intrusion Detection Systems for IoT–Edge Environments: A PRISMA-ScR Systematic Review of Deployability Evidence and a Unified Assessment Framework"** — this is an actual systematic **survey** (78 studies, 2017–2026), the best structural match for a "survey" citation slot. https://doi.org/10.3390/fi18060300
2. **"An Edge-Deployable Lightweight Intrusion Detection System for Industrial Control"** — Electronics (MDPI), directly on-topic for industrial/SCADA edge deployment rather than generic IoT. https://doi.org/10.3390/electronics15030644
3. **"Lightweight and Energy-Aware Intrusion Detection for Industrial IoT Using TinyML and Edge AI"** — Scientific Reports (Nature). Could alternatively serve Slot 3 (TinyML) instead — overlaps both slots, pick one. https://www.nature.com/articles/s41598-026-50690-0

### Slot 3 — `tinyml-ids` (line 512): TinyML / resource-constrained detection reference
1. **"TinyML-Based Intrusion Detection System for In-Vehicle Network Using Convolutional Neural Network on Embedded Devices"** — IEEE Embedded Systems Letters / IEEE Xplore. Concrete resource numbers (20.44 kB flash, 26.44 kB RAM on nRF52840), strong fit for a "resource-constrained" citation. https://ieeexplore.ieee.org/document/10706824/
2. **"TinyML in Network Security: A Lightweight and Real-Time Intrusion Detection Framework"** — IEEE Conference Publication. https://ieeexplore.ieee.org/document/11203466/
3. **"Lightweight and Energy-Aware Intrusion Detection for Industrial IoT Using TinyML and Edge AI"** — Scientific Reports (Nature), same candidate as Slot 2 option 3; use in whichever slot it fits better, not both. https://www.nature.com/articles/s41598-026-50690-0

### Slot 4 — `missing-features-ml` (line 515): inference-time missing features / feature robustness in tabular ML
1. **"Dynamic Instance-Wise Classification in Correlated Feature Spaces"** — introduces a classifier (IFC²F) explicitly evaluated for robustness up to 10% missing features at inference time, closest direct topical match found. https://arxiv.org/pdf/2106.04668
2. **"On Missingness Features in Machine Learning Models for Critical Care: Observational Study"** — PMC/NCBI. Different domain (healthcare) but directly about missingness-as-a-feature in ML models, a well-cited methodological reference for handling missingness. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8701717/
3. **"Tabular and Graph-based Representations for Noise and Missing Data in Robust Machine Learning"** — ScienceDirect, 2026. Directly compares representations under missing-data conditions. https://www.sciencedirect.com/science/article/pii/S2590005626000202

**Sources used to find these candidates (full search-result URL lists, for your own re-search if desired):**
- [Performance Analysis of IEC 61850 Process Bus and Interoperability Test among Multi-Vendor System](https://research.manchester.ac.uk/en/studentTheses/performance-analysis-of-iec-61850-process-bus-and-interoperabilit/)
- [(PDF) Performance Analysis of IEC 61850 Sampled Value Process Bus Networks](https://www.academia.edu/47389911/Performance_Analysis_of_IEC_61850_Sampled_Value_Process_Bus_Networks)
- [(PDF) Direct Evaluation of IEC 61850-9-2 Process Bus Network Performance](https://www.academia.edu/99300211/Direct_Evaluation_of_IEC_61850_9_2_Process_Bus_Network_Performance)
- [IEC 61850 Process Bus Communication Decrypted - OMICRON](https://www.omicronenergy.com/en/news/details/iec-61850-process-bus-communication-decrypted/)
- [A methodology for the evaluation of the message transmission delay over IEC 61850 communication network](https://www.sciencedirect.com/science/article/abs/pii/S2352467721001260)
- [Towards Latency Bypass and Scalability Maintain in Digital Substation Communication Domain with IEC 62439-3 Based Network Architecture](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9269785/)
- [Lightweight Intrusion Detection Systems for IoT–Edge Environments: A PRISMA-ScR Systematic Review](https://doi.org/10.3390/fi18060300)
- [An Edge-Deployable Lightweight Intrusion Detection System for Industrial Control](https://doi.org/10.3390/electronics15030644)
- [Lightweight and Energy-Aware Intrusion Detection for Industrial IoT Using TinyML and Edge AI](https://www.nature.com/articles/s41598-026-50690-0)
- [TinyML-Based Intrusion Detection System for In-Vehicle Network Using CNN on Embedded Devices (IEEE Xplore)](https://ieeexplore.ieee.org/document/10706824/)
- [TinyML in Network Security: A Lightweight and Real-Time Intrusion Detection Framework (IEEE Xplore)](https://ieeexplore.ieee.org/document/11203466/)
- [Dynamic Instance-Wise Classification in Correlated Feature Spaces](https://arxiv.org/pdf/2106.04668)
- [On Missingness Features in Machine Learning Models for Critical Care: Observational Study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8701717/)
- [Tabular and graph-based representations for noise and missing data in robust machine learning](https://www.sciencedirect.com/science/article/pii/S2590005626000202)
- [Direct Evaluation of IEC 61850-9-2 Process Bus Network Performance (IEEE Xplore)](https://ieeexplore.ieee.org/document/6244823/)

---

## 6. Model scope confirmation

**Five models are named in the abstract (line 59), introduction (line 80), and contributions list (line 97) — "five lightweight detectors": Logistic Regression, Random Forest, XGBoost, compact MLP, and feature-space 1D-CNN. Only four of these five were actually re-evaluated by this session's new experiments (LR, RF, XGBoost, compact MLP) — the CNN was cut (Section 3.4) and appears only via the original paper's pre-existing, non-recomputed numbers.** Table-by-table breakdown: **Table IV (baseline)** includes all 5 — RF/XGBoost/CNN rows are the **original paper's** validated values (traced in Section 1.10 to `outputs/metrics_naive.csv` and `outputs/metrics_groupkfold_macro.csv` in the repo root, not `RTS_Paper/`), while the LR and compact-MLP rows are this session's new harness output (`RTS_Paper/outputs/metrics/phase_b_loro_clean_macro.csv`). **Table V (latency)** and the **robustness table** (§ Availability Robustness) both list all 5 models but mark the CNN row "Not evaluated (excluded; see Limitations)" — a label that currently points to a Limitations paragraph that does not exist yet (Section 3.4). **The feature-budget table, placement table, threshold table, and SHAP/explanation table cover only 4 models (LR, RF, XGBoost, compact MLP) or fewer** — the SHAP table is RF+XGBoost only, by original design, not related to the CNN cut. **The three "five lightweight detectors" sentences (lines 59, 80, 97) are the specific spots you'll need to decide on:** either edit them to "four fully evaluated detectors, plus a fifth (CNN) reported from prior work and excluded from new experiments" or leave "five" and rely on the Limitations section once it's written to carry that distinction — right now neither the sentence wording nor the Limitations section resolves this, so a reader would reasonably expect all 5 to have new results throughout.

---

## 7. Full file manifest — every CSV/JSON/parquet this paper's numbers depend on

All paths relative to repo root. Grouped by where they're used.

**Original paper outputs (reused, not recomputed — feeds Table IV's RF/XGBoost/CNN rows only):**
- `outputs/metrics_naive.csv`
- `outputs/metrics_groupkfold_macro.csv`

**New harness — Phase B baseline (128-feature central budget):**
- `RTS_Paper/outputs/metrics/phase_b_loro_clean_fold_metrics.csv`
- `RTS_Paper/outputs/metrics/phase_b_loro_clean_macro.csv`
- `RTS_Paper/outputs/metrics/phase_b_loro_clean_pooled.csv`
- `RTS_Paper/outputs/predictions/phase_b_loro_clean_predictions.csv`

**New harness — structural feature budgets:**
- `RTS_Paper/outputs/metrics/phase_b_loro_clean_fold_metrics_edge_r1.csv`
- `RTS_Paper/outputs/metrics/phase_b_loro_clean_macro_edge_r1.csv`
- `RTS_Paper/outputs/metrics/phase_b_loro_clean_pooled_edge_r1.csv`
- `RTS_Paper/outputs/metrics/phase_b_loro_clean_fold_metrics_pmu_only.csv`
- `RTS_Paper/outputs/metrics/phase_b_loro_clean_macro_pmu_only.csv`
- `RTS_Paper/outputs/metrics/phase_b_loro_clean_pooled_pmu_only.csv`
- `RTS_Paper/outputs/metrics/phase_b_loro_clean_fold_metrics_cyber_log_only.csv`
- `RTS_Paper/outputs/metrics/phase_b_loro_clean_macro_cyber_log_only.csv`
- `RTS_Paper/outputs/metrics/phase_b_loro_clean_pooled_cyber_log_only.csv`

**New harness — ranked feature budgets (64/32/16):**
- `RTS_Paper/outputs/metrics/feature_budget_ranked_fold_metrics.csv`
- `RTS_Paper/outputs/metrics/feature_budget_ranked_macro.csv`
- `RTS_Paper/outputs/predictions/feature_budget_ranked_predictions.csv`

**Payload and latency:**
- `RTS_Paper/outputs/metrics/payload_model.csv`
- `RTS_Paper/outputs/metrics/latency_and_envelope_verdicts.csv`
- `RTS_Paper/outputs/metrics/envelope_verdicts.csv`
- `RTS_Paper/outputs/metrics/latency_by_budget.csv`
- `RTS_Paper/outputs/timings/raw_latency_samples.parquet`
- `RTS_Paper/outputs/timings/environment.json`

**Robustness / availability degradation:**
- `RTS_Paper/outputs/robustness/missingness_results.csv`
- `RTS_Paper/outputs/robustness/group_unavailability_results.csv`
- `RTS_Paper/outputs/robustness/relay_loss_results.csv`
- `RTS_Paper/outputs/robustness/logs_unavailable_results.csv`
- `RTS_Paper/outputs/robustness/pmu_group_unavailable_results.csv` (produced by `RTS_Paper/scripts/run_pmu_ablation.py`; reproducibility confirmed bit-for-bit against the original inline run — see Section 1.7)

**Placement:**
- `RTS_Paper/outputs/metrics/placement_comparison_table.csv`

**Thresholds:**
- `RTS_Paper/outputs/metrics/threshold_stability_fold_results.csv`
- `RTS_Paper/outputs/metrics/threshold_variability_summary.csv`
- `RTS_Paper/outputs/metrics/threshold_rule_macro_summary.csv`

**SHAP / explanation overhead:**
- `RTS_Paper/outputs/metrics/shap_timing.csv`
- `RTS_Paper/outputs/metrics/shap_selective_strategies.csv`

**Statistics:**
- `RTS_Paper/outputs/metrics/wilcoxon_model_comparisons.csv`

**Feature mapping / audit:**
- `RTS_Paper/outputs/audit/feature_source_map.csv`
- `RTS_Paper/outputs/audit/baseline_verification.json`
- `RTS_Paper/outputs/audit/repository_inventory.json`
- `RTS_Paper/outputs/audit/resulttodo_map.csv`

**Generated but not directly cited by number in the current draft text (macros exist as an alternate/unused path — the draft was filled by hand-transcribing values into prose and tables, not by `\input{results_macros}`, which remains commented out at line 26 of the `.tex`):**
- `RTS_Paper/paper/results_macros.tex` — exists, was generated by `build_macros.py`, but is **not currently `\input`ed by the draft** (verify: `grep -n "input{results_macros}" RTS_Paper/paper/ieee_rtc_telemetry_budgets_draft.tex` returns only the commented-out line). This means the macros file and the draft's actual hard-coded numbers are two independent representations of the same underlying data — they were cross-checked by eye during writing but there is no automated guarantee they stay in sync if either changes.

---

*End of packet. No `.tex` edits were made while assembling this document.*
