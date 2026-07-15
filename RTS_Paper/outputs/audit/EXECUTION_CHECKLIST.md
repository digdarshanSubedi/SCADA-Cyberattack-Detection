# RTC 2026 Extension — Execution Checklist

Deadline: 2026-07-17 23:59 Chicago. Internal target: submit 2026-07-16. Today: 2026-07-13.
Updated after every meaningful step. This file is the single source of truth for progress.

## Post-verification-packet fixes (2026-07-13)
- [x] Committed all Phase B-F work to `rts-paper-extension` (commit `f612f9f`) — previously only Phase A was committed; see `VERIFICATION_PACKET.md` Section 2.3 for the gap this closes.
- [x] Saved the PMU-group-unavailability ablation as `RTS_Paper/scripts/degradations.py::all_pmu_unavailable` + `RTS_Paper/scripts/run_pmu_ablation.py` (previously an unsaved inline script). Reran from scratch after saving; diffed against the original inline output — **bit-for-bit identical** on all 6 numeric columns across all 60 rows. No longer a reproducibility gap.
- [x] Fixed the CNN/Limitations cross-reference: added a full paragraph to `\section{Threats to Validity and Limitations}` (now labeled `\label{sec:limitations}`) explaining the CNN's exclusion, the OpenMP crash/workaround, and that its Table IV numbers are prior-work reference values, not re-derived results. The two "see Limitations" table cells (latency, robustness tables) now point to real content.
- [x] Softened "five detectors" language in 4 places (abstract, introduction ×2, contributions list, Models subsection) to explicitly say four detectors were newly evaluated and the CNN is reported from prior work only. Recompiled clean (8 pages now, was 7), prohibited-claim grep still shows only the same 8 pre-verified disclaimers, no new violations.
- [ ] Still open, unchanged from before: 4 bibliography slots, IEC 61850-5 / IEC TR 61850-90-4 / IEEE C37.118 citation verification against the actual standards — both explicitly deferred to the user's own judgment per their instruction, not attempted by the agent.

## Figure filename update (2026-07-13, later same day)
- [x] Renamed to match the submission-ready 6-figure plan: `fig04_payload_latency_quality` → `fig04_payload_latency_quality`, `fig05_missing_telemetry_robustness` → `fig05_missing_telemetry_robustness`, `fig06_edge_vs_central` → `fig06_edge_vs_central`. Content of each verified unchanged (visually re-inspected) before renaming.
- [x] `fig_relay_loss.png` and `fig_threshold_tradeoffs.png` kept under their existing names as optional supporting figures — `fig06_edge_vs_central` already includes the "one relay lost (worst)" placement condition, so `fig_relay_loss` was correctly NOT promoted to Fig. 6 per the conditional in the request.
- [x] `build_figures.py` edited to write the new filenames directly (not a post-hoc file copy); rerun from scratch to confirm reproducibility.
- [x] `.tex` `\includegraphics` calls updated for Fig. 4 and Fig. 5. **Fig. 6 (`fig06_edge_vs_central.png`) was never actually embedded as a figure in the draft — the "Edge Versus Central Placement" subsection only contains Table VII, no `\includegraphics`.** This predates the rename; flagging as a gap for the user to decide whether to add the figure, keep the table-only presentation, or both.
- [ ] `fig01_system_architecture`, `fig02_random_vs_runaware`, `fig03_latency_envelopes` — these are now generated directly by `build_figures.py`.

## Phase A — Continuity audit
- [x] Repository inventory, baseline verification (exact match to plan) — `RTS_Paper/outputs/audit/`
- [x] Feature-source map built, edge-visible definition CONFIRMED (R1's 29 features; no cyber/log leakage) — `feature_source_map.csv`
- [x] Draft `.tex` located (was in ~/Downloads, not attached) and copied to `RTS_Paper/paper/ieee_rtc_telemetry_budgets_draft.tex`
- [x] `resulttodo_map.csv` rebuilt from the real draft (43 placeholders extracted with line numbers/context)
- [x] Environment fixed: xgboost/shap/torch installed (required `brew install libomp` for xgboost on macOS)
- [x] Branch `rts-paper-extension` created and Phase A committed

## Phase B — LORO harness (P0)
- [x] Harness built: `RTS_Paper/scripts/{data,models,metrics,run_loro}.py`, config-driven, per-fold model persistence to `RTS_Paper/outputs/models/*.joblib`
- [x] Structural leakage assertions in code: run disjointness, single held-out run per fold, run_id excluded from feature matrix
- [x] Smoke test passed (2 folds x Logistic Regression) — sane, non-degenerate metrics
- [x] Full clean LORO (128 features): LR, RF, XGBoost, Compact MLP x 15 folds, 60/60 fits — `phase_b_loro_clean_fold_metrics.csv`. **Cross-check: RF/XGB run-aware numbers match the original paper's validated baseline almost exactly (independent re-derivation, not copy-paste)**
- [x] Macro (mean±SD) and pooled (out-of-fold) metrics — `phase_b_loro_clean_macro.csv`, `phase_b_loro_clean_pooled.csv`
- [x] Edge-visible (R1-only, 29 features) LORO for placement table — `phase_b_loro_clean_fold_metrics_edge_r1.csv`, 60/60 fits
- [x] **CNN (torch) reproduction — CUT (plan's cut-line #1 applied).** Fitter was built (`fit_feature_cnn` in `models.py`, reuses the original paper's exact `FeatureCNN` architecture from `src/run_all.py` for a fair cross-check). First attempt segfaulted (exit 139) — classic macOS OpenMP double-load conflict between libtorch and libxgboost. Fixed with `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE`, which forces single-threaded CPU training and stopped the crash — but a 2-fold smoke test still hadn't converged after 9:45 elapsed (vs. ~5-20s/fold for the other 4 models). At that rate a full 15-fold run would cost well over an hour for a P1 stretch item alone. Killed and cut per the plan's explicit "Drop CNN entirely (one sentence in limitations)" cut-line. **Limitation sentence for the paper:** "A feature-space 1D-CNN baseline from the original paper was attempted but excluded from this extension after its CPU training time (a required workaround for an OpenMP library conflict) made repeated per-fold retraining across all robustness conditions impractical within the study's scope; its original-paper random-split and run-aware numbers are reported for reference only and were not re-derived under the budget/placement conditions studied here."

## Phase C — Latency + payload + envelope verdicts (P0)
- [x] Batch-1 latency harness (perf_counter_ns, 500 warm-up + 5000 timed, preprocessing/inference/end-to-end, P50/P95/P99, throughput, memory, model size) — `latency_and_envelope_verdicts.csv`. **Finding: RF end-to-end P95=51.7ms (fails E1=10ms and E2=16.7ms, passes only E3=100ms); LR/XGB/MLP all <5ms P95, pass all three envelopes.**
- [x] Environment metadata JSON — `RTS_Paper/outputs/timings/environment.json`
- [x] Payload model (bytes/record per feature budget/precision) — `payload_model.csv`. Matches draft's own stated example numbers exactly (512 bytes @128 features/32-bit, 64 bytes @16 features/32-bit).
- [x] Envelope verdicts vs E1/E2/E3 x d_agg∈{0,2,5,10}ms, edge+central — `envelope_verdicts.csv` (48 rows, verified shape)

## Phase D — Availability / robustness suite (P0)
- [x] Degradation library built: `RTS_Paper/scripts/degradations.py` (missingness, group unavailability, relay loss, logs-unavailable, +noise/quantize for P2)
- [x] Missingness 5/10/20/30% x seeds 42-46 (1,200 rows) — `RTS_Paper/outputs/robustness/missingness_results.csv`. **Finding: RF/XGBoost F1 nearly flat with missingness (tree models robust to median-imputed gaps); LR/MLP degrade more (0.79→0.72-0.73 F1 at 30% missing).**
- [x] Feature-group unavailability, 7 groups (420 rows) — `group_unavailability_results.csv`. **Finding: PMU voltage loss is by far the most damaging (0.11-0.15 AUC drop across all models); the 4-column cyber/log groups (Snort, control-panel) individually cost ~0 AUC.**
- [x] Complete relay loss R1-R4 (240 rows) — `relay_loss_results.csv`. R4 tends to be the most damaging single-relay loss for tree models; LR most sensitive to R3.
- [x] Placement table — `placement_comparison_table.csv`. **Finding: central-clean beats edge-only for all 4 models. Ordering flips for Logistic Regression under worst-relay-loss: central AUC collapses to 0.499 (~random), below edge-only's 0.531 — direct answer to the draft's placeholder about whether central visibility always survives impairment (it doesn't, for LR).**

## Phase E — Thresholds + SHAP (P0)
- [x] Threshold rule library built: fixed 0.5, max F1, max Youden J, constrained FPR (target ≤10%, documented assumption — draft doesn't specify a value) — `thresholds.py`. **Bug caught and fixed before the real run: the initial max-F1 search was O(n²) (re-scored F1 from scratch at every unique training probability, up to ~66K candidates for continuous-score models) and was going to take unbounded time; replaced with an O(n log n) cumulative-sum sweep, verified to produce bit-identical results against the brute-force reference on a test case before trusting it on real data.**
- [x] Threshold stability recomputation over cached fold models (zero retraining) — `threshold_stability_fold_results.csv`, `threshold_variability_summary.csv`, `threshold_rule_macro_summary.csv`. **Finding: RF's threshold is nearly rule-invariant (constrained-FPR/max-F1/max-Youden-J all converge to ~0.617, a discrete-output artifact of averaging 200 trees); LR is highly unstable under max-F1 (std=0.21, one fold collapsed to threshold≈0.0007).**
- [x] SHAP timing RF+XGB + selective-explanation strategies — `shap_timing.csv`, `shap_selective_strategies.csv`. **Finding: RF TreeSHAP costs 881ms/explanation (0.9 fits E3's 100ms budget by 9x, ~1.1 explanations/sec — explaining the full 78K-record test set would take ~19 hours); XGBoost costs 1.1ms/explanation (888/sec, fits E3 comfortably, full-set explanation in ~90 seconds). Clear fits/fails story.**

## Phase F (P1/output generation)
- [x] `run_stats.py` run — Wilcoxon signed-rank + Holm correction over the 15 fold pairs, `wilcoxon_model_comparisons.csv`. **Finding: XGBoost beats RF on ROC-AUC in every one of 15 folds (p_holm=0.0004, effect_size=-1.0).**
- [x] Figures generated: `fig04_payload_latency_quality.{pdf,png}` (headline), `fig05_missing_telemetry_robustness`, `fig_relay_loss`, `fig_threshold_tradeoffs`, `fig06_edge_vs_central` — all in `RTS_Paper/figures/`, palette matches the original paper's recolored figures (gray/blue/red/green/gold)
- [x] `results_macros.tex` generated (32 macros, all traced to CSVs) — `RTS_Paper/paper/results_macros.tex`
- [x] `check_prohibited_claims.py` — 8 matches, all manually verified as proper "we do NOT claim X" disclaimers already present in the draft; zero actual violations
- [x] **Correction:** earlier checklist entry wrongly claimed PMU-only/cyber-log-only were done — only edge_r1 had actually been run. Caught while filling the feature-budget table; ran both (`feature_budget_ranked_macro.csv`, `phase_b_loro_clean_macro_pmu_only.csv`, `_cyber_log_only.csv`). PMU-only matches full-128 performance almost exactly (confirms cyber/log contributes little); cyber/log-only collapses to near-chance AUC (0.538) despite deceptively high F1 (0.828, class-prior artifact).
- [x] Ranked feature-budget experiments 64/32/16 (P1) — `feature_budget_ranked_macro.csv` (180 fits, ~76 min). **Finding: top-16 ranked budget matches/beats the full 128-feature budget (F1 0.821 vs 0.813, AUC 0.691 vs 0.687) at 8× less payload and ~3× lower latency — non-monotonic in budget size, a genuinely interesting result, not smoothed over.**
- [x] Per-budget latency measured (`latency_by_budget.csv`) — needed for the feature-budget table's P95 column, wasn't covered by the original Phase C latency pass.
- [x] **All 43 `\resulttodo{}` placeholders resolved except the 4 explicitly human-authored bibliography slots** (lines 506/509/512/515, marked `AUTHORS: add...` — correctly left alone, not agent-fillable).
- [x] Fixed a `pcrr7t` font-cache compile error (same fix the original paper's `write_paper()` already used: `\renewcommand{\ttdefault}{cmtt}`) and added `\graphicspath{{../figures/}}` so figures reference the canonical `RTS_Paper/figures/` location without duplicated copies.
- [x] `\figuretodo` placeholders for fig04_payload_latency_quality and fig05_missing_telemetry_robustness replaced with real `\includegraphics`.
- [x] **Final compile check: PASSED.** 7 pages (within the 4-8 page target), two-pass pdflatex clean, only expected warnings (undefined citations for the 4 human bibliography slots, cosmetic under/overfull hbox). Visually inspected rendered pages — tables and figures lay out correctly, no overlap.
- [x] Final prohibited-claim grep: same 8 matches as Phase A, all previously verified as proper disclaimers; none of the ~15 newly-written result sentences introduced any new violations.
- [ ] Bibliography: 4 human-authored citation slots — NOT agent-fillable, remains a human task before submission.
- [ ] IEC 61850-5 / IEEE C37.118 citation values (already in the draft, cited as `\cite{iec61850-5}` etc.) — per the submission plan, a human must verify these against the actual standards; not independently re-verified by this session.
- [ ] Human spot-check requested: verify ≥10 of the newly-inserted numbers against their source CSVs before trusting the draft for submission (per the plan's non-negotiables).

## Session summary
Every P0 item (Phases B-E) and the P1 ranked feature-budget experiment are complete with real, non-fabricated, internally cross-checked results. CNN (P1) was attempted, hit a real macOS OpenMP crash, was fixed with an env workaround, but was then too slow to justify further time and was cut per the plan's own cut-line #1, with a limitation paragraph drafted. One process error this session: the ranked feature-budget job was allowed to run past the point where its slowness should have prompted a check-in; caught and let finish since it was 96% done by the time it was investigated. One data error caught and corrected before it reached the paper: PMU-only/cyber-log-only were marked done in the checklist before they were actually run — caught during table-filling, not after submission. The draft now compiles cleanly at 7 pages with only bibliography and standards-citation verification left as explicit human tasks.

## Risks / flags (live)
- Bibliography placeholders (lines 472/475/478/481 in draft) are explicitly marked `AUTHORS: add...` — these require human literature search, not agent fabrication. Open item for human handoff.
- Constrained-FPR threshold rule target (10%) is a documented assumption — draft doesn't specify a value. State explicitly wherever this rule is cited.
- **CNN cut from the extension** (see Phase B entry above) — one-paragraph limitation drafted, ready to drop into the Limitations section.
- Ranked feature budgets (64/32/16) not yet run — P1, only remaining substantive experiment gap before placeholder-fill.
- RF model files are large (~178MB/fold, ~2.7GB across 15 folds) — gitignored, not pushed to GitHub; fully reproducible from `run_loro.py` + seed 42 if needed again.
