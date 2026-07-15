# IEEE Float Sequence Validation Report

## Outputs

- Corrected TeX file: `RTS_Paper/paper/ieee_rtc_telemetry_budgets_draft.tex`
- Compiled PDF: `RTS_Paper/paper/ieee_rtc_telemetry_budgets_draft.pdf`
- Rendered-page review artifact: `/tmp/rtc_pdf_final_technical_verify/contact_sheet.png`

## Confirmed Figure Path

- The TeX file is in `RTS_Paper/paper/`.
- The figure folder is one level above it at `RTS_Paper/figures/`.
- The correct active graphics path is `\graphicspath{{../figures/}}`.
- `\graphicspath` was retained.
- The `\paperfigure` macro now checks `../figures/#1` with `\IfFileExists`, loads the same explicit path with `\includegraphics{../figures/#1}`, and falls back to a framed placeholder if a figure is missing.

## Final Figure Order

1. Figure 1 -- `fig01_system_architecture.png` -- `fig:architecture`
2. Figure 2 -- `fig02_random_vs_runaware.png` -- `fig:random-runaware`
3. Figure 3 -- `fig03_latency_envelopes.png` -- `fig:latency`
4. Figure 4 -- `fig04_payload_latency_quality.png` -- `fig:tradeoff`
5. Figure 5 -- `fig05_missing_telemetry_robustness.png` -- `fig:missingness`
6. Figure 6 -- `fig06_edge_vs_central.png` -- `fig:placement`

## Final Table Order

1. Table I -- `tab:model-config`
2. Table II -- `tab:baseline`
3. Table III -- `tab:latency`
4. Table IV -- `tab:feature-budget`
5. Table V -- `tab:availability`
6. Table VI -- `tab:placement`

## Table Placement Changes

- Table II was kept as a single-column `table` and scaled with `\resizebox{\columnwidth}{!}{...}`.
- Table III was wrapped with `\resizebox{\columnwidth}{!}{...}` to prevent column overflow.
- Tables IV, V, and VI were kept as single-column `table` environments, changed from `\tiny` to `\scriptsize`, and wrapped with `\resizebox{\columnwidth}{!}{...}` using reduced `\tabcolsep`.
- Tables using `\resizebox{\columnwidth}{!}{...}`: II, III, IV, V, and VI.
- No table data, captions, labels, citations, equations, or numerical results were changed.

## Packages And Barriers

- Added/used `float` for `[H]` placement.
- Removed `flafter`; `[H]` placement and `\FloatBarrier` controls preserve the required order without it.
- Updated `placeins` to `\usepackage[section]{placeins}`.
- Added `\FloatBarrier` at logical section/subsection boundaries, including before Section 3, before Section 4, before Method subsections 4.2 and 4.3, before Section 5, between Results subsections, before Discussion, before Threats, before Conclusion, and before References.
- No `[H]` placements were changed; fixed local placement remains intentional to preserve the strict sequence.

## Validation Checks

- Compiled successfully with TeX Live.
- Compiled twice after edits.
- Final PDF page count: 6.
- Actual PDF pages were rendered and visually inspected.
- No figures or tables appear in the front matter, Introduction, or Related Work.
- No Results floats appear in Discussion, Threats to Validity and Limitations, Conclusion, or References.
- Figure numbering is Figure 1 through Figure 6.
- Table numbering is Table I through Table VI.
- Log check found no LaTeX errors, missing figures, undefined references, undefined citations, duplicate labels, overfull boxes, float overflow errors, or float-only pages.
- Remaining warnings are only ordinary underfull box/spacing warnings from the strict IEEE two-column layout and fixed placement.
- No excessive white space, blank pages, clipped tables, or clipped figures were found in the rendered PDF review.
- Manuscript wording and numerical results were preserved; the final pass only changed float/package/barrier structure.
