# Official template comparison and review changes

Source: [public DSC Capstone Overleaf template](https://www.overleaf.com/read/swwcqhtsdwbg), resolved project6556cf1443f932295cf9d008. Independently inspected through Computer/Overleaf on2026-11-03. See `docs/audit_academic.md` for access evidence and limitations.

Verified official instructions: use the template; remove blue instructional prose; headings may change; Q1 includes a code link and no website link. Actual source uses12pt letterpaper,1in margins, custom title/abstract and paragraph spacing, natbib, XCharter main text/math, Arial sans and Fira Code mono, with a custom bibliography style. The Q1 assignment explicitly requires the template for the final; the methodology page describes its use for Q1 reports generally.

## Current local adaptation

`reports/latex/style/dsc180reportstyle.sty` remains an attributed typography subset, not an unchanged full archive. It retains the used title/abstract/margin/paragraph/natbib formatting and removes unused demonstration/appendix/algorithm/contents helpers. The audit restores **XCharter** main text and Charter math, replacing the initial Pagella substitution. Arial is used where installed, with portable Heros fallback. TeX Gyre Cursor remains the monospaced fallback, and `plainnat` remains the bibliography style. These residual deviations are disclosed. Main/math font restoration is verified by successful compilation and rendered inspection, not merely source editing.

The direct source-ZIP URL returned HTTP403. In the independent browser audit, the download call stalled and was interrupted without an archive path. The read-only editor and rendered instructions were accessible. No remote template was modified. An exact whole-source/style match is therefore **not claimed**. If the mentor/TA requires the full unmodified style or custom bibliography, import the original files into Overleaf before submission. The current checkpoint matches the verified used formatting substantially more closely than the initial version, but acceptance of residual deviations is not independently established.

## Build evidence

Tectonic builds the multi-file editable LaTeX source; Poppler renders every page for visual inspection. Runtime/font dependencies may be downloaded by Tectonic during a first build. Scientific computation does not depend on PDF tooling. The final build log/page inspection state is recorded in `CHECKPOINT_STATUS.md` and `results/audit/verification_summary.json`.
