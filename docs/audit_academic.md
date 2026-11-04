# Independent academic and course-compliance audit

Audit date: November 3, 2026 (America/Los_Angeles). Scope: original checkpoint snapshot as initially read; the root reviewer is revising the report and experimental design concurrently. This document does not certify the later PDF, tests, repository publication, or submission. Locations below refer to the original checkpoint LaTeX (81 lines) unless stated otherwise.

## Independently verified official requirements

The live [2026–27 Q1 assignment](https://dsc-capstone.org/2026-27/assignments/projects/q1/) was opened in this audit. Its project schedule gives **Friday, November 6, 2026** for the checkpoint and **Friday, December 4, 2026** for the project. Its checkpoint report requires title, abstract, and introduction covering motivation, prior literature, and relevant data. Mentor additions can apply. The report is a PDF submitted to Gradescope. The code checkpoint is a public GitHub repository link; README instructions must cover data, software dependencies, and commands reproducing results. The page explicitly requires the linked Overleaf LaTeX template for the final report. No length limit or exact submission hour appears there. The [live home page](https://dsc-capstone.org/2026-27/) independently agrees on November 6 for the checkpoint.

The original report's title, abstract and three introduction subsections satisfy the public content outline. Original report lines 11, 24 and 76 explicitly state that no live model experiment occurred. This is honest preliminary reporting; absence of LLM results is not, by itself, a violation of the public checkpoint outline. Domain-specific mentor approval and private Gradescope/Piazza instructions remain unverified.

## Actual template inspection and retrieval outcome

The official [read-only Overleaf link](https://www.overleaf.com/read/swwcqhtsdwbg) resolved in Computer to project `6556cf1443f932295cf9d008`, titled **DSC Capstone LaTeX Template**. The visible file tree includes `report.tex`, `reference.bib`, `style/dsc180reportstyle.sty`, and `style/dsc180bibstyle.bst`.

The rendered original template states that headings can be added/removed with mentor discretion; blue instructional prose must be removed; Q1 should link code rather than a website. The source editor visibly confirmed letter paper, one-inch margins, natbib, `aysep={}`, bibliography spacing of `-.1pt`, and additional packages/helpers absent from the local subset. Thus local attribution is supported, but the local subset is not an exact copy of the official style.

**Concrete retrieval limitation:** File → Download → Download as source (.zip) was selected through Computer with a download wait. That tool call never returned a local path and was subsequently aborted after approximately 1004 seconds. No complete unchanged official source file, ZIP, or fingerprint was obtained by this reviewer. No `tmp/audit_course_style.sty` exists from this audit. Web access to the public read URL returned an error page rather than source. No remote template content was edited. The exact original font configuration was not independently captured in full during this audit; the prior adaptation document's XCharter/Arial/Fira Code statement should not be presented as newly reverified here.

## Prioritized findings on the original snapshot

### P1-A — Official style conformance remains partial

- **Location:** `reports/latex/style/dsc180reportstyle.sty:1–8,47–53`; `reports/latex/checkpoint_report.tex:79`; `docs/template_audit.md`.
- **Evidence:** Local file explicitly calls itself a typography subset and replaces fonts. The report uses `plainnat`, while the actual official tree includes `dsc180bibstyle.bst`. Original source includes packages/helpers not retained locally. Basic margins, title, abstract and citation spacing match observed template elements.
- **Why it matters:** Public final-report requirements specify the supplied template. No evidence establishes that a font/bibliography substitution is approved. The checkpoint can honestly be marked partially conformant; it cannot be certified exact.
- **Proposed fix:** Preserve source provenance, obtain full official style/bibliography via a functioning download, and apply only necessary portability patches with an explicit diff. If using local installed fonts, prefer original font files where available and keep substitutions documented. Do not invent an original-style hash or claim exact conformance.
- **Verification:** Diff saved files against the unmodified downloaded originals, compile, inspect all final pages and verify no blue instructional text, missing references, clipping, or broken URLs. The root reviewer owns any final style changes.

### P1-B — H2 identifies answer incorporation unless copying controls are added

- **Location:** original report lines 24, 27, 69, 73, 76; `docs/research_design.md`, Inference control and Limits.
- **Evidence:** Tool context supplies the same wrapper's full pillar answer. The original report correctly warns that an improvement could be copying/formatting rather than stronger reasoning. That disclaimer prevents a misleading result claim, but the original two-arm design cannot distinguish these mechanisms.
- **Why it matters:** A calendar-only answer task plus a full correct answer can yield high accuracy without demonstrating LLM reasoning or autonomous tool use. PAL and Toolformer involve substantially different interventions.
- **Proposed fix:** Include a deterministic no-LLM renderer, partial context, and controlled corrupted-context arms. Score faithful reproduction separately from reference accuracy and unjustified acceptance of corrupted context. Use corrupted contexts only as an explicit robustness manipulation and retain the unmodified gold answer for evaluation. Avoid claiming a corruption arm establishes reasoning by itself.
- **Verification:** Validate intervention payloads and immutable scoring labels; compare paired case/group outcomes; inspect whether full-context performance exceeds the no-LLM renderer for any outcome that actually requires a model. Root reports implementing controls; this reviewer has not independently executed the new implementation.

### P1-C — H3 was not identified by the original experiment

- **Location:** original report lines 24, 61, 69, 73; `docs/research_design.md`, Additional experiment needed for H3.
- **Evidence:** Both original conditions provide the same explicit convention rules; no policy factor changes. H3 is explicitly future work, so there is no fabricated H3 finding.
- **Why it matters:** Comparing direct and full-context conditions changes answer availability, not just convention representation. Few ambiguous cases make aggregate refusal improvements easy to misinterpret.
- **Proposed fix:** Cross minimal/explicit policy with context on matched omitted/selected-profile inputs; include fully specified controls and report over-refusal. Phrase the estimand as the effect of explicit policy instructions on policy compliance, rather than discovery of a universal correct chart. Freeze cases before measuring a model.
- **Verification:** Assert identical input/schema and context within each policy contrast; hash prompts and inspect the intentional instruction delta. Report ambiguity and answered-case coverage together. Root reports adding a matched H3 factor; not independently executed by this reviewer.

### P1-D — H1 and generalization are limited by sampling

- **Location:** original report lines 47–56, 73; research design Dataset and provenance.
- **Evidence:** Original suite has 20 ordinary dates restricted to noon/day 12–20 and 24 boundary cases over six Jie in each of two years. Multiple cases share a transition or date; these are not 96 independent draws. Year, time of day and prompt/task complexity differ across strata.
- **Why it matters:** A raw error-rate contrast is a descriptive comparison of this suite, not a causal boundary effect or population accuracy estimate. The original text already correctly acknowledges this.
- **Proposed fix:** Expand all 12 Jie, pair near/far observations with comparable years/clock/conventions, preserve group IDs, use grouped uncertainty and category counts. Keep small-stratum limitations even after expansion.
- **Verification:** Print a coverage matrix including independent groups, dates and profile counts; check split/bootstrap units; ensure no subgroup inference treats the 184 field checks as independent observations. Root reports expanding coverage; not independently verified here.

### P2-A — Broad calendar convention sentence has stronger support from code than HKO

- **Location:** original report line 18 (`hko2025,lunar`).
- **Evidence:** A February calendar can anchor published days and Lichun time, but does not, by itself, document every month boundary or every BaZi convention. The sentence is carefully scoped to implementation profiles and has a dependency citation, so this is a precision improvement, not a disproved claim.
- **Proposed fix:** Attach the implementation source specifically to the selected Lichun/Jie rules and reserve HKO for its exact independently checked fields.
- **Verification:** Trace each convention to the pinned implementation/API; cite HKO beside the particular published facts. Avoid presenting observed convention agreement as evidence for destiny prediction.

### P2-B — Final model evidence must stay distinct from software/astronomical diagnostics

- **Location:** original report lines 11, 64–76; `docs/source_audit.md`.
- **Evidence:** Original writing clearly distinguishes 184 selected field checks, partial shared-lineage validation, unknown timestamp cause, and untested H1–H3. No false live-model result was found in the report. The root independently reports a new Skyfield/JPL check in `results/audit/independent_solar_terms.json`; this reviewer has not independently run that calculation.
- **Proposed fix:** Include exact checked fields, exclusions, versions, selection and timestamp residual range when incorporating new diagnostics. Use “independent ephemeris comparison” with method assumptions, not “proven astronomical truth.” Do not diagnose the 14-second difference without additional evidence.
- **Verification:** Reproduce the saved independent astronomy command, inspect frame/time-scale assumptions and sign of residuals, and retain raw outputs. Continue to label charts as implementation-profile references.

## Important citations independently checked against primary pages

| Citation | Primary source checked | Audit conclusion |
|---|---|---|
| PAL | [PMLR record](https://proceedings.mlr.press/v202/gao23f.html) | Eight authors, ICML 2023, volume 202, pp. 10764–10799 match. Program generation plus interpreter description is accurate. |
| Toolformer | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html) | Nine authors and NeurIPS 2023/volume 36 match. Learning API selection/integration, including a calendar tool, is supported. |
| TimeBench | [ACL Anthology](https://aclanthology.org/2024.acl-long.66/) | Seven authors, ACL 2024, pp. 1204–1228 match. Hierarchical tasks and category variation are supported; it does not establish H1 for this suite. |
| BaziQA | [arXiv 2602.12889](https://arxiv.org/abs/2602.12889) | Jiangxi Chen and Qian Liu, 2026, 200 contest questions from 2021–2025 and fixed symbolic chart setting match. Keep v1 paper and expanded repository versions separate. |

The local prior source snapshots and source-audit statements were read, but this reviewer did not independently re-fetch every upstream source, reproduce the benchmark-stub claim, check every HKO glyph, or establish repository licenses. Those claims retain their existing evidence and should not be described as fully reverified by this narrower audit.

## Original PDF inspection

`pdfinfo` confirmed that the initial `reports/checkpoint_report.pdf` is six letter-size pages (612×792 points), produced by LaTeX/xdvipdfmx. Existing original raster pages 1, 3 and 6 were visually inspected: title/abstract/body, data table, and bibliography were legible with no apparent clipping or overlap. Page 6 is mostly empty due to bibliography spillover; this is an aesthetic issue, not a public requirement violation. Other original pages and the revised PDF require the root's complete final inspection. An attempted `pdftotext` command failed because that executable is unavailable, so no successful extraction is claimed here.

## Academic assessment

The original package is a legitimate prospective reliability study with useful implementation diagnostics, but no empirical finding about LLM performance yet. Its defensible contribution is an auditable instrument and controlled future comparison; it is not a first BaZi benchmark and cannot claim general reasoning improvement from full-answer context. The original report already discloses most of these limitations. The feasible December scope is frozen measurement, representative real same-model experiments if authorized access exists, controls/ablation, clustered analysis, and error inspection. Fine-tuning should remain lower priority until that evidence exists.

No independently established P0 fabricated result, incorrect citation metadata, or violated checkpoint-content requirement was found in this audit. Exact template conformance and private mentor/submission requirements remain unresolved. The root's later implementation, test and astronomy findings are separate evidence and must be integrated without treating this document as their independent replication.
