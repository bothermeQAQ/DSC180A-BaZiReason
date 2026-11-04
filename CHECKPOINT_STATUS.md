> **Historical status for commit cf3e829 (audit_v2).** The current suite, report and verification are described in [README.md](README.md); the report hash below refers to the earlier PDF.

# Audited checkpoint status

Updated **November 3, 2026**, America/Los_Angeles. Original snapshot: **a50f967**, preserved at **`audit/pre-review-a50f967`**. This is a prepared/reviewed package, not a Gradescope submission or a completed live-model study.

Repository: https://github.com/bothermeQAQ/DSC180A-BaZiReason

## Public requirement compliance

| Requirement | Status | Evidence / remaining action |
|---|---|---|
| PDF title, abstract, introduction with problem/prior work/data | PASS | Revised seven-page PDF, compiled/text-checked and all pages visually inspected |
| Public GitHub code and reproducible README | Existing public repo; revised upload verified below | Source, original/v2 data, declared controls, tests and exact commands |
| Official template style | PARTIAL | XCharter text/math and Arial restored; used margins/title/abstract/citations retained; mono/plainnat deviations disclosed |
| Deadline dates | VERIFIED PUBLICLY | November 6 checkpoint; December 4 final; exact hour not established publicly |
| Mentor scope/additions and private instructions | NOT VERIFIED | Student checks Professor Shang, Gradescope/Piazza |
| Course submission | NOT SUBMITTED | No submission action was attempted |

## Current artifacts and absolute paths

Root: `/Users/botherme/Desktop/DSC180A_Quarter1_Project`

- PDF: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/reports/checkpoint_report.pdf`
- Editable report: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/reports/latex/checkpoint_report.tex`
- Prioritized findings: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/AUDIT_FINDINGS.md`
- Independent assessment: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/ULTRA_REVIEW_REPORT.md`
- Revised dataset: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/data/synthetic/audit_v2.jsonl`
- Original preserved dataset: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/data/synthetic/checkpoint.jsonl`
- Current verification: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/results/audit/verification_summary.json`
- Main calendar checks: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/results/audit/calendar_diagnostics_v2.json`
- Distinct astronomy check: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/results/audit/independent_solar_terms.json`
- Instructions: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/README.md`

## Actual verification

- Original Git archive + fresh install: **42 PASS**, not assumed from the prior log.
- Revised full suite: **106 PASS, 0 FAIL** on Python 3.14.2 and fresh Python 3.12.14 installation.
- Ruff lint/format and dependency checks: **PASS**.
- Both generated datasets reproduce committed bytes. V1 remains unchanged; V2 has **152 unique input/profile cases**, 126 scorable, 16 ambiguous, 6 unsupported, 4 invalid.
- V1's 184 agreeing cells reduce to 117 distinct reference queries. V2 has 270/270 cells but 169 distinct queries; 126 separate matched-hour checks agree. All exclusions and shared-lineage limits are retained.
- Skyfield 1.53/JPL DE421 independently computes 24 Jie 2024-2025: lunar-python residual −1.168115 to +2.261093 seconds; sxtwl residual −15.783111 to −13.018692 seconds. Not full-chart proof or root-cause diagnosis.
- HKO checks: two day values and minute-resolution Lichun pass.
- Deterministic copier: 152 records, **zero model calls**,126/126 exact matches by construction. No LLM or external-validation inference.
- Representative combined smoke: 154 observations, 132 synthetic completion invocations plus 22 copy controls. Separate Python H3 wrapper smoke: 88 synthetic invocations. These are software fixtures.
- Report: seven letter pages, all visually inspected; no overfull/clipped content, unresolved citations or placeholders. One harmless underfull-line warning remains. Residual template differences are documented.

## What changed

Fixed timezone-cache/source dependence, extreme-date overflow, invalid-profile output representation, weak schema/duplicate JSON handling, aliased identities/method collection validation, late/lost manifests and evaluation selection mismatch. Added all 12Jie, actual adjacent midnight cases, distinct timezone groups, 24 matched missing/complete H3 cases, explicit/minimal policy, partial/corrupted context, zero-call copying, profile-aware accuracy, coverage and whole-group paired estimates. See `AUDIT_FINDINGS.md` for exact original locations, evidence and regressions.

**NOT RUN / BLOCKED:** live LLM pilot/H1/H2/H3. No usable configured model endpoint/credentials or installed local inference were found in the inspected environment. No paid model calls, model-weight downloads or training occurred. No fine-tuning gain, live model score or astronomical certainty is invented. Original 42 tests and same-engine replay establish software behavior, not these hypotheses.

## Reproduction commands executed

```bash
bazireason generate --dataset-version checkpoint_v1 --output tmp/audit/regenerated_v1.jsonl
bazireason generate --dataset-version audit_v2 --output tmp/audit/regenerated_v2.jsonl
bazireason diagnostics --data data/synthetic/audit_v2.jsonl --output results/audit/calendar_diagnostics_v2.json
python scripts/check_external_anchors.py
python scripts/check_solar_terms.py
pytest -q --junitxml=results/audit/revised_pytest.xml
ruff check src scripts tests
ruff format --check src scripts tests
tmp/review-py312/bin/python -m pip install -r requirements.txt
tmp/review-py312/bin/python -m pytest -q -p no:cacheprovider
python scripts/run_pilot.py --backend mock --design h3 --max-calls 88 --output tmp/audit/wrapper_h3_smoke.jsonl
tectonic --keep-logs --outdir reports/latex reports/latex/checkpoint_report.tex
cp reports/latex/checkpoint_report.pdf reports/checkpoint_report.pdf
pdftoppm -r 85 -png reports/checkpoint_report.pdf tmp/audit-report-final/page
```

The astronomy command requires optional dependencies and an explicitly downloaded DE421 kernel. Exact setup, kernel hash, definitions and versions are retained. Original archive reproduction and zero-call/combined mock commands are in the audit documents. Secret scans exclude ignored research downloads/venvs; committed artifacts contain no real credentials. Git uses the existing GitHub noreply identity. No global Python/system configuration was modified.

## Remaining submission steps

1. Review the PDF and audit; understand the profile, reference and metric limits. Confirm topic scope and mentor additions with Professor Shang.
2. Confirm template acceptance or import the full official style/custom bibliography through a functioning Overleaf download. XCharter restoration improves conformance but is not proof of an unchanged full template.
3. Verify exact **November 6, 2026** hour and report/code assignment identities in Gradescope/Piazza.
4. Submit the seven-page PDF and the public repository URL to the correct checkpoint assignments; inspect uploaded rendering/visibility. No submission has been made here.
5. Obtain an authorized model endpoint for a bounded real pilot, freeze a grouped final evaluation and follow the prioritized December plan in `ULTRA_REVIEW_REPORT.md`. Do not report mocks or copying as LLM research results.

## Publication verification

The initial publication log in `results/publication_verification.json` describes a50f967 and the old report hash. It must not be used to certify the revised report. Revised remote/public visibility, report hash and preserved tag are checked after upload and recorded in `results/audit/publication_verification.json` when successful. Local/remote equality and a clean working tree are checked at the end of this review.

Revised publication **PASS**: public unauthenticated metadata, remote main equality, original tag and exact report bytes verified. The current PDF SHA-256 is `848ae9a5bf2c55a13ef54aef01fe2182a6c8b895b947afff7337250840cb78c1`. See `results/audit/publication_verification.json` for the checked artifact commit.
