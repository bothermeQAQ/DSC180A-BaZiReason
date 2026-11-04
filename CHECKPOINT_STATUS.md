# Checkpoint status

Snapshot: **2026-11-03, America/Los_Angeles**. This is a prepared submission package, not a completed course submission. Student review, mentor-specific checks and Gradescope submission remain.

Public repository: **https://github.com/bothermeQAQ/DSC180A-BaZiReason**. Publication verification is recorded below.

## Official compliance matrix

| Requirement | Status | Evidence / remaining action |
|---|---|---|
| Title, abstract, introduction PDF | PASS | Six-page report, literature/data included; generated, opened, text-checked and visually inspected |
| Public code repository URL | PASS | Code, synthetic data, results, tests and README |
| Data/dependency/reproduction README | PASS | Offline commands and bounded live-run instructions |
| Course LaTeX template | PARTIAL | Attributed typography subset, portable font/plainnat changes; exact-style acceptance unverified |
| Mentor-specific expectations/topic scope | NOT VERIFIED | User reports mentor openness; confirm final scope with Professor Shang |
| Nov 6 checkpoint / Dec 4 final dates | VERIFIED PUBLICLY | Project and home pages agree; writing lesson has stale date text |
| Exact submission hour/private instructions | NOT VERIFIED | Student must inspect Gradescope/Piazza |
| Gradescope PDF and code submissions | NOT SUBMITTED | No course submission was attempted |

## Completed deliverables and absolute paths

Project root: `/Users/botherme/Desktop/DSC180A_Quarter1_Project`

- Report: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/reports/checkpoint_report.pdf`
- Editable report: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/reports/latex/checkpoint_report.tex`
- Code and reproduction guide: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/README.md`
- Dataset: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/data/synthetic/checkpoint.jsonl`
- Computational observations: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/results/calendar_diagnostics.json`
- External checks: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/results/external_anchor_checks.json`
- Test log: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/results/pytest.txt`
- Requirements: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/docs/course_requirements.md`
- Literature: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/docs/literature_review.md`
- Design: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/docs/research_design.md`
- Source audit: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/docs/source_audit.md`
- Training plan: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/docs/finetuning_plan.md`
- Review handoff: `/Users/botherme/Desktop/DSC180A_Quarter1_Project/CLAUDE_REVIEW.md`

## Verification actually performed

| Check | Result |
|---|---|
| pytest (calendar, validation, boundaries, determinism, schema, metrics, prompts, HTTP fixtures, CLI smoke) | **42 PASS, 0 FAIL** |
| Ruff lint and format | **PASS** |
| Second clean venv installation + pytest | **42 PASS, 0 FAIL**, same host/Python |
| Seeded dataset byte regeneration | **PASS** (tested) |
| HKO external anchors | **PASS**: two civil-day values and one minute-resolution Lichun anchor |
| Cross-implementation partial checks | **184 agree / 184**, correlated and shared astronomical lineage |
| Solar-term precision comparison | **DISAGREEMENT**: twelve timestamps differ by 13.729553-17.126991 seconds |
| PDF compilation/rendering/text/layout | **PASS**, six pages; harmless underfull line warning, no overfull boxes or unresolved references |
| Live LLM pilot / H1 / H2 / H3 | **NOT RUN**: no configured model access detected |
| Fine-tuning | **NOT RUN**: intentionally deferred |
| Full independent chart correctness | **NOT ESTABLISHED** |

The numerical evidence is computational verification only. Eight mock observations and their scores are explicitly synthetic software smoke artifacts. No real LLM performance numbers, training gains or API costs are invented. No paid model calls were made. The 96 cases contain 80 labeled computations and 16 non-answer-policy cases (6 ambiguous, 6 unsupported, 4 invalid). All expected labels retain single-engine provenance.

## Main executed commands

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/bazireason generate --seed 1802026
.venv/bin/bazireason diagnostics
.venv/bin/python scripts/check_external_anchors.py
.venv/bin/pytest -q --junitxml=results/pytest.xml
.venv/bin/ruff check src scripts tests
.venv/bin/ruff format --check src scripts tests
.venv/bin/bazireason run --backend mock --limit 4 --max-calls 8 --output results/mock_smoke.jsonl
.venv/bin/bazireason evaluate --limit 4 --predictions results/mock_smoke.jsonl --output results/mock_smoke_metrics.json
python3 -m venv tmp/clean-venv
tmp/clean-venv/bin/python -m pip install -r requirements.txt
tmp/clean-venv/bin/python -m pytest -q
tectonic --keep-logs --outdir reports/latex reports/latex/checkpoint_report.tex
cp reports/latex/checkpoint_report.pdf reports/checkpoint_report.pdf
pdftoppm -r 85 -png reports/checkpoint_report.pdf tmp/pdfs/verified
```

Source research used live official course pages, GitHub repository snapshots and publication sources. Commands were scoped to this project; downloaded research code and venvs are ignored. No global Python/system settings were modified. No real secrets are in the committed files; fixture credentials and blank examples are marked accordingly.

## Limitations and blockers

A live model experiment is BLOCKED by absent configured inference access, while its software pipeline is implemented and tested with fixtures. H3 needs a distinct policy ablation. Independent hour/full-chart validation and second-precision astronomy remain open research work. The fixed UTC+8 reference-clock contract is narrow and explicitly rejects local-clock/true-solar conventions. sxtwl's audited fork and installed distribution differ; see source audit. The course template was adapted rather than redistributed unchanged. These limits are stated in the report.

## Exact remaining submission steps

1. Read and understand the PDF/code; independently review with `CLAUDE_REVIEW.md`, especially clock and sect rules, source lineage and unsupported claims.
2. Confirm final topic scope and report additions with Professor Shang. Confirm whether the adapted typography is acceptable; if exact files are required, import the full official style/bibliography in Overleaf and recompile/reinspect.
3. Verify the **November 6, 2026** deadline's exact time and the correct checkpoint report/code assignments in Gradescope/Piazza. Public pages do not establish the hour.
4. Submit `reports/checkpoint_report.pdf` to the report checkpoint and the public repository URL to the code checkpoint. Check uploaded PDF rendering and repository visibility. No submission has been made here.
5. Preserve this checkpoint snapshot; later real runs must use new named result files and honest experimental status.

## Publication and final integrity verification

Publication was verified as recorded below. If publication ever needs to be repeated from a clone with no remote, the commands are `gh auth login`, `gh repo create DSC180A-BaZiReason --public --source=. --remote=origin --push`. Do not run repository creation against an existing unrelated repository.

Publication **PASS**: created the new public repository and pushed `main`. An unauthenticated GitHub API request confirms `private=false`; downloading the public report returns the exact local PDF bytes. Evidence: `results/publication_verification.json`. The committed report SHA-256 is `b7247df972a4a5da0c97f83fc7094d35b3081627378a660e9d90a346941a65a0`.

Additional access inspection confirms no configured standard/provider or BaZiReason API variables and no reachable Ollama (11434) or LM Studio-compatible (1234) model endpoint. See `results/model_access_inspection.json`; no credential values were printed or stored.

Git commands executed: `git init -b main`, local author identity configured with GitHub noreply email, `git add .`, meaningful commits, `gh repo create DSC180A-BaZiReason --public --source=. --remote=origin --push`, and `git push`. Temporary source downloads, venvs and caches are excluded by `.gitignore`. The final remote/local commit comparison and clean working-tree check were performed after the documentation commit.
