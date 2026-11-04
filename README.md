# BaZiReason

**Evaluating Tool-Augmented LLM Reliability in Traditional Chinese Calendar Reasoning**
Yonghao Wang · DSC 180A, Fall 2026 · Domain D25 · Advisor: Jingbo Shang

[Report](reports/checkpoint_report.pdf) · [Independent audit](AUDIT_FINDINGS.md) · [Review assessment](ULTRA_REVIEW_REPORT.md) · [Submission status](CHECKPOINT_STATUS.md)

This project measures calendar-answer reliability under explicit implementation profiles. It does not study destiny prediction. It is currently **evaluation infrastructure with computational evidence**, not a completed empirical LLM study. No live model performance or training result exists; H1-H3 remain untested.

An independent audit reproduced the original 42 tests, found real validation/design weaknesses, and implemented repairs and controls. The original 96-case checkpoint and source remain preserved at Git tag **`audit/pre-review-a50f967`**. The revised suite has **152 unique input/profile cases**:126 scorable,16 ambiguous,6 unsupported,4 invalid.

Calendar evidence includes270 agreeing day/year/month cells (169 distinct reference queries),126 separately counted hour comparisons, and an independent JPL/Skyfield check of24 solar-term timestamps. These are partial, correlated checks, not independent proof of complete charts. [Calendar audit](docs/audit_calendar.md) explains exclusions and actual discrepancies.

## Research design

- **H1:** descriptive direct-model error comparison between ordinary and boundary cases; stronger causal claims require matched strata.
- **H2:** same-model full-context accuracy, with deterministic-copy, timestamp-only and corrupted-hour controls. Full context supplies the answer: accuracy alone cannot establish improved internal reasoning.
- **H3:** explicit vs minimal nonanswer policy, holding inputs, computation rules and schema fixed. Report ambiguity-only definitive-answer rate together with policy correctness and scorable coverage.

[Design and estimands](docs/research_design.md) · [Experimental implementation audit](docs/audit_experiments.md) · [Literature](docs/literature_review.md)

## Installation

Python 3.11+ is the declared target; actual audit verification uses **CPython 3.14.2 and 3.12.14 on macOS arm64**. Other platforms are untested. sxtwl may require a C++ compiler if a suitable wheel is unavailable. No global Python/system settings are changed by setup.

```bash
git clone https://github.com/bothermeQAQ/DSC180A-BaZiReason.git
cd DSC180A-BaZiReason
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Direct dependencies are pinned in `pyproject.toml`; `requirements-lock.txt` records the original tested transitive environment. For that lock, install `pip install -r requirements-lock.txt`, then `pip install --no-deps -e .` (normal build isolation installs declared build tools). Optional astronomy dependencies are separate and are not needed for the standard suite.

## Reproduce the audited dataset and diagnostics

Run from the repository root:

```bash
bazireason generate --dataset-version audit_v2
bazireason diagnostics --data data/synthetic/audit_v2.jsonl --output results/audit/calendar_diagnostics_v2.json
python scripts/check_external_anchors.py
pytest -q
ruff check src scripts tests
ruff format --check src scripts tests
```

Seed 1802026 is the default. `generate --dataset-version checkpoint_v1` regenerates the original 96-case file. Both generators are byte-deterministic under the pinned dependencies. `data/synthetic/audit_pilot_ids.json` fixes a representative22-case pilot across all categories/statuses; it is not a powered sample. Original v1 diagnostics and mocks remain historical artifacts. New audit evidence stays under `results/audit/`.

## Calendar input contract

```python
from bazireason.calendar_engine import compute_chart
from bazireason.conventions import Convention

result = compute_chart(
    {"datetime": "2025-02-03T22:12:28", "timezone": "Asia/Shanghai"},
    Convention(sect=2).to_dict(),
)
```

Inputs use naive ISO local wall time with seconds, an IANA zone and optional fold. Explicit tzdata TZif loading avoids host path/cache state. A repeated DST time needs fold 0/1; a nonexistent time is invalid. Legal civil instants are converted to **fixed UTC+08:00 for all pillars**. International records therefore test this reference-clock task, not local birthplace charting.

Year/month use lunar-python 1.4.8 Lichun/Jie instants. Sect 1 advances the day at 23:00; sect 2 at midnight. Both retain its 23:00-advanced hour stem, including sect 2. Missing profiles request clarification. True solar time, local civil-clock charts, lunar-year/month profiles and normalized years outside 1901-2099 are unsupported. The admitted range is a study restriction, not an upstream guarantee. Outputs include `ok/ambiguous/unsupported/invalid`, optional pillars, reason codes, conventions and provenance. Invalid profiles have no fictitious selected convention.

## Offline controls and synthetic smoke

```bash
bazireason run --data data/synthetic/audit_v2.jsonl --limit 152 \
  --methods copy_context --max-calls 0 --output results/my_copy.jsonl
bazireason evaluate --data data/synthetic/audit_v2.jsonl \
  --predictions results/my_copy.jsonl --output results/my_copy_scores.json
python scripts/run_pilot.py --backend mock --design combined --max-calls 132 \
  --output results/my_mock_pilot.jsonl
```

`copy_context` makes **zero model calls** and copies a fresh engine result. Its perfect reference agreement is tautological. Mock predictions use `SYNTHETIC-MOCK-NOT-AN-LLM`; their scores verify software only. Neither is LLM evidence. Prediction/manifest files are protected from overwrite; use new filenames for reruns.

A planned manifest is written before inference. Evaluation automatically uses its declared cases, methods and trials, verifies the dataset hash, and rejects conflicting overrides. Interrupted/missing observations remain in denominators. Pre-audit artifacts without manifests retain legacy explicit-design evaluation support.

## Real pilot commands (NOT RUN)

Configure an already authorized OpenAI-compatible chat endpoint supporting JSON mode and the requested parameters. The program reads exported variables, not `.env` files. Never commit credentials.

```bash
export BAZIREASON_API_KEY='your-private-key'
export BAZIREASON_BASE_URL='https://your-provider.example/v1'
export BAZIREASON_MODEL='your-exact-model-version'
python scripts/run_pilot.py --backend api --design h3 --max-calls 88 \
  --output results/live_h3_pilot.jsonl
python scripts/run_pilot.py --backend api --design h2 --max-calls 88 \
  --output results/live_h2_controls.jsonl
```

Replace the illustrative provider values. The H3 pilot crosses direct/full context with minimal/explicit policy (88 calls). H2 uses direct, full context, timestamp-only and corrupted context (88 calls) plus22 zero-call copy records. A combined design uses132 completion calls plus22 controls. To minimize provider drift, prefer one combined run when affordable. These are invocation and output-token caps, **not dollar caps**; check current pricing before running. No paid API call was made in this audit.

`run_pilot.py` reads the22 IDs directly, runs the design, and scores the manifest into `OUTPUT.scores.json`. For repeated trials, set `--trials 3` and multiply the cap by 3. For full-suite experiments, use `bazireason run --data ... --limit 152 --methods ... --trials ... --max-calls ...`; the CLI refuses excess calls. Native Anthropic/Ollama protocols are not separately implemented. Local OpenAI-compatible servers may use a dummy key if required by their interface.

Available model conditions: `direct`, `tool_context`, `tool_facts`, `tool_corrupted`, and their `_minimal` variants. Same computation rules/schema are used across policy versions. Rotation balances order; records retain requested/served model, versioned prompt hash, attempts, latency, usage and provider finish reason. No hidden repair/retry is performed. Missing/invalid outputs fail appropriate correctness/coverage metrics. API cost is null unless measured; errors omit sensitive bodies/headers.

## Independent solar-term reproduction (optional)

```bash
python -m pip install -e '.[dev,astronomy]'
mkdir -p tmp/astronomy
curl -fL https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de421.bsp -o tmp/astronomy/de421.bsp
python scripts/check_solar_terms.py
```

This downloads a public astronomical kernel, not model weights. The script performs no implicit download and logs kernel SHA-256, versions, reference frame/time scale, search precision and all 24 Jie 2024-2025 comparisons. Optional exact package versions are in `results/audit/astronomy_environment.txt`. Skyfield 1.53 apparent geocentric ecliptic-of-date results differ from lunar-python by −1.168 to +2.261 seconds; sxtwl is 13.019-15.783 seconds earlier. This is a separate numerical path supporting those selected timestamps, not proof of all chart fields or all years. No labels are silently corrected.

## Metrics and evidence

Exact and per-pillar accuracy use126 scorable observations per full v2 trial. Joint exact additionally requires matching convention metadata. Nonanswer policy correctness uses26 cases; ambiguity-only definitive frequency uses16. Coverage prevents blanket refusal from appearing successful. Missing output is not credited as a correct policy response. Reasons are format-checked, not semantically adjudicated. Descriptive paired group-bootstrap intervals preserve `group_id` dependence; they do not establish population generalization.

| Artifact | Meaning |
|---|---|
| `data/synthetic/checkpoint.jsonl` | Immutable historical 96-case dataset |
| `data/synthetic/audit_v2.jsonl` | Revised152-case suite with all 12 Jie and matched H3 groups |
| `results/audit/original_*` | Independent original test reproduction and bug/coverage evidence |
| `results/audit/calendar_diagnostics_v2.json` | Main270 cells, unique coverage/exclusions, separate126 hour checks |
| `results/audit/independent_solar_terms.json` | Distinct JPL/Skyfield comparison and kernel/version provenance |
| `results/audit/deterministic_copy*` | Zero-call tautological copy control |
| `results/audit/mock_controls*` | Synthetic operational smoke; no LLM results |
| `results/audit/revised_*`, `clean_py312_*` | Executed test/lint/install evidence |
| `AUDIT_FINDINGS.md`, `ULTRA_REVIEW_REPORT.md` | Prioritized defects, fixes and academic assessment |

## Report and submission

```bash
tectonic --keep-logs --outdir reports/latex reports/latex/checkpoint_report.tex
cp reports/latex/checkpoint_report.pdf reports/checkpoint_report.pdf
mkdir -p tmp/report
pdftoppm -r 100 -png reports/checkpoint_report.pdf tmp/report/page
```

Tectonic/Poppler are needed only for authoring/rendering, not experiments. [Template audit](docs/template_audit.md) records the exact source and any remaining font/bibliography deviations. See [course requirements](docs/course_requirements.md) and [checkpoint status](CHECKPOINT_STATUS.md) for current readiness. Gradescope/Piazza instructions and mentor-specific expectations still require student verification; no course submission has been attempted.

## Scope, attribution and remaining work

The study needs actual same-model observations before scientific conclusions. The small grouped sample, reference lineage, limited conventions, public-fact contamination, provider drift and answer-transmission effects remain limits. No fine-tuning or model weights were downloaded. [Training plan](docs/finetuning_plan.md) remains optional and deferred.

Project code/synthetic cases: MIT. lunar-python:6tail/MIT; sxtwl and Xu Jianwei's original ShouXing astronomy are acknowledged. Consult distribution notices and [source audit](docs/source_audit.md) for provenance/license distinctions. HKO calendar facts and course style are attributed; complete third-party datasets and personal birth records are not redistributed. AI assistance was used extensively; Yonghao Wang must review and understand the submitted work and follow applicable disclosure requirements.
