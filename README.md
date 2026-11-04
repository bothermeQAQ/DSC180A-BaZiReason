# BaZiReason

**Evaluating Tool-Augmented LLM Reliability in Traditional Chinese Calendar Reasoning**

Yonghao Wang · DSC 180A, Fall 2026 · Domain D25 · Advisor: Jingbo Shang

[Checkpoint PDF](reports/checkpoint_report.pdf) · [Research design](docs/research_design.md) · [Submission status](CHECKPOINT_STATUS.md) · [Independent review handoff](CLAUDE_REVIEW.md)

This academic project studies deterministic calendar computation and model reliability. It makes no claims about predicting destiny. The question is whether supplying deterministic calendar context improves a model's four-pillar agreement, consistency and handling of missing conventions.

**Current evidence:** a seeded 96-case synthetic suite, working calendar wrapper and paired inference/scoring pipeline. There are 80 labeled computations, 6 ambiguous requests, 6 unsupported requests and 4 invalid inputs. Calendar diagnostics show 184/184 selected cross-implementation checks agree, while 12 solar-term timestamps differ by 13.73-17.13 seconds. These partial, correlated checks share astronomical lineage; they do not establish full-chart ground truth. Two day anchors and one minute-level solar-term anchor were visually checked against HKO. **No live LLM results or training results exist. H1-H3 remain untested.**

## Research questions

1. Do direct models make more errors near boundaries than on ordinary dates (H1)?
2. Does same-model tool-provided context improve exact pillar accuracy (H2)?
3. Does explicit ambiguity/convention representation reduce unsupported answers (H3)?

H1 needs a properly sampled model run; H2 has implemented paired conditions; H3 additionally requires a prompt-policy ablation. See [design](docs/research_design.md) and [prior work](docs/literature_review.md). The treatment is application-provided context, not an autonomous tool-calling agent.

## Install

Python **3.11+** is the supported target; verified locally on CPython **3.14.2**, macOS arm64. Standard-library `zoneinfo` uses the pinned tzdata wheel. `sxtwl` may need a C++ compiler when no wheel exists for your platform. Other platforms/Python versions are not yet tested.

```bash
git clone https://github.com/bothermeQAQ/DSC180A-BaZiReason.git
cd DSC180A-BaZiReason
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`pyproject.toml` pins direct runtime and verification dependencies; `requirements-lock.txt` records the tested Python environment's transitive versions. To reproduce that environment on a compatible platform, install the lock first, then `pip install --no-deps -e .`. PDF compilation additionally uses Tectonic (or a suitable XeLaTeX/BibTeX installation); PDF rendering uses Poppler. Neither is needed for calendar experiments. No API key, external dataset or model download is needed for the offline commands.

## Reproduce checkpoint diagnostics

Run from the repository root:

```bash
bazireason generate --seed 1802026 --output data/synthetic/checkpoint.jsonl
bazireason diagnostics --data data/synthetic/checkpoint.jsonl --output results/calendar_diagnostics.json
pytest -q
ruff check src scripts tests
ruff format --check src scripts tests
```

Generation overwrites the named dataset deterministically. The committed suite is original synthetic data, not personal records. No network is used by generation or diagnostics. Regenerating with the pinned dependencies reproduces its bytes; [SHA-256 manifest](results/artifact_hashes.json) records the snapshot. The CLI scripts under `scripts/` are equivalent wrappers (`python scripts/generate_dataset.py`, etc.).

## Input contract

```python
from bazireason.calendar_engine import compute_chart
from bazireason.conventions import Convention

result = compute_chart(
    {"datetime": "2025-02-03T22:12:28", "timezone": "Asia/Shanghai"},
    Convention(sect=2).to_dict(),
)
print(result)
```

The input is a naive local wall time with seconds, an IANA zone, optional `fold=0/1`, and explicit convention. Repeated DST times require fold; nonexistent times are invalid. All pillars use the civil instant converted to **fixed UTC+08:00**. This reference-clock task does not implement arbitrary local-clock charts. Historical legal DST is normalized before computation.

Year/month use lunar-python's Lichun/Jie instants. Sect 1 advances the day at 23:00; sect 2 at midnight. **Both retain the library's 23:00-advanced hour stem**, including sect 2. True solar time, local civil-clock charting, lunar-year/month profiles and reference years outside 1901-2099 are unsupported. Missing time/zone/profile prompts clarification. The admitted year range is a study restriction, not a validated range guarantee from upstream.

Outputs distinguish `ok`, `ambiguous`, `unsupported`, `invalid`, with reason codes, convention metadata and engine provenance. Only ok outputs contain pillars. See [conventions and limitations](docs/research_design.md).

## Synthetic smoke run

```bash
bazireason run --backend mock --limit 4 --max-calls 8 --output results/my_mock.jsonl
bazireason evaluate --limit 4 --predictions results/my_mock.jsonl --output results/my_mock_metrics.json
```

Existing prediction files are protected from overwrite; choose a new filename for each run. The mock is labeled `SYNTHETIC-MOCK-NOT-AN-LLM` and returns a fixed schema fixture without reading labels. Its metrics only test execution and **must not be cited as model performance**. Committed `mock_smoke*` artifacts demonstrate this path. HTTP adapter tests also use in-memory fixtures, not real paid calls.

## Real model pilot (not run here)

Configure an OpenAI-compatible chat endpoint supporting JSON mode and the requested sampling fields. The program does not auto-load `.env` files. Copying `.env.example` alone does not configure access; export your own credentials locally. Never commit a real key.

```bash
export BAZIREASON_API_KEY='your-private-key'
export BAZIREASON_BASE_URL='https://your-provider.example/v1'
export BAZIREASON_MODEL='your-exact-model-id'
bazireason run --backend api --limit 4 --max-calls 8 --trials 1 \
  --temperature 0 --max-tokens 512 --output results/pilot.jsonl
bazireason evaluate --limit 4 --trials 1 --predictions results/pilot.jsonl \
  --output results/pilot_metrics.json
```

Replace the illustrative URL/model with the provider's actual values. Use the same model for both methods. The first four cases are ordinary: this is a network/schema pilot, **not a boundary experiment**. For a full 96-case paired trial use `--limit 96 --max-calls 192`; for three repeats use `--trials 3 --max-calls 576` and evaluate with `--trials 3`. Check current provider pricing before expanding. Call and output-token bounds do not guarantee a dollar budget. No implicit retry or answer-repair is performed. Local OpenAI-compatible servers can use a dummy credential if required by their interface; support depends on endpoint fields. Native Anthropic and Ollama APIs are not separately implemented.

The runner alternates condition order, logs requested/served model and fingerprint when returned, prompt hashes, dataset hash, attempts, usage and latency, and flushes each completed observation. Error classes are retained without provider response bodies or credentials. Monetary cost remains null. Review provider errors/truncation before interpretation. A failed or missing call still counts as a failure in the declared evaluation denominator.

## Output files and metrics

| Path | Meaning |
|---|---|
| `data/synthetic/checkpoint.jsonl` | 96 generated cases, group IDs, profiles and single-engine reference labels |
| `results/calendar_diagnostics.json` | Actual partial sxtwl comparisons and solar-term discrepancies; no LLM scores |
| `results/external_anchor_checks.json` | HKO factual anchor checks, precision and source hash |
| `results/mock_smoke*` | Clearly synthetic execution records and scorer smoke output |
| `results/pytest.txt`, `pytest.xml`, `ruff.txt` | Executed verification evidence |
| `results/artifact_hashes.json` | Dataset, report and diagnostics snapshot hashes |
| `reports/checkpoint_report.pdf` | Checkpoint report |
| `reports/latex/` | Editable report, bibliography and adapted course typography style |
| `docs/sources/` | Source URLs, access metadata, repository commits and external anchor provenance |

Exact match and per-pillar accuracy use only the 80 labeled observations per full trial. Status/schema/convention metrics use all 96. Uncertainty policy accuracy uses the 16 non-answer cases and does not semantically grade explanations. Repeated consistency compares complete valid structured responses. Full definitions, exclusions and prospective grouped inference are in [research_design.md](docs/research_design.md). Descriptive Wilson intervals do not account for grouped dependence.

## Build and inspect the report

```bash
tectonic --keep-logs --outdir reports/latex reports/latex/checkpoint_report.tex
cp reports/latex/checkpoint_report.pdf reports/checkpoint_report.pdf
pdftoppm -r 100 -png reports/checkpoint_report.pdf tmp/report
```

Create `tmp/` before rendering if it does not exist. The report uses an attributed typography subset of the official course template; font and bibliography substitutions are documented in [template_audit.md](docs/template_audit.md). Verify mentor/TA expectations if exact unmodified template files are required. The final PDF was rendered and visually inspected. The student must review its claims and the private submission instructions before submitting.

## Known limitations and next work

No live LLM, H3 ablation, training, independent full-chart audit, hour cross-validation or second-precision astronomical adjudication has been performed. The current sample is small and correlated. Shared ShouXing reference lineage, limited profiles, public-source contamination and answer-copying effects limit conclusions. The optional [fine-tuning plan](docs/finetuning_plan.md) is deferred. Use [CLAUDE_REVIEW.md](CLAUDE_REVIEW.md) for independent review, and [CHECKPOINT_STATUS.md](CHECKPOINT_STATUS.md) for exact submission actions.

## Attribution and license

Project code and original synthetic suite: MIT. lunar-python: 6tail, MIT. sxtwl and the original ShouXing astronomical calendar by Xu Jianwei are acknowledged; consult upstream distribution notices (metadata and fork-license differences are in the source audit). HKO anchors are small attributed calendar facts; upstream PDFs and contest data are not redistributed. Literature and official course template links are in the documentation. AI assistance was used for research, implementation, testing and drafting; Yonghao Wang must verify and understand the submitted work and follow the course's applicable disclosure rules.
