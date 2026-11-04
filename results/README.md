# Evidence interpretation

## benchmark_v3 (current)

- `benchmark_v3/deterministic_baselines.jsonl` and its manifest and scores hold zero-call controls on all 261 cases: `copy_context` (the answer-copy ceiling) and three zero-knowledge floors (`heuristic_surface`, `always_abstain`, `always_answer`). They are executed deterministic baselines, **not language-model results**.
- `benchmark_v3/mock_combined_pilot.jsonl` and its manifest and scores are a synthetic mock run of the 44-case combined pilot (352 mock completions plus 88 deterministic records). They verify the software path only.
- `benchmark_v3/independent_boundaries.json` is the JPL DE421/Skyfield side check of the 40 cases within minutes of a Jie (18 events). lunar-python differs from JPL by -3.12 to +1.32 s, with no disagreements.
- `benchmark_v3/calendar_diagnostics.json` is the same-lineage sxtwl comparison (539/539 day/year/month cells, 225/225 hours), and it replays all 261 labels.

The files below describe the historical 96-case (v1) and 152-case (v2) suites and are kept unchanged.


`calendar_diagnostics.json` and `external_anchor_checks.json` contain real executed **calendar** checks, not model experiments. The former has shared-astronomy limitations; the latter checks only two day labels and minute-level Lichun plausibility. Dataset expected labels are still single-engine references.

`mock_smoke.jsonl`, its manifest and metrics are **SYNTHETIC SOFTWARE FIXTURES**. Do not use their scores as evidence for H1-H3. They do not represent inference by Codex, OpenAI, Anthropic, or a local language model. The `synthetic` flag is preserved in evaluation output.

`pytest.txt`/`pytest.xml`, `ruff.txt` and `format.txt` record actual commands. A failure or missing observation cannot be removed from scientific denominators to inflate scores. Live model experiments and training are NOT RUN. API cost is unmeasured, not zero per model call; no paid model calls were executed in this checkpoint session.
