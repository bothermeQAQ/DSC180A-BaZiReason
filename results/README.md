# Evidence interpretation

## Phase 2 (`phase2/`): real local model results

`phase2/RESULTS.md` is generated from the files below by `scripts/phase2/make_report.py`. These are real inference results from a local MLX model (`record_kind: llm`, `synthetic: false`); every run has a manifest with the model and adapter hashes.

- `phase2/model_selection/`: measured bake-off of five candidate models.
- `phase2/training/<run>/`: training plan, every loss report, summary and loss curve for both rounds. Weights are not in Git.
- `phase2/heldout/`: the generated held-out splits. `base`, `tuned` (round 1) and `tuned_v2` on the first test split; `fresh_*` on the split generated after round 1; `valid2_*` on the validation split used to choose the adapter.
- `phase2/v3/`: benchmark_v3 predictions, manifests and scores from the unchanged `bazireason evaluate`. `*_strict` uses the checkpoint's prompts; `*_reasoned` adds one sentence allowing visible reasoning (prompt version `v4-reasoned-information-ladder`).
- `phase2/baziqa/`: the four conditions with the round-1 adapter; `phase2/baziqa_v2/`: the round-2 adapter (its `base_*` files are copies of the same base runs); `phase2/baziqa_budget1024/`: one run of the untuned model with a larger token budget. Replies quote contest questions only through question IDs; the question text stays in the fetched benchmark.

Missing, truncated or unparseable replies count as failures in every table. BaziQA accuracy is agreement with published contest keys, not evidence about destiny.

## benchmark_v3 (current)

- `benchmark_v3/deterministic_baselines.jsonl` and its manifest and scores hold zero-call controls on all 261 cases: `copy_context` (the answer-copy ceiling) and three zero-knowledge floors (`heuristic_surface`, `always_abstain`, `always_answer`). They are executed deterministic baselines, **not language-model results**.
- `benchmark_v3/mock_combined_pilot.jsonl` and its manifest and scores are a synthetic mock run of the 44-case combined pilot (352 mock completions plus 88 deterministic records). They verify the software path only.
- `benchmark_v3/independent_boundaries.json` is the JPL DE421/Skyfield side check of the 40 cases within minutes of a Jie (18 events). lunar-python differs from JPL by -3.12 to +1.32 s, with no disagreements.
- `benchmark_v3/calendar_diagnostics.json` is the same-lineage sxtwl comparison (539/539 day/year/month cells, 225/225 hours), and it replays all 261 labels.

The files below describe the historical 96-case (v1) and 152-case (v2) suites and are kept unchanged.


`calendar_diagnostics.json` and `external_anchor_checks.json` contain real executed **calendar** checks, not model experiments. The former has shared-astronomy limitations; the latter checks only two day labels and minute-level Lichun plausibility. Dataset expected labels are still single-engine references.

`mock_smoke.jsonl`, its manifest and metrics are **SYNTHETIC SOFTWARE FIXTURES**. Do not use their scores as evidence for H1-H3. They do not represent inference by Codex, OpenAI, Anthropic, or a local language model. The `synthetic` flag is preserved in evaluation output.

`pytest.txt`/`pytest.xml`, `ruff.txt` and `format.txt` record actual commands. A failure or missing observation cannot be removed from scientific denominators to inflate scores. Live model experiments and training are NOT RUN. API cost is unmeasured, not zero per model call; no paid model calls were executed in this checkpoint session.
