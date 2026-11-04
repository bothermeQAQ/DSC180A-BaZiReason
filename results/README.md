# Evidence interpretation

`calendar_diagnostics.json` and `external_anchor_checks.json` contain real executed **calendar** checks, not model experiments. The former has shared-astronomy limitations; the latter checks only two day labels and minute-level Lichun plausibility. Dataset expected labels are still single-engine references.

`mock_smoke.jsonl`, its manifest and metrics are **SYNTHETIC SOFTWARE FIXTURES**. Do not use their scores as evidence for H1-H3. They do not represent inference by Codex, OpenAI, Anthropic, or a local language model. The `synthetic` flag is preserved in evaluation output.

`pytest.txt`/`pytest.xml`, `ruff.txt` and `format.txt` record actual commands. A failure or missing observation cannot be removed from scientific denominators to inflate scores. Live model experiments and training are NOT RUN. API cost is unmeasured, not zero per model call; no paid model calls were executed in this checkpoint session.
