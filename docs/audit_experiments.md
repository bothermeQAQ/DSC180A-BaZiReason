> **Historical (audit_v2, prompt v2-policy-controls).** The current conditions, metrics and data are described in [research_design.md](research_design.md).

# Independent audit of inference and evaluation

Audit date: 2026-11-03. Modules read: `schema.py`, `metrics.py`, `evaluator.py`,
`baselines.py`, `llm_clients.py`, `prompts.py`, `cli.py`, and all pipeline tests.
No real-model result is reported in this document. Original locations below refer
to the preserved checkpoint source; fixes refer to the current working tree.

## Findings and implemented fixes

| Priority | Original location and evidence | Consequence | Fix and verification |
|---|---|---|---|
| P0 if interpreted as reasoning improvement | `prompts.py:30-34` serializes `model_view(compute_chart(...))`, containing exactly all target fields. | H2 can be solved perfectly by copying. It cannot identify improved internal calendar reasoning. This is deliberate assisted context, not direct gold-file leakage; changing expected annotations does not change the prompt. | Retain the condition with a restricted interpretation and add deterministic copy, preprocessing-only, and corrupted-context controls. `test_controls_do_not_read_gold_or_mutate_reference`, `test_copy_control_never_calls_model_or_reads_expected`, and `test_h2_corruption_adherence_is_distinct_from_accuracy`. |
| P1 | Original `prompts.py` has only direct/tool-context conditions; `docs/research_design.md` explicitly says H3 ablation is additional work. | No contrast identifies the effect of explicit nonanswer policy. No H3 result had actually been claimed. | Implement explicit/minimal policy crossed with each model context condition. Tests require identical user messages and computation rules; the only system-message difference is the explicit nonanswer policy. |
| P1 | `metrics.py:45-49` counts only non-ok status with non-null pillars as a contradiction. An `ok` answer on ambiguous input is not counted. | The existing metric does not measure H3's unsupported definitive-answer outcome. | Add nonanswer and ambiguity-only definitive-answer frequencies, valid scorable coverage, and paired policy-correctness changes. Detect definitive claims even when the response fails schema. Existing contradiction metric retains its narrow definition. |
| P1 | `schema.py:16-18,27`: any dictionary, including `{}`, passes the convention type check; `reason_codes=[""]` is accepted for nonanswer. | Supposedly strict format validity accepts missing profile fields and blank reasons. | Require exact profile keys, integer sect representation, nonempty string profile values, nonblank reason strings. Profile support and exact correctness remain separate scoring questions. |
| P1 | `evaluator.py:12-20` has no method-list validation and uses numeric comparison for trial identity. `True` and `0.0` can alias integer trial slots. | Invalid designs can collapse or miscount experimental observations. | Validate unique nonempty method names and genuine integer trial IDs; reject malformed reference schemas and duplicate/unexpected identities. Regression tests cover booleans, floats, strings and duplicate methods. |
| P1 | Original `cli.py:89-124` writes the run manifest only after all inference finishes. Evaluation also accepts an independently selected prefix. | An interrupted run can lose its authoritative selected IDs and declared denominator; scoring only returned IDs can hide missing observations. | Write the planned manifest before the first call, then bind evaluation to its dataset hash, selected IDs, conditions and trials. Explicit mismatches are rejected. Regression tests verify interruption survival and that truncating four planned observations to one still yields three missing outputs. |
| P1 | Original `baselines.py:46` uses normal `json.loads`. Duplicate object keys are silently overwritten. | Ambiguous responses can be scored as valid using only the last value. | Reject duplicate keys at every object level; separately retain schema-invalid parsed outputs with `error=invalid_schema`. |
| P2 | Original evaluation reports only discordant counts, and Wilson intervals ignore shared groups. | Precision estimates can overstate evidence if related dates/profiles/trials are treated independently. | Add seeded paired group bootstrap intervals, capped at 10,000 resamples (default 1,000). Preserve all observations in the estimand and resample whole group IDs. Intervals are descriptive, not population inference. |

`exact_accuracy` was intentionally a four-pillar-only metric. Its exclusion of
profile correctness is a definition, not an arithmetic bug. The added
`joint_exact_accuracy` requires correct pillars and exact convention metadata.

## Implemented conditions and estimands

| Condition name | What the model receives | Interpretation |
|---|---|---|
| `direct` | Input, selected profile, JSON schema, computation rules, explicit nonanswer policy | Same-model unaided calendar task. |
| `tool_context` | Direct prompt plus complete engine response | Assisted answer transmission. Full accuracy alone demonstrates neither reasoning nor validation of the engine. |
| `tool_facts` | Direct prompt plus normalized UTC+08 timestamp and engine provenance | Preprocessing-only help; no pillars, result status or reason codes. A null normalized timestamp still reveals that preprocessing did not yield an admitted instant. |
| `tool_corrupted` | Same text and structure as full context; for scorable cases only, shift the hour cycle by 12 | Preserves a valid pair and the hour branch, but changes the hour stem. Measures whether the model follows an erroneous supplied hour rather than the reference. Other fields remain correct. Nonanswer records are uncorrupted. |
| `copy_context` | No model call or prompt | Deterministically returns a fresh engine response. Its perfect reference agreement is tautological; it is an implementation ceiling/control, not LLM evidence. |
| `direct_minimal`, `tool_context_minimal`, `tool_facts_minimal`, `tool_corrupted_minimal` | Same corresponding user message, profile, calendar/range rules and JSON schema; omit only explicit missing/unsupported/invalid handling instructions | Isolates the additional nonanswer instruction. The minimal condition still knows the schema and calendar convention; it is not a rule-free model. |

The JSON schema itself requires null pillars for a non-ok answer in both policy
levels. The policy ablation changes when to select a non-ok status, not the output
format. Prompt version is `v2-policy-controls`; historical v1 and v2 results should
not be pooled as one prompt condition.

H2 primary descriptive contrast remains `tool_context - direct` exact accuracy.
Compare `tool_facts - direct` to assess preprocessing help without supplying
pillars. Compare corrupted-context accuracy and `corrupted_hour_following_rate`
to assess reliance on supplied errors. A corruption effect tests this particular
hour-error mechanism; it does not prove broad reasoning ability or robustness to
other tool failures.

H3 primary outcome is the explicit-minus-minimal difference in
`ambiguous_definitive`, restricted to references with `status=ambiguous`.
Negative is favorable. `unsupported_definitive` uses all non-ok references and is
a secondary pooled diagnostic; the name means a definitive claim unsupported by
the admitted input, not only the literal `status=unsupported` stratum. Both are
implemented within each context condition. Primary H3 analysis should use direct
policy pairs; the full-context pair additionally reveals how policy interacts
with an already supplied nonanswer, which is an easier task.

The new balanced omitted-versus-specified groups in `audit_v2` provide related
scorable controls. Report `ambiguous_policy_correct`, `nonanswer_policy_correct`,
scorable coverage, joint exact accuracy, missing counts and format validity with
H3. A system cannot be credited with useful caution merely because it refuses
every scorable question, fails to return output, or times out. Missing output
counts as no definitive answer in that frequency metric, and as failure on
policy correctness/coverage. It must not be interpreted as a successful safe
answer. Reasons are format-checked but their semantic utility is not graded.

All paired intervals resample whole `group_id`s, retaining cases and repeated
trials in the sampled groups. The observation-weighted point estimate and
percentile interval condition on this synthetic diagnostic suite. A single group
has no interval. Small numbers of groups, shared engine lineage, purposive case
selection and multiple comparisons prevent a strong generalization claim.

## Reproducible commands

Run from the repository root after installing the locked environment. The
following offline control made zero model calls:

```sh
.venv/bin/bazireason generate --dataset-version audit_v2 --output data/synthetic/audit_v2.jsonl
.venv/bin/bazireason run --data data/synthetic/audit_v2.jsonl --output results/audit/deterministic_copy_control.jsonl --limit 152 --methods copy_context --max-calls 0
.venv/bin/bazireason evaluate --data data/synthetic/audit_v2.jsonl --predictions results/audit/deterministic_copy_control.jsonl --output results/audit/deterministic_copy_scores.json --methods copy_context
```

The checked artifact has 152 explicitly marked `deterministic_control` records,
126 scorable references, 26 nonanswers and no model invocations. The 126/126
four-pillar agreement and 26/26 nonanswer agreement only demonstrate that copying
the same engine reproduces its labels. They provide no independent calendar
validation and no evidence supporting H1, H2 or H3 for an LLM. Prediction files
and manifests are never overwritten; use a new name to rerun.

When the sibling prediction manifest exists, evaluation automatically restores
its planned selection, methods and trials. Explicit flags must agree with that
declaration; the dataset file hash must match. This applies to interrupted
artifacts as well as complete runs. Legacy files without a manifest require an
explicitly declared evaluation selection. `evidence_status=no_observations`
distinguishes an empty artifact from an observed experiment.

An exact representative selection is stored in
`data/synthetic/audit_pilot_ids.json` (22 cases). `--case-ids` preserves the given
order and overrides the four-case prefix default:

```sh
AUDIT_IDS="$(.venv/bin/python -c 'import json; ids=json.load(open("data/synthetic/audit_pilot_ids.json")); print(",".join(ids))')"
```

For a software-only smoke run, use `--backend mock`. For a real H3 pilot, after
confirming an authorized endpoint/model and the provider's cost, the exact
four-condition design is:

```sh
.venv/bin/bazireason run --backend api --model "$BAZIREASON_MODEL" --base-url "$BAZIREASON_BASE_URL" --data data/synthetic/audit_v2.jsonl --case-ids "$AUDIT_IDS" --methods direct direct_minimal tool_context tool_context_minimal --trials 1 --max-calls 88 --max-tokens 512 --output results/audit/live_h3_pilot.jsonl
.venv/bin/bazireason evaluate --data data/synthetic/audit_v2.jsonl --case-ids "$AUDIT_IDS" --methods direct direct_minimal tool_context tool_context_minimal --trials 1 --predictions results/audit/live_h3_pilot.jsonl --output results/audit/live_h3_scores.json
```

The matched H2-controls pilot uses 88 model calls plus 22 zero-call control
records. It is a separate run; use the same pinned model/settings and regard
cross-run comparisons as susceptible to service drift:

```sh
.venv/bin/bazireason run --backend api --model "$BAZIREASON_MODEL" --base-url "$BAZIREASON_BASE_URL" --data data/synthetic/audit_v2.jsonl --case-ids "$AUDIT_IDS" --methods direct tool_context tool_facts tool_corrupted copy_context --trials 1 --max-calls 88 --max-tokens 512 --output results/audit/live_h2_controls.jsonl
.venv/bin/bazireason evaluate --data data/synthetic/audit_v2.jsonl --case-ids "$AUDIT_IDS" --methods direct tool_context tool_facts tool_corrupted copy_context --trials 1 --predictions results/audit/live_h2_controls.jsonl --output results/audit/live_h2_scores.json
```

`BAZIREASON_API_KEY` is read only by the API client and never written into
prediction artifacts. A six-model-condition combined pilot (`direct`,
`direct_minimal`, `tool_context`, `tool_context_minimal`, `tool_facts`,
`tool_corrupted`, plus `copy_context`) costs at most 132 model invocations per
trial, with 154 observations. These are call and output-token bounds, not dollar
caps. No such live run was performed by this audit.

## What the original tests established

The existing metric tests meaningfully checked hand-constructed denominator and
failure behavior. Prompt isolation checked a real leakage boundary. API fixtures
checked protocol shape, error handling and credential omission. Those are useful
software assertions. Dataset regeneration, matching labels to the same engine,
and repeated copies of gold responses establish internal consistency, not
independent calendar correctness or model performance. None of the tests had
validated an actual LLM's boundary reasoning or the H3 intervention.

The added tests exercise distinct controls, profile-aware success, refusal
coverage, malformed definitive answers, identity aliasing, duplicate JSON keys,
policy-only message differences, deterministic bypass, grouped estimates,
selected IDs and interruption survival. HTTP transport fixtures remain offline.
Current inference adapter behavior remains one bounded HTTP attempt with no
repair/retry; response usage, latency and finish reason are recorded when
supplied. A truncated but parseable output is still assessed by schema and
correctness, while its provider finish reason remains available for inspection.
