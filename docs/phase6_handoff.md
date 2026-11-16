# Phase 6 handoff: state, findings and how to continue

Last updated: Sunday 15 November 2026, 23:10 PST, by the lead agent (Claude Fable 5.2). Branch `phase6/model-capability`; commit locally, never push. Read this file first, then [`phase6_log.md`](phase6_log.md) (what happened, in order) and [`phase6_preregistration.md`](phase6_preregistration.md) (the frozen rules; sections 10 to 12 are the current plan).

## 1. What is running

One GPU job at a time. Two background shell scripts (started with `nohup`, they survive the session):

| Script | Log | Does |
|---|---|---|
| `scripts/phase6/queue_s2s.sh` | `artifacts/logs/p6_queue_s2s.log` | trains S2s (started 22:57 on 15 Nov, about 4 h), scores it on dev6 |
| `scripts/phase6/queue_main.sh` | `artifacts/logs/p6_queue_main.log` | waits for S2s, then runs everything in section 2 |

Check status (read-only):

```bash
ps -eo pid,etime,command | grep -E "queue_|train_lora|run6|own_replies|build_train6|gen_blind6" | grep -v grep
```

```bash
sed 's/\x1b\[[0-9;]*m//g' artifacts/logs/p6_queue_main.log | tr '\r' '\n' | grep -a -E "^===|STEP_FAILED|ARM_SKIPPED|QUEUE_|U1_STOPPED|continuation|prompt" | tail -30
```

Markers: `QUEUE_S2S_DONE` / `QUEUE_S2S_FAILED`; `QUEUE_MAIN_FAILED` (a step of parts 1 to 3 failed: the queue stopped); `QUEUE_MAIN_CHAIN_DONE` (selection can be prepared); `QUEUE_MAIN_DONE`.

## 2. The plan the queue executes (pre-registration 12.4)

| Step | Run | What it is | Expected end (PST) |
|---|---|---|---|
| 0 | S2s | S2's recipe, seed and 6.0 M budget with the 36-token system prompt (H7 twin) | Mon 16 Nov ~03:35 |
| 1 | U2 | S2's recipe at 15 M tokens; prompt chosen by `decide6.py prompt` (short unless S2s is clearly worse than S2) | Mon ~15:45 |
| 2a | U3, U3v | the fork: weighted continuations of U2, derivation version 1 against 2 (H9) | Tue 17 Nov ~03:00 |
| 2b | U4 to U10 | weighted continuations from the latest level; stops after two levels without a new best on dev6 | Tue ~15:00 to Wed 18 Nov ~21:00 |
| 3 | valid6 for S2 and S2s | selection inputs | +1.2 h |
| 4 | U1 | the registered run (S3's recipe, 15 M, long prompt) with its fit check | +13.5 h |
| 5 | fit sample B for every U run | diagnostic | +0.6 h each |

Each continuation: the parent answers fit sample B (2,000 fresh questions, never training input), the question types are reweighted by its errors, 6.0 M tokens at learning rate 1e-4. Decisions are made by `scripts/phase6/decide6.py` and logged in `results/phase6/round2.json`.

**If the queue stops** (`QUEUE_MAIN_FAILED`): read the log, fix the cause, then restart it; it skips built sets, finished training runs and scored parts.

```bash
nohup sh scripts/phase6/queue_main.sh >> artifacts/logs/p6_queue_main.log 2>&1 &
```

Never edit a queue script while its process runs (the shell reads it as it goes): stop the process first. Python scripts can be edited at any time; they are read when a step starts.

## 3. What to do when the queue has finished

1. **Check** that `QUEUE_MAIN_DONE` (or at least `QUEUE_MAIN_CHAIN_DONE` plus U1's `valid6` files, or `U1_STOPPED`) is in the log, and that every candidate has `results/phase6/valid6/<label>/*_L0.jsonl` (rule parts and calendar).
2. **Rebuild the tables** (CPU): `PYTHONPATH=src .venv-mlx/bin/python scripts/phase6/truncation6.py`, then `PYTHONPATH=src .venv/bin/python scripts/phase6/make_report6.py`. Read `results/phase6/RESULTS.md`.
3. **Optional, before selection:** U1-hi by hand (`scripts/phase6/queue_u1hi.sh`, 16 to 21 h) only under the rule of pre-registration section 5 (U1's clean fit share below 95%) and only if it can start by 27 November. If it is not run, report H5 as untested.
4. **Selection, once:** `PYTHONPATH=src .venv/bin/python scripts/phase6/decide6.py select` writes `results/phase6/selection.json`. This opens the sealed split: do not run it before every candidate's valid6 files exist, and do not rerun it.
5. **Sealed runs, once:** `nohup sh scripts/phase6/queue_sealed6.sh > artifacts/logs/p6_queue_sealed6.log 2>&1 &` (about 30 h; it stops at the first failed step and can be restarted; every access is logged in `results/phase6/sealed_access.jsonl`).
6. **Report:** rebuild the tables, then write sections 4 onwards of `docs/phase6_report.md` from `results/phase6/RESULTS.md` (every number from a saved file; L0 is the headline; L5 last and apart). Then the LaTeX draft next to `reports/latex/checkpoint_report.tex` (leave that file untouched), README sections, `ruff`, the full test suite, local commits.

Deadlines: the last training run must start by 27 November; the report is due Friday 4 December 2026.

## 4. Results so far (dev6, model alone, L0)

| Model | All rule items (822) | Calendar (219) | Note |
|---|---:|---:|---|
| Untuned base | 14.1% | 1.8% | half its replies hit the token cap |
| Phase 2 | 23.6% | 74.0% | calendar reference |
| A1 (best earlier adapter) | 60.5% | 62.1% | |
| U0 (earlier data, one joint run, 6 M) | 40.3% | 62.1% | |
| S1 (small steps, 6 M) | 66.8% | 58.4% | |
| **S2 (S1 + natural wordings, 6 M)** | **79.9%** | 63.5% | best so far; +19.5 [15.8, 23.4] over A1 |
| S3 (S2 + evidence, self-check, twins, 6 M) | 77.6% | 53.0% | combinations 38 vs 50, calendar −10.5 [−17.1, −3.8] against S2 |
| S1-8B (S1's data on Qwen3-8B) | 66.4% | 63.0% | −0.4 [−3.7, +3.1] against S1: no gain from the larger base |

Where S2 still fails (`results/phase6/first_error/dev6_p6s2.json`): the hidden-stem step of area-of-life questions and the 天克地冲 step of flow years (both rewritten in derivation version 2, tested by the fork), relations, overview loops, held-out combinations (50%), and calendar decisions (it gives a definite chart in 50 of 74 requests where none exists; families `zi_transition` 13/40, `invalid` 9/20, `tz_needs_tool` 0/8).

## 5. The failure of 15 November and the audit

- **What failed:** U1's data build stopped at the leakage check, because 1 of the 4,002 new training-writer questions (`train6b`) is on a chart that is also an evaluation chart. `queue_u1.sh` did not stop: later steps failed in seconds and it printed its end marker, U1-hi failed at its first step, and the round-2 queue waited for a marker that never came. The GPU was idle from 21:02 to 22:57.
- **Fixed:** the build leaves out a training-writer question on an evaluation chart (none in `train6`, so S2 and S3 rebuild bit for bit); every queue step is followed by a stop (tested); the failed logs are kept as `artifacts/logs/*_failed_20261115.log`.
- **Audit (all clean):**
  - No row of U0, S1, S2 or S3, and no row of the earlier training files behind A1, G1 and Phase 2, is on a Phase 6 evaluation chart (every family checked).
  - No candidate has a valid6 score yet (only Phase 2's calendar rows); sealed6 is closed (`results/phase6/selection.json` does not exist).
  - The screening decision (`results/phase6/screening.json`) was applied by the frozen rule: nothing dropped, best S run S2.
  - The 8B pilot ran as registered (batch 1 after the memory probe, 5.9 h); the prompt is rendered identically in training and evaluation.
- **Fixed in passing:** `make_report6.py` would have crashed on the chain's decision log (it read a key the loop no longer writes).

## 6. Decisions taken by the lead agent (pre-registration section 12)

1. **The main line uses S2's recipe, not S3's.** S3's added components gave no gain alone and significantly lowered calendar and combination accuracy. The registered U1 (S3's recipe) is still trained, after the chain.
2. **H7 is tested as a 6 M-token twin** (S2s against S2), which costs 4.6 h instead of 21 h and decides U2's prompt before the 15 M-token run.
3. **No further run on Qwen3-8B** (H10 not supported; the rule of section 11).
4. **U1-hi is not queued**; by hand only, under its registered condition.

## 7. Ideas not yet implemented, in order of expected value

Each needs a pre-registration entry before its results exist, and can be added as an extra arm of a chain level (`LEVELS` and `ARM_OPTIONS` in `scripts/phase6/decide6.py`, an option in `build_train6.py`; stop `queue_main.sh` before editing it, Python files can be edited while it runs).

1. **Calendar families weighted by measured error** in the continuations, as the procedure types already are. Evidence: every Phase 6 model is about 10 points below Phase 2 on calendar, and the errors sit in a few families. Needs a fresh calendar fit sample (`TD.generate_split` with a new seed), a `--sample c` in `own_replies.py`, and weights in `train_data6.calendar_rows`.
2. **Self-check and evidence rows without the calendar twins**, as an arm: S3 no longer breaks right answers on a re-check (0 of 278 against 21 of 290 for S2) and is hurt less by raw facts, but its mixture cost calendar and combinations. An arm with small shares (for example 3% each) would test whether the robustness can be had without the cost.
3. **Composition rows:** questions that join two trained procedures in ways other than the four held-out combinations (S2 50%, A1 48%). The held-out kinds must stay out of training.
4. **A routing line** as the first step of every derivation (name the task from the question and the JSON hint), if held-out wording stays near S2's 71%.
5. **More training writers** (local Gemma 4 and MiniCPM5; never Qwen3.5, the evaluation writer) and overview paraphrases.
6. **The user's own questions:** `data/phase6/user_questions/questions.txt` (format in its README); no person has written a test question yet.

## 8. Rules that must hold

- Commit locally; no push, merge or pull request. Leave the user's untracked notes alone. Weights and caches stay out of Git.
- sealed6 is scored once, after `selection.json`; valid6 is for selection only; dev6 is read freely.
- Every number in a document comes from a saved file via a script. System accuracy (L5) is never called model accuracy.
- Labels come from the rule code or lunar-python only. BaziQA is never training data. No life-prediction claims.
- No downloads without the user's approval (Qwen3-8B was approved and is in `models/qwen3-8b-mlx-4bit`).
