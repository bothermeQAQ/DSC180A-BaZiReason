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
   - **UD (pre-registration 12.14) runs next by itself:** `scripts/phase6/queue_ud.sh` (started 18 November 11:03, log `artifacts/logs/p6_queue_ud.log`) waits for `QUEUE_MAIN_DONE`, then runs `decide6.py ud` (parent: the chain's best run; twin: its child level), the build with `--steps 3 --seed-from <twin>`, training (about 4.2 h), dev6 and valid6, and prints `QUEUE_UD_DONE` (or `QUEUE_UD_FAILED`). Wait for it before the selection: UD is a candidate. Do not start another GPU job while it runs. After its dev6: truncation, report, `first_error.py --labels p6ud`, `decision_first.py dev6 p6ud <twin>`; H12 is UD minus its twin (`RESULTS.md`, round-2 table).
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

## 9. State on Monday 16 November, 04:30 PST (supersedes the times in sections 1 and 2)

- **S2s finished** (03:39): 658/822 on dev6 alone against S2's 657 (H7: +0.1 [−2.9, +3.2]; calendar −0.9 [−7.6, +6.0]). No gain from 45% more rows; not worse, so U2 uses the short prompt.
- **U2 is training** since 04:04 (29,775 rows, 15.0 M tokens, 14,887 iterations; expected end about 14:35, scored by about 15:45). A first build was stopped after six minutes because its calendar part had under-filled; see the log.
- **Then, automatically:** the fork U3 / U3v (about 16:30 to Tuesday 04:00), the chain, valid6 for S2 and S2s, U1 as registered, diagnostics. Only `queue_main.sh` is running; `queue_s2s.sh` has ended.
- **Verified on 16 November:** the dry run of U2, the fork and a later level with both kinds of weights (no row on a held-out chart, fork arms paired, rebuild identical); every existing set (U0, S1, S2, S3, S2s) rebuilds bit for bit with the final code; the full test suite.
- **Derivation version 2 was extended before the fork** (pre-registration 12.6): the life-stage walk lists all twelve stages, the pattern step writes the 比劫 check for every stem that appears, and the relations step lists all eight groups of three. `results/phase6/steps_v2_check.json`: 15,000 derivations, every answer unchanged, no wrong statement.
- **Composition rows are now an arm of the second chain level** (pre-registration 12.7, H11): U4 against U4c, which has 10% composition rows (`bazireason.compose6`, six kinds, never a held-out combination). The chain's levels and arm options are in `LEVELS` and `ARM_OPTIONS` of `scripts/phase6/decide6.py`: level 3 is U3 / U3v, level 4 is U4 / U4c, then single runs to U10.
- **What the evidence says about strategy:** more rows of the same mixture no longer help at this size (S2s against S2), and the 8B base did not help either. The remaining errors are slips at steps that must decide where to stop or what to leave out (fixed in version 2, tested by the fork), unseen wordings (held-out template wording about 70%), held-out combinations (50 to 55%), and calendar decisions (weights by cell, 12.5). Of the ideas in section 7, calendar weights (1) and composition rows (3) are now in the chain; the routing line (4) and a robustness arm (2) are the next candidates, for level 5 or later.
- **Checked at 08:55 on 16 November:** U2 at iteration 6,720 of 14,887, no error in the log. The two fit-sample measurements ran end to end on the CPU with a stand-in model. `docs/phase6_report.md` now has the development results (section 4); section 5 onwards waits for the sealed runs.
- **Expected (PST):** U2 trained about 14:45 and scored about 15:55 on Monday; fork done about 05:00 Tuesday; U4 / U4c about 17:00 Tuesday; the chain ends between Tuesday evening and Thursday morning; then valid6 for S2 and S2s, U1 (13.5 h) and diagnostics; selection and the sealed runs (about 40 h with the added L0 runs) follow by hand.
- **Checked at 10:25 on 16 November:** U2 at iteration 8,920 of 14,887 (0.40 iterations per second), validation loss 0.0089 at iteration 8,500, peak memory 22.3 GB, no error. Updated times (PST): training ends about 14:35, dev6 scores about 15:05, valid6 about 15:35, then the fork starts; U3 about 22:15 Monday, U3v about 03:30 Tuesday, U4 / U4c about 15:30 Tuesday.
- **valid6 stays unread until the selection** (added 16 November, before any candidate had a valid6 score). The queue scores every candidate on valid6 right after dev6. `run6.py` leaves the count of right answers out of the log line for valid6, and `make_report6.py` writes that split's tables only when `results/phase6/selection.json` exists; before that it only names the runs that are scored. Do not open the files in `results/phase6/valid6/` by hand: `decide6.py select` reads them once.
- **U2 is done (16 November, 15:06).** dev6 alone: 733/822 rule items (89.2%; S2 657, S2s 658), calendar 182/219 (83.1%; Phase 2 162). U2 minus S2s: +9.1 [+6.1, +12.2] and +20.5 [+12.7, +27.8]. **Budget matters:** the earlier reading that more of the same mixture does not help held only at a fixed token budget. Details in the log.
- **What is left in U2:** 89 wrong answers, of which 50 are held-out combinations (unchanged at 50/100: H11, U4c) and 15 are area-of-life items. In 35 of 46 area-of-life replies the hidden-stem step names a stem the branch does not hide (S2s 22, S2 14): version 1's filtered list. Version 2 rewrites that step (H9, U3v), and pre-registration 12.9 adds a second reading for H9 on the statements in the replies (`make_report6.statement_pair`, written into `RESULTS.md` when both arms exist).
- **After each chain level:** `PYTHONPATH=src .venv-mlx/bin/python scripts/phase6/truncation6.py`, then `PYTHONPATH=src .venv/bin/python scripts/phase6/make_report6.py`, then `first_error.py --labels <the new runs>`; write the dev6 numbers into the log and commit. Read dev6 only.
- **The fork started at 15:37 on 16 November** (level 3: U3 and U3v from U2). Expected (PST): fit samples of U2 until about 17:05, U3 trained about 21:20 and scored about 22:20, U3v scored about 03:35 Tuesday, U4 / U4c about 15:35 Tuesday.
- **Restarting mid-level is safe since 16 November 15:45** (pre-registration 12.10): when the latest level has an arm without a dev6 result, `decide6.py next` asks for that arm once more with the same parent (decision `complete` in `round2.json`), then the chain goes on. A half-trained run is trained again from its start (its data folder is reused).
- **From level 4 on, flow-year and area-of-life questions ask about years of the chart's life** (pre-registration 12.11): `decide6.py` prints `u4 <parent> --years life` and `u4c <parent> --compose-share 0.10 --years life`; later levels inherit `years` from the parent's manifest. Read it in the `Question years` table of `RESULTS.md` (inside or outside 2024-2049). U1 keeps the fixed window as registered.
- **The user asked to be told when the model is well trained** (16 November, 21:00) and accepted the timeline, so U1 and the diagnostics stay before the selection. The test is pre-registration 12.12: four conditions on dev6 (chain ended; every trained type with 30+ items at 90% or more; years outside 2024-2049 within 10 points of inside; calendar levelled off). When `QUEUE_MAIN_CHAIN_DONE` appears, check them from `round2.json` and the `By question type`, `Question years` and `Calendar requests` tables of `RESULTS.md`, then tell the user either way (push notification and message) before the selection. If one fails, find its cause and register one targeted level (it must start by 27 November).
- **U3 scored on dev6 (16 November, 21:16):** 722/822 rule items (U2 733), calendar 189/219 (U2 182). Not a new best. Two side effects of the continuation weights: element counts fell to 35/44 (a life-stage phrase, "从木起顺数", replaces the first step in 10 replies, on independent wordings) and the calendar decision moved near solar terms (a definite chart in 12 of 18 requests that need the tool; U2 3). U3v (derivation version 2, same weights) trains from 21:49; its dev6 result is due about 02:30 Tuesday.
- **From level 4 on, calendar rows are weighted by family** (pre-registration 12.13): every cell of a family gets the family's weight, so the share of each expected answer inside a family stays as generated. `decide6.py` prints `--calendar-weight-by family` for U4 and U4c; later levels inherit it (`calendar_weighting.by` in the manifest). Read it in the `Calendar requests by kind` table of `RESULTS.md`.
- **U3v scored on dev6 (17 November, 02:25):** 730/822 rule items (U2 733, U3 722) and 201/219 calendar requests, the best calendar of any run (balanced decision accuracy 92.8%). Not a new best, so level 3 has none, and level 4 starts from U3v. H9: +0.6 [−2.2, +3.4] on the items whose derivation changed. Second reading: −4.5 [−6.3, −2.7], with wrong hidden-stem statements in area-of-life replies down from 30/46 to 0/46. Area of life is still 31/46. 10 of its 15 errors are year arithmetic for years outside 2024-2049, which level 4's life years aim at. Held-out combinations fell to 39/100. **If level 4 has no new best, the chain ends and condition 4 of 12.12 fails** (level 3 raised the calendar count by 19), so a targeted level follows: find the cause, register it, and start it by 27 November.
- **U4 scored on dev6 (17 November, 08:28):** 740/822 rule items, a new best (U2 733), so the chain goes on: level 5 starts from the better of U4 and U4c on dev6. Calendar 200/219 (U3v's 201 is still the best). The life years worked: no wrong year arithmetic is left, area of life is 40/46 (31 before), and years outside 2024-2049 are 69/77 right against 29/29 inside. Against 12.12, conditions 2 (area of life, two items short of 90%) and 3 (a gap of 10.4 points, one item short) still fail. The six area-of-life errors are later steps: female children charts given the marriage gods, a half-combination counted as 宫动, and misread palace cards (see the log). Held-out combinations fell to 35/100. U4c's dev6 result is due about 13:30; each later level takes about 6 hours.
- **U4c scored on dev6 (17 November, 13:30):** 739/822 rule items, one below U4, so level 5 continues from U4 (no composition rows; derivation version 2, years of the chart's life and calendar weights by family are inherited from U4's manifest). H11: +1.0 [−12.0, +14.0] on the held-out combinations (36 against 35), −0.1 [−2.2, +1.8] on all rule items: no gain. Side effect: six pattern items on one held-out wording answered in the composition rows' short shape (Ten God of the month's main stem, no 格). Expected (PST): U4's fit samples until about 15:00, U5 trained about 19:15, scored on dev6 about 19:45; each later level about 6 hours.
- **U5 training since 15:02 (17 November):** level 5 from U4, with U4's options inherited (derivation version 2, years of the chart's life, no composition rows, calendar weights by family). U4's calendar fit sample shows a regression dev6 is too small to see: Shanghai daylight-saving requests that need the tool 7/32 (U3v 26/32); U4 assumes UTC+8 on 1986-1989 summer dates. U5's family weights give that family 89 rows (U4 37). Expected (PST): trained about 19:10, scored on dev6 about 19:40. If the Shanghai cell has not recovered in U5's own fit sample (run before U6 is built), it is a candidate cause for a targeted level under 12.12.
- **U5's first attempt ran out of GPU memory (17 November, 15:13)** because a local model server (Ollama, started by the user's ChatGPT/Codex app at 15:08) loaded a 7.6 GB model beside it. The queue stopped; the user asked for an immediate restart, and it was restarted at 16:18 with Ollama still open but no model loaded. If the queue is still stopped, check `pgrep -x ollama` and `memory_pressure` first: a 20 GB training run does not fit beside a local model. The restart trains U5 again from its start (same data, config, seed). Expected (PST): trained about 20:25, scored on dev6 about 20:55.
- **U5 scored on dev6 (17 November, 21:01):** 740/822, the same as U4, so not a new best: one level without a new best, and level 6 starts from U5. For 12.12 the chain's best run stays U4 (ties go to the earlier run). U5 fixes area of life (43/46) and the year window (gap 6.5), but relations falls to 56/64, all on one held-out wording, at the step that collects the relations. Calendar falls to 193/219: the larger Shanghai daylight-saving share made U5 refuse answerable Shanghai dates from May to October of other years, and some other zones. If U6 (dev6 about 03:30 Wednesday) is not above 740, the chain ends; then check 12.12 on U4 and tell the user. The candidate causes are in the log entry.
- **U6 training since 22:32 (17 November):** level 6 from U5, built by the registered procedure from U5's fit samples (relations weight up to 0.118; Shanghai daylight-saving rows down to 28, plain charts and solar-term boundary rows up). 5,164 iterations; expected trained about 02:45 and scored on dev6 about 03:15 Wednesday. U5's sample B shows the relations drop on training wordings too (11/100 wrong, U4 1).
  - A cause for the calendar errors is measured (log, "How the calendar replies decide"; `scripts/phase6/decision_first.py`): the training answers decide in their first sentence, before any check. About half of each run's dev6 calendar errors are refusals that contradict their own check: 9 of U3v's 18, 5 of U4's 19, 11 of U5's 26.
  - The candidate fix is not registered: write the zone rule and the window comparison before the decision. Consider it when the chain ends.
- **U6 scored on dev6 (18 November, 03:10): 747/822, a new best** (U4 and U5 740), so the chain goes on and level 7 starts from U6 (U6's fit samples, then U7: built about 04:45, scored on dev6 about 09:30 Wednesday). Calendar 203/219, the best of any run. On U6, conditions 2, 3 and 4 of 12.12 hold (lowest trained type: area of life 42/46; year gap 6.5 points; best calendar +2 over levels 5 and 6); only condition 1, the chain's end, is open. The chain's best run is now U6. If U7 and U8 are not above 747, the chain ends after level 8: check 12.12 on U6 and tell the user. Both relations errors and 3 of the 4 area-of-life errors start at a collecting step (the final relation list, the year rows); 5 calendar errors are still window refusals of dates outside the window. Wrong statements rose only inside held-out combinations (Ten God slips, 42 statements). Details in the log.
- **U7 training since 04:42 (18 November):** level 7 from U6, built by the registered procedure from U6's fit samples (checked: U6's adapter as the start, data hashes match). 5,283 iterations; expected trained about 09:00 and scored on dev6 about 09:30 Wednesday. U6's sample B: 1,984/2,000 (relations back to 2 wrong). U6's sample C: 685/780 (U5 704).
  - **The calendar weights oscillate:** the Shanghai cell that needs the tool goes 10, 26, 7, 29, 11 of 32 from U2 to U6, each time following the family weight of the level before (3.41, 0.83, 2.86, 0.55; U7's data 2.44). The models learn a refusal rate, not the rule. This is more evidence for writing the rule before the decision (the unregistered candidate change).
- **U7 scored on dev6 (18 November, 09:22): 739/822, not a new best** (U6 747 stays the chain's best run; one level without a new best). Calendar 202/219. All three relations errors on training wordings start at the collecting step again. U8 trains from U7 since 10:50 (5,208 iterations; dev6 about 15:30). If U8 is not above 747, the chain ends after level 8: check 12.12 on U6 and tell the user (push and message), before the selection.
- **U8 scored on dev6 (18 November, 15:23): 754/822, a new best** (U6 747). The gain is all in the held-out combinations (50/100, U6 36): U8 checks each pillar against the void pair before answering. On the other 722 items it is 7 below U6, and relations fell to 57/64 (89.1%), so 12.12 condition 2 fails on U8. Calendar 204/219, the best count so far. The chain goes on: level 9 (U9) from U8 (dev6 about 21:30 Wednesday), then level 10 from U9 (dev6 about 03:30 Thursday), the last level. When U10 is scored the chain has ended: check 12.12 on the chain's best run and tell the user (push and message), before the selection. UD's parent is then the chain's best run (U8 unless U9 or U10 passes 754) and its twin the level built from it; `QUEUE_MAIN_DONE` moves to about 20:00 Thursday, UD's dev6 to about 01:00 Friday.
- **UD is prepared (pre-registration 12.14, the user's request of 18 November 08:39):** derivation version 3 writes each decision after its check (calendar 时区 and 交节 steps; relations keys copied from the pairwise checks; area of life with both sexes' gods and a verdict per year). Code in commit `e611be5`, checks in the log (288 tests; every set rebuilds bit for bit; `results/phase6/steps_v3_check.json`; a paired dry build). `queue_ud.sh` waits for `QUEUE_MAIN_DONE` (see section 3).
- **If `QUEUE_MAIN_FAILED` appears:** read the step above it in `artifacts/logs/p6_queue_main.log`, fix, restart with the command in section 2.
