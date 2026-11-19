# Phase 6 pre-registration

Written on 14 November 2026, before any Phase 6 model was trained and before anything was scored on `data/phase6/valid6` or `data/phase6/sealed6`. The commit that adds this file is the time stamp; later changes are listed in section 9 with their reasons. Plan: [`phase6_execution_prompt.md`](phase6_execution_prompt.md). Audit: [`phase6_capability_audit.md`](phase6_capability_audit.md).

## 1. Question

Can the language model itself, answering alone, become substantially better at BaZi rule reasoning than the best existing adapter (A1, about 69% on Phase 5 sealed questions), without losing the calendar skill? And which change is responsible?

| | Hypothesis | Tested by |
|---|---|---|
| H1 | Sequential rounds cause the skill trade-off | U0 (joint, existing data) against A1 (rules) and Phase 2 (calendar) |
| H2 | Coarse procedure steps cause the computed-step errors | S1 against U0: statement accuracy of computed steps, L0 accuracy |
| H3 | Narrow wording causes the natural-question gap | S2 against S1: blind-wording accuracy, seen against held-out wording |
| H4 | The model can learn to use evidence and check itself | S3 against S2: L1 − L0, L4 fix and break rates, L3w |
| H5 | Adapter capacity limits the result | U1-hi against U1 (run only under the rule in section 5) |
| H6 | The base model limits the result | fresh in-distribution accuracy (`scripts/phase6/diag_fit.py`) |

## 2. Data status

| Material | Status |
|---|---|
| Every Phase 2-5 split (including the Phase 5 sealed split), benchmark_v3, BaziQA | development or historical; never training data for evaluation wordings |
| `data/phase6/dev6/*`, `data/phase6/blind/dev6.jsonl` | development (read freely) |
| `data/phase6/valid6/*`, `data/phase6/blind/valid6.jsonl` | selection only |
| `data/phase6/sealed6/*`, `data/phase6/blind/sealed6.jsonl` | **sealed**: scored once, after `results/phase6/selection.json` exists; every access is logged in `results/phase6/sealed_access.jsonl` |

- **Year blocks** (`bazireason.splits6.phase6_years`): dev6 uses one year in each validation triple (1910, 1930, ...), valid6 the other two, sealed6 the test block. Training charts come from the training years. Every chart of an earlier evaluation split is excluded from valid6 and sealed6, and every evaluation chart is excluded from Phase 6 training (`tests/test_phase6.py`).
- **Parts:**

  | Part | dev6 | valid6 | sealed6 | What it is |
  |---|---:|---:|---:|---|
  | rules | 294 | 289 | 462 | Phase 3 rule families with an exact oracle plan; `meta.template == 2` marks the held-out wording |
  | indep | 140 | 140 | 280 | labels from lunar-python, a chart format and wordings never used in training |
  | blind | 188 | 184 | 379 | natural questions written by an untuned Qwen3.5-4B (the Phase 5 writer and filters) |
  | combo | 100 | 100 | 160 | held-out combinations of trained rules (flow year + life stage, 旬空 + positions, revealed hidden stems, Ten-God counts) |
  | cf | 100 | 100 | 160 | counterfactual pairs: one character changed, the answer changes |
  | calendar | 219 | 269 | 816 | chart requests with answerable and non-answerable twins |

- **Training wordings** come from a different writer: the untuned base Qwen3-4B (`data/phase6/blind/train6.jsonl`, 3,286 questions) with different descriptions and personas. No evaluation question, combination wording, counterfactual fact wording, independent template or L4 check request occurs in any training set (tested).
- **No user-written questions** were available (`data/phase6/user_questions/`).

## 3. Levels (`bazireason.ladder`)

| Level | The model receives |
|---|---|
| L0 | the question and the four pillars (independent questions keep their own chart format) |
| L1 | plus each symbol's element, polarity and hidden stems; no derived fact (checked mechanically; hidden stems are left out for hidden-stem questions) |
| L2 | plus generic rule text chosen by question type, identical for every chart |
| L3 | plus the rule code's conclusion; **L3w** the same words with a wrong conclusion |
| L4 | its own L0 reply and a check request: a, re-check; b, "there is an error"; c, "this field is wrong". Every wrong L0 answer and 60 right ones per part |
| L5 | the Phase 5 hybrid system (routing, value repair, engine fallback) |

- Levels are never averaged together. No repair, fallback, loop guard or routing touches L0-L4.
- **Primary level: L0.**
- **Decoding:** greedy, 1,100 new tokens for rule items and 700 for calendar rows, batch 32 with the GPU token budget of Phase 5.

## 4. Systems and training

- **Fixed baselines:** untuned base, Phase 2 (calendar reference), A1 (best model alone), G1 (conclusion following).
- **New adapters:** all trained from the untuned base with the chain's adapter settings (rank 16, scale 2, last 16 of 36 layers), learning rate 2e-4 with cosine decay, one pass, sequences of at most 2,048 tokens (longer rows dropped, never truncated). Budgets are exact token counts.

  | Run | Budget | Data (`bazireason.train_data6`) |
  |---|---:|---|
  | U0 | 6.0 M | the union of all earlier training files, sampled in proportion |
  | S1 | 6.0 M | small-step procedures on fresh charts, template wordings; fresh calendar rows; 12% earlier "other" tasks |
  | S2 | 6.0 M | S1 plus natural wordings (45% of procedure rows), other chart formats, contrast rows |
  | S3 | 6.0 M | S2 plus evidence rows, reference checking, self-check rows from S1's own replies on training charts, matched calendar twins |
  | U1 | 15 M | the recipe chosen by the rule in section 5 |
  | U1-hi | 15 M | U1 with all 36 layers (run only under section 5) |

## 5. Decisions on development data (frozen)

- **Recipe for U1:** start from S3. A component is dropped from U1 if its screening run is worse than the run before it on dev6 L0 rule accuracy (rules + indep + blind + combo + cf) with a paired 95% upper bound below 0. The component's own target counts as support, not as a reason to drop it: computed-statement accuracy for S1, blind accuracy for S2, L1 − L0 and the L4a fix rate for S3.
- **Calendar check:** if every S run is more than 10 points below U0 on dev6 calendar rows, U1 raises the calendar share from 28% to 35%.
- **U1-hi** is trained only if U1 answers fewer than 95% of fresh training-distribution procedure questions correctly at L0 (a sign of under-fitting) and the run can start by 27 November.
- **Stopping:** no further run after U1 (and U1-hi) unless one change gave a paired dev gain with a lower bound above 0 and a further change of the same kind is untested; never after 27 November.

## 6. Selection on valid6 (frozen)

- **Candidates:** U1, U1-hi if trained, and the best S run on dev6.
- **Calendar gate:** valid6 calendar accuracy minus Phase 2's, 95% lower bound at least −7 points. 269 rows cannot resolve ±5.
- **Rule:** among candidates that pass the calendar gate, the highest valid6 L0 accuracy on rules + indep + blind + combo + cf. Ties go to the smaller adapter, then to the earlier run. If none passes, the highest L0 candidate is selected and the calendar claim is reported as failed.
- `scripts/phase6/select6.py` applies the rule and writes `results/phase6/selection.json`.

## 7. Primary and secondary comparisons on sealed6

All are paired, with row-group bootstrap 95% intervals. M is the selected model.

- **P-1 (primary):** M minus A1 at L0, on rules + indep + blind + combo + cf (1,441 items).
  - **Success:** the lower bound is above 0. Target: +10 points.
  - **Power:** with about 30% discordant pairs the half-width is about 3 points.
- **P-2:** M minus A1 at L0 on the blind questions only (379; half-width about 6 points).
- **P-3:** M minus the untuned base at L0.
- **P-4 (calendar):** M minus Phase 2 on 816 sealed calendar rows, margin −5 points (half-width about 3.8). Balanced decision accuracy on twins is compared with Phase 2.
- **P-5 (evidence):** L1 minus L0 for M (success: above 0), next to A1 and G1.
- **P-6 (self-correction):** L4a fix rate and break rate for M against A1. Targets: fix at least 25%, break at most 2%.
- **P-7 (faithfulness):** L0 replies with a contradicted checkable statement. Target: M at most half of A1's rate.
- **H1 on sealed:** U0 against A1 at L0 and against Phase 2 on calendar.
- **Secondary:**
  - L2, L3, L3w (follows the wrong conclusion / corrects it);
  - combo and cf (pairs right on both charts);
  - seen against held-out wording; by question type;
  - statement-level accuracy and the first wrong statement; loops and truncation;
  - tokens and seconds per reply;
  - L5 for G1 and M, reported last as the system.

  No multiplicity correction; P-1 is the one success criterion.

## 8. Sealed runs (once)

- **L0:** base, Phase 2, A1, G1, U0, M; calendar for the same models.
- **L1, L2, L3, L3w, L4:** A1, G1, M; base L1-L3 only.
- **L5:** G1 and M.

## 9. Changes after this commit

Each entry was written before the result it could affect existed.

- **14 November, before any screening run was trained:**
  - **Script name.** The screening and selection rules are applied by `scripts/phase6/decide6.py screen` and `decide6.py select` (not `select6.py`). The rules are those of sections 5 and 6, unchanged.
  - **More training wordings for U1.** At 15 M tokens, U1 would reuse each of the 3,286 training-writer questions about 2.5 times. U1 adds `data/phase6/blind/train6b.jsonl`: the same writer (untuned Qwen3-4B), descriptions and personas as `train6`, a different seed, 220 per class. No other recipe changes.
  - **Self-check input for U1.** As for S3, U1's self-check rows are built from S1's own replies (`results/phase6/own_replies_p6s1.jsonl`, 2,400 replies); its 6% share needs about 600 of them.
  - **U1-hi memory fallback.** A probe (`scripts/phase6/probe_memory.py`: 8 real steps on the 16 longest U1 rows; fits when it ends normally with a peak of at most 25 GB) decides the set-up, in this order:
    1. all 36 layers at rank 16, batch 2, as planned;
    2. the same with gradient checkpointing (`grad_checkpoint` in the training config: activations are recomputed in the backward pass, so the gradients are the same and only time and memory change);
    3. rank 64 on the last 16 layers, reported under that description.
  - U1-hi uses exactly U1's data (same recipe, seed and options); the queue checks that the two training files are identical.
  - **valid6 runs.** Only what the selection rule reads: the candidates at L0 (rule items and calendar) and Phase 2 on calendar. The existing models' L1-L4 run on sealed6 (section 8), and on dev6 for A1 and U0.
- **14 November, 20:47, while S1 was training (no screening result, no valid6 or sealed6 score):**
  - **Fit share for the U1-hi rule.** The 2,400 fit questions (`scripts/phase6/own_replies.py`, sample A) are not all fresh for S2, S3 and U1. About 45% are training-writer questions on their original charts, which these runs train on, and S1's replies to the first part of the sample are the input of the self-check rows. The U1-hi rule of section 5 therefore reads the share over the sample-A questions whose chart is not in U1's training file (`scripts/phase6/fit_clean.py`). The full-sample share is still reported.
  - **A second round of training is added** at the user's request: section 10. It changes the candidates of section 6 and adds sealed runs to section 8; no other rule changes. The sealed split stays closed until the second round has ended.

## 10. Round 2 (added 14 November, 20:47)

Written while S1 was training. The dev6 results of the base, Phase 2, A1, G1 and U0 existed; no S result, no valid6 score and no sealed6 score existed. The user asked for improvements to the training and approved this round; it replaces the stopping rule of section 5 for the runs listed here, and no training run follows it.

### 10.1 Measurements behind it

- **Compute goes to a fixed prompt.** In the S1 training file, 41% of the content tokens are system prompts and 43% are the final reply, the only part with a training loss (`results/phase6/token_share.json`). The main system prompt (279 tokens, the same in 89% of the rows) is 34% of the tokens.
- **The models under-fit.** A1 answers 80.8% of its own training rows (`results/phase6/diag_fit.json`). U0, with 6.0 M tokens, is 20.2 points below A1 on dev6, and A1 is the end of six sequential rounds (the seven rounds up to G1 used about 21.7 M tokens).
- **Tokenization is not a cause:** each stem and branch is one token and no stem-branch pair is merged (`token_share.json`).

### 10.2 Runs

| Run | Starts from | Data | Budget |
|---|---|---|---:|
| U2 | the untuned base | U1's recipe and options with the short system prompt | 15 M sequence tokens, as U1 |
| U3, U4, U5 | the parent's adapter | the parent's recipe, question types reweighted by the parent's measured errors | 6.0 M each |

- **Short system prompt:** `你是 BaZiReason，可核验的八字历法与规则推理助手。先写依据，最后一行以“结论JSON:”开头给出 JSON 对象。` (36 tokens).
  - It replaces the main system prompt in every training row that has it. The system prompts of the strict-format calendar tasks and of the multiple-choice task are unchanged, because they define another output format.
  - It contains no chart-specific information. At every level, in calendar rows and inside L5, a model receives the system prompt it was trained with; nothing else in a prompt changes.
- **U2:** seed 2026111715; U1's settings.
  - At the same token budget it holds about 1.4 times as many rows as U1.
  - To keep U1's 45% share of natural wordings, it adds a third set of training-writer questions, `train6c` (same writer, descriptions and personas; another seed; 220 per class).
- **Adapter structure of round 2:** rank 16 on the last 16 layers. If U1-hi was trained and U1-hi minus U1 on dev6 L0 (all rule items) has a paired 95% lower bound above 0, round 2 uses U1-hi's structure instead.
- **Fit sample B** (`own_replies.py --sample b`): 2,000 questions, 100 for each of the generator's 20 question types, in training template wordings on fresh training-year charts (seed 2026111791).
  - No row of it is ever training input. Its charts are excluded from every round-2 training set; for runs built earlier, a question whose chart is in the run's training file is left out of the score.
- **Continuations (U3, U4, U5):**
  - **Parent of U3:** the run with the highest dev6 L0 accuracy on all rule items among U1, U1-hi and U2. Ties go to the smaller adapter, then to the earlier run.
  - **Weights:** the parent answers sample B. For question type *h* with *n* scored questions and *w* wrong, the error is (*w* + 1) / (*n* + 2). The type's weight in the procedure rows is its recipe weight times max(error, 0.05), for template and natural wordings alike. When the unused natural questions of a type run out, template questions on fresh charts replace them.
  - **Unchanged:** every other component keeps the parent recipe's share; a continuation keeps its parent's options, system prompt and adapter structure; the parent's files are not modified.
  - **Settings:** learning rate 1e-4 with cosine decay to 1e-5 and 40 warm-up updates; seeds 2026111716, …717 and …718; otherwise as U1.
  - **Loop:** a continuation whose dev6 L0 accuracy on all rule items is higher than its parent's becomes the next parent; otherwise the loop ends. At most three continuations. No run starts after 27 November.
- **Runs after each training run:** dev6 L0 and valid6 L0 on rule items and calendar rows, and sample B for a parent.

### 10.3 Hypotheses (secondary)

| | Hypothesis | Tested by |
|---|---|---|
| H7 | At equal compute, a short system prompt gives a better model, because more of the budget is training examples | U2 against U1 at L0 (rule items; calendar reported next to it) |
| H8 | Continuing with the failing question types weighted up improves the model further | each continuation against its parent at L0 |

Both are read on dev6 during the round and confirmed on sealed6 (supported when the paired 95% lower bound is above 0). P-1 remains the one success criterion.

### 10.4 Selection and sealed runs

- **Candidates (replaces the list in section 6):** the best S run on dev6, U1, U1-hi if trained, U2 and every continuation. The calendar gate and the rule are unchanged. "Smaller adapter" means fewer trainable parameters.
- **Sealed runs (added to section 8):** L0 on rule items and calendar rows for U1, U2 and every ancestor of M. On sealed6, H7 is U2 minus U1, and H8 is M minus the parent of the first continuation in its chain, if M is a continuation.
- `selection.json` is written once, after round 2 has ended, and the sealed split is opened only after it.

### 10.5 Changes to round 2 (14 November, 23:19)

Written after S1's dev6 result and before any other screening result, any valid6 score and any round-2 run. The user asked for every justified improvement, including more training.

- **What was known:** S1 answers 66.8% of the dev6 rule items alone (A1 60.5%).
  - On the template rule items it answers 79.9% of the training wordings and 37.3% of the held-out wording.
  - `scripts/phase6/first_error.py` compares each reply with the rule code's derivation. On training wordings the wrong answers start at four kinds of step: the season and support steps of strength (得令, 得势), the hidden-stem step of area-of-life questions, the clash step of flow years (天克地冲), and the pair and trio steps of relations.
- **Derivation version 2** (`bazireason.steps6`). Three steps are written out further; the rule code and every answer are unchanged.
  - **Strength:** the positions of the five elements seen from the day master are written first (as in the Ten-God table), and 得令 and 得势 are read from them.
  - **Areas of life:** the hidden stems of all four branches are listed with their Ten Gods before the relevant ones are picked; the palace branch's card is written before the yearly look-ups.
  - **Flow years:** the pillars whose branch clashes with the flow branch are named, and the 相克 order is written, before 天克地冲 is judged.
- **Fork at the first continuation:** U3 as in 10.2, and U3v with the same parent, weights, seed and budget but derivation version 2.
  - **H9:** U3v minus U3 at L0 on the strength, 扶抑, area-of-life, flow-year and overview items (the items whose derivation changed), on dev6 and then on sealed6.
  - The chain continues from whichever of the two has the higher dev6 accuracy on all rule items (a tie goes to U3) and keeps that run's derivation version.
- **Longer loop:** up to eight levels (U3 to U10; seeds 2026111716 to …723) instead of three.
  - Each continuation starts from the latest one (at the fork, from the better arm), with weights from that run's replies to fit sample B.
  - The loop ends when two levels in a row fail to exceed the best dev6 accuracy on all rule items reached so far in the chain (the first parent included), or when a run could not start by 27 November.
- **Selection and sealed runs:** every run of the chain is a candidate, as before. On sealed6, L0 is also run for U3 and U3v (H9).

## 11. A larger base model (added 15 November, 04:13)

Written while S2 was being scored on dev6 (no S2 result had been read) and before any run on the larger model. The user approved the download on 14 November.

- **Model:** Qwen3-8B in the 4-bit MLX weights published by Qwen (`Qwen/Qwen3-8B-MLX-4bit`, 4.4 GB): the family and architecture of the 4B base, with 36 layers. It wrote no question of any set.
- **Pilot (H10, base size):** S1's training file, seed and settings on Qwen3-8B ("S1-8B"), against S1.
  - **Same:** rank 16 on the last 16 layers, the learning rate and schedule, one pass, the long system prompt, thinking mode off. The evaluation prompt is rendered exactly as the training prompt.
  - **Batch:** set by the memory probe, in this order: 2 as for S1; 1 with gradient accumulation 4 (the same number of updates); the same with gradient checkpointing.
  - **H10:** S1-8B minus S1 at L0 on all dev6 rule items (paired). It is supported when the 95% lower bound is above 0. Calendar is reported next to it.
- **Decision:** if H10 is supported, further runs on the larger model use the best 4B recipe of round 2; they are recorded here before they start. If it is not supported, no further run uses the larger model.
- **Selection and claims:** runs on the larger model are selected separately, by the rule of section 6 (M8). P-1 to P-7 stay with the 4B selection M, whose base is A1's. On sealed6 the base-size comparisons are S1-8B minus S1 and, if M8 exists, M8 minus M, at L0.
- **Order:** the GPU runs one job at a time. The pilot runs after the screening queue and before U1. If it stops for a technical reason, the queue goes on to U1.

## 12. Changes after the screening and the pilot (15 November, 22:55)

Written after the screening queue and the larger-base pilot had finished, and before any of the runs named here existed. The user asked a new lead agent to investigate, recover the queue and change the plan where the evidence justifies it.

### 12.1 What was known

- **dev6 results:** every level of S1, S2 and S3; L0 of S1-8B; `results/phase6/screening.json`.
- **valid6:** only Phase 2's calendar rows (176/269). No candidate had any valid6 score. sealed6 was closed.
- **No run of round 2 existed.** U1's data build had stopped at the leakage check, so U1, U1-hi and everything after them had not run.

### 12.2 The failure, and what it did not affect

- **Cause:** one of the 4,002 new training-writer questions (`train6b`) is on a chart that is also an evaluation chart (the same four pillars recur on another date). The build's check stopped before any file was written.
- **The queue did not stop:** the U1 script went on through every later step, each failing in seconds, and printed its end marker. The GPU was idle from 21:02.
- **No result is affected.** No row of any trained set (U0, S1, S2, S3), and no row of the earlier training files behind A1, G1 and Phase 2, is on a Phase 6 evaluation chart (checked for every family, including the earlier tasks that the build's check leaves out).
- **Fix:** a training-writer question on an evaluation chart is left out when a set is built (none in `train6`, so S2 and S3 rebuild bit for bit; one in `train6b`). Every queue now stops at its first failed step.

### 12.3 Screening and pilot results that change the plan

| Comparison on dev6 | Points [95%] |
|---|---|
| S3 minus S2, all rule items at L0 | −2.3 [−5.2, +0.5] |
| S3 minus S2, combinations of trained rules | −12.0 [−23.0, −1.0] |
| S3 minus S2, calendar rows | −10.5 [−17.1, −3.8] |
| S1-8B minus S1, all rule items at L0 (H10) | −0.4 [−3.7, +3.1] |

- **The frozen rule of section 5 drops nothing**, so the registered U1 has S3's recipe.
- **S3's added components gave no gain alone and cost calendar and combination accuracy.** What they bought: raw facts hurt less (L1 minus L0 −1.9 against −7.7 for S2), and a re-check no longer breaks right answers (0 of 278 against 21 of 290) but fixes almost none (2 of 184).
- **H10 is not supported.** By section 11 no further run uses the larger model.

### 12.4 The revised plan

- **Main line on S2's recipe.** The runs that the final model comes from use the recipe of the best screening run on dev6 (S2), not S3's.
- **H7, restated as a twin at 6.0 M tokens:** S2s is S2's recipe, seed and budget with the short system prompt; H7 is S2s minus S2 at L0 on dev6 (rule items; calendar next to it), and on sealed6.
- **U2:** S2's recipe at 15 M tokens (seed 2026111715; `train6`, `train6b` and `train6c`), rank 16 on the last 16 layers. Its system prompt is the short one unless S2s minus S2 on dev6 has a 95% upper bound below 0 on the rule items or on the calendar rows.
- **Chain:** as in 10.2 and 10.5 (the fork U3 / U3v, then up to U10 with the two-level stop). Its first parent is the best run on dev6 among U2 and any of U1 and U1-hi that exist at that time.
- **U1 as registered** (S3's recipe, 15 M tokens, long prompt) is trained after the chain. **U1-hi** is not queued; it may be started by hand under the condition of section 5, and if it is not run H5 is reported as untested.
- **Candidates for selection (replaces 10.4):** the best S run on dev6, S2s, U1, U1-hi and U2 if trained, and every run of the chain. The gate and the rule are unchanged.
- **Sealed L0 runs (replaces the additions of 10.4, 10.5 and 11):** S2 and S2s (H7), U3 and U3v (H9), S1 and S1-8B (H10), U1, U2 and every ancestor of M.

### 12.5 Calendar decisions weighted by measured error (15 November, 23:03)

Written while S2s was training (no result of it read) and before any continuation existed.

- **Why:** every Phase 6 model is about 10 points below Phase 2 on the dev6 calendar rows, and the errors sit in a few kinds of request. S2 answers 13 of 40 day-boundary requests (Phase 2: 24), 9 of 20 invalid requests (16) and none of 8 requests that need a time-zone tool (7), and it gives a definite chart in 50 of the 74 requests where none exists.
- **Fit sample C** (`own_replies.py --sample c`): fresh calendar requests from the training generator (seed 2026111792), up to 40 for each cell, a cell being a request family with its expected status (for example day boundary with status ambiguous). It is never training input: a training request with the same text is left out.
- **Weights:** the parent answers sample C. For a cell with *n* scored requests and *w* wrong, the error is (*w* + 1) / (*n* + 2); the cell's weight is max(error, 0.05) divided by the mean of that value over the cells. In a continuation the calendar rows are drawn from a pool twice as large as before, with these weights; rows of other calendar tasks keep weight 1. The calendar share of the mixture does not change.
- **Scope:** every continuation of the chain, both arms of the fork included (they get the same weights). If the sample cannot be scored for a technical reason, the continuation is built without calendar weights and the manifest says so.
- **Reading:** H8 (each continuation against its parent) now covers both kinds of weights; the calendar difference is reported next to the rule items.

### 12.6 After the H7 twin: U2 as built, and derivation version 2 extended (16 November, 04:07)

Written after S2s's dev6 result and while U2 was training (no result of it), before the fork (U3, U3v) existed. Version 2 had not been used in any run.

- **H7 on dev6:** S2s minus S2 is +0.1 points [−2.9, +3.2] on the rule items (658 against 657 of 822) and −0.9 [−7.6, +6.0] on the calendar rows. The short prompt held 45% more rows in the same budget and gave no gain; it is not worse either, so by 12.4 U2 uses it.
- **U2 as built:** 29,775 rows, 15.0 M tokens, with the planned shares. A first build held only 20% calendar rows instead of 30%, because the calendar generator's pool was exhausted (a short prompt holds more rows in the same tokens). It was stopped after six minutes of training, before any result, and rebuilt; the pool now grows with the budget, and every earlier set rebuilds bit for bit.
- **What S2s's errors showed** (`results/phase6/first_error/dev6_p6s2s.json`): on the counterfactual life-stage items it answers 4 of 14 (S1 13 of 14). It writes the walk through the twelve stages correctly but ends it at the wrong branch; S1 wrote all twelve stages and read the answer off. On training wordings it also takes a 比肩 or 劫财 stem as the pattern (5 of 32 pattern items) and names a group of three branches that is not in the chart.
- **Principle:** in version 2 no step chooses, while it is written, where to stop or what to leave out. Three steps are added to the three of 10.5; the rule code and every answer are unchanged.
  - **Life stage:** all twelve stages are listed from the day master's 长生, then the branch is looked up.
  - **Pattern:** for every hidden stem that appears among the stems, the text says whether its Ten God is 比劫 before it is taken or not.
  - **Relations:** all eight groups of three are listed with the branches that are present, then the complete ones are named.
- **H9's items** are those whose derivation differs between the versions: strength, 用神, area-of-life, flow-year, overview, pattern, relations and life-stage items.

### 12.7 Composition rows as an arm of the second chain level (16 November, 04:16)

Written while U2 was training, before any continuation existed.

- **Why:** the held-out combinations of trained rules are the weakest rule items of every model (S2 50 of 100, S2s 55, A1 48). The replies start the right sub-procedures and fail where they join. Neither more rows of the same mixture (S2s) nor a larger base (S1-8B: 41) helped.
- **Composition rows** (`bazireason.compose6`): questions that carry the result of one trained procedure into another, written out in small steps, with every value from the rule code. Six kinds:
  - the life stage of the day master at all four branches;
  - the branch that clashes with a given branch, and the Ten Gods of its hidden stems;
  - the direction of the luck cycle, its first pillar and that pillar's Ten God;
  - the most frequent element of the eight characters and its Ten-God class;
  - the relation between two given branches and the Ten Gods of their main hidden stems;
  - the life stage and the seasonal state of the day master in the month.
- **The held-out kinds stay held out:** no row is a flow-year life stage, a 旬空 position question, a revealed-hidden-stem question or a Ten-God count, and no kind ends with one of those four last operations on the same inputs. The existing test that no evaluation wording occurs in a training set covers these rows.
- **Arm:** U4 as in 10.2 and 10.5, and U4c with the same parent, weights, seed and budget, in which 10% of the tokens are composition rows taken from the procedure share.
  - **H11:** U4c minus U4 at L0 on the held-out combination items (dev6: 100; sealed6: 160), with all rule items next to it.
  - The chain continues from whichever of the two has the higher dev6 accuracy on all rule items (a tie goes to U4) and keeps that run's mixture.
- **Sealed L0 runs:** U4 and U4c are added.

### 12.8 How U2 is read, and valid6 kept unread until the selection (16 November, 10:40)

Written while U2 was training (iteration 9,220 of 14,887), before any result of it and before any candidate had a valid6 score.

- **U2 against the 6.0 M runs is a description, not a hypothesis.** Two paired differences at L0 are reported on dev6 and on sealed6, on all rule items with the calendar rows next to them:
  - **U2 minus S2s:** the same recipe and system prompt. U2 has 2.5 times the tokens, another seed and two more files of natural wordings (`train6b`, `train6c`), so the difference is not attributed to one of the three.
  - **U2 minus S2:** the best screening run; it also differs in the system prompt.
- **Why no hypothesis:** the budget is not varied alone, and no target was set before the run was built.
- **valid6 is not shown before the selection.** The queue scores every candidate on valid6 after dev6. From now on `run6.py` prints the valid6 line without the count of right answers, and `make_report6.py` writes the valid6 tables only when `results/phase6/selection.json` exists. The records on disk, the gate and the selection rule are unchanged.

### 12.9 After U2: a second reading for H9 (16 November, 15:20)

Written after U2's dev6 result and before the fork existed: no data, weights or run of U3 or U3v, and no fit sample of U2.

- **U2 on dev6, model alone:** 733 of 822 rule items (89.2%) and 182 of 219 calendar rows (83.1%). U2 minus S2s: +9.1 points [+6.1, +12.2] on the rule items and +20.5 [+12.7, +27.8] on the calendar rows. Of its 89 wrong answers, 50 are held-out combinations and 15 are area-of-life items.
- **What the replies show:** in 35 of U2's 46 area-of-life replies, the step that names the hidden stems carrying the wanted Ten Gods names a stem that the branch does not hide (S1 14, S2 14, S2s 22 of 46). The answer is still right in 24 of the 35, because the final JSON does not carry that step. With more training on version 1 this step got worse; version 2 (10.5) writes the hidden stems of all four branches before it selects.
- **Second reading for H9, on dev6 and sealed6 at L0:** U3v minus U3 in
  - the share of rule-item replies with at least one wrong statement by the claim checker (`bazireason.claim_check`; repetitions are not statements), and
  - the share of area-of-life replies with a wrong hidden-stem statement,
  paired by item with the same bootstrap as the accuracy differences (`make_report6.statement_pair`).
- **The test of H9 is unchanged:** answer accuracy on the items whose derivation changed (12.6). The second reading is reported next to it and decides nothing in the chain.
- **Nothing else changes:** the fork, its weights, seeds and budgets, and the chain's rules are as registered.

### 12.10 A level whose arm has no result (16 November, 15:39)

Written after the queue had started the fork (15:37, level 3: U3 and U3v from U2) and before any data, weights or result of it existed.

- **Found by reading the code:** `decide6.py next` counted a level as reached as soon as one of its arms had a dev6 result. A queue restarted between the two arms of a pair would have gone on to the next level; the second arm (U3v or U4c) and its hypothesis would have been lost. The same held for an arm that failed.
- **Rule from now on:** when the latest level has an arm without a dev6 result, that arm is asked for once more, with the same parent, options, seed and budget (decision `complete` in `results/phase6/round2.json`). If it still has no result after that, the level stands with the arms it has, as before.
- **Unchanged:** the run a level starts from, the arms of each level, the two-level stop and the cutoff date.

### 12.11 Years of the chart's life in flow-year and area-of-life questions (16 November, 18:48)

Written while U3 was training (iteration 2,900 of 5,471), after U2's dev6 result and before any data of level 4 existed. U3v's data is not affected.

- **Found:** every Phase 6 training row of these two types asks about the same window: flow years 2024-2049, area-of-life periods starting 2024-2045 (`train_data6.concrete_intent`). The evaluation questions use years of the chart's life, drawn by the Phase 3 generator (flow years 8 to 69 years after the birth year, first years 16 to 49 after it). U2 on dev6:
  - flow-year items: 16 of 16 right with the year inside 2024-2049, 39 of 44 outside;
  - area-of-life items: 13 of 13 inside, 18 of 33 outside (S2: 7 of 13 and 18 of 33).
  - All 20 of U2's wrong answers on these two types are on years the training rows never asked about. Fit sample B uses the window too, so the continuation weights cannot see this (U2 answers 1,960 of its 2,000 questions).
- **Change:** from level 4 on, the template-wording questions of these two types ask about years of the chart's life, drawn as the Phase 3 generator draws them (`build_train6.py --years life`; the birth year is the date the training chart was drawn from). Paraphrased questions keep their years. Both arms of level 4 get it (`ARM_OPTIONS` in `decide6.py`), so H11 stays paired, and every later level inherits it from its parent.
- **Checked before use:**
  - 6,000 derivations (1,500 training charts, both types, both derivation versions) with years 1910-2160: every final JSON equals the rule code's answer, the claim checker finds no wrong statement, and the longest reply is 929 tokens of the 1,100 cap.
  - With the default (`fixed`), every existing set (U0, S1, S2, S3, S2s, U2, U3) rebuilds bit for bit, and U3v builds identically with and without the change.
  - A dry run of level 4 (stand-in parent U2): 546 of U4's 638 template rows of these types are outside 2024-2049 (years 1915-2152); no row on a held-out chart; both arms have the same calendar requests; U4c's procedure questions are a subset of U4's.
- **How it is read (a description, not a hypothesis):** on dev6 and sealed6, flow-year and area-of-life items with the year inside and outside 2024-2049, for every run (`make_report6.years_section`). Level 4 also changes the parent and the weights, so the change is not separated from the continuation; the items inside the window are the comparison.
- **Unchanged:** fit samples B and C, the evaluation splits, the chain's rules, and U1 as registered (fixed years).

### 12.12 When the model counts as well trained (16 November, 21:07)

Written while U3 was being scored on dev6, before its result was read, after the user asked to be told when the model is well trained.

- **Why the stop rule is not enough:** two levels without a new best on dev6 say that more of the same continuation stops paying. They do not say whether what is left could be fixed by better data. U2 answered 1,960 of 2,000 fresh training-distribution questions, yet 20 of its 89 wrong dev6 answers came from years the training never asked about (12.11).
- **The lead agent calls the model well trained only when all four hold**, read on dev6, model alone, for the chain's best run on dev6:
  1. **The chain has ended** by its rule (two levels without a new best) or after U10.
  2. **Every trained question type is learned:** each type with at least 30 dev6 rule items is at 90% or more. The four held-out combinations are not counted. U2: only area of life is below (31/46).
  3. **The year window is closed:** on flow-year and area-of-life items together, the share right with the year outside 2024-2049 is within 10 points of the share inside. U2: 57/77 outside, 29/29 inside.
  4. **Calendar has levelled off:** the runs of the last two levels raised the best dev6 calendar count by fewer than 5 of 219 requests (about 2 points) over the best before them.
- **Not a condition:** the held-out combinations. They are never trained (H11 tests whether composition rows help), so they are reported as the limit of generalization, not counted against training. U2: 50/100.
- **If 1 holds and 2, 3 or 4 does not,** the plateau points to a gap in the data, not to the limit of the model. The lead agent looks for the cause (as for the year window), registers a targeted change in this section before any of its data or results exist, and runs one more level from the best run if it can start by 27 November. If no cause is found, the model is reported as trained as far as this recipe goes, with the failed condition named.
- **The user is told** when the chain ends, either way, before the selection.
- **Unchanged:** the chain's rules, the candidates, the selection rule and the sealed runs. This decides only whether a targeted level is added before the selection.

### 12.13 Calendar weights by family from level 4 on (16 November, 22:07)

Written after U3's dev6 result, before U3v's result and before any data of level 4 existed. U3v's data was built before the change.

- **Found in U3's calendar requests on dev6** (`Calendar requests by kind` in `RESULTS.md`): U3 answers 189/219 against U2's 182, but part of the gain is a moved decision. Near solar terms, of the 18 requests that need the tool U3 answers 6 right and gives a definite chart in 12 (U2: 15 right, 3 definite); solar-term windows with a definite chart went from 5/10 to 9/10. Definite charts where none exists rose from 13 to 22 of 74, answerable requests refused fell from 14 to 3 of 145, and balanced decision accuracy fell from 85.7% to 83.4%.
- **Cause:** 12.5 weights each cell (a family with one expected answer) by the parent's errors on that cell alone. U2 was wrong on 25 of 40 solar-term-window requests with a definite chart in fit sample C and on 7 of 40 needing the tool, so U3's rows were drawn with weights 3.12 and 0.96: about three times as many of the first. The model learned to answer definitely, not to tell the two apart. Each later level would push the decision back and forth in the same way.
- **Change:** from level 4 on, a cell's error is that of its whole family (wrong and n summed over the family's cells). Every cell of a family then has the same weight: families the parent fails get more rows, and the share of each expected answer inside a family stays as the generator makes it. `build_train6.py --calendar-weight-by family`, recorded as `"by": "family"` in the manifest's `calendar_weighting`; later levels inherit it. Both arms of level 4 get it (`ARM_OPTIONS` in `decide6.py`), so H11 stays paired. The default (`cell`) is unchanged.
- **Checked before use:** every existing set (U0, S1, S2, S3, S2s, U2, U3 and U3v) rebuilds bit for bit with the new code and the default. A dry run of level 4 with U2's fit samples as stand-ins (U3's do not exist yet) gives every cell of a family one weight; it draws 177 solar-term-window requests with a definite chart against 160 needing the tool (cell weights: 221 against 82) and 36 Shanghai daylight-saving requests with a definite chart against 35 needing the tool (cell weights: 15 against 44). U4 and U4c share their calendar requests, no row is on a held-out chart, and a dry U5 built from the dry U4 inherits family weights and life years without being told. The full test suite passes (281 tests, one new for the family weights).
- **How it is read (a description, not a hypothesis):** the calendar results by kind for every run of the chain (`make_report6.calendar_kinds_section`), above all the cells near solar terms. Level 4 also changes the parent, the question years and the procedure weights, so the change is not separated from them.
- **Unchanged:** fit sample C, the floor of 0.05, the procedure-type weights, the evaluation, U1 as registered and the chain's rules.

### 12.14 Derivation version 3: each decision after its check (18 November, 08:50)

Written while U7 was training, before U7's dev6 result and before any data, weights or result of the run below existed. The user asked for it to be prepared as an extra level after the chain (18 November, 08:39).

- **Found** (`docs/phase6_log.md`, 17-18 November; `scripts/phase6/decision_first.py`, `first_error.py`; dev6 and fit samples only):
  - **Calendar:** in the training answers without tool facts (`worked.explain_request`, mode `none`), the first sentence already carries the decision. No earlier sentence states the zone's rule or compares the date with the solar-term windows. The models state the right window but refuse dates outside it, or refuse zones whose offset the rule gives. These two kinds are 9 of U3v's 18 dev6 calendar errors, 5 of U4's 19, 11 of U5's 26 and 5 of U6's 16.
  - **The calendar weights cannot teach the rule:** in fit sample C, Shanghai daylight-saving requests that need the tool are 10, 26, 7, 29 and 11 of 32 right from U2 to U6, each time following the family weight of the level before (3.41, 0.83, 2.86, 0.55). The models learn a refusal rate, not the rule.
  - **Relations:** in the wrong replies read, the pairwise checks are right. The errors are in the final list (a full trio dropped, a trio added, an invented 自刑) or in copying it into the JSON. U6's two training-wording errors start at the final list.
  - **Area of life:** U4's six dev6 errors are two female children charts given the female marriage gods in the first line, two half combinations counted as 宫动, and two misread palace cards. U6's training-wording errors start at the year rows (2) and at the final list of years (1).
- **Change: derivation version 3** (`build_train6.py --steps 3`; `steps6.set_steps(3)`, which also turns on `worked`'s check-first calendar answers during the build). Only training answers change. The questions, the final JSON objects (up to the order of relation lists, which scoring ignores: `p2_eval._same`, `p3_eval.UNORDERED`) and scoring are unchanged. Versions 1 and 2 are unchanged.
  1. **Calendar without tool facts**, for every request whose answer depends on the UTC offset (well-formed, parsable, inside the study range):
     - a 时区 step first: the zone's rule as one fixed sentence built from `known_offset`'s tables, then the request's year and month compared with it, ending "→ UTC+8" (or the zone's offset) or "→ 偏移需工具";
     - if the offset is known, the 参考时刻 step as before, then (unless the 23-o'clock sect question decides first, as in `no_tool_expected`) a 交节 step: the reference date compared with the windows of the solar terms around it, ending "不在交节窗口内" or "在{节}窗口内 → 需工具";
     - then the decision: a refusal, the existing non-answer text, or the four pillars as before. The verify rows without tool facts get the same two steps.
  2. **Relations** (`steps6.interact_lines`): every pairwise result (stems and branches; adjacency marked on both) ends with its JSON keys or "无"; the 三合/三会 line ends with its keys; the final two lines copy the keys in the order they appeared, each once; the JSON copies those two lines.
  3. **Area of life** (`steps6.domain_lines`): the first line writes the domain's correspondence for both sexes before taking the chart's; each year row applies the palace rule in words (冲 or 六合 → 宫动; any other relation → 不算) and ends with a verdict (计入 or 不计); the final list copies the years marked 计入.
- **The run, UD** (recipe `ud`, label `p6ud`), queued by `scripts/phase6/queue_ud.sh` to start after `QUEUE_MAIN_DONE` (after U1 and its diagnostics), by 27 November:
  - **Parent:** the chain's best run on dev6 (all rule items, L0; `decide6.best` over the chain's runs and its first parent: ties to the smaller adapter, then the earlier run), fixed once the chain has ended (`decide6.py ud`, recorded in `results/phase6/round2.json`).
  - **Twin:** the chain level built from that parent. UD has the twin's seed, the parent's fit samples B and C, the registered continuation procedure (6.0 M tokens, learning rate 1e-4 to 1e-5, the parent's adapter as start) and every option inherited from the parent's manifest. Only `--steps 3` differs. If the parent has no child level (it is U10), UD uses seed 2026111724 and has no twin.
  - **Runs after training:** dev6 and valid6 at L0 on every part, as for every level. valid6 stays unread until the selection.
- **H12 (decide after the check):** UD minus its twin at L0, paired by item, (a) on the calendar requests and (b) on the relations and area-of-life items, with all rule items next to them. Read on dev6, confirmed on sealed6; supported when the paired 95% lower bound is above 0 (as H7-H11). Without a twin, UD is read against its parent and H12 is reported as untested.
- **Descriptions, not hypotheses:** answerable calendar requests refused (of 145 on dev6) and definite charts where none exists (of 74); for UD and its twin, where wrong replies first leave the derivation (`first_error.py`).
- **Selection and sealed runs:** UD is a candidate like every continuation (sections 6 and 10.4: the calendar gate, then the best valid6 L0 accuracy on all rule items; ties to the smaller adapter, then the earlier run, UD being the latest). On sealed6, L0 is also run for UD and its twin (H12).
- **12.12 is unchanged:** "well trained" is judged on the chain's best run when the chain ends, and the user is told then, before UD. UD's result is reported on the same four conditions.
- **Checks before the build:**
  - the test suite;
  - every existing set rebuilds bit for bit with the default (`build_train6.py --check`);
  - on thousands of charts, every version-3 derivation's final JSON equals the rule code's answer (relation lists as sets), and the claim checker finds no wrong statement in its prose;
  - on thousands of random calendar requests, the version-3 text decides as `no_tool_expected` does: a refusal exactly when the tool is needed, the 时区 conclusion equal to `known_offset`, the 交节 conclusion equal to `jie_window_of`;
  - a dry build of UD in a scratch folder, with the current best run as a stand-in parent.
- **Unchanged:** the chain and its rules, U1, the candidates already registered, the selection rule, the gate, and the sealed split's rules.
- **Amendment (18 November, 09:45), before any of UD's data existed:** calendar replies are scored with a cap of 700 tokens (`run6.MAX_TOKENS`). As first written, the version-3 calendar answers with a definite chart ran up to 707 tokens before the final JSON, so many would have been cut off. They are now written compactly:
  - the 时区 step gives the zone's rule in one clause ("Asia/Shanghai 除 1919 年、1940–1949 年和 1986–1991 年 4–9 月外为 UTC+8；1961 年 6 月不在例外中 → UTC+8");
  - the 参考时刻 step no longer repeats the zone and its offset;
  - the 交节 step compares the date with the nearest window bounds ("6 月 27 日晚于芒种最晚的 6 月 7 日、早于小暑最早的 7 月 6 日 → 不在交节窗口内");
  - the month step refers to the 交节 step ("上一个节是芒种（见交节）") instead of restating the windows. The year, day and hour steps are as before.
  - Measured with the final JSON line (`results/phase6/steps_v3_check.json`): answers with a chart at most 624 tokens of 700 (mean 555; version 2: 542 and 480); area of life at most 1,001 of 1,100 (version 2: 928); relations at most 931 (803). No training answer is over a cap.

### 12.15 Level 10 waits for the user's choice (18 November, 16:10)

The user wrote at 16:05 on 18 November: "once u9 is finished do not start u10 yet, update me, and i will decide whether to start u10 or not". This is recorded before any of U9's data exists: U8's fit samples were being drawn, and U9 is built after them.

- **The hold:** U9 (level 9) is built, trained and scored on dev6 and valid6 as registered. After that, `decide6.py next` does not start level 10 by itself. It waits until the user's choice is recorded in `results/phase6/round2.json` by `decide6.py user 10 start|stop "<the user's words>"`. While it waits it writes "HOLD level 10" to the queue log, and the GPU is idle. Nothing else in the queue changes.
- **The user decides on dev6** (read freely), like every chain decision so far. valid6 stays unread until the selection.
- **start:** level 10 runs as registered (sections 10.2 and 10.5, 12.11, 12.13): U10 from U9, with U9's fit samples.
- **stop:** the chain ends after level 9 by the user's choice. This is recorded as the loop's end, with `"ended_by": "user"`.
  - It is the chain's end for 12.12: condition 1 holds, and condition 4 reads levels 8 and 9.
  - It is the chain's end for UD (12.14). UD's parent is the chain's best run on dev6 among U2 (the first parent) and levels 3 to 9; its twin is the level built from that parent.
  - If the parent is U9, UD has no twin. It then uses seed 2026111724 and is read against its parent, and H12 is reported as untested (12.14).
- **Code:** `decide6.py` holds in `next` and has a new `user` command. `make_report6.py` lists the choice and how the chain ended among the round-2 decisions. Tests cover the wait, both choices, a choice recorded only once, and UD after a stop.
- **Unchanged:** the rules of every level that runs, U1, UD, the selection rule and the gate, and the sealed split's rules.
