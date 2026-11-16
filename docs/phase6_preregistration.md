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
