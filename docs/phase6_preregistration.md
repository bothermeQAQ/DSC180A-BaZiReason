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
  - **U1-hi memory fallback.** If a short probe shows that all 36 layers at rank 16 do not fit in the 32 GB machine at batch 2, U1-hi uses rank 64 on the last 16 layers instead and is reported under that description.
  - **valid6 runs.** Only what the selection rule reads: the candidates at L0 (rule items and calendar) and Phase 2 on calendar. The existing models' L1-L4 run on sealed6 (section 8), and on dev6 for A1 and U0.
