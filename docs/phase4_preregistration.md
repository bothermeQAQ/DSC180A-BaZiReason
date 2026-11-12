# Phase 4 pre-registration

Written and committed on 12 November 2026, **before** any Phase 4 candidate was trained or scored on the Phase 4 validation split, and before any row of the Phase 4 final test split was read. The commit that adds this file is the time stamp. Any later change to it is a new commit, listed at the end with its reason.

## 1. Status of earlier material

- **Inspected (development material in Phase 4):**
  - every Phase 2 and Phase 3 split, including the Phase 3 test split and `p2check_v1`;
  - benchmark_v3;
  - BaziQA.

  Phase 3 round 3 was chosen after its test results were inspected, so its Phase 3 test numbers are not an untouched estimate. These sets are used in Phase 4 for error analysis and tuning only. Phase 4's error analysis is `results/phase4/error_analysis.json`.
- **Adapters:** the Phase 3 round 3 adapter stays the experimental demo model, and round 2 stays the pre-registered validation winner of Phase 3. Nothing earlier is overwritten.

## 2. New data (hashes in each folder's `manifest.json`)

| Split | Years | Parts (rows) | Use |
|---|---|---|---|
| `data/phase4/train_v1` | training | 3,715 train / 179 valid (loss only) | approach A |
| `data/phase4/valid` | validation | `p3` 240, `p2` 306, `indep` 140 | adapter selection (section 4) |
| `data/phase4/final_test` | test | `p3` 788, `p2` 484, `indep` 280 | scored once, after selection |

- **Parts:**
  - `p3`: the Phase 3 rule-reasoning families.
  - `p2`: the Phase 2 calendar families.
  - `indep`: rule questions labelled by lunar-python, in seven kinds, with wordings and a chart format never used in any training set.
- **Exclusions:** prompts identical to any inspected row are excluded.
- **Known overlap:** 3 validation and 5 final-test prompts also occur in training, all table drills with a finite question space.

## 3. Candidates and workflows

- **Adapters:**
  - P2: `bazireason-qwen3-4b-lora-v2`.
  - R2: `-p3-r2`.
  - R3: `-p3-r3`.
  - A1: `-p4-a1`, which continues R3 on `p4_train_v1`.

  At most one more approach-A adapter (A2) may be trained. Its data may only follow from errors read on development material, never on `data/phase4/valid`, and it enters the same selection.
- **Workflows, fixed now:**
  - **W0, model alone:** the prompts of Phases 2 and 3.
  - **W1, verify and repair (approach B):** deterministic code (the calendar engine and the project's rule code) recomputes every field of the reply's final JSON that it can decide. If any field differs, the model gets one more turn that names the fields that differ, but not their values, and asks it to recheck and answer again. The scored answer is the second one. For `indep` rows the checker uses `bazireason.rules`, not the lunar-python labels.
  - **W2, independent tool check (approach B):** where a tool answer is supplied (`strict_shift`, benchmark_v3 `tool_corrupted`), code checks it against rules that do not use the tool. The checks:
    - 五鼠遁 and 五虎遁;
    - clock time against the hour branch;
    - date against the year pillar away from 立春;
    - an independent day-number formula;
    - the request's own ambiguity.

    A failed check adds one line to the prompt naming the pillars that failed, without the corrected values.
  - **W3, contest guard (approach B, BaziQA only):**
    - Inputs: the verified chart plus the rule code's deep-structure lines.
    - Loop guard: when a reply repeats a span three times and has no parseable option, it is cut at the loop and the model is asked for the option letter only, in at most 16 tokens.
    - Citation check: titles that are not in the rule cards are counted and flagged.
- **Engine display (system ceiling, not a model result):** for fields the code can decide, the chatbot shows the engine's value. This is correct by construction and is reported only as such.

## 4. Adapter selection rule (frozen)

Implemented in `scripts/phase4/select.py`; inputs are W0 runs on `data/phase4/valid` only.

1. **Calendar non-inferiority.** On `valid/p2.jsonl` rows with `task == "chart"`, the paired difference of the candidate minus P2 must have a 95% interval lower bound ≥ −5.0 points. The interval comes from a row-group bootstrap, `p2_eval.paired_difference`, with 2,000 resamples and a fixed seed.
2. **Among eligible adapters,** choose the one with the most correct answers over all validation rows (`p3` + `p2` + `indep`).
3. **Ties** keep R3. If no adapter is eligible, R3 stays the demo model, and the report says that no adapter met the calendar constraint.

## 5. Final test plan (scored once, after `results/phase4/selection.json` exists)

- **Runs:** W0 for P2, R2, R3 and A1 (and A2 if trained), plus W1 for R3 and the selected adapter, on all three parts. Every access to the final test is appended to `results/phase4/final_test_access.jsonl`, and the runner refuses to start before the selection file exists.
- **Primary comparisons** (selected adapter, W0; paired bootstrap 95% intervals):
  - **P-1:** calendar rows (`p2`, `task == "chart"`) against P2, with a non-inferiority margin of −5 points.
  - **P-2:** Phase 3 reasoning items (`p3`, excluding `strict_shift`) against R3.
  - **P-3:** independent rule questions (`indep`) against R3 and against P2.
- **Secondary comparisons (reported with intervals, no multiplicity correction, called exploratory):**
  - exact four pillars;
  - rule questions in the Phase 2 wording;
  - the unseen-wording subset;
  - false refusals and definite answers to requests without an answer;
  - claims and ambiguity measures;
  - `strict_shift` corrected and followed, W0 against W2;
  - the 2 × 2 of adapter {R3, selected} × workflow {W0, W1};
  - benchmark_v3 `tool_corrupted`, W0 against W2 (inspected benchmark);
  - BaziQA, W0 against W3, for R3 and the selected adapter (inspected benchmark; nothing is tuned on its keys, and accuracy is reported as agreement with contest keys only).
- **Not done:** no prompt, threshold or checkpoint is changed after a final-test result is seen. If a bug is found after scoring, the fix and both results are reported.

## 6. Changes after this commit

Each change below was made before any final-test row was generated or scored.

1. **Harness bug in the independent part, found on validation.**
   - **The bug.** The `indep` rows end with the question, because their labels are in `meta`. `run_eval.py` dropped the last message of every row as if it were the reference answer, so the models received only the system prompt. Every adapter scored 0/140, with the same reply to every question.
   - **The fix.** `workflow.prompt_messages` drops the last message only when it is an assistant reply. The W1 repair turn had the same flaw and uses the same helper. A test checks that every validation prompt ends with the user's question.
   - **What was rerun.** The broken outputs and the selection computed from them are kept in `results/phase4/valid/superseded_question_not_sent/`. The `indep` part was rerun for all four adapters, and the selection was rerun with the same rule. The fix cannot change eligibility, which depends only on calendar rows.
2. **The frozen rule selected P2.** P2 is eligible by construction, since its comparison is with itself, and no other adapter's lower bound reached −5 points (`results/phase4/selection.json`). With 168 calendar rows in 120 groups, the interval is about ±7 points, so the rule could admit only an adapter that was measurably better than P2 on calendar rows. This is a power limit of the design. It is reported as such, and the rule is not changed. What follows applies section 5 as written:
   - W1 and the BaziQA contest guard (W3) run for R3 and for the selected adapter, P2. A1 gets the same runs as exploratory extras.
   - P-1, and P-3 against P2, become P2 against itself. The same calendar non-inferiority comparison is reported for every other adapter as exploratory, on the final test, which has more calendar rows.
   - The chatbot keeps R3 as its default demo model, as the brief asks, and offers P2 and A1 next to it.
3. **No A2.** The eligibility failure is a power limit, not a measured calendar loss: R3's point estimate is +0.6 points against P2. A1 replayed calendar tasks in 60% of its rows, yet its calendar point estimate moved down (−3.6), not up. The error analysis therefore points to no training change expected to clear a bound that needs a gain over P2. Training another round to clear this threshold would tune against the selection split.
