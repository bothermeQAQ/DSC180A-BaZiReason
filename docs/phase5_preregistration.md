# Phase 5 pre-registration

Written on 13 November 2026, before any Phase 5 system was scored on `data/phase5/valid` or `data/phase5/sealed`. The commit that adds this file is the time stamp; later changes are listed in section 8 with their reasons. Plan and background: [`phase5_execution_prompt.md`](phase5_execution_prompt.md). Working log: [`phase5_log.md`](phase5_log.md).

## 1. Question

Does an engine-first system beat every fine-tuned model used alone, on accuracy and on faithfulness to verified facts? In this system the code computes every fact it can, the model explains, and the code checks the reply. Phase 5 also asks:
- what fine-tuning still adds once the facts are supplied;
- whether a merged adapter can hold both the calendar and the rule skills.

## 2. Data status

| Material | Status in Phase 5 |
|---|---|
| Every Phase 2–4 split, including `data/phase4/final_test`, plus benchmark_v3 and BaziQA | **development** (read freely) |
| `data/phase5/blind/dev.jsonl`, `data/phase5/convo/dev.jsonl` | development |
| `data/phase5/valid/*`, `data/phase5/blind/valid.jsonl`, `data/phase5/convo/valid.jsonl` | **selection only** |
| `data/phase5/sealed/*`, `data/phase5/blind/sealed.jsonl`, `data/phase5/convo/sealed.jsonl` | **sealed**: scored once, after `results/phase5/selection.json` exists; every access is logged in `results/phase5/sealed_access.jsonl` |

**Sealed parts:**
- `rules`: 411 rows, Phase 3 rule families with an exact oracle plan.
- `indep`: 280 rows, lunar-python labels.
- `blind`: 314 rows, questions written by an untuned Qwen3.5-4B from plain descriptions.
- `calendar`: 818 rows, Phase 2 chart requests.
- `convo`: 60 five-turn dialogues.

The sealed split uses the test-year block and new seeds, and excludes every prompt of any earlier split. Hashes are in each manifest.

## 3. Systems and conditions

The base model is always Qwen3-4B-Instruct-2507 (4-bit, MLX), served with adapters swapped in place on one loaded base (`bazireason.adapter_bank`). Greedy decoding throughout. Validation and sealed runs use batch 32; development runs used 16.

- **Adapters:**
  - `base`, untuned;
  - `p2`, `p3r3` and `p4a1`, the existing adapters;
  - **M\***, an interpolation `(1 − α)·R3 + α·A1`, with α chosen on development data (section 4);
  - **G1** (`p5g1`), the one Phase 5 training run (section 7).
- **Conditions** (`bazireason.grounded`, prompt version 2):
  - **w0, model alone:** the four pillars and the question.
  - **gf, full fact sheet:** every verified fact of the chart.
  - **go, oracle plan:** the fact sheet, the question, then the rule code's conclusion for the true intent with an instruction to adopt it.
  - **gr, routed:** the same with the intent chosen by the frozen keyword router `bazireason.intent.route`, which was tuned on development data only.
  - **grv, routed and verified:** gr; if the reply's conclusion or any fixed-form statement disagrees with the code, one repair turn that gives the code's values; if the second reply still disagrees, the engine's own derivation is shown.
- **The system** is grv with generator G\*.
- **Prompt version 2.** It was chosen on development data: with A1, oracle-routed agreement on rule rows rose from 197 to 203 of 294 and contradicted statements fell from 11 to 4 (`docs/phase5_log.md`).

## 4. Selection on the validation split (frozen)

1. **M\*:** the α in {0.15, 0.25, 0.35} with the most correct development answers (model alone: Phase 4 final-test calendar rows, rule rows and independent questions, plus blind dev questions). Ties go to the smaller α.
2. **M\* is a standalone candidate only if,** on validation (w0), all three hold:
   - (a) the 95% lower bound of calendar rows minus P2 is at least −7 points. The validation set has 388 rows, so ±5 cannot be resolved there.
   - (b) M\*'s correct rules + independent + blind answers are within 3 points of A1's.
   - (c) M\*'s balanced decision accuracy on calendar rows is above that of P2, R3 and A1. The balanced accuracy is the mean of the right-status rate on answerable rows and on rows without an answer.

   If M\* fails, it is reported on validation only and P-3 is not tested.
3. **G\*:** among {base, R3, A1, G1, and M\* if a candidate}, the generator with the most first-pass correct answers under gr on validation (rules + indep + blind). Ties are broken by fewer replies with a contradicted statement, then by A1.
   - **Conversation guard:** a candidate whose grv dialogue success on the validation conversations is more than 3 dialogues (of 30) below A1's is excluded.

`scripts/phase5/select.py` applies these rules and writes `results/phase5/selection.json`.

## 5. Primary comparisons (sealed split; paired, row-group bootstrap 95% intervals)

- **P-1, system versus the best model alone.** Final correct answers of the system (grv, G\*) minus A1 alone (w0), on sealed rules + indep + blind (1,005 items).
  - **Expected:** a positive difference.
  - **Success:** the lower bound is above 0.
- **P-2, what training adds once facts are supplied.** First-pass correct answers under gr: G\* minus base, on the same items. If G\* is base, the comparison is A1 minus base.
  - Reported with its interval, whatever the sign.
  - If G\* is G1, G1 minus A1 under gr is reported as well (what the Phase 5 training added).
- **P-3, merged adapter, calendar non-inferiority** (only if M\* is a candidate). M\* minus P2 on sealed calendar rows (w0, 818 rows), with a margin of −5 points.
  - **Success:** the lower bound is at least −5.
  - **Power:** with about 25% discordant pairs and a design effect of 1.2, the half-width is about 3.8 points. If the true difference is 0, power is about 0.75. This is stated in advance.

**Sealed runs (once):**
- **w0:** A1 on rules, indep and blind; P2, R3, A1, M\* (if a candidate) and G1 on calendar.
- **gr and grv:** G\* and base, plus A1 if G\* is not A1.
- **go:** G\* (it differs from gr only on misrouted items).
- **Conversations:** G\* under w0 and grv, and A1 under grv.

## 6. Gates and secondary measures (exploratory, no multiplicity correction)

**Gates** (pass or fail is part of the result):
- Router intent accuracy on the sealed blind questions is at least 90%.
- No computable question is routed to the prediction refusal.
- Of the system's final displayed replies, fewer than 3% contain a contradicted fixed-form statement. The first-pass rate before repair is reported next to it.
- The median interactive reply time of the system is at most twice that of the model alone (`scripts/phase5/latency.py`).

**Secondary measures:**
- **Routing and grounding:** gf, go and gr for G\*. These separate the contribution of the facts, of routing and of the router's errors.
- **The 2 × 3 table:** {base, R3, A1} × {w0, gr} on validation.
- **Conversations:** dialogue success (all five turns pass, no contradiction) for w0 and grv, plus per-turn success.
- **Model alone:** calendar accuracy and balanced decision accuracy for P2, R3, A1 and M\*; false refusals; definite answers where none exists.
- **Breakdowns:** by intent; blind versus template wording.
- **The checker's own false-alarm rate,** from a read sample of flags on validation replies.
- **Costs:** latency, memory and swap cost.
- **BaziQA:** not used. It is near chance and inspected; Phase 4 holds the record.

## 7. Training

Development data showed a specific failure that prompting did not fix. A1 disagrees with a verified conclusion supplied in its prompt on 31–33% of rule questions (97 of 294 with prompt version 1, 91 with version 2), and the untuned base does not adopt it either. The value-supplied repair fixes the final answer, but costs a second generation for about a third of questions.

One training run, **G1**, was therefore added before any validation scoring:
- **Start:** continued from A1, for time and to keep A1's knowledge; a joint run from the base was not attempted.
- **Data:** `p5_train_v1`, 2,260 rows, all written by the rule code:
  - 700 routed grounded answers and 400 full-sheet answers;
  - 260 two-turn follow-ups that keep the verified answer;
  - 100 declined predictions;
  - 800 replayed `p4_train_v1` rows.

  Wordings are training-only.
- **Configuration:** `configs/phase5/qwen3_4b_lora_p5_g1.yaml`.
- **Selection:** G1 enters the same rules as every other adapter.
- **Expected effect:** fewer first-pass disagreements and repair turns. It is not expected to change final accuracy, which the repair already secures.

## 8. Changes after this commit

1. **13 November 2026, 23:30: GPU memory budget for long batches (harness only; no selection rule changed).**
   - **What happened:** during validation, the untuned base ran out of GPU memory at batch 32 on its repair turns. Its first replies run to the 800-token cap, so 32 repair prompts of about 2,000–2,450 tokens each, plus 800 new tokens, did not fit. Every other run had finished.
   - **Change:** a chunk of a batch now holds at most 32 conversations and at most 80,000 padded tokens (chunk size × (longest prompt + max_tokens); `run5.TOKEN_BUDGET`, `MLXChatModel.generate_batch(token_budget=…)`). Finished chunks are saved to the reply cache at once.
   - **Effect:** ordinary first-pass chunks are unchanged or split slightly smaller; only long prompts (repairs, late conversation turns) run in smaller chunks. Decoding stays greedy; batch composition can change only last-digit arithmetic.
   - **Scope:** the base's validation gr/grv runs were rerun with it, reusing its stored first-pass replies. Every validation run after 23:22 and every sealed run use it.
