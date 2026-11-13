# BaZiReason Phase 4 report: training versus deterministic verification

Yonghao Wang · DSC 180A/B Capstone, Domain D25 · 13 November 2026
Branch `phase3/deep-reasoning`. Every earlier adapter, dataset and result is unchanged. Phase 4 writes only to `data/phase4/`, `configs/phase4/`, `results/phase4/` and a new adapter folder.

All numbers come from files under `results/phase4/`; the full tables are in [`results/phase4/RESULTS.md`](../results/phase4/RESULTS.md), built by `scripts/phase4/make_report.py`. Every model result is real greedy inference with MLX on an Apple M5 MacBook Pro (32 GB). Every score is an exact comparison with the calendar engine, the project's rule code or lunar-python; free text is never graded by similarity. Intervals are paired 95% bootstrap intervals over row groups (BaziQA: over charts). The design was frozen in advance in [`phase4_preregistration.md`](phase4_preregistration.md); its section 6 lists every later change.

## 1. Summary

| Question | Answer |
|---|---|
| Best model overall | **No adapter wins everywhere.** **A1** (new) is best at rule reasoning on untouched data: +5.4 points over round 3 on Phase 3 rule items [+1.6, +9.0] and +13.9 points on independent questions labelled by lunar-python [+8.2, +19.3]. But it now refuses 34.8% of answerable chart requests. **Round 3** is the best all-rounder when the model must compute charts itself: calendar rows −0.8 points against Phase 2 [−5.3, +3.8]. The **pre-registered rule selected Phase 2**, because on validation no rule-trained adapter could show calendar non-inferiority with only 168 rows. That says more about the rule's power than about the adapters. For the chatbot, where the engine supplies the chart, A1 with the checks is the strongest system measured. Round 3 stays the demo default, as asked |
| Calendar skills recovered? | **Round 3: yes, within error.** On the untouched 248 calendar rows: Phase 2 71.4%, round 3 70.6%, round 2 63.7% (−7.7 [−12.9, −2.6]). **A1: no**, 67.3%. Its targeted data fixed what round 3 had lost: unknown time zones 0/8 → 8/8, asks for the tool 0/30 → 29/30, solar-term boundaries 60.6% → 90.9%, definite charts for requests without one 52.9% → 11.5%. But it overshot: exact pillars without tool facts fell from 80.7% to 32.5%, mostly by answering "needs tool" (42 of its 56 false refusals) |
| Rule reasoning generalized? | **A1: yes. Earlier rounds: only to their own templates.** On wordings never used in training, A1 gets 58.8% against 49.6% for round 3. On the independent questions (new wordings, a new chart format and third-party labels), round 3 is no better than Phase 2 (57.9% vs 59.3%); A1 gets 71.8%. The gains are largest where procedures used to be confused: strength 53 → 80%, interactions 50 → 63%, flow years 46 → 64%, element counts 20 → 75% |
| Corrupted tools | **Detection solved by code; the model's own correction did not improve.** The independent check flags 150/150 corrupted tool answers and 0/30 clean ones on the final test. On benchmark_v3 it catches every hour-stem, day-rollover and confident-but-ambiguous corruption, flags none of the 17 clean cases, and misses month-boundary corruptions, which need the exact solar-term time. Told which pillar failed but not the value, the models still copy most wrong answers: round 3 corrects 21/150 (from 9), Phase 2 13/150 (from 4), A1 10/150 (no change). Training did not help (A1 10/150 vs round 3 9/150 without the check) |
| Repetition and citations | **On rule tasks, already rare after round 3.** In 1,552 replies each: loops Phase 2 17, round 3 3, A1 3; invented titles Phase 2 2, round 2 14, round 3 1, A1 0. **On the contest, A1 regressed:** with the verified chart it loops in 94 replies (round 3: 53) and names 10 titles (5 not in the rule cards). The contest guard cuts that to 38 loops, 42 unanswered (from 92) and 0 unknown titles |
| Contest (BaziQA, inspected) | **No improvement.** Best condition: Phase 2 with the contest guard, 32.9%, which is +1.0 over Phase 2 with the verified chart [−1.6, +3.5]. A1 alone: 25.5%, −5.1 against round 3 [−8.0, −2.2]. With the verified chart: 29.8%. Chance is 24.7%; human champions average 55.4% |
| Training versus tools | **Rule accuracy came from training; reliability came from code.** Training (A1 vs round 3): +5.4 reasoning, +13.9 independent. Verify-and-repair alone (round 3, W1): +2.1 [+1.2, +3.5] and +0.7, never breaking a right answer. Both: +7.4 [+3.6, +11.0] and +14.6 [+9.3, +20.0]. Every detection result is from code: corrupted tools (100% of table-visible kinds), ambiguity of the request, wrong fixed-form claims in prose (25/25 sampled flags real), loops and invented citations. Turning a detection into a right answer needed the code's value; the model given only the location of the error mostly failed |
| Strongest contribution | A **verifiable** rule-reasoning assistant: a 4B local model trained only on code-generated, rule-checked data, plus a code layer that checks every fixed-form claim. On untouched data with third-party labels, A1 answers 71.8% of rule questions; the untuned general-purpose base model of the same size, with the same prompts, answers 22.9% (+48.9 points [+42.9, +55.0]). The checker flags the 16% of replies whose prose states a wrong rule fact, including about 1 in 10 replies the structured score counts as right. No general-purpose LLM tested here, and none in the published BaziQA comparison, offers that guarantee. The project does **not** show better contest prediction than general-purpose models |

## 2. What Phase 3 left (error analysis before any Phase 4 training)

`scripts/phase4/error_analysis.py` read only saved outputs on inspected material (the Phase 2 and 3 test splits, `p2check_v1`, benchmark_v3 and BaziQA); it ran no model. Output: `results/phase4/error_analysis.json`.

| Weakness named in the brief | What the answers showed (round 3, inspected splits) |
|---|---|
| Lost calendar skills | Unknown time zones 2/28 right (Phase 2: 18/28): offsets converted from memory instead of asking for the tool. Solar-term boundaries 66/112 (Phase 2: 75), partly by claiming "seconds required" when the 节 is minutes or hours away. Verification 147 vs 154 of 184; counterfactuals 68 vs 74 of 84 |
| Poor transfer between wordings | Seen wordings 271/366 (74%), unseen 133/255 (52%) |
| Confused procedures | 9 of 14 wrong five-element counts actually ran the strength procedure |
| Corrupted tool answers | Followed 98.0% (benchmark_v3 strict); corrected 8/150 on the Phase 3 split |
| Wrong abstention | Answerable charts refused with "seconds required" near 节 |
| Loops and citations | BaziQA with tools: 53 loops, 43 of them ending with no answer; round 2 named 《穷通宝鉴》 125 times plus invented titles (《六亲定论》, 《滴天灵》…) |
| Contest | 30.3–30.6% for round 3 against 30.6–31.9% for Phase 2 |

## 3. Design

- **New data** (hashes in each folder's `manifest.json`, documented in [`data/phase4/README.md`](../data/phase4/README.md)):
  - `train_v1`: approach A training data, 3,715 rows.
  - `valid`: selection only, 686 rows.
  - `final_test`: 1,552 rows: Phase 3 rule families 788, Phase 2 calendar families 484, independent rule questions 280.
- **The independent questions:** labels from lunar-python's `EightChar`, not from the project's rule code. Seven kinds, two wordings each, written for Phase 4, and a bare chart format ("辛卯 乙未 庚辰 庚辰") that occurs in no training set.
- **Untouched test:** prompts identical to any inspected row were removed. `run_eval.py` refused to score the final test until `results/phase4/selection.json` existed, and logged every access (`final_test_access.jsonl`).
- **Candidates:** Phase 2 (`bazireason-qwen3-4b-lora-v2`), Phase 3 rounds 2 and 3, and A1. The untuned base model was added after the candidate runs, as a general-purpose baseline.
- **Workflows:**
  - **W0:** model alone.
  - **W1, verify and repair:** code recomputes every field of the final JSON it can decide. On a mismatch the model gets one more turn naming the differing fields, never their values.
  - **W2, independent tool check:** a supplied tool answer is checked with rules that do not use the tool (五鼠遁, 五虎遁, clock against hour branch, year away from 立春, a Julian-day formula for the day pillar, and the request's own ambiguity). A failed check adds one line naming the pillars, without values.
  - **W3, contest guard:** verified chart plus the rule code's structure lines. A loop is cut and the model asked for the option letter only. Titles outside the rule cards are flagged.
- **Selection rule, frozen before scoring:** eligible if, on validation calendar rows, the paired difference against Phase 2 has a 95% lower bound of at least −5 points. Among eligible adapters, the most correct validation answers wins.

**Changes after the pre-registration**, all made before any final-test row was generated:
1. **Harness bug.** The independent rows end with the question, but the runner dropped the last message of every row as if it were the reference answer, so validation sent no question (0/140 for every adapter). Fixed in `workflow.prompt_messages`, which the W1 repair turn also uses; the independent part was rerun for every adapter. The broken outputs are kept in `results/phase4/valid/superseded_question_not_sent/`.
2. **Selection outcome.** The rule selected Phase 2 (section 6). As written in the plan, W1 and W3 then also ran for Phase 2, and A1 kept them as exploratory runs.
3. **No A2.** A second approach-A round was allowed, but the data gave no reason for one (pre-registration, section 6).

**Changes after final-test results were seen**, none of which touches a model, prompt or frozen threshold:
- A W2 runner bug wrote the plain v3 prompt version into each record, while the manifest named the W2 prompt. The scorer refused the file; the field was corrected in the three saved files, with the outputs unchanged and the correction noted in each manifest.
- The untuned base model was added as a baseline.
- The claim checker's two false-alarm classes were fixed on validation replies before section 10 was computed. A 14-flag sample of final-test output had pointed to them.

## 4. Approach A: what A1 was trained on, and the run

The data follow the error analysis (details in `data/phase4/README.md`):
- 2,245 rows (60%) replay Phase 2 calendar families, weighted toward those that regressed: `tz_needs_tool` 120 with `tz_known` 100 as contrast, `jie_boundary` 220, `jie_window` 120, `verify` 220, `counterfactual` 160, `internal` 100, and 300 rule rows including `rule_elements` 50 and `rule_strength` 40.
- 1,470 rows are Phase 3 rule families.
- Every rule answer starts with a procedure header (`【规则】五行计数：…`) so the model names the procedure before applying it.
- 337 rule questions are rewritten: other pillar formats, other lead-ins, 日元 for 日主.
- No answer names a book.

| Run | Starts from | Rows | Iterations (updates) | Minutes | Validation loss | Peak memory |
|---|---|---:|---:|---:|---|---:|
| `bazireason-qwen3-4b-lora-p4-a1` | Phase 3 round 3 | 3,715 | 1,857 (928) | 115 | 0.091 → 0.0080 | 21.0 GB |

LoRA rank 16, scale 2, last 16 layers, peak learning rate 5e-5. Configuration: `configs/phase4/qwen3_4b_lora_p4_a1.yaml`. Adapter SHA-256 `89311020…0d91`.

## 5. Approach B: what was built

- **`src/bazireason/workflow.py`:**
  - W1: `check_reply`, `repair_turn`.
  - W2: `tool_check`. It uses a day formula and the hour, month and year rules without any calendar library, at the fixed UTC+8 clock; it does not convert through Asia/Shanghai, whose 1986–91 daylight time would give false alarms.
  - W3: `cut_at_loop`, `unknown_titles`.
- **`src/bazireason/claim_check.py`:** rechecks every fixed-form statement in free text against the engine and the rule code: pillars, day master, flow-year pillar, Ten Gods, branch and stem relations, hidden stems, element counts, 旬空, life stages, stars (day or year basis), the strength label, a named pattern, the spouse star against the gender entered, citations outside the rule cards, definite fortune claims, loops.
- **Chatbot** (`bazireason.demo_app`): every reply now has a "逐条核对（规则代码，不经模型）" list showing each claim, whether the code agrees and the code's value. The model menu offers round 3 (default), A1, Phase 2 and the untuned model.
  - **Race fixed while checking the page in a browser.** A question sent while a new chart was still being computed went out with the previous chart, while the page then displayed the new one. Sending now waits for the chart in progress.
  - **Checks confirmed on a made-up test birth.** A1 wrote "甲辛相合" (the stem pair that combines is 丙辛); the list marked it "2/4 条一致" and showed the code's value.

## 6. Selection on the validation split

| Adapter | Phase 3 part | Phase 2 part | Independent | Total | Calendar rows minus Phase 2 [95%] | Eligible |
|---|---:|---:|---:|---:|---|---|
| Phase 2 **(selected)** | 48/240 | 206/306 | 82/140 | 336 | 0 (itself) | yes |
| Round 2 | 184/240 | 191/306 | 78/140 | 453 | −9.5 [−15.2, −3.9] | no |
| Round 3 | 185/240 | 214/306 | 80/140 | 479 | +0.6 [−6.6, +7.8] | no |
| A1 | 193/240 | 211/306 | 98/140 | **502** | −3.6 [−11.8, +5.1] | no |

With 168 calendar rows in 120 groups, the interval is about ±7 points, so the rule could only admit an adapter measurably better than Phase 2 on the calendar. The rule was not changed. Its outcome is reported as the pre-registered selection, and every adapter was scored on the final test, which has more calendar rows.

## 7. The untouched final test (model alone)

| Measure | Untuned base | Phase 2 | Round 2 | Round 3 | A1 |
|---|---:|---:|---:|---:|---:|
| **Phase 3 rule reasoning** (608) | 6.2% | 14.5% | 63.3% | 62.5% | **67.9%** |
| … wording never seen in training (274) | 3.6% | 10.6% | 49.3% | 49.6% | **58.8%** |
| **Independent rule questions** (280) | 22.9% | 59.3% | 56.8% | 57.9% | **71.8%** |
| **Calendar rows** (248) | 1.6% | **71.4%** | 63.7% | 70.6% | 67.3% |
| Exact four pillars, no tool facts (83) | 0.0% | 74.7% | 61.4% | **80.7%** | 32.5% |
| Answerable chart refused (lower is better) | 0.6%* | 11.2% | 14.9% | **3.7%** | 34.8% |
| Definite chart where none exists (lower is better) | 0.0%* | 42.5% | 39.1% | 52.9% | **11.5%** |
| Ambiguity, rule questions: right decision (54) | 16.7% | 68.5% | 87.0% | 92.6% | 92.6% |

\* **About the untuned base model:**
- **Its low refusal and over-answering rates are not good behaviour.** It gives a parseable conclusion in only 77 of 484 calendar replies, so it rarely commits to any status at all.
- **It was added after the other runs**, as the general-purpose baseline: the same 4B instruct model, with the same prompts and no adapter.
- **On the independent questions,** A1 minus base is +48.9 points [+42.9, +55.0].

Pre-registered primary comparisons, for the selected adapter (Phase 2):
- **P-1:** calendar against Phase 2 is 0 by construction.
- **P-2:** rule reasoning against round 3 is −48.0 points [−52.6, −43.8].
- **P-3:** independent questions against round 3 are +1.4 [−5.0, +7.5]; against Phase 2, 0 by construction.

The comparisons that carry information are exploratory:

| Paired difference [95%] | Calendar rows | Rule reasoning | Independent |
|---|---|---|---|
| A1 minus round 3 | −3.2 [−10.8, +4.1] | **+5.4 [+1.6, +9.0]** | **+13.9 [+8.2, +19.3]** |
| A1 minus Phase 2 | −4.0 [−11.1, +2.9] | +53.5 [+49.0, +57.9] | **+12.5 [+6.1, +18.6]** |
| Round 3 minus Phase 2 | −0.8 [−5.3, +3.8] | +48.0 [+43.8, +52.5] | −1.4 [−7.5, +5.0] |
| Round 2 minus Phase 2 | **−7.7 [−12.9, −2.6]** | +48.8 [+44.4, +53.3] | −2.5 [−8.9, +3.9] |

By independent-question kind (40 each), A1 against round 3:
- element count 75% vs 20%;
- luck direction 95% vs 83%;
- Ten God of a branch 68% vs 58%;
- Ten God of a stem 73% vs 63%;
- hidden stems 100% vs 98%;
- twelve stages 85% vs 85%;
- 旬空 8% vs 0%.

Phase 2 gets 20% on 旬空 and 68% on element counts. The 旬空 kind is near zero for every Phase 3-based adapter: its new wording names the day pillar and the whole chart, and the models answer with the wrong decade.

## 8. Training versus tools

| Setting | Rule reasoning | Independent | Calendar rows |
|---|---|---|---|
| Round 3 alone (W0) | 62.5% | 57.9% | 70.6% |
| Training only: A1 alone | 67.9%; +5.4 [+1.6, +9.0] | 71.8%; +13.9 [+8.2, +19.3] | 67.3%; −3.2 [−10.8, +4.1] |
| Tools only: round 3 + W1 | 64.6%; +2.1 [+1.2, +3.5] | 58.6%; +0.7 [0.0, +1.8] | 70.6%; 0 |
| Both: A1 + W1 | 69.9%; +7.4 [+3.6, +11.0] | 72.5%; +14.6 [+9.3, +20.0] | 68.5%; −2.0 [−9.6, +5.3] |

- **W1 never broke a right answer.** Across the three adapters it re-asked 1,581 rows and fixed 42. With only the names of the wrong fields, the model rarely finds the right values.
- **The engine as a ceiling.** Showing the engine's value for every field the code decides (what the chatbot does for the chart and the structure panel) is correct by construction. That is a system property, not a model result.

## 9. Corrupted tool answers

**Final test, `strict_shift` (150 corrupted, 30 clean, one-pass JSON):**

| Adapter | Check flags (corrupted / clean) | Corrected, W0 → W2 | Followed the error, W0 → W2 |
|---|---|---|---|
| Phase 2 | 150/150 / 0/30 | 4 → 13 | 146 → 135 |
| Round 3 | 150/150 / 0/30 | 9 → 21 | 137 → 125 |
| A1 | 150/150 / 0/30 | 10 → 10 | 136 → 131 |

**benchmark_v3 `tool_corrupted` (inspected, 244 corrupted + 17 clean):**
- **The check:** flags hour_stem 81/81, day_rollover 77/77 and default_guess 19/19, but jie_side only 1/67; clean 0/17.
- **The models with W2:** they still follow 100% of the flagged hour-stem and day-rollover corruptions. Only the default-guess cases change: the model declines once told the request is ambiguous (0% followed for Phase 2 and round 3, 26% for A1).
- **Overall following, W0 → W2:** Phase 2 98.8% → 92.2%, round 3 98.0% → 91.4%, A1 98.8% → 94.3%.

Detection is a solved problem for every corruption visible from the tables. Correction in a single strict-JSON pass is not something these 4B models do from a value-free hint. The reliable remedy is to let the code supply the value.

## 10. Loops, citations and the claim checker

| Final test, 1,552 replies each | Untuned base | Phase 2 | Round 2 | Round 3 | A1 |
|---|---:|---:|---:|---:|---:|
| Replies with a loop | 9 | 17 | 2 | 3 | 3 |
| … of them without an answer | 7 | 13 | 2 | 1 | 2 |
| Name a title | 162 | 2 | 14 | 1 | 0 |
| Title not in the rule cards | 10 | 2 | 0 | 0 | 0 |

| BaziQA with the verified chart, 1,665 replies | Phase 2 | Round 3 | A1 | Phase 2 + W3 | Round 3 + W3 | A1 + W3 |
|---|---:|---:|---:|---:|---:|---:|
| Replies with a loop | 30 | 53 | 94 | 22 | 15 | 38 |
| Unanswered | 17 | 53 | 92 | 20 | 45 | 42 |
| Titles named / not in the cards | 0 / 0 | 1 / 0 | 10 / 5 | 1 / 0 | 2 / 0 | 0 / 0 |

**Claim checker on the final-test prose** (330 free-text rule replies per adapter: Ten Gods, strength, pattern, 用神, relations, flow years, areas of life, overviews, questions without a unique answer):

| Adapter | Claims checked | Replies with a wrong claim | … scored right by the JSON | Scored-wrong replies flagged |
|---|---:|---:|---:|---:|
| Untuned base | 890 | 97 (29.4%) | 0 | 97/329 |
| Phase 2 | 677 | 138 (41.8%) | 3 | 135/315 |
| Round 2 | 1,021 | 62 (18.8%) | 14 | 48/154 |
| Round 3 | 1,005 | 54 (16.4%) | 22 | 32/152 |
| A1 | 1,108 | 53 (16.1%) | 21 | 32/114 |

- **Precision.** A random sample of 25 final-test flags was read one by one. The reader was the AI assistant, not a domain expert. All 25 are departures from the project's declared tables: wrong relations, stars, Ten Gods, patterns, strength labels and hidden stems.
- **Conventions.** Three of the 25 depend on a convention that some schools state differently: 申辰 as 半合, 阳刃 for yin stems, and 月劫 vs 阳刃 naming.
- **Hidden errors.** About 10% of the replies that the structured score counts as right (22 of 228 for round 3, 21 of 216 for A1) state a wrong rule fact in their prose. The chatbot shows these, and an exact-match score alone would miss them.

## 11. BaziQA (inspected; 555 questions, three sampled runs, chance 24.7%)

| Condition | Phase 2 | Round 3 | A1 |
|---|---:|---:|---:|
| No tools | 30.6% | 30.6% | 25.5% |
| Verified chart | 31.9% | 30.3% | 29.8% |
| Verified chart + contest guard (W3) | **32.9%** | 28.5% | 28.9% |

Paired differences (chart-cluster bootstrap):
- **A1 against round 3:** no tools −5.1 [−8.0, −2.2]; with the chart −0.5 [−3.8, +3.0].
- **W3:** on Phase 2 +1.0 [−1.6, +3.5], on round 3 −1.7 [−4.8, +1.3], on A1 −0.9 [−4.1, +2.3].

The untuned base model gets 25.6% / 31.2%. Five hosted frontier models in the BaziQA paper got 32–37% on an earlier data version with a different protocol. That is context, not a like-for-like comparison. Contest keys encode practitioners' whole-chart judgments, and agreement with them is not evidence that destiny can be predicted.

## 12. Limitations

- **Selection power.** The frozen calendar constraint could not distinguish adapters within ±7 points on 168 rows. A future rule should size the calendar set for its margin, or use the larger final split's design.
- **Label independence is partial.** lunar-python and the project's rule code follow the same conventions, and the tables agree entry by entry. The independent questions test new wordings, a new chart format and a separate implementation, not a different school.
- **W1 on generated rows.** There the checker and the labels come from the same rule code. On the independent questions the check uses `bazireason.rules` and the labels come from lunar-python.
- **Who checked the checker.** The claim-checker precision rests on 25 flags read by the AI assistant against the project's own tables; no domain expert reviewed them.
- **Inspected benchmarks.** benchmark_v3 and BaziQA were inspected in earlier phases; nothing was tuned on their keys.
- **Scope.** One base model (Qwen3-4B-Instruct-2507, 4-bit) and one seed per adapter.

## 13. Reproduce

```bash
.venv/bin/python scripts/phase4/error_analysis.py           # saved outputs only
.venv/bin/python scripts/phase4/build_splits.py --check     # valid and final_test, byte for byte
.venv/bin/python scripts/phase4/build_dataset.py --check    # p4_train_v1
nohup caffeinate -i .venv-mlx/bin/python scripts/phase2/train_lora.py \
  --config configs/phase4/qwen3_4b_lora_p4_a1.yaml > artifacts/logs/train_p4_a1.log 2>&1 &
scripts/phase4/queue_valid.sh       # validation W0 for every adapter, then select.py
scripts/phase4/queue_final.sh       # final test W0/W1/W2, benchmark_v3, BaziQA W3
.venv-mlx/bin/python scripts/phase4/run_eval.py --label base --split final_test   # baseline
.venv/bin/python scripts/phase4/score_baziqa4.py
.venv/bin/python scripts/phase4/make_report.py               # RESULTS.md and summary.json
```

No runner overwrites an existing output. `tests/test_phase4.py` checks split hashes, regeneration, leakage, label agreement, the seal, the prompts and the training headers.
