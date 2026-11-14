# BaZiReason Phase 5 report: an engine-first grounded system

Yonghao Wang · DSC 180A/B Capstone, Domain D25 · 14 November 2026
Branch `phase5/grounded-system`. Every earlier adapter, dataset and result is unchanged. Phase 5 writes only to `data/phase5/`, `configs/phase5/`, `results/phase5/`, `scripts/phase5/`, new modules in `src/bazireason/` and one new adapter folder.

All numbers come from files under `results/phase5/`. The full tables are in [`results/phase5/RESULTS.md`](../results/phase5/RESULTS.md), built by `scripts/phase5/make_report5.py` from saved records, which run no model. Every model result is real greedy inference with MLX on an Apple M5 MacBook Pro (32 GB). Every score is an exact comparison with the rule code, the calendar engine or lunar-python; free text is never graded by similarity. Intervals are paired 95% bootstrap intervals over row groups. The design was frozen in [`phase5_preregistration.md`](phase5_preregistration.md) before any validation or sealed scoring; its section 8 lists the one later change. The working log is [`phase5_log.md`](phase5_log.md).

## 1. Summary

| Question | Answer |
|---|---|
| Does the engine-first system beat every model alone? | **Yes, by a wide margin.** On the 1,005 sealed rule questions (template rules 411, independent 280, blind wording 314), the system (routed, verified, generator G1) ends with **98.6%** correct, against **68.5%** for A1 alone, the best model alone in Phase 4: **+30.1 points [+27.2, +33.1]** (P-1). Its first answer, before any repair, is already +22.9 [+19.7, +26.1] |
| Faithfulness | **Replies with a contradicted rule fact: A1 alone 18.1%, the system 0.0%** (0 of 1,005 final replies; 2 of 1,005 first answers). The target was under 3%. The claim checker's flags were read on a random sample of 30: 29 real errors, 1 borderline, 0 false alarms (upper 95% bound about 10%) |
| What does fine-tuning add once the facts are supplied? | **A lot.** With the same routed prompt, G1 minus the untuned base is **+52.1 [+49.1, +55.1]** (P-2). The base cannot use a supplied conclusion: 6.6% on template rules, mostly because it writes free-form output (84% of its replies are well-formed, and only 46.6% of those are right). The Phase 4 training already gave +41.7 (A1 minus base); the Phase 5 run G1 adds **+10.4 [+8.1, +12.8]** over A1 (P-2b) |
| What does verification add? | It turns the remaining disagreements into right answers: G1 +7.3 [+5.8, +9.0], A1 +17.7, base +59.3. With verification, every generator ends at 98.5–98.6%: **the final accuracy no longer depends on the adapter**. What the adapter changes is how often a second generation is needed (repairs: base 603, A1 186, G1 73 of 1,005) |
| Merged adapter (R3↔A1) | **Not a candidate.** The best development point was α = 0.15. On validation it was the best model alone on calendar rows (+4.6 [+0.8, +8.7] over Phase 2) and in balanced decision accuracy. But it lost **−5.6 [−9.6, −1.6]** on rule questions against A1, beyond the frozen 3-point limit. P-3 was therefore not tested |
| Adapter routing | **Negligible.** A per-intent table chosen on dev, simulated from stored replies: +5 answers of 498 over A1 alone (validation); +8 of 1,005 over G1 under the routed system (sealed). Not adopted |
| Conversations (five turns) | **G1 holds a dialogue at first pass; A1 needs the repairs.** Sealed, 60 dialogues: system with G1 59/60 (first pass 50/60); with A1 59/60 (first pass 23/60); G1 alone 8/60. The one failed dialogue is a router error |
| Calendar (model alone, 818 sealed rows) | **Every rule-trained adapter is still below Phase 2.** Phase 2 72.9%; R3 −1.3 [−3.7, +1.0]; G1 −3.2 [−6.7, +0.4]; A1 −5.9 [−10.0, −1.8]. G1 refuses fewer answerable requests than A1 (125 vs 198 of 532). In the chatbot the engine computes the chart, so this skill is not used there |
| Router | **96.2%** intent accuracy on the sealed blind questions (target ≥ 90%); no computable question was sent to the prediction refusal. **All 14 remaining errors of the system are routing errors** |
| Cost | Median interactive reply **3.28 s for the system against 3.24 s for A1 alone** (40 validation questions, one at a time). Prompts are 3.4× longer (1,272 vs 369 tokens); peak memory 3.61 vs 3.33 GB. One base serves every adapter; a swap takes about 4 ms |
| What did not improve | Declining: the system declines fewer of the questions that should be declined than A1 alone (56 vs 64 of 65), because the router misses some prediction phrasings. G1 alone is slightly weaker than A1 alone on rule questions (−1.2 points on validation): it has learnt to lean on the supplied conclusion. In the chatbot, G1 answers "why" and "say it more simply" by restating the verified conclusion: the answer is kept, but there is little new explanation |

## 2. What was built

| Work package | What exists now |
|---|---|
| WP0 record | `results/phase5/addenda/indep_rescore.json`: the Phase 4 independent questions rescored with strict and content-lenient 旬空 scoring. A1 201 → 212 of 280, R3 162 → 169; no ranking changes. The Phase 4 final test is now development material |
| WP1 evaluation suite v2 | `data/phase5/` (see its README): validation and sealed splits in their own year blocks; blind questions written by an untuned model; five-turn dialogues; a calendar set sized for its margin. Hashes in manifests, checked by tests |
| WP2 one base, many adapters | `bazireason.adapter_bank.AdapterBank`: one loaded Qwen3-4B; adapters swapped in place with `load_weights`; "base" sets every `lora_b` to zero; `mix:<a>:<b>:<α>` interpolates. Swapped replies equal fresh loads for all six adapters (`results/phase5/bank_check.json`). The chatbot's `Models` class runs on it |
| WP3 cheap experiments | Interpolation grid on dev (`dev_alpha.json`) and the merged adapter's gates on validation; adapter routing simulated from stored replies (`addenda/extras.json`) |
| WP4 grounded system | `bazireason.facts` (the fact sheet: every verified fact of a chart, labelled 确定 / 约定 / 分流派); `bazireason.intent` (a keyword router with 20 intent classes and `plan_for`, which builds the rule code's conclusion per intent); `bazireason.grounded` (prompts for w0, gf, go, gr; verification of the conclusion and of every fixed-form statement; a repair turn that gives the code's values; the engine's own derivation as the fallback; follow-up turns) |
| WP5 training | One run, G1, justified by a measured failure (section 4) |
| WP6 BaziQA | Not run, by the pre-registration: near chance and inspected; Phase 4 holds the record |
| WP7 chatbot | Grounded mode by default (router, fact sheet, verification with repair and fallback, labels per question type), G1 the default generator, model-alone mode kept for comparison (section 10) |
| WP8 | Sealed evaluation, `RESULTS.md`, `summary.json`, this report, tests |

## 3. Design

- **Data** (`data/phase5/README.md`):

  | Split | Rules | Independent | Blind | Calendar | Dialogues | Role |
  |---|---:|---:|---:|---:|---:|---|
  | dev | 294 | 280 | 195 | 248 | 20 | the Phase 4 final test plus new blind and dialogue sets; read freely |
  | valid | 202 | 140 | 156 | 388 | 30 | selection only |
  | sealed | 411 | 280 | 314 | 818 | 60 | scored once, after `selection.json` |

  - **Rules:** Phase 3 rule families whose rule-code plan reproduces the stored label exactly.
  - **Independent:** rule questions labelled by lunar-python, in wordings never used in training.
  - **Blind:** questions written by an untuned Qwen3.5-4B from plain descriptions of what a user wants to know; it never saw a project template. Filters were fixed before generation. MiniCPM5-2B, the other family available, produced unusable questions, so the writer is a newer Qwen generation, not a different family.
  - **Dialogues:** a blind question, then "why" or "simpler", a pushback with a wrong value, a prediction request and a different computable question; each turn has a check the code decides.
  - **Your own questions:** `data/phase5/blind_user/` explains the format; Phase 5 ran without them.
- **Conditions** (prompt version 2, chosen on dev):
  - **w0:** model alone.
  - **gf:** the full fact sheet.
  - **go:** the fact sheet plus the rule code's conclusion for the true intent, with "adopt this, do not re-derive".
  - **gr:** the same with the routed intent.
  - **grv:** gr, then the code checks the conclusion and every fixed-form statement. On a disagreement the model gets one repair turn with the code's values; if it still disagrees, the engine's derivation is shown.
- **Systems:** base, P2, R3, A1, M\* (interpolated), G1. The system is grv with the generator G\* chosen on validation.
- **Change after the pre-registration** (section 8): the untuned base ran out of GPU memory on its validation repair turns at batch 32; chunks are now capped at 80,000 padded tokens. No selection rule changed. After the sealed results, one gate computation was corrected: three template questions "能不能长寿？" have no rule answer (`ambig:no_rule`), so they are not computable questions; the router sent them to the prediction refusal. They are listed next to the gate and stay scored wrong.

## 4. Development findings, and why G1 was trained

On dev (A1, prompt version 1 unless noted):

| Part | Alone (w0) | Full sheet (gf) | Oracle-routed (go) | Routed + verified, final |
|---|---:|---:|---:|---:|
| Template rules (294) | 191 | 165 | 197 | 294 |
| Independent (280) | 212 | 219 | 243 | 280 |
| Blind (195) | 108 | 122 | 159 | 190 |

- **The full fact sheet alone does not help.** A1 ignores it, re-derives from its memorised tables and errs differently in the long context (template rules 191 → 165).
- **The routed conclusion helps most on blind questions** (108 → 159), and verification secures the final answer. These counts are after the list-scoring fix (`rescore5.py`), which moved one blind item per file; the log quotes them before it.
- **The remaining failure is compliance.** Even with the conclusion supplied and prompt version 2 ("请直接采用上面这个已核验的结论作答"), A1 disagrees with it on 91 of 294 template rule rows. The untuned base adopts it even less (12/294 under go, mostly free-form output).
- **G1** was trained for this one failure, before any validation scoring:
  - continued from A1 on `p5_train_v1`: 2,260 code-written rows (700 routed answers, 400 full-sheet answers, 260 follow-ups that keep the verified answer, 100 declined predictions, 800 replayed Phase 4 rows);
  - 1,130 iterations in 92 minutes, peak memory 23.6 GB;
  - no blind question, independent template or dialogue phrasing in the data (checked by a test).
- **G1 on dev:** disagreement with the supplied conclusion fell from 91 to 38 of 294 rule rows and from 20 to 0 of 280 independent questions. Model alone it lost 15 template and 13 independent rows.
- **Interpolation (dev, w0):** α = 0.15 had 639 correct answers, against 637 for 0.25 and 626 for 0.35. The 0.25 point the review asked about is not better than its neighbours.

## 5. Selection on validation (`results/phase5/selection.json`)

| | Base | R3 | A1 | G1 | M\* (0.15) |
|---|---:|---:|---:|---:|---:|
| First pass correct, routed (of 498) | 203 | 386 | 408 | **461** | 387 |
| Replies with a contradicted statement | 23 | 9 | 9 | 2 | 8 |
| Final correct, routed + verified | 489 | 491 | 491 | 491 | 491 |
| Dialogues fully right / at first pass (of 30) | 29 / 1 | 29 / 11 | 29 / 10 | 29 / 25 | 29 / 12 |
| Calendar alone (of 388; P2 254) | | 253 | 249 | 264 | 272 |
| Balanced decision accuracy (P2 69.4%) | | 67.4% | 71.5% | 72.7% | 72.7% |

- **M\*** passed gates (a) and (c) but failed (b): 319 rule answers alone against A1's 347. It is not a candidate.
- **G\* = G1**, by first-pass correct answers. The conversation guard excluded no one.
- **G1 against the adapter checks the plan asks of any new adapter** (frozen for M\* only, so applied to G1 afterwards; they did not affect selection): on validation it passes all three (calendar +2.6 [−2.2, +7.6] over P2; −1.2 points against A1 alone; balanced decision accuracy 72.68% against 71.48%). On sealed, its calendar lower bound −6.7 is inside the −7 margin but not the −5 one, and its balanced decision accuracy (73.39%) is below Phase 2's (74.46%).

## 6. Sealed results

### Primary comparisons (pre-registered)

| Comparison | Difference, points [95%] | Outcome |
|---|---|---|
| P-1 system minus A1 alone | +30.1 [+27.2, +33.1] | success (lower bound > 0) |
| P-2 G1 minus base, both routed | +52.1 [+49.1, +55.1] | reported |
| P-2b G1 minus A1, both routed | +10.4 [+8.1, +12.8] | reported |
| P-3 merged adapter on calendar | not tested | M\* was not a candidate |

### By part

| System | Template rules (411) | Independent (280) | Blind (314) |
|---|---:|---:|---:|
| A1 alone | 66.2% | 75.7% | 65.0% |
| Base, routed | 6.6% | 83.9% | 42.0% |
| A1, routed | 69.6% | 92.1% | 85.7% |
| G1, routed (first pass) | 86.4% | 100.0% | 90.1% |
| G1, oracle-routed | 87.1% | 100.0% | 93.6% |
| **System final** (any generator) | 99.3% | 100.0% | 96.5% (base 96.2%) |

### By intent (A1 alone → G1 first pass → system final)

- **Large gains:** domain 24% → 77% → 98%; overview 43% → 100% → 100%; interactions 54% → 74% → 100%; flow years 48% → 63% → 100%; single facts 74% → 99% → 99%.
- **Weakest first pass:** flow years (62.8%) and interactions (73.6%): G1 still re-derives these lists, and verification fixes them.
- **Regressions:** predictions 100% → 69% → 69% and ambiguous questions 98% → 92% → 92%. A1 alone declines every prediction; the system misses five because the router sends "什么时候容易生病", "哪一年运势会好转" and similar to `unknown` (4) or to the wealth domain (1). In three of the four `unknown` cases the prose still declines, but the conclusion field is not the decline label.

### Remaining errors of the system

All 14 wrong final answers have a wrong intent:
- 5 prediction requests not recognised;
- 3 "能不能长寿" questions sent to the prediction refusal instead of "no rule answer";
- 3 "日主X生在月令Y" life-stage questions routed to the day pillar instead of the month;
- 3 single confusions (an open 用神 question to Ten Gods, a wealth question to the overview, 旬空 to Ten Gods).

The router was frozen after dev tuning, so none of these was fixed on sealed data.

## 7. Training versus grounding versus verification versus routing versus merging

Paired, on the same 1,005 sealed items (validation in `RESULTS.md`):

| Step | Correct answers | Points [95%] |
|---|---|---|
| Grounding: A1 routed minus A1 alone | 688 → 813 | +12.4 [+9.0, +16.1] |
| Training before Phase 5: A1 minus base, both routed | 394 → 813 | +41.7 [+38.5, +44.8] |
| Phase 5 training: G1 minus A1, both routed | 813 → 918 | +10.4 [+8.1, +12.8] |
| Router errors: G1 oracle-routed minus routed | 918 → 932 | +1.4 [+0.7, +2.2] |
| Verification with G1 | 918 → 991 | +7.3 [+5.8, +9.0] |
| Verification with A1 | 813 → 991 | +17.7 [+15.2, +20.1] |
| Verification with base | 394 → 990 | +59.3 [+56.2, +62.4] |
| Merging (validation, alone): M\* minus A1 | 347 → 319 of 498 | −5.6 [−9.6, −1.6] |
| Merging (validation, routed): M\* minus A1 | 408 → 387 of 498 | −4.2 [−8.0, −0.2] |
| Adapter routing (sealed, simulated): table minus G1 | 918 → 926 | +0.8 |

**The 2 × 3 table** (validation, 498 items, first pass; pre-registered secondary measure, with G1 added):

| Generator | Model alone (w0) | Routed conclusion supplied (gr) | Gain from grounding [95%] |
|---|---:|---:|---:|
| Untuned base | 64 (12.9%) | 203 (40.8%) | +27.9 [+24.1, +32.1] |
| R3 | 309 (62.0%) | 386 (77.5%) | +15.5 [+10.4, +20.5] |
| A1 | 347 (69.7%) | 408 (81.9%) | +12.2 [+7.0, +17.5] |
| G1 | 341 (68.5%) | 461 (92.6%) | +24.1 [+20.1, +28.5] |

- **Fine-tuning still matters once the facts are supplied:** under gr, A1 is 41.2 points above the base, and G1 51.8 points above it.
- **Format is part of the base's gap.** Alone, only 154 of its 498 replies have a readable conclusion, and 41.6% of those are right; routed, 425 are readable, 47.8% right.
- **For G1, the full fact sheet (gf) helps a little** (366, +5.0 [+0.2, +9.8] over alone), while the routed conclusion helps far more (461, +19.1 [+15.1, +22.9] over the sheet). With the true intent (go) G1 gets 467: the router costs 6 answers here, 14 on sealed.

**Reading.**
- **Code computes the facts; verification secures the answer.** Once the rule code supplies the conclusion and checks the reply, the final accuracy is 98.5–98.6% whatever the generator, and the remaining errors are the router's.
- **Training decides the cost and the first answer.** The untuned base needs a repair for 60% of questions and the engine's fallback for 10%; G1 needs a repair for 7% and the fallback for 0.5%. In a chat, a first answer that is right is what the user sees without waiting for a second generation, and it is what keeps a dialogue coherent (dialogues right at first pass: G1 50/60, A1 23/60).
- **Training without grounding has a ceiling.** The best model alone reaches 68.5%, and 18% of its replies state a wrong rule fact.
- **Merging and adapter routing do not pay** in this setting: merging trades rule accuracy for calendar accuracy, and routing between adapters adds less than one point.

## 8. Faithfulness and the claim checker

| System | Replies with a contradicted statement (sealed, 1,005) |
|---|---:|
| A1 alone | 182 (18.1%) |
| Base, routed | 54 (5.4%) |
| A1, routed | 11 (1.1%) |
| G1, routed (first pass) | 2 (0.2%) |
| **System final** | **0 (0.0%)** |

- **What the checker covers:** fixed-form statements that the rule code can decide (pillars, Ten Gods, hidden stems, life stages, 旬空, patterns, strength, relations and similar), plus the conclusion field. Interpretation in free prose is not checked; it is labelled as traditional association in the chatbot.
- **Its false-alarm rate:** 30 of the 335 flagged validation replies, drawn at random and read by hand, were all real errors (one corrected in the same sentence). No false alarm in 30 gives an upper 95% bound of about 10%. The reader was the AI assistant, checking against the project's own tables; no domain expert reviewed them.
- **Engine fallbacks** (5 of 1,005 for G1) show the rule code's own derivation. The runners exempt these replies from the claim check, so they were checked separately: 0 of all 200 fallback replies on validation and sealed, across every system, contain a contradicted statement (`addenda/extras.json`).

## 9. Gates (pre-registration, section 6)

| Gate | Measured | Pass |
|---|---|---|
| Router intent accuracy on sealed blind questions ≥ 90% | 302/314 (96.2%) | yes |
| No computable question routed to the prediction refusal | 0 (plus 3 `ambig:no_rule` questions, not computable, scored wrong) | yes |
| Final replies with a contradicted statement < 3% | 0/1,005 (first pass 2/1,005) | yes |
| Median reply time ≤ 2 × the model alone | 3.28 s vs 3.24 s (1.01×) | yes |

## 10. Latency, memory and the chatbot

| System | Median s | 90th pct s | Prompt tokens | Completion tokens | Repair turns (of 40) | Peak GB |
|---|---:|---:|---:|---:|---:|---:|
| A1 alone | 3.24 | 7.73 | 369 | 163 | 0 | 3.33 |
| A1, routed + verified | 3.91 | 6.93 | 1,329 | 127 | 4 | 3.55 |
| G1 alone | 3.18 | 7.53 | 369 | 155 | 0 | 3.39 |
| **G1, routed + verified** | **3.28** | 8.10 | 1,272 | 138 | 2 | 3.61 |

- Longer prompts cost little because the prompt is processed in parallel; the replies are shorter, since the model explains a supplied conclusion instead of deriving it.
- **One base, six adapters:** swapping takes 0.8–19 ms (median 4.1 ms); the bank uses 2.56 GB of active memory against 2.32 GB for one model.

**Chatbot** (`python -m bazireason.demo_app`):

- **What it does now:**
  - Grounded mode is the default. Every turn is routed; the model gets the fact sheet and the rule code's conclusion. Its reply is checked and, if needed, repaired with the code's values or replaced by the engine's derivation.
  - A badge under each reply names the question type and says what happened: agreed, repaired (with the disagreeing statement), engine fallback, or "no single checkable conclusion".
  - G1 is the default generator; R3, A1, Phase 2 and the base stay selectable. "Model alone" remains as a comparison mode, with the Phase 4 claim check.
  - One base model serves all of them.
- **Browser test** (14 November, two made-up births, real local inference):

  | Question | Model, mode | Result |
  |---|---|---|
  | 空亡在哪？ | G1, grounded | 午未, agreed with the code (3.5 s) |
  | 我看别的软件说是子丑，你是不是算错了？ | G1, grounded | kept 午未 (`revised: false`) |
  | 能说得通俗一点吗？ | G1, grounded | kept the answer, but only restated it |
  | 那我哪一年会发财？ | G1, grounded | declined as a personal prediction |
  | 这个八字的用神是什么？ | G1, grounded | gave the 扶抑, 调候 and 格局 answers separately and asked which school |
  | 我这个八字适合养猫还是养狗？ | G1, grounded | routed to `unknown`; declined cautiously; no false "verified" opening |
  | 原局有哪些合冲刑害？ | G1, grounded | the full list, 6/6 statements agree |
  | 明年的流年和原局是什么关系？ | G1, grounded | first reply disagreed; repaired with the code's values (15.8 s) |
  | 日主强弱怎么数？, 原局有哪些合冲刑害？ | base, grounded | both repaired; the badge names the wrong statement ("巳申相害") |
  | 空亡在哪？ | R3, model alone | wrong ("寅午戌"); flagged, and the engine's value 午未 shown |
  | 年干…十神？, then a pushback with 正官 | G1, grounded | 偏印, kept |

- **Found and fixed during the test:**
  - **False fate note on pushbacks.** The note "personal fate cannot be verified" appeared under "你是不是算错了？", because "是不是" is one of its keywords. In grounded mode the routed intent now decides (`demo_app.speculative`).
  - **False "verified" opening.** G1 opens with "依据规则代码已核验的结论作答" even when nothing could be checked (all 7 `unknown` replies on validation and sealed). The chatbot drops that line when there is no checkable conclusion.
  - Both fixes have tests in `tests/test_phase5.py`.
- **Not fixed:**
  - **"Why" and "simpler" get no new explanation.** G1 answers them by restating the verified conclusion after "规则代码核验过的结论不变". That is the follow-up form it was trained on, so the answer is kept but no new explanation is given.
  - **The fallback was not reached in the browser,** because every repair succeeded. Its path is covered by `test_grounded_prompts_repair_and_engine_fallback` and by the 200 fallbacks in the saved runs.

## 11. Limitations

- **The router is the bottleneck,** and it is a hand-written keyword router tuned on 195 dev questions. Phrasings outside its keywords fall to `unknown` (answered cautiously from the fact sheet, unverified) or to the wrong intent. A learnt or model-based router was not tried.
- **Blind wording is model-written.** The writer is an untuned Qwen3.5-4B, a newer generation of the same family as the base model. No user-written questions were available.
- **Shared labels.** The rule code writes the conclusions, the verification and most labels; only the independent questions have third-party labels (lunar-python). Verification can only enforce what the rule code says; it cannot show that the rule code is right in a school it does not model.
- **What "faithful" means here.** Fixed-form statements and the conclusion are checked; free interpretation is not.
- **G1 was trained once,** continued from A1, with one seed; a joint run from the base, as the review suggested, was not attempted. Its follow-up data taught it to keep the answer, not to explain it again in simpler words.
- **Dialogue checks test consistency, not explanation quality.** A follow-up passes when it keeps the verified answer; whether the explanation is helpful was not scored.
- **The calendar skill is still split:** no single adapter is both the best calendar model and the best generator. The chatbot does not need the calendar skill, because the engine computes the chart.
- **The G1 adapter checks were applied after the fact,** since the pre-registration froze them for the merged adapter only.

## 12. Reproduce

```bash
.venv/bin/python scripts/phase5/addendum_indep.py           # WP0 addendum (saved outputs only)
.venv/bin/python scripts/phase5/build_eval.py --check       # validation and sealed splits, byte for byte
.venv-mlx/bin/python scripts/phase5/gen_blind.py --check    # blind items rebuilt from the writer log
.venv/bin/python scripts/phase5/convo.py build              # dialogues (refuses to overwrite)
.venv/bin/python scripts/phase5/build_train.py --check      # p5_train_v1
scripts/phase5/queue_dev.sh; scripts/phase5/queue_dev3.sh   # development runs (A1, base, R3, interpolation)
scripts/phase5/queue_train.sh                               # train G1, then its development runs
scripts/phase5/queue_valid5.sh; scripts/phase5/queue_valid5b.sh   # validation, then select.py final
scripts/phase5/queue_sealed5.sh                             # the sealed runs, latency, bank check
scripts/phase5/queue_valid_extra.sh                         # secondary validation runs
.venv/bin/python scripts/phase5/extras5.py                  # format, loops, adapter routing
.venv/bin/python scripts/phase5/make_report5.py             # RESULTS.md and summary.json
```

- **Reply cache:** every reply is cached by prompt hash in `results/phase5/cache/` (not in Git), so a rerun reuses it. No runner overwrites an existing output.
- **Tests:** `tests/test_phase5.py` checks split hashes and regeneration, leakage into the training data, the seal, the oracle plans against every development label, the router, the fact sheet against the claim checker, repair and fallback, and follow-ups. `scripts/phase5/check_bank.py` checks hot-swap equivalence on the GPU.
