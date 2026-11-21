# Phase 6, round 2: the whole report

Written on 19 November 2026 after level 10, when everything stopped as the user chose on 18 November (pre-registration 12.16). **Section 0, the final test on `sealed6`, was added on 21 November 2026.** On 19 November the user chose "Final test only". Sections 1 to 8 are the dev6 report of 19 November. They are unchanged apart from short notes where they are out of date.

**How to read the numbers.**

- **Section 0 uses `sealed6`:** 1,441 rule items and 816 calendar requests. No training run, choice or selection used them, so these are the unbiased numbers.
- **Sections 1 to 8 use `dev6`,** the split the chain used to make its choices, so they are optimistic.
- **Levels.** Every score is for the model alone (L0) unless a table names another level: no tool, no fact sheet and no code in the answer. L5, the Phase 5 hybrid system, is system accuracy, not model accuracy, and is always shown apart.
- Every number was produced by a script from saved replies. The commands are at the end.
- Intervals are paired 95% bootstrap intervals, in points.

## 0. The final test (`sealed6`), 21 November 2026

**How it ran.** The order was fixed in 12.16, before any result was known.

1. **valid6 for S2 and S2s** (19 November, 08:39 to 09:43).
2. **`decide6.py select`, run once** (09:44), picked **U8**.
   - The rule: the calendar gate, then the best valid6 L0 accuracy on all rule items.
   - U8 had 736 of 813 (90.5%). Next came U5, U6 and U7 with 728 each. S2 and S2s failed the calendar gate.
3. **The sealed runs, once** (`scripts/phase6/queue_sealed6.sh`, 19 November 09:44 to 21 November 02:02):
   - L0 for 18 runs;
   - L1 to L4 for U8, A1 and G1, and L1 to L3 for the untuned base;
   - L5 for G1 and U8.
   - **Access log:** every access to the split is a line in `results/phase6/sealed_access.jsonl`. It has five: the first start, one restart, and one for each of the three later steps.
   - **One interruption.** The queue ran out of GPU memory once (19 November, 20:22), when another program on the machine loaded a model.
     - It restarted at 20:35, after that program had been idle for 5 minutes.
     - The restart skipped every finished part.
     - The one unfinished part, G1's calendar requests, was scored from the start. Replies generated before the failure were reused from the cache; decoding is greedy, so they are the same.

### 0.1 The headline

**U8, model alone (L0), on `sealed6`:**

| | Right | Share |
|---|---:|---:|
| All rule items | 1,336 of 1,441 | 92.7% |
| Without the combinations | 1,259 of 1,281 | 98.3% |
| Combinations (held out from training) | 77 of 160 | 48.1% |
| Calendar requests | 751 of 816 | 92.0% |

- **The dev6 numbers were not inflated by the chain's choices.**
  - Rule items: 91.7% on dev6, 90.5% on valid6 and 92.7% on sealed6.
  - Calendar: 93.2% on dev6 and 92.0% on sealed6.
- **U8's lead in the combinations held on fresh items.** On dev6 it looked fragile (section 1).
  - On sealed6 U8 has 77 of 160 combinations; U4 to U7 have 46 to 60.
  - U8 minus U6: +2.0 [+0.7, +3.3] on all rule items, but +0.6 [−0.2, +1.4] without the combinations.

### 0.2 The pre-registered comparisons (pre-registration section 7)

| | Comparison | Result | Target | Met |
|---|---|---|---|---|
| P-1 | U8 − A1, all rule items | +32.1 [+29.4, +34.7] | lower bound > 0 (aim +10) | yes |
| P-2 | U8 − A1, blind wording | +37.5 [+32.7, +42.5] | secondary | yes |
| P-3 | U8 − untuned base, all rule items | +79.8 [+77.4, +82.1] | lower bound > 0 | yes |
| P-4 | U8 − Phase 2, calendar | +20.6 [+17.1, +24.0] | lower bound ≥ −5 | yes |
| P-5 | U8 with raw facts (L1) − alone (L0) | −1.9 [−3.3, −0.6] | lower bound > 0 | no |
| P-6 | U8 self-correction (L4a) | fixes 11 of 105 (10.5%), breaks 16 of 300 (5.3%) | fix ≥ 25%, break ≤ 2% | no |
| P-7 | Replies with a contradicted checkable statement | U8 7.2%, A1 18.0% | at most half of A1's rate | yes |

- **P-4 holds on balanced decision accuracy as well:** 94.7% for U8, against 75.3% for Phase 2.
- **P-5 and P-6 fail.** U8 does not make use of supplied facts, and it cannot repair its own errors (section 0.5).
- **H1, from round 1, fails on sealed6 as on dev6.** U0 − A1 is −21.3 [−24.1, −18.4] on rule items, and U0 − Phase 2 is −5.8 [−9.1, −2.5] on calendar.

### 0.3 Every run on `sealed6` (model alone)

| Run | All rule items (1,441) | Without combinations (1,281) | Combinations (160) | Calendar (816) |
|---|---:|---:|---:|---:|
| **U8 (selected)** | **1,336 (92.7%)** | **1,259 (98.3%)** | 77 | 751 (92.0%) |
| U4c | 1,310 (90.9%) | 1,250 (97.6%) | 60 | 753 (92.3%) |
| U4 | 1,307 (90.7%) | 1,258 (98.2%) | 49 | 750 (91.9%) |
| U6 | 1,307 (90.7%) | 1,251 (97.7%) | 56 | 754 (92.4%) |
| U3v | 1,305 (90.6%) | 1,237 (96.6%) | 68 | 727 (89.1%) |
| U5 | 1,305 (90.6%) | 1,249 (97.5%) | 56 | 717 (87.9%) |
| U2 | 1,300 (90.2%) | 1,212 (94.6%) | 88 | 668 (81.9%) |
| U7 | 1,299 (90.1%) | 1,253 (97.8%) | 46 | 748 (91.7%) |
| U3 | 1,269 (88.1%) | 1,192 (93.1%) | 77 | 668 (81.9%) |
| S2s | 1,185 (82.2%) | 1,104 (86.2%) | 81 | 481 (58.9%) |
| S2 | 1,147 (79.6%) | 1,073 (83.8%) | 74 | 459 (56.2%) |
| S1-8B | 1,015 (70.4%) | 943 (73.6%) | 72 | 509 (62.4%) |
| S1 | 996 (69.1%) | 933 (72.8%) | 63 | 452 (55.4%) |
| P4 A1 | 874 (60.7%) | 797 (62.2%) | 77 | 527 (64.6%) |
| P5 G1 (the chatbot's model) | 829 (57.5%) | 772 (60.3%) | 57 | 550 (67.4%) |
| U0 | 567 (39.3%) | 497 (38.8%) | 70 | 536 (65.7%) |
| Phase 2 | 357 (24.8%) | 315 (24.6%) | 42 | 583 (71.4%) |
| Untuned base | 186 (12.9%) | 130 (10.1%) | 56 | 15 (1.8%) |

**U8 minus each run, paired:**

| U8 minus | All rule items | Without combinations | Calendar |
|---|---|---|---|
| P4 A1 | +32.1 [+29.4, +34.7] | +36.1 [+33.3, +38.9] | +27.5 [+23.2, +31.5] |
| P5 G1 | +35.2 [+32.4, +38.0] | +38.0 [+35.3, +40.8] | +24.6 [+20.9, +28.2] |
| Phase 2 | +67.9 [+65.4, +70.4] | +73.7 [+71.2, +76.2] | +20.6 [+17.1, +24.0] |
| Untuned base | +79.8 [+77.4, +82.1] | +88.1 [+86.3, +89.9] | +90.2 [+88.1, +92.3] |
| U2, where the chain started | +2.5 [+1.0, +4.1] | +3.7 [+2.4, +5.0] | +10.2 [+7.3, +13.0] |
| U5 | +2.2 [+0.8, +3.5] | +0.8 [+0.0, +1.6] | +4.2 [+1.8, +6.6] |
| U6 | +2.0 [+0.7, +3.3] | +0.6 [−0.2, +1.4] | −0.4 [−1.9, +1.2] |
| U7 | +2.6 [+1.2, +3.9] | +0.5 [−0.5, +1.5] | +0.4 [−1.3, +2.1] |

### 0.4 The hypotheses: dev6 against `sealed6` (model alone)

| Hypothesis | Comparison | dev6, rule items | sealed6, rule items | sealed6, calendar | Reading on sealed6 |
|---|---|---|---|---|---|
| H7: short system prompt at equal compute | S2s − S2 | +0.1 [−2.9, +3.2] | +2.6 [+0.5, +4.8] | +2.7 [−0.6, +6.1] | a small gain on rule items that dev6 was too small to show |
| More data: 15 M against 6.0 M tokens | U2 − S2s | +9.1 [+6.1, +12.2] | +8.0 [+6.0, +10.0] | +22.9 [+19.4, +26.3] | the largest gain, confirmed |
| H8: the whole weighted-continuation chain | U8 − U2 | +2.6 [+0.2, +4.7] | +2.5 [+1.0, +4.1] | +10.2 [+7.3, +13.0] | small but real, confirmed |
| H9: derivation version 2 | U3v − U3 | +1.0 [−1.1, +3.1] | +2.5 [+0.7, +4.3] | +7.2 [+4.3, +10.1] | helps rule items and calendar |
| H10: Qwen3-8B against Qwen3-4B | S1-8B − S1 | −0.4 [−3.7, +3.1] | +1.3 [−1.2, +3.9] | +7.0 [+3.7, +10.7] | no gain on rule items; a gain on calendar |
| H11: composition rows | U4c − U4 | −0.1 [−2.2, +1.8] | +0.2 [−1.1, +1.5] | +0.4 [−1.3, +2.0] | no effect; combinations only +6.9 [−2.5, +16.2] |
| H12: each decision after its check (UD) | – | – | – | – | not tested (UD was not run) |

**H9's second reading (statements) on sealed6:**

- Replies with a wrong statement fall from 161 to 95 of 1,441: −4.6 [−6.1, −3.1].
- Wrong hidden-stem statements in area-of-life replies fall from 44 to 0 of 77.

### 0.5 Help, self-correction and the Phase 5 hybrid system

**Levels of help, on all rule items.** L3w keeps the 1,329 items for which a wrong conclusion can be given.

| Model | L0 alone | L1 raw facts | L2 rule text | L3 right conclusion | L3w: right / follows the wrong conclusion |
|---|---:|---:|---:|---:|---:|
| U8 | 1,336 (92.7%) | 1,309 (90.8%) | 1,158 (80.4%) | 1,304 (90.5%) | 1,050 (79.0%) / 122 (9.2%) |
| P4 A1 | 874 (60.7%) | 803 (55.7%) | 604 (41.9%) | 1,010 (70.1%) | 640 (48.2%) / 207 (15.6%) |
| P5 G1 | 829 (57.5%) | 719 (49.9%) | 549 (38.1%) | 1,006 (69.8%) | 124 (9.3%) / 814 (61.2%) |
| Untuned base | 186 (12.9%) | 227 (15.8%) | 359 (24.9%) | 364 (25.3%) | – |

- **U8 does best alone.** Each level minus L0, paired: L1 −1.9 [−3.3, −0.6], L2 −12.4 [−14.6, −10.0], L3 −2.2 [−3.7, −0.8].
  - Help raises U8 only on the combinations: 77 of 160 alone, 101 with rule text and 95 with the right conclusion.
  - **What changes:** 189 answers on the template rules and blind wording are right at L0 but wrong with rule text. 22 of them have no readable conclusion. The rest are complete replies whose values change, across interactions, flow-year specials, Ten-God stems, trigger years, stars and patterns.
  - **The text is not new to U8.** From U2 on, its training mixture has rows with raw facts or the same rule text in the prompt, and rows that check a supplied reference conclusion by deriving it. The cause of the drop is not established.
- **A1 and G1 gain from the right conclusion** (+9.4 [+6.8, +12.1] and +12.3 [+8.6, +15.8]), but they lose points with raw facts or rule text.
- **The untuned base gains from rule text and from the right conclusion** (+12.0 [+9.8, +14.0] and +12.4 [+10.1, +14.5]). About half its replies run into the 1,100-token cap at every level.
- **Misleading conclusions.** G1, which Phase 5 trained to use a supplied conclusion, follows a wrong one in 61.2% of items. U8 follows it in 9.2% and keeps its own derivation.

**Self-correction (L4).** Every wrong L0 answer is checked, plus 60 right ones per part.

| Model | a: re-check (fixed / broke) | b: told there is an error | c: told which field is wrong |
|---|---|---|---|
| U8 | 11/105 (10.5%) / 16/300 (5.3%) | 16/105 (15.2%) / 26/300 (8.7%) | 11/105 (10.5%) / 22/300 (7.3%) |
| P4 A1 | 11/567 (1.9%) / 5/300 (1.7%) | 12/567 (2.1%) / 13/300 (4.3%) | 22/567 (3.9%) / 14/300 (4.7%) |
| P5 G1 | 5/612 (0.8%) / 0/297 (0.0%) | 4/612 (0.7%) / 2/297 (0.7%) | 14/612 (2.3%) / 9/297 (3.0%) |

- No model corrects itself usefully.
- U8's fixes are almost all combinations: 10 of 11 with check a, 15 of 16 with check b.
- U8 is right far more often than wrong, so any of the three checks would lose more right answers than it repairs.

**L5, the Phase 5 hybrid system.** Its parts are a router, the rule code's conclusion, value repair and an engine fallback. It covers the template rules, independent and blind parts (1,121 items). This is system accuracy, not model accuracy.

| | Template rules (462) | Independent (280) | Blind wording (379) | All three (1,121) |
|---|---:|---:|---:|---:|
| The system, with G1 or with U8 inside | 459 | 280 | 360 | 1,099 (98.0%) |
| U8 alone (L0) | 452 | 275 | 377 | 1,104 (98.5%) |
| U8 alone minus the system with U8 inside | −1.5 [−2.8, −0.2] | −1.8 [−3.6, −0.4] | +4.5 [+2.4, +6.9] | +0.4 [−0.6, +1.5] |
| U8 alone minus the system with G1 inside | −1.5 [−2.8, −0.2] | −1.8 [−3.6, −0.4] | +4.5 [+2.4, +6.9] | +0.4 [−0.6, +1.4] |

- **U8 alone is level with the system.**
  - The system scores the same with either model inside, because the rule code supplies and repairs the values. The two systems differ on only 2 of the 1,121 items, both blind wordings, one each way.
  - The system is ahead on the template rules and the independent items. U8 alone is ahead on the blind wording.
- **The system's limit is its router.** It sends 21 of the 379 blind wordings to the wrong question type.
- **Inside the system, U8's own replies are worse than alone:** 930 of 1,121, against 1,104 alone, the same pattern as the levels of help. G1's are 1,011. The system's repair makes up the difference.

### 0.6 What U8 still gets wrong on `sealed6`

- **Combinations: 77 of 160, the clearest limit.** These questions chain two trained steps in a way no training row shows.

  | Kind (40 items each) | Right |
  |---|---:|
  | Flow year with life stage | 5 |
  | Void positions | 20 |
  | Revealed stems | 24 |
  | God counts | 28 |

- **Trained types.** Every trained type with 30 or more items is at 90% or higher.
  - The lowest are element counts (77 of 85, 90.6%) and relations (100 of 105, 95.2%). All others are at 97% or higher.
  - So the type bar of 12.12, read on sealed6, holds. On dev6 it failed by one item, on relations. 12.12 is registered on dev6, so its verdict (not met) does not change.
- **Calendar: 65 errors of 816.**
  - Answerable requests refused: 28 of 531.
  - A definite chart where none exists: 15 of 285.
  - The right decision with wrong pillars: 22. In all 22 the day pillar is wrong, and in 21 the hour pillar as well, since the hour stem follows from the day stem. The errors are in the day-count arithmetic.
  - Balanced decision accuracy: 94.7%.
- **Statements.** 7.2% of U8's replies contain a statement the claim checker contradicts (A1: 18.0%). No look-up statement is wrong (0 of 3,544).

### 0.7 What this means

- **The final model is U8.** It is Qwen3-4B with a rank-16 LoRA adapter.
  - On questions that no choice used, alone, it answers 92.7% of rule items, 98.3% without the held-out combinations, and 92.0% of calendar requests.
  - It is 32 points above A1, the Phase 4 model, and 35 points above G1, the chatbot's current model.
  - It is level with the Phase 5 hybrid system, which answers with the rule code's values.
- **For the chatbot (Quarter 2).** G1 is the chatbot's default, as the user confirmed in Phase 5. On these numbers, U8 alone is as accurate as the hybrid system and better on unfamiliar wording. Whether to switch is the user's decision; nothing has been changed.
- **What the numbers do not show.**
  - Each run was trained once, with one seed. Seed-to-seed variation was not measured, so differences of a point or two between runs may be within it.
  - The labels come from the project's rule code and lunar-python. The sealed items use new charts and new wordings, but the same conventions as the training labels. The scores measure how well a model follows those conventions, not agreement with any human school of BaZi.
  - Nothing here measures whether BaZi predicts anything about a person's life, and no such claim is made.

## 1. The short answer on dev6 (19 November)

- **The best run of round 2 on dev6 is U8.** It is Qwen3-4B with a rank-16 LoRA adapter, trained in seven stages: U2, then the continuations U3v, U4, U5, U6, U7 and U8.
  - **Rule items:** 754 of 822 (91.7%).
  - **Calendar requests:** 204 of 219 (93.2%).
- **Gains, on the same items:**

  | U8 minus | Rule items | Calendar |
  |---|---|---|
  | P5 G1, the chatbot's current model | +36.9 [+33.2, +40.5] (754 vs 451) | +27.9 [+20.2, +35.0] (204 vs 143) |
  | The untuned base | +77.6 [+74.4, +80.6] (754 vs 116) | – |
  | U2, where the chain started | +2.6 [+0.2, +4.7] (754 vs 733) | +10.0 [+5.2, +15.3] (204 vs 182) |

- **Well trained (12.12)? Not by the test registered on 16 November.** Three of its four conditions hold on U8. The fourth fails by one item: relations are 57 of 64 (89.1%), and the bar is 90% (58 of 64). Section 4 has the details.
- **The lead at the top is fragile.** U8's lead comes from the 100 held-out combinations (50 of 100, against 36 for U6). No run trains on these items, and their score swings by up to 26 items from one level to the next.
  - On the other 722 items, U6 scores highest (711). U8 has 704, and U8 minus U6 is −1.0 [−2.3, +0.3]. U9 has 706, and U9 minus U6 is −0.7 [−1.8, +0.4].
  - So U6, U8 and U9 are statistically level on everything the runs were trained for.
- **The final model is not chosen yet.** The frozen rule picks it on `valid6`, among all candidates. It may not be U8. *(21 November: the rule picked U8. Section 0 has the final test.)*

## 2. Every round-2 run (dev6, model alone)

| Run | From | All rule items (822) | Without combinations (722) | Combinations (100) | Calendar (219) |
|---|---|---:|---:|---:|---:|
| P5 G1 (chatbot today) | – | 451 (54.9%) | 418 (57.9%) | 33 | 143 (65.3%) |
| U2 | untuned base | 733 (89.2%) | 683 (94.6%) | 50 | 182 (83.1%) |
| U3 | U2 | 722 (87.8%) | 679 (94.0%) | 43 | 189 (86.3%) |
| U3v | U2 | 730 (88.8%) | 691 (95.7%) | 39 | 201 (91.8%) |
| U4 | U3v | 740 (90.0%) | 705 (97.6%) | 35 | 200 (91.3%) |
| U4c | U3v | 739 (89.9%) | 703 (97.4%) | 36 | 200 (91.3%) |
| U5 | U4 | 740 (90.0%) | 705 (97.6%) | 35 | 193 (88.1%) |
| U6 | U5 | 747 (90.9%) | **711 (98.5%)** | 36 | 203 (92.7%) |
| U7 | U6 | 739 (89.9%) | 708 (98.1%) | 31 | 202 (92.2%) |
| **U8** | U7 | **754 (91.7%)** | 704 (97.5%) | 50 | 204 (93.2%) |
| U9 | U8 | 730 (88.8%) | 706 (97.8%) | 24 | **207 (94.5%)** |
| U10 | U9 | 733 (89.2%) | 698 (96.7%) | 35 | 205 (93.6%) |

**What the runs are:**

- **U2:** 15 M tokens of training data and the short system prompt, trained from the untuned base.
- **U3 onward:** each continuation is trained from its parent's adapter on 6 M new tokens. Question types and calendar requests are weighted by the parent's own errors on fresh practice questions (fit samples B and C). From level 4, the calendar weights are set by family (12.13).
- **U3v onward:** written steps of derivation version 2.
- **U4c:** U4 plus composition rows (H11).
- **The 822 rule items** are template rules (294), independent rules (140), blind wordings (188), held-out combinations (100) and counterfactual pairs (100).

**Runs before the chain** (same split and scoring):

| Run | Rule items | Calendar |
|---|---:|---:|
| S2, wording | 657 | 139 |
| S2s, S2 with the short prompt | 658 | 137 |
| S1-8B, S1's data on Qwen3-8B | 546 | 138 |
| S1, small steps | 549 | 128 |

## 3. What the chain decided

The chain rule: each level continues from the previous level's best arm. It ends after two levels in a row without a new best, after level 10, or after 27 November.

| Level | Built from | Run(s) | Rule items | New best? | Paired difference against the level before (rule items; calendar) |
|---|---|---|---:|---|---|
| 3 | U2 | U3, U3v | 722, 730 | no (U2 733) | U3 − U2 −1.3 [−3.6, +0.9]; +3.2 [−2.5, +8.5]. U3v − U2 −0.4 [−2.6, +1.7]; +8.7 [+4.0, +13.8] |
| 4 | U3v | U4, U4c | 740, 739 | yes (U4) | U4 − U3v +1.2 [−0.7, +3.2]; −0.5 [−4.3, +3.3] |
| 5 | U4 | U5 | 740 | no (tie) | +0.0 [−1.8, +1.7]; −3.2 [−6.9, +0.5] |
| 6 | U5 | U6 | 747 | yes | +0.9 [−0.9, +2.5]; +4.6 [+1.5, +7.9] |
| 7 | U6 | U7 | 739 | no | −1.0 [−2.7, +0.7]; −0.5 [−2.1, +1.0] |
| 8 | U7 | U8 | 754 | yes | +1.8 [−0.1, +3.7]; +0.9 [−1.2, +3.2] |
| 9 | U8 | U9 | 730 | no | −2.9 [−4.9, −1.0]; +1.4 [−0.5, +3.5] |
| 10 | U9 | U10 | 733 | no, chain ends | +0.4 [−1.5, +2.2]; −0.9 [−3.5, +1.5] |

- **Level 10** started because the user chose it (12.16). The chain then ended for two reasons: it was the last level, and it was the second level in a row without a new best.
- **The pattern:** on rule items, no level is clearly better than the one before it, and level 9 is clearly worse. On calendar, levels 3 (U3v) and 6 are clearly better. The chain as a whole gained over U2: +2.6 [+0.2, +4.7] on rule items and +10.0 [+5.2, +15.3] on calendar.
- **Most of the rule-item gain came early.** Without combinations, U2 scores 683, U4 705 and U6 711. After U6 the score stays between 698 and 711.

## 4. The well-trained test (pre-registration 12.12), on U8

| Condition | Result on U8 | Holds? |
|---|---|---|
| 1. The chain has ended | ended after level 10, its last level, with two levels in a row without a new best | yes |
| 2. Every trained question type with at least 30 dev6 items is at 90% or more | relations 57 of 64 (89.1%); every other type is at 90% or more | **no, by one item** |
| 3. Flow-year and area-of-life items: the share right with the year outside 2024-2049 is within 10 points of the share inside | inside 29/29, outside 74/77 (96.1%); a gap of 3.9 points (U2: 26.0) | yes |
| 4. The last two levels raised the best calendar count by fewer than 5 | levels 9 and 10 raised it from 204 to 207 (+3) | yes |

- **U6 meets conditions 2 and 3:** no type is below 90%, and the year gap is 6.5 points. Conditions 1 and 4 concern the chain and hold. But U6 is not the chain's best run, so the registered test does not apply to it.
- **What 12.12 says when condition 1 holds and another fails:** register a targeted change, then run one more level from the best run if it can start by 27 November. Under the user's choice to stop everything, no such level was run. Section 8 describes this option.

## 5. What the best runs still get wrong

**1. Decisions written before their check.** This is the error that recurs most across runs and task kinds.

- **Revealed stems:** U9 lists the hidden stems' Ten Gods, then writes "都透" without checking the other stems, and answers wrongly.
- **Pattern questions in the held-out wording:** U10 gets 6 of 12, against 12 of 12 for U9. It skips the Ten-God table that U9 writes out, then names the main hidden stem's Ten God from memory with the wrong polarity (戊 for a 辛 day master as 偏印, not 正印).
- **Calendar:** refusals of a solar-term window whose own stated date lies outside the window. U8 has 4, U9 2 and U10 5.
- Derivation version 3 (UD, 12.14) was written to put each decision after its check. It was not run.

**2. Relations.** U8 gets 57 of 64, and its wrong relation statements are 56, against 37 for U9 and 18 for U10. U8's errors start at the step that collects the pairs and at the trio step. Relations by run: U6 62/64, U9 61/64, U10 59/64.

**3. Held-out combinations: 24 to 50 of 100.** These questions chain two trained steps in a way no training row shows. The runs score as follows:

| Kind (25 items each) | U2 | U6 | U8 | U9 | U10 |
|---|---:|---:|---:|---:|---:|
| Void positions | 10 | 5 | 12 | 6 | 12 |
| Revealed stems | 15 | 9 | 17 | 4 | 13 |
| God counts | 18 | 12 | 16 | 9 | 6 |
| Flow year with life stage | 7 | 10 | 5 | 5 | 4 |

Composition rows did not help (H11, section 6). This is the clearest limit of generalization in the current recipe.

**4. Calendar.** U8 makes 15 errors:

- answerable solar-term windows: 3 of 10 wrong;
- daylight-saving folds: 3 of 11;
- invalid inputs: 2 of 12;
- Shanghai daylight saving that needs the tool: 2 of 4;
- one each in plain charts, known time zones, time zones that need the tool, solar-term windows that need the tool, and solar-term boundaries that need the tool.

Answerable requests refused: 7 of 145. Definite charts given where none exists: 7 of 74. Balanced decision accuracy: 92.9%.

**5. What is fixed.**

- **The year window:** U2 got 57/77 outside 2024-2049; U8 gets 74/77.
- **Wrong statements:** replies with a wrong statement fell from 88/822 for U2 to 60 for U8. Look-up statements are almost never wrong (0 of 2,089 on U8).
- **Area of life:** 31/46 for U2, 44/46 for U8.
- **Overview:** 18/20 for U2, 19/20 for U8.

## 6. Hypotheses of round 2 (dev6, model alone, all rule items; calendar)

| Hypothesis | Comparison | Rule items | Calendar | Reading |
|---|---|---|---|---|
| H7: short system prompt at equal compute | S2s − S2 | +0.1 [−2.9, +3.2] | −0.9 [−7.6, +6.0] | no effect |
| More data: 15 M against 6.0 M tokens, same recipe and prompt | U2 − S2s | +9.1 [+6.1, +12.2] | +20.5 [+12.7, +27.8] | the largest single gain of the round |
| H8: weighted continuation | each level against the one before (section 3) | at most +1.8 per level; U8 − U2 +2.6 [+0.2, +4.7] | U8 − U2 +10.0 [+5.2, +15.3] | small but real over the whole chain |
| H9: derivation version 2 | U3v − U3 | +1.0 [−1.1, +3.1] | +5.5 [+0.5, +10.3] | helps calendar |
| H10: base size (Qwen3-8B against Qwen3-4B, S1's data) | S1-8B − S1 | −0.4 [−3.7, +3.1] | +4.6 [−1.0, +10.1] | no gain from the larger base |
| H11: composition rows | U4c − U4 | −0.1 [−2.2, +1.8]; combinations only +1.0 [−12.0, +14.0] | +0.0 [−2.5, +2.7] | no effect |
| H12: each decision after its check (UD) | – | – | – | not tested (not run) |

## 7. What was not run, and the state of the splits

*State on 19 November. Since then valid6 has been read for the selection, the selection has picked U8, and `sealed6` has been scored once (section 0).*

**Not run, by the user's choice of 18 November (12.16):**

- valid6 for S2 and S2s;
- U1 as registered and its diagnostics;
- the fit samples B for runs without one;
- UD (12.14), so H12 is untested;
- U1-hi.

**How the stop happened:**

- **04:23, 19 November:** after U10's valid6 run, `decide6.py next` recorded the chain's end in `results/phase6/round2.json`. That is level 11, with no runs and two levels in a row without a new best. It then held, as registered.
- **04:24:** the main queue and `queue_ud.sh` were stopped. Each log ends with a stop line.

**The splits:**

- `valid6` has been scored for U2 through U10 by the queue, but no score has been read: the waiters and reports leave it out.
- `sealed6` is still sealed. `results/phase6/selection.json` does not exist.
- Labels come only from the rule code and lunar-python. BaziQA was never training data.

## 8. What is left, and how to run it

*Since 19 November, item 1, the final test, has run (section 0). Item 2, UD, was not run: on 19 November the user chose "Final test only", so H12 stays untested.*

**1. The final test.** The report needs it for an unbiased headline. The order was fixed in 12.16, before any of this was known:

1. valid6 for S2 and S2s, about 1 h;
2. `decide6.py select`, under the frozen rule: the calendar gate, then the best valid6 L0 accuracy on all rule items;
3. the sealed runs once (`scripts/phase6/queue_sealed6.sh`, about 30 h).

U1, U1-hi and UD have no valid6 result, so they are not candidates. Writing starts on 29 November and the report is due on 4 December, so there is time.

**2. Optional, if it can start by 27 November: UD (12.14).**

- **What it is:** U8 continued with derivation version 3, about 5.5 h with dev6 and valid6.
- **Its twin:** U9, the level built from U8, so H12 can be read.
- **Why run it:** it rewrites the collecting and trio steps where U8's relations errors start, which is where 12.12 condition 2 fails. It also targets the decide-before-check errors of section 5.
- **Selection:** it would be a candidate in the selection like every continuation, so it would have to run before the final test.

**3. Nothing more.** Section 7 lists what stays untested.

## How these numbers were produced

```bash
PYTHONPATH=src .venv/bin/python scripts/phase6/make_report6.py          # results/phase6/RESULTS.md, summary.json
PYTHONPATH=src .venv-mlx/bin/python scripts/phase6/truncation6.py       # replies cut off at the cap
PYTHONPATH=src .venv/bin/python scripts/phase6/first_error.py --split dev6 --labels p6u10
PYTHONPATH=src .venv/bin/python scripts/phase6/decision_first.py dev6 p6u8 p6u9 p6u10
PYTHONPATH=src:scripts/phase6:scripts/phase5 .venv/bin/python scripts/phase6/sealed_extras6.py   # section 0's tables and breakdowns
```

- **Section 0:** the registered comparisons, the L0 tables, the levels of help, self-correction and L5 are in the `sealed6` part of `results/phase6/RESULTS.md`, written by `make_report6.py`. `sealed_extras6.py` prints the rest from the same saved records: every run's table, the paired differences against U8 (`decide6.paired("sealed6", ...)`), level minus L0, L5 against U8 alone paired by item, and the error breakdowns in 0.5 and 0.6. The breakdowns are descriptive counts, not registered measures.

- **Paired differences:** `decide6.paired("dev6", a, b, parts=...)`, with `parts` set to all rule parts, to the rule parts without `combo`, or to `("calendar",)`.
- **The table in section 2:** the saved dev6 L0 records, counted by part with `make_report6.load`.
- **The well-trained conditions:** counted from the same records, by question type and by question year.
- **Day-by-day record:** `docs/phase6_log.md`.
- **The rules:** `docs/phase6_preregistration.md` (12.12 to 12.16).
