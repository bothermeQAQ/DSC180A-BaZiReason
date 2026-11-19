# Phase 6, round 2: the whole report

19 November 2026. Written after level 10, when everything stopped as the user chose on 18 November (pre-registration 12.16).

- **04:23:** the queue recorded the chain's end and held.
- **04:24:** both queues were stopped. No training or scoring job of this project is running.

**How to read the numbers.**

- Every number is from `dev6`, the split the chain used to make its choices, so the numbers are optimistic. An unbiased number needs the final test, which has not been run: `valid6` has not been read and `sealed6` is still sealed.
- Every score is for the model alone (L0): no tool, no fact sheet and no code in the answer.
- Every number was produced by a script from saved replies. The commands are at the end.
- Intervals are paired 95% bootstrap intervals, in points.

## 1. The short answer

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
- **The final model is not chosen yet.** The frozen rule picks it on `valid6`, among all candidates. It may not be U8.

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
```

- **Paired differences:** `decide6.paired("dev6", a, b, parts=...)`, with `parts` set to all rule parts, to the rule parts without `combo`, or to `("calendar",)`.
- **The table in section 2:** the saved dev6 L0 records, counted by part with `make_report6.load`.
- **The well-trained conditions:** counted from the same records, by question type and by question year.
- **Day-by-day record:** `docs/phase6_log.md`.
- **The rules:** `docs/phase6_preregistration.md` (12.12 to 12.16).
