# BaZiReason Phase 3 progress report: verifiable traditional rule reasoning

Yonghao Wang · DSC 180A/B Capstone, Domain D25 · 12 November 2026
Branch `phase3/deep-reasoning`. Phase 1 (tag `checkpoint-2026-11-06`) and every Phase 2 adapter, dataset and result are unchanged.

All numbers come from files under `results/phase3/` (full tables in [`results/phase3/RESULTS.md`](../results/phase3/RESULTS.md)) and `results/phase2/`. Every model result is real local inference with MLX on an Apple M5 MacBook Pro with 32 GB. Every score is an exact structured comparison with the calendar engine or the rule code; free text is never graded by similarity.

## 1. Summary

| Question | Answer |
|---|---|
| Starting point | Phase 2 adapter `bazireason-qwen3-4b-lora-v2` on `Qwen/Qwen3-4B-Instruct-2507` (4-bit). LoRA rank 16 on the last 16 layers, 14.68 M trainable parameters |
| Training | Three rounds, each warm-started from the one before. 14,400 training rows in total, 7,199 iterations (3,599 optimizer updates), 393 minutes, 10.5 M sequence tokens, peak memory 20.6 GB |
| Final adapter | `artifacts/adapters/bazireason-qwen3-4b-lora-p3-r3/adapters.safetensors` (56 MB, SHA-256 `b559c219…6f600dfa6d`). Round 3 was chosen over round 2 after the test splits, by the brief's calendar constraint. The validation rule had preferred round 2 by 357 to 353 (section 5) |
| Rule reasoning | **Improved.** On the Phase 3 test split (621 reasoning items, unseen years): untuned 6.0%, Phase 2 15.8%, Phase 3 65.1%, i.e. +49.3 points over Phase 2 [45.2, 53.5]. Single table lookups: 26.7% → 87.3%. Rule questions in the Phase 2 wording: 122 → 149 of 240 |
| Ambiguity | **Improved.** Definite answers where none exists: 11/32 → 1/32 on the Phase 3 split. On benchmark_v3 strict: 76.2% → 28.6% of ambiguous requests, while the right status rose from 89.3% to 92.7% |
| Strict corrupted charts | **Not meaningfully improved.** In one-pass JSON, wrong tool answers corrected: 4/150 → 8/150. On benchmark_v3, corrupted answers followed: 98.8% → 98.0% |
| BaziQA | **No improvement over Phase 2.** Without tools: 30.6% vs 30.6%. With verified charts: 30.3% vs 31.9%, a difference of −1.6 points [−4.5, +1.2]. Chance is 24.7%; human champions average 55.4% |
| Regressions | Five-element counts in the Phase 2 wording (14 → 6 of 20); counterfactual comparisons (74 → 68 of 84); false alarms on sound charts (9 → 14 of 91); repetition loops in contest replies with tools (30 → 53 of 1,665); a small, non-significant calendar dip on the clean check split (−4.0 points [−9.2, +1.3] on calendar rows). Round 2's larger calendar regression was the reason it was not kept |
| Demo | `.venv-mlx/bin/python -m bazireason.demo_app`, then open `http://127.0.0.1:8765`. It offers the Phase 3, Phase 2 and untuned models |

## 2. Why Phase 2 fell short (diagnosed before any new training)

The Phase 2 answers on its held-out split and on benchmark_v3 were read item by item. Five weaknesses had four causes.

| Phase 2 weakness | Root cause found in the answers |
|---|---|
| Rule look-ups right 47/96 | **Table recall, not reasoning.** Errors were a wrong 生克 direction, a wrong 刑 or 害 partner, or a Ten God named in one sentence without working out the relation. Every branch-relation question was wrong (0/8): the model listed relations it half remembered instead of checking each pair. |
| Corrupted tool answer followed 98.8% of the time (strict JSON) | **Copying.** The strict protocol allows no visible steps, and the model had never been trained to recheck a supplied chart in one pass. Of the corruption kinds, only `default_guess` is visible from the request alone. `jie_side` cannot be seen without the solar-term instant. |
| Definite answers to ambiguous inputs (29% and 41% on the Phase 2 held-out splits, 76.2% on benchmark_v3 strict) | **No contrast in training.** Most training answers were charts, and few ambiguous rows had a near-identical answerable twin. The model learned to always give a chart. |
| No BaziQA gain | **The contest asks for something never taught.** Contest keys map a chart to life events through a practitioner's whole-chart judgment. Phase 2 taught calendar computation and single look-ups. Its charts were right (80/116), but nothing linked them to an answer option. |
| Brittle free-form interpretation | **No chained rules.** Strength, pattern, 用神 and flow-year questions need four to six rule applications in order; Phase 2 had one look-up per question. |

These findings set the Phase 3 design:
- rule tables are written out before they are used;
- whole-chart procedures are chained one step at a time;
- ambiguous questions are paired with answerable twins;
- claims are checked one by one;
- the strict audit is trained.

Interpretation that no rule settles is declined, not invented.

## 3. What was built

- **Sourced rule cards** (`data/phase3/knowledge/rules_v2.json`, built by `scripts/phase3/build_knowledge.py`). There are 12 new cards, in addition to the 18 from Phase 2: 3 deterministic tables, 8 declared conventions and 1 traditional association.
  - They carry 20 short quotations from 《渊海子平》, 《三命通会》, 《滴天髓》 and 《穷通宝鉴》 (public domain, as transcribed on zh.wikisource.org). Each quotation is checked against the downloaded page, and the page's SHA-256 is recorded.
  - 《子平真诠》 is cited without a quotation because no transcription was available.
  - Rule statements are this project's own wording; no domain expert has reviewed them.
- **Rule code** (`src/bazireason/deep_rules.py`):
  - Ten-God classes and their 相生 / 相克 relations, plus the ten named pairings (食神生财, 官印相生, 伤官见官 and so on).
  - A branch card of 合 / 冲 / 刑 / 害 / 破 partners, with pairs read by position.
  - 透干 and 通根; the strength count (得令 / 得地 / 得势); the month-order pattern by 透干.
  - **用神 under three schools kept apart:** 扶抑, a simplified 调候 (winter → 火, summer → 水) and 格局 (顺用 / 逆用).
  - Stars (禄, 阳刃, 天乙贵人, 驿马, 桃花, 华盖), with the basis pillar declared.
  - Flow years against a chart: 伏吟, 天克地冲, 值太岁 and 冲太岁, 岁运并临.
  - A declared trigger rule for areas of life: an area's star appears in the flow year, or its palace is clashed or six-combined.
- **Task builders** (`deep_tasks.py`). There are 15 drill kinds and 13 reasoning families. Each answer writes the table rows it uses and ends with a JSON line that the scorer compares exactly. The families include:
  - claims to check (correct, wrong with a correction, or unverifiable);
  - nine kinds of question without a unique answer, each paired with an answerable twin;
  - nine kinds of rule-keyed multiple choice;
  - multi-turn dialogues: follow-up, unsupported pushback, an injected error, a prediction request;
  - whole-chart overviews.
- **Datasets and splits:**
  - three datasets: `p3_train_v1` (6,933 / 302 / 801 rows), `p3_train_v2` (4,122 / 181) and `p3_train_v3` (3,345 / 191);
  - a development split (`dev_v1`, 627 rows from the validation years);
  - a calendar check split (`p2check_v1`, 485 rows).

  All regenerate byte for byte. Details: [`data/phase3/README.md`](../data/phase3/README.md).
- **Evaluation:**
  - `p3_eval.py`: exact scorers for every family, with grouped bootstrap intervals.
  - `run_p3_eval.py`, plus the unchanged Phase 2 runners for the Phase 2 splits, benchmark_v3 and BaziQA.
  - `score_baziqa3.py`: four tracks (all, BaZi-only, official keys, paper years) with person-cluster intervals.
  - `count_citations.py`: book names and repetition loops in replies.
- **Demo** (`demo_app.py`, `demo_deep.py`): three models, a verified deep-structure panel, routing of free questions to trained formats, and a rule-code check of each reply.
- **Tests:** 25 new, 229 in total, all passing; Ruff is clean.

## 4. Data and leakage

- **Disjoint years and exclusions.** Year blocks and exclusion lists are those of Phase 2: train 119 years, validation 30, test 50, disjoint. No date lies within seven days of a benchmark_v3 case or within one day of a BaziQA birth date. A test checks every ISO date in every user turn.
- **No benchmark or contest text.** No BaziQA question, option, key or explanation and no benchmark_v3 prompt is read by any generator, and tests check this.
- **Unseen wordings.** Each question has three wordings, and the third occurs only in the test split (255 of 621 reasoning items).
- **Unseen tool errors.** The test-only `strict_shift` family uses two kinds of tool error that never occur in training.
- **Split roles:**
  - Errors were read on `dev_v1`.
  - Rounds were compared on validation splits by rules written before the candidates were scored.
  - The Phase 3 test split was first scored after round 2 had been chosen over round 1.
  - The Phase 2 test splits were read row by row while diagnosing round 2. `p2check_v1` was therefore generated for the round-3 comparison and scored without inspection.
- **Labels from code only.** Every label comes from the engine or the rule code. Areas of life have one target: the traditional correspondence, where it sits in the chart, which years match the declared trigger rule, and an explicit statement that no event is predicted. No personal outcome is labelled.

## 5. Training and selection

| Round | Starts from | Data | Why this round | Iterations (updates) | Minutes |
|---|---|---|---|---:|---:|
| 1 | Phase 2 v2 adapter | `p3_train_v1`: 6,933 rows, 683 of them Phase 2 replay | Teach the procedures | 3,466 (1,733) | 178 |
| 2 | Round 1 | `p3_train_v2`: 4,122 rows, 412 replay | Round-1 errors on `dev_v1`: Ten Gods guessed inside chart tasks (4/35), pairs misread from correct cards (133/133 cards right), silent year arithmetic in multiple choice. Answers now write the Ten-God table first, check every pairing and pair, and list each option's pillar | 2,061 (1,030) | 120 |
| 3 | Round 2 | `p3_train_v3`: 3,345 rows, 1,715 replay | Round-2 regressions: calendar forgetting, explicit tables unused under Phase 2 prompts, book names in answers. Replayed Phase 2 rule answers now start with their table; no book title anywhere; half the learning rate | 1,672 (836) | 95 |

LoRA settings as in Phase 2: rank 16, scale 2.0, last 16 layers, batch 2 × gradient accumulation 2, AdamW with warm-up and cosine decay. The peak learning rate was 1e-4 in rounds 1–2 and 5e-5 in round 3.

**How the final round was chosen.**

1. **Round 1 vs round 2 (validation rule, fixed in advance).** Round 2 won: 227 vs 198 of 298 Phase 3 validation items.
2. **Round 2 vs round 3 (second validation rule, fixed before round 3 was scored).** The rule added the correct answers on the Phase 3 validation split (298) and on the Phase 2 validation split (203), to weigh calendar skills alongside rule reasoning. Round 2 won by 4 items:

| Round | Phase 3 validation | Phase 2 validation | Total |
|---|---:|---:|---:|
| 2 | 227 | 130 (exact four pillars 47/74) | 357 |
| 3 | 215 | 138 (exact four pillars 57/74) | 353 |

3. **Calendar constraint (applied after the test splits were scored).** The brief says not to sacrifice the Phase 2 calendar skills. On the never-inspected check split, round 2's calendar rows were 11.2 points below Phase 2 [−16.2, −6.6]; round 3's were −4.0 [−9.2, +1.3]. Round 2 therefore breaks the constraint and round 3 does not, while round 3's rule reasoning is within noise of round 2's (−1.6 points [−5.2, +1.8]). `scripts/phase3/final_choice.py` records this decision and its evidence in `results/phase3/final_choice.json`. BaziQA was not consulted.

**Caveat.** This choice was made after the test splits were scored, so round 3's test numbers below are post-selection. With two candidates the selection effect is small, but the numbers are not a pre-registered estimate. The validation rule failed to protect calendar skills because its Phase 2 validation split has only 74 exact-pillar items, and the rule counted any item as one point.

## 6. Results

### 6.1 Rule reasoning (Phase 3 test split, unseen years)

| Family | Untuned | Phase 2 | Round 1 | Round 2 | **Round 3 (final)** |
|---|---:|---:|---:|---:|---:|
| Single table lookup (`drill`) | 11/150 | 40/150 | 133/150 | 133/150 | **131/150** |
| Ten Gods and pairings | 0/40 | 0/40 | 3/40 | 23/40 | **19/40** |
| Strength count | 0/30 | 4/30 | 22/30 | 18/30 | **15/30** |
| Pattern by 透干 | 2/30 | 8/30 | 11/30 | 22/30 | **24/30** |
| 用神 under a named school | 0/30 | 0/30 | 20/30 | 26/30 | **27/30** |
| All relations with adjacency | 0/40 | 0/40 | 9/40 | 13/40 | **16/40** |
| Flow year against the chart | 0/50 | 0/50 | 10/50 | 22/50 | **21/50** |
| Traditional view of an area of life | 0/36 | 0/36 | 2/36 | 8/36 | **10/36** |
| Rule-keyed multiple choice | 19/63 | 25/63 | 26/63 | 42/63 | **36/63** |
| Second and third turns | 5/30 | 9/30 | 13/30 | 15/30 | **14/30** |
| Whole-chart overview | 0/20 | 0/20 | 5/20 | 10/20 | **7/20** |
| **All reasoning items** | **37/621 (6.0%)** | **98/621 (15.8%)** | **325/621 (52.3%)** | **414/621 (66.7%)** | **404/621 (65.1%)** |
| Wording never seen in training | 6/255 | 28/255 | 115/255 | 129/255 | **133/255** |

Round 3 minus Phase 2: +49.3 points [45.2, 53.5] (row-group bootstrap). Long chains are still the weak point. A whole-chart relation list, a flow year or an area of life needs every one of 6–20 checks right, and one slip makes the item wrong.

**Rule questions in the Phase 2 wording** (the three Phase 2-style splits, 240 items, 20 per rule): Phase 2 122, round 2 130, round 3 149. On the first split, round 3 minus Phase 2 is +15.6 points [5.2, 26.0].

| Rule (of 20) | Phase 2 | Round 3 | | Rule (of 20) | Phase 2 | Round 3 |
|---|---:|---:|---|---|---:|---:|
| Ten God of a stem | 13 | 16 | | Seasonal state | 15 | 18 |
| Strength | 7 | 13 | | Hidden stems | 9 | 13 |
| Branch relations | 2 | 8 | | Luck pillars | 16 | 17 |
| Flow year | 2 | 8 | | Pattern, relatives, life stage | 10, 10, 19 | 10, 10, 19 |
| 旬空 | 5 | 11 | | **Five-element count** | **14** | **6** |

The table prefix of round 3 (section 5) worked where it was added: Ten Gods, relations, flow years and 旬空. Five-element counts got worse: 9 of the 14 wrong round-3 answers ran the strength procedure (得令 / 得地 / 得势) instead of counting.

### 6.2 Ambiguity, claims and declining

| Phase 3 test split | Untuned | Phase 2 | Round 2 | **Round 3** |
|---|---:|---:|---:|---:|
| Right decision on paired questions (answer or decline) | 10/54 | 37/54 | 50/54 | **52/54** |
| Definite answer where none exists (lower is better) | 4/32 | 11/32 | 4/32 | **1/32** |
| Declined although an answer exists (lower is better) | 6/22 | 6/22 | 0/22 | **1/22** |
| Single claim verdicts right | 1.0% | 53.1% | 93.8% | **94.3%** |
| False statement accepted as correct (lower is better) | 1/84 | 15/84 | 6/84 | **5/84** |
| Unverifiable prediction accepted as correct | 0/30 | 0/30 | 0/30 | **0/30** |

| benchmark_v3 strict, `direct` | Untuned | Phase 2 | Round 2 | **Round 3** |
|---|---:|---:|---:|---:|
| Right status (261) | 0.0% | 89.3% | 75.1% | **92.7%** |
| Definite chart for an ambiguous request (lower is better) | 90.5% | 76.2% | 42.9% | **28.6%** |
| Answerable request refused (lower is better) | 0.0% | 0.0% | 18.7% | **1.8%** |

All 44 requests for a definite personal prediction on the three Phase 2-style splits were declined by Phase 2 and by round 3 alike. Round 2 had over-shot into refusing answerable calendar requests (18.7%); round 3 keeps most of the gain on ambiguous requests and refuses only 1.8% of answerable ones.

### 6.3 Strict corrupted-chart detection

| One-pass audit of a supplied wrong answer | Untuned | Phase 2 | Round 2 | **Round 3** |
|---|---:|---:|---:|---:|
| Phase 3 split, wrong answers corrected | 0/150 | 4/150 | 11/150 | **8/150** |
| … `year_shift` / `default_guess` | 0/10, 0/15 | 0/10, 4/15 | 6/10, 5/15 | 4/10, 4/15 |
| … every other kind: 五鼠遁 or 五虎遁 violation, day rollover, hour branch, shifted day (125 rows) | 0 | 0 | 0 | 0 |
| benchmark_v3 `tool_corrupted`, wrong answer followed (lower is better) | 100.0% | 98.8% | 96.7% | **98.0%** |

Strict corrupted-chart detection did not meaningfully improve. In a single JSON object with no room for steps, the model does not recompute a stem from the 五鼠遁 or 五虎遁 table. This held after 1,300 strict-audit training rows in rounds 1 and 2 (about half with a wrong supplied answer) and 200 more in round 3. The only errors it sometimes catches are visible without arithmetic: an ambiguous request answered confidently, and a year pillar that contradicts the year.

With visible steps, detection is near ceiling for every model. On the three Phase 2-style verification tasks, round 3 detects 90/93 corrupted charts against 91/93 for Phase 2. It raises more false alarms on sound charts: 14/91 against 9/91.

### 6.4 Calendar skills

| | Phase 2 | Round 2 | **Round 3 (final)** |
|---|---:|---:|---:|
| **Check split, never inspected, all rows (485)** | 349 | 314 (−7.2 [−11.2, −3.4]) | **345 (−0.8 [−4.7, +2.9])** |
| … calendar rows, difference from Phase 2 | – | −11.2 [−16.2, −6.6] | **−4.0 [−9.2, +1.3]** |
| … exact four pillars (159) | 130 | 105 | **129** |
| Phase 2 first test split, all rows (633) | 456 | 437 (−3.0 [−6.0, 0.0]) | **467 (+1.7 [−1.4, +5.1])** |
| … exact four pillars (190) | 152 | 141 | **161** |
| Phase 2 fresh test split, all rows (486) | 336 | 310 (−5.3 [−9.3, −1.5]) | **343 (+1.4 [−2.0, +5.0])** |
| … exact four pillars (162) | 122 | 106 | **133** |
| benchmark_v3 reasoned, exact, `direct` (225) | 156 | 130 | **157** |
| benchmark_v3 reasoned, exact, `tool_intermediate` (225) | 204 | 167 | **200** |
| BaziQA charts computed without the tool, exactly right (116) | 80 | 77 | **89** |

Round 2 had forgotten derivation steps that Phase 2 knew. Its errors on the first Phase 2 split:
- the leap-day term dropped from the day count;
- a negative offset added as a positive one;
- the hour stem of the late 子 hour taken from the wrong day.

Round 3 replayed the Phase 2 families at half the learning rate. It is back at the Phase 2 level on the clean check split and above it on the two older splits, which were read during diagnosis and are therefore less clean.

### 6.5 BaziQA contest questions (1,024-token budget, three runs per tuned condition)

| Model | No tools | Verified chart | Unanswered (no tools / tools) | Replies naming a book (of 1,665) | Repetition loops |
|---|---:|---:|---:|---:|---:|
| Untuned (1 run) | 25.6% | 31.2% | 15.1% / 4.3% | 1 / 0 (of 555) | 0 / 0 |
| Phase 2 | 30.6% | 31.9% | 0.8% / 1.0% | 0 / 0 | 24 / 30 |
| Phase 3 round 2 | 27.6% | 30.1% | 6.3% / 7.0% | 87 / 92 (37 / 24 non-existent) | 113 / 135 |
| **Phase 3 round 3** | **30.6%** | **30.3%** | 2.2% / 3.2% | 7 / 1 (3 / 0 non-existent) | 28 / 53 |

- **No gain over Phase 2.** Round 3 minus Phase 2 is 0.0 points [−3.3, +3.2] without tools and −1.6 [−4.5, +1.2] with tools (person-cluster bootstrap). BaZi-only questions: 28.9% / 30.2%. Official keys: 31.5% / 31.3%. On years with an archived champion score: 31.5% / 30.7%, against a champion average of 55.4%.
- **Round 3 fixed most of round 2's contest-reply defects.** Book names fell from 87 / 92 to 7 / 1. Loops fell from 113 / 135 to 28 / 53, so fewer questions go unanswered.
- **Grounding is high.** With the verified chart, 95.8% of checkable chart statements in round-3 replies match the chart (Phase 2: 94.1%). Without the tool, 81.3% match (Phase 2: 69.6%).
- **No gain was expected from rules alone.** Contest keys reward whole-chart judgment about life events, and Phase 3 deliberately teaches no event labels. Agreement with a contest key would not, in any case, be evidence that destiny can be predicted.

## 7. What regressed, and the tradeoffs

**Round 3 (final) against Phase 2:**

| Regression | Size | Likely cause |
|---|---|---|
| Five-element count, Phase 2 wording | 14 → 6 of 20 | 9 of the 14 wrong answers ran the strength procedure: a Phase 3 procedure captured a nearby question |
| Counterfactual comparison of two charts | 74 → 68 of 84 | Not diagnosed |
| False alarms when verifying a sound chart | 9 → 14 of 91 | More suspicion after audit training |
| Repetition loops in contest replies, with tools | 30 → 53 of 1,665 | Long templated answers; fewer than round 2 (135) |
| Calendar rows, check split | −4.0 points [−9.2, +1.3] | Not significant; exact pillars 129 vs 130 |
| BaziQA with verified charts | −1.6 points [−4.5, +1.2] | Not significant |

**Round 3 against round 2** (the price of the calendar recovery):

| Measure | Round 2 | Round 3 |
|---|---:|---:|
| Phase 3 test, reasoning items | 414/621 | 404/621 (−1.6 [−5.2, +1.8]) |
| Rule-keyed multiple choice | 42/63 | 36/63 |
| Whole-chart overview | 10/20 | 7/20 |
| Reasoning with the verified tool block | 88/147 | 78/147 |
| Strict audit corrections | 11/150 | 8/150 |

**The tradeoff, investigated.** Rule training and calendar computation compete for the same 14.7 M LoRA parameters:
- **Round 2:** 10% replay bought the best rule scores at a calendar cost that is significant on every split.
- **Round 3:** 51% replay and half the learning rate restored calendar skills and fixed the citation, loop and over-abstention defects. Rule reasoning stayed within 1.6 points of round 2; the losses are concentrated in multiple choice and overviews.

The brief ranks calendar skills as not to be sacrificed, which is why round 3 is final. Round 2 stays on disk for anyone who weighs rule reasoning higher.

## 8. What distinguishes BaZiReason from a general-purpose LLM

1. **Verified computation, not recall.** Charts come from a calendar engine checked against independent astronomy, handling solar-term instants, both 子-hour conventions, time zones and daylight saving. The model's own chart is compared with it. Without the tool, the untuned base model gets 0 of 116 contest charts exactly right; Phase 3 gets 89 of 116.
2. **Rules with provenance and a declared kind.**
   - Every rule card states whether it is a table, a convention or a traditional association.
   - Quotations are checked against a public-domain transcription; a source that could not be checked is cited without a quote.
   - Schools are named and never merged into one "用神".
   - On rule-reasoning items, the base model gets 6.0% right and Phase 3 65.1%.
3. **Checkable answers.** Every answer ends in a structured conclusion that code compares with the rule engine. The demo does this live and also checks fixed-form claims in the prose: pillars, Ten Gods, luck direction, the named pattern, and the spouse star against the gender entered.
4. **Calibrated refusal.**
   - It says when a question has no unique answer: 用神 without a school, an unknown hour or gender, "is this a good chart?". Only 1 of 32 such test questions got a definite answer.
   - It accepts no unverifiable prediction (0 of 30).
   - It declines to predict personal events (44 of 44).
5. **Measured, with its failures.** Every claim above has a held-out number and an interval. The weak strict audit, the regressions and the absent contest gain are reported, not hidden.

## 9. Demo

```bash
.venv-mlx/bin/python -m bazireason.demo_app                  # http://127.0.0.1:8765
.venv-mlx/bin/python -m bazireason.demo_app --preload all    # load all three models at start (~9 GB)
.venv-mlx/bin/python -m bazireason.demo_app --adapter artifacts/adapters/bazireason-qwen3-4b-lora-p3-r2   # round 2
```

**Using it.**
1. Enter a birth date, time, time zone, sect and, optionally, gender, then press 计算四柱.
2. Pick a model: Phase 3 (round 3) by default, Phase 2, or the untuned base.
3. Ask a question or use one of the 12 quick questions.

Each reply shows the engine's chart and the deep-structure panel computed by the rule code. It also shows the exact prompt the model received and a verdict that compares the reply's conclusion and fixed-form claims with the rule code. `--share-lan` serves the demo on the local network behind an access code. Nothing is stored unless the user ticks the consent box.

**Checked end to end in the browser with round 3** (1990-05-17 14:30, Asia/Shanghai, no gender):
- **Strength count:** correct and passed the check.
- **Flow year 2027 (丁未):**
  - **Correct:** the pillar, the Ten God and both 午未 combinations.
  - **Missed:** a 伏吟 with the hour pillar 丁未.
  - **Check:** flagged the miss and showed the engine's derivation.
- **"我什么时候会结婚？":** declined, but took the female spouse stars although no gender was entered.
- **"这个八字好不好？", first turn:** correctly declined as having no unique answer. In passing it called the chart 伤官格 (the rule code says 偏印格).
- **"这个八字好不好？", later turn:** repeated the previous turn's answer instead.
- **Marriage question:** asked for the missing gender, but named two Ten Gods wrongly, and the check flagged them.

After this run I added two prose checks: the named pattern, and the spouse star against the gender entered. On a repeat run in the browser the pattern check flagged the 伤官格 claim. A unit test confirms that the spouse-star check flags the female-only reply when no gender is entered. The demo is useful because it does not trust the model.

## 10. Limits

- **Templated Chinese.** The model is better inside the trained formats than outside: wording seen in training 74.0%, unseen 52.2%. Later turns can copy the previous answer.
- **This project's conventions.** The strength count, the 透干 pattern rule, the simplified 调候 and the trigger rule are declared conventions; a practitioner may disagree, and none has reviewed them.
- **Small sets.** Several families have 20–50 items, so a few items' difference is noise. Intervals are reported for the main contrasts.
- **Post-selection test numbers.** The final round was chosen after the test splits were scored (section 5).
- **BaziQA is not an untouched test.** It was evaluated in Phase 2. Nothing was tuned on it, but it was not unseen by the researcher.
- **One machine, one base model.** A 4-bit 4B model with LoRA; a larger model may behave differently.

## 11. Next steps

1. **Train on the mixture.** Mix Phase 2 and Phase 3 data from the start, rather than correcting forgetting afterwards, and route procedures by an explicit task label so that a five-element question does not trigger the strength procedure.
2. **Give the strict audit room to compute.** Add a short structured check field before the final JSON, rather than more one-pass rows.
3. **Harden multi-turn.** Add dialogue rows whose later question changes format, so the model stops copying the previous answer.
4. **Get an expert review.** Ask a practitioner to review the 12 rule cards and the declared conventions.

## 12. Reproduce

[`docs/phase3_training.md`](phase3_training.md) has every command:
- the knowledge base;
- the datasets (`--check` regenerates byte for byte);
- the three training configurations;
- both selection scripts and `final_choice.py`;
- every evaluation, the report and the demo.

`scripts/phase3/queue_round3.sh` runs the last round and all its comparisons unattended.
