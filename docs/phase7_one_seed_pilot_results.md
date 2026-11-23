# Phase 7 one-seed pilot results

**Development experiment · Qwen3-4B-Instruct-2507 · Apple M5, 32 GB**

## Decision

Explicit-state training (T) scored **112/140 (80.0%)** on fresh held-out graph questions. Its advantage over ordinary replay (R) was **+67.9 percentage points**. Relative to the original U8, familiar-rule accuracy changed by **-0.8** and calendar accuracy by **-0.8** points. The predeclared pilot screen **passed**. The thresholds were +4 over replay and retention losses no greater than 1.5 and 2 points, respectively. This experiment used one training seed. The reserved graph families were not scored.

Training on worked graph questions accounted for most of the gain. Conventional derivations (G) gained 64.3 points over replay. Explicit source and value fields (T) added five correct answers over G on the primary panel, all in two families. G answered seven more alternate-wording questions and four more changed-source questions than T. Keep U8 as the reference and carry G forward as the simpler research candidate. The evidence for preferring T depends on the panel.

## Prerequisite diagnostic

| U8 condition | Correct /120 |
|---|---|
| Ordinary question | 17/120 (14.2%) |
| Same chart, changed source | 11/120 (9.2%) |
| Both members right | 3/120 (2.5%) |
| Isolated source lookup | 38/120 (31.7%) |
| Isolated downstream lookup | 15/120 (12.5%) |
| Both isolated lookups | 3/120 (2.5%) |
| Correct intermediate supplied | 32/120 (26.7%) |
| True distractor supplied | 13/120 (10.8%) |
| Intent cue supplied | 24/120 (20.0%) |

Among 103 ordinary misses, 2 occurred with both isolated prerequisites correct. On the others, 6 missed only the producer, 31 only the consumer, and 64 both. Providing the branch fixed 24 initially wrong answers and broke 9 initially right ones. A secondary content check found 0 strict misses with a recoverable correct final value. An oracle cue can simplify routing as well as supply the missing value.

| Diagnostic family | Full | Producer | Consumer | With gold branch | Misses with a failed prerequisite |
|---|---|---|---|---|---|
| 六合→god | 1/17 (5.9%) | 8/17 (47.1%) | 1/17 (5.9%) | 5/17 (29.4%) | 15/16 |
| 冲→stage | 4/18 (22.2%) | 7/18 (38.9%) | 2/18 (11.1%) | 3/18 (16.7%) | 14/14 |
| 害→positions | 1/17 (5.9%) | 4/17 (23.5%) | 3/17 (17.6%) | 6/17 (35.3%) | 16/16 |
| 桃花→hidden | 0/17 (0.0%) | 4/17 (23.5%) | 1/17 (5.9%) | 0/17 (0.0%) | 17/17 |
| 破→count | 6/17 (35.3%) | 1/17 (5.9%) | 6/17 (35.3%) | 5/17 (29.4%) | 10/11 |
| 破→element | 2/17 (11.8%) | 1/17 (5.9%) | 0/17 (0.0%) | 8/17 (47.1%) | 15/15 |
| 驿马→main | 3/17 (17.6%) | 13/17 (76.5%) | 2/17 (11.8%) | 5/17 (29.4%) | 14/14 |

The diagnostic triggered the prerequisite review. On 101 of 103 ordinary misses, U8 also failed an isolated lookup for the same chart. This prevents attribution to a failure to combine two already-mastered steps. The new prompts also differed from the familiar training tasks, so performance here cannot distinguish incomplete rule knowledge from difficulty applying a known rule to a new instruction. After inspecting the replies, we kept the three approved arms and their frozen evaluation intact to test end-to-end transfer. [Prerequisite decision](../results/phase7/prerequisite_review.json)

## What was trained

R continued on familiar procedures. G trained on new graph questions with numbered conventional derivations. T used exactly G’s questions, gold values and final-answer schema, exposing the source and branch consumed by each step in labeled fields. All arms started from the identical U8 adapter and shared 4,683 replay rows. The remaining 3,221 rows differed by arm. Each trained once with seed 710223. Rows were paired across G and T, with matched batches, fixed endpoints and the historical optimizer settings.

| Arm | Rows | Updates | Sequence tokens | Loss-bearing tokens (logged) | Padded tokens | Train hours | Peak GB | Final iteration |
|---|---|---|---|---|---|---|---|---|
| R | 7904 | 1976 | 3,000,713 | 1,795,245 | 3,141,600 | 1.47 | 19.88 | 3952 |
| G | 7904 | 1976 | 2,994,562 | 1,783,693 | 3,133,728 | 1.48 | 19.88 | 3952 |
| T | 7904 | 1976 | 3,009,352 | 1,798,483 | 3,153,248 | 1.48 | 19.88 | 3952 |

The shared token allocations were 30% calendar, 38% familiar procedures, 8% contrast and 4% other replay. The experimental substitution was approximately 20%. The maximum pairwise discrepancies were 0.50% for sequence tokens, 0.84% for loss-bearing tokens and 0.63% for padded tokens. Optimizer updates matched exactly. Reference labels were obtained independently for each graph instance and compared to the project’s own rule code. Legacy calendar answers use the existing calendar scorer.

## Controlled comparison

| Panel | U8 | R | G | T |
|---|---|---|---|---|
| Held-out graphs, primary | 16/140 (11.4%) | 17/140 (12.1%) | 107/140 (76.4%) | 112/140 (80.0%) |
| Held-out graphs, changed source | 21/140 (15.0%) | 12/140 (8.6%) | 118/140 (84.3%) | 114/140 (81.4%) |
| Held-out graphs, new wording | 13/140 (9.3%) | 15/140 (10.7%) | 108/140 (77.1%) | 101/140 (72.1%) |
| Natural draws, trained graph / familiar wording | 20/70 (28.6%) | 10/70 (14.3%) | 64/70 (91.4%) | 64/70 (91.4%) |
| Natural draws, trained graph / new wording | 9/70 (12.9%) | 6/70 (8.6%) | 65/70 (92.9%) | 60/70 (85.7%) |
| Natural draws, unseen graph / familiar wording | 12/70 (17.1%) | 5/70 (7.1%) | 60/70 (85.7%) | 60/70 (85.7%) |
| Natural draws, unseen graph / new wording | 7/70 (10.0%) | 10/70 (14.3%) | 59/70 (84.3%) | 55/70 (78.6%) |
| Familiar rules | 238/240 (99.2%) | 238/240 (99.2%) | 239/240 (99.6%) | 236/240 (98.3%) |
| Calendar | 216/242 (89.3%) | 203/242 (83.9%) | 215/242 (88.8%) | 214/242 (88.4%) |

| Held-out primary contrast | Difference (pp) | 95% paired chart interval | 95% family + chart interval | Leave-one-family-out range |
|---|---|---|---|---|
| T-R | +67.9 | [+61.4, +74.3] | [+44.3, +88.6] | [+62.5, +75.8] |
| T-G | +3.6 | [+0.7, +6.4] | [+0.0, +10.7] | [+1.7, +4.2] |
| G-R | +64.3 | [+58.6, +70.0] | [+36.4, +88.6] | [+58.3, +74.2] |
| R-U8 | +0.7 | [-5.7, +7.1] | [-10.0, +11.4] | [-1.7, +3.3] |
| G-U8 | +65.0 | [+58.6, +70.7] | [+37.1, +89.3] | [+59.2, +75.0] |
| T-U8 | +68.6 | [+62.1, +75.0] | [+45.0, +89.3] | [+63.3, +76.7] |

The primary panel weights seven families equally. Each contributes 20 independent chart groups. Intervals use 4,000 paired bootstrap draws. Family resampling shows sensitivity to which operations are represented. It does not quantify training-seed variation. G−R tests the substitution of the synthetic graph data. T−G isolates the target rendering within the tested graph set. U8 is a frozen reference. Secondary contrasts are descriptive.

## Source switches and error patterns

| Held-out edge | U8 full / both | R full / both | G full / both | T full / both |
|---|---|---|---|---|
| 六合→stage | 1/20 · 0/20 | 1/20 · 0/20 | 6/20 · 2/20 | 8/20 · 3/20 |
| 华盖→hidden | 4/20 · 2/20 | 6/20 · 2/20 | 20/20 · 20/20 | 20/20 · 20/20 |
| 害→god | 0/20 · 0/20 | 0/20 · 0/20 | 1/20 · 0/20 | 4/20 · 1/20 |
| 桃花→main | 0/20 · 0/20 | 0/20 · 0/20 | 20/20 · 20/20 | 20/20 · 20/20 |
| 破→positions | 3/20 · 0/20 | 6/20 · 0/20 | 20/20 · 20/20 | 20/20 · 20/20 |
| 驿马→count | 5/20 · 3/20 | 4/20 · 0/20 | 20/20 · 20/20 | 20/20 · 20/20 |
| 驿马→element | 3/20 · 0/20 | 0/20 · 0/20 | 20/20 · 20/20 | 20/20 · 20/20 |

| Baseline or diagnostic | Primary challenge panel | Natural unseen panel |
|---|---|---|
| mode | 16/140 (11.4%) | 12/70 (17.1%) |
| natal | 16/140 (11.4%) | 6/70 (8.6%) |

The mode baseline uses the most frequent training answer for each downstream operation. The natal baseline feeds the selected natal branch directly to that operation, skipping the requested producer. Challenge twins were sampled to have different gold values. The natural draw had no such filter. The alternate Chinese wording comes from a held-out template, with the same author as the training templates.

G and T solved both members of every pair in five of the seven families. Across all seven, G solved both on 102/140 charts and T on 104/140, compared with 2/140 for R and 5/140 for U8. The difficult families asked for the day master’s life stage at the harmony partner branch (六合→stage) and the Ten God of the harm partner’s main hidden stem (害→god). Both depend on the day stem as well as the derived branch.

On the ordinary primary misses, the first written step named the correct derived branch on 33/33 G replies and 28/28 T replies. The first-step extraction pattern matched 33 and 28 replies, respectively. This is a post-hoc check of the written work. It localizes observable errors without assuming the text reveals the model’s internal computation.

## Sensitivity to supplied context

The same 56 fresh charts appeared in each condition, eight per development family. L1 supplied chart facts, L2 supplied a rule explanation, and L3 supplied the correct final answer under a verified-result heading. L3w used the same heading with a plausible wrong answer. Results are descriptive. The pilot decision used the unaided primary panel.

| Context condition, out of 56 | U8 | R | G | T |
|---|---|---|---|---|
| No context, correct | 4 | 7 | 46 | 42 |
| Raw chart facts, correct | 6 | 7 | 43 | 41 |
| Rule explanation, correct | 6 | 6 | 22 | 46 |
| Verified correct answer, correct | 7 | 10 | 48 | 44 |
| False verified answer, still correct | 2 | 4 | 43 | 43 |
| False verified answer, copied its value | 5 | 9 | 4 | 3 |

Rule explanations produced the largest context difference. On the same 56 charts they moved G from 46 to 22 correct and T from 42 to 46. G and T each answered 43 correctly under a false verified heading. These findings qualify the choice of G as the simpler unaided candidate when the intended application also supplies rules.

## Retention

| Arm | Familiar-rule Δ vs U8, 95% | Calendar Δ vs U8, 95% | False refusals /150 | Definite charts on nonanswerable /92 |
|---|---|---|---|---|
| R | +0.0 [-1.2, +1.2] | -5.4 [-10.3, -0.4] | 21 | 8 |
| G | +0.4 [-0.8, +2.1] | -0.4 [-5.4, +4.6] | 20 | 2 |
| T | -0.8 [-2.5, +0.8] | -0.8 [-5.7, +4.0] | 19 | 4 |

The rule panel sampled 12 fresh questions per familiar task type (240 total). The calendar panel retained 242 rows, clustered by generator family and event/twin ID. Deltas use row accuracy with a paired group bootstrap. A small point loss may be within sampling error. Demonstrating the specified non-inferiority margins requires a larger independently held-out study.

## Recommendation

1. Keep U8 as the reference. Both graph-trained adapters warrant further study. G is the simpler candidate and led on new wording. T passed the stated point thresholds, but its family-resampled advantage over G includes zero, and both retention intervals cross their respective loss margins.

2. Prioritize the two day-stem-dependent operations. A follow-up diagnostic should pair each complete question with isolated source and downstream lookups using the same operand and independently written instructions. This would test whether a model that names the correct branch can reliably apply the downstream rule.

3. If more research time is approved, compare G and T over multiple seeds with independent wording and additional held-out families. A matched control with new single-step examples would help separate the value of multi-step training from extra exposure to the lookup tables. G versus R here changes the example mix, wording and worked answers together. No replication or extended training was part of this pilot.

## Reproducibility and boundaries

The development evaluation used greedy generation, a single model and its adapter, the U8 system prompt, and no answer replacement. The primary scorer requires one labeled JSON answer, the specified type, and exact value. Historical Phase 6 scores were not used as Phase 7 outcomes. The seven development edges and seven diagnostic edges are disjoint from the 22 training edges. Twelve other edges remain reserved. This tests a finite family of symbolic transformations on synthetic charts. It does not establish transfer across arbitrary BaZi reasoning, independent human phrasing, distribution shift in deployment or another training seed.

The protocol, data hashes, raw replies, fitted-adapter summaries, bootstrap results and family counts are available locally. [Protocol](phase7_pilot_protocol.md) · [Input audit](../results/phase7/input_audit.json) · [Training budgets and hashes](../data/phase7/training_manifest.json) · [Diagnostic counts and examples](../results/phase7/diagnostic_summary.json) · [Full comparison](../results/phase7/comparison.json) · [Raw replies and training records](../results/phase7)

## Completion and verification

The 24-hour budget began at 12:19 p.m. Pacific on November 21. The final model reply was recorded at 21:19 Pacific that evening. The three training runs used about 4 hours 26 minutes in total. Closeout re-scored all 6,688 archived replies, checked one reply per input ID per model and panel, and verified the input, configuration and adapter hashes. [Closeout counts and hashes check](../results/phase7/closeout.json)
