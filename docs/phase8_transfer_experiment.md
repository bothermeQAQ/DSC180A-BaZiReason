# Phase 8 · Transfer to unseen BaZi rule combinations

**Primary leader · G.** The targeted continuation (A) solved 144/240 (60.0%) independently worded questions, compared with 150/240 (62.5%) for its matched active control and 151/240 (62.9%) for its parent. A−B was **-2.5 percentage points**, with a paired chart 95% interval of [-7.9, +3.3] and a family-and-chart interval of [-12.1, +7.1]. The predeclared screen **did not pass**.

## Why this experiment

Phase 7’s conventional worked examples (G) raised primary accuracy from 17/140 under ordinary replay to 107/140. Explicit operands (T) reached 112/140. Every G and T primary miss named the correct intermediate branch. On all 26 life-stage errors across the two adapters the text also gave the right start, direction and target. On all 35 Ten-God errors it named the correct hidden stem. This motivated a direct test of whether strengthening the last operation would carry through to a fresh multi-step question.

G was the parent because its unaided transfer held up across wording changes. T was evaluated as a second frozen contender. A practiced extracting the day stem from the chart and applying it to an explicitly supplied branch. B practiced main or all hidden stems, matching element counts and positions on the same charts and branch operands. B controls for continued optimization, new chart exposure and worked lookups. Its practice can also improve performance on the eight other families.

## Training and timing

Each arm trained once, with seed 810117, from the exact Phase 7 G weights. Each used 3,040 common replay examples plus 1,920 fresh single-step examples. Each of the 120 day-stem × branch combinations appeared eight times in A for each target operation. Neither arm trained a new two-step graph family. Both used rank-16 LoRA, a fresh AdamW optimizer, learning rate 0.00005 with cosine decay to 0.000005 and the fixed final checkpoint. No evaluation answer selected a checkpoint.

| Arm | Rows | Updates | Sequence tokens | Supervised tokens | Padded tokens (plan) | Minutes | Peak GB | Final train loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 4960 | 1240 | 1,512,734 | 810,267 | 1,614,944 | 47.0 | 19.57 | 0.0055 |
| B | 4960 | 1240 | 1,506,180 | 801,313 | 1,590,752 | 45.7 | 19.57 | 0.0019 |

The budget started at 01:12 Pacific on November 22. GPU training began at 02:19. The report was finalized at 06:38 Pacific, 5.45 hours after the start, including preparation, idle time, inference, verification and reporting.

## Transfer and retention

Each new primary question combines two familiar operations along a directed edge held outside Phase 7 training, diagnostics and development. The panel covers all 12 reserved edges, with 20 independently drawn chart groups per edge. Year/day source position and two new wordings were crossed within edge. A separate evaluation agent wrote the new Chinese wording. All models used the same unaided greedy decoder and answer contract. Each model generated one reply per prompt.

| Panel | G · parent | T · explicit fields | B · control | A · targeted |
| --- | --- | --- | --- | --- |
| New wording, primary | 151/240 (62.9%) | 145/240 (60.4%) | 150/240 (62.5%) | 144/240 (60.0%) |
| Familiar wording, same charts | 186/240 (77.5%) | 183/240 (76.2%) | 189/240 (78.8%) | 193/240 (80.4%) |
| New wording, four day-dependent families | 22/80 (27.5%) | 18/80 (22.5%) | 20/80 (25.0%) | 25/80 (31.2%) |
| New wording, eight other families | 129/160 (80.6%) | 127/160 (79.4%) | 130/160 (81.2%) | 119/160 (74.4%) |
| Changed source, four targeted families | 19/80 (23.8%) | 25/80 (31.2%) | 22/80 (27.5%) | 24/80 (30.0%) |
| Correct branch given, downstream only | 30/80 (37.5%) | 38/80 (47.5%) | 52/80 (65.0%) | 73/80 (91.2%) |
| Familiar rules, fresh | 394/400 (98.5%) | 393/400 (98.2%) | 392/400 (98.0%) | 394/400 (98.5%) |
| Calendar, fresh | 361/400 (90.2%) | 359/400 (89.8%) | 362/400 (90.5%) | 361/400 (90.2%) |
| Phase 7 development, original wording | 107/140 (76.4%) | 112/140 (80.0%) | 114/140 (81.4%) | 118/140 (84.3%) |
| Phase 7 development, source switched | 118/140 (84.3%) | 114/140 (81.4%) | 116/140 (82.9%) | 124/140 (88.6%) |
| Phase 7 development, alternate wording | 108/140 (77.1%) | 101/140 (72.1%) | 113/140 (80.7%) | 112/140 (80.0%) |

| Primary contrast | Δ pp | 95% chart interval | 95% family + chart interval | Removing one family, range | Left only / right only |
| --- | --- | --- | --- | --- | --- |
| A-B | -2.5 | [-7.9, +3.3] | [-12.1, +7.1] | [-5.0, -0.5] | 23 / 29 |
| A-G | -2.9 | [-8.7, +2.5] | [-11.7, +7.1] | [-5.9, -0.9] | 23 / 30 |
| A-T | -0.4 | [-7.1, +6.2] | [-14.2, +11.2] | [-2.7, +4.5] | 37 / 38 |
| B-G | -0.4 | [-5.8, +4.6] | [-7.1, +6.7] | [-1.8, +0.9] | 20 / 21 |
| B-T | +2.1 | [-4.2, +7.9] | [-10.0, +14.2] | [-0.9, +5.5] | 35 / 30 |
| G-T | +2.5 | [-3.8, +8.8] | [-8.8, +13.7] | [-0.5, +5.5] | 37 / 31 |

The paired intervals use 10,000 bootstrap draws. The chart interval conditions on the 12 observed families. The broader interval also resamples families. Each model contributed a single fixed training realization. Phase 7 development outcomes are descriptive because they informed the intervention.

## Which operations transferred

| Source → downstream lookup | G | T | B | A |
| --- | --- | --- | --- | --- |
| 六合→hidden | 18/20 | 15/20 | 18/20 | 19/20 |
| 六合→number | 17/20 | 19/20 | 18/20 | 18/20 |
| 华盖→main | 14/20 | 14/20 | 11/20 | 13/20 |
| 害→number | 20/20 | 19/20 | 20/20 | 20/20 |
| 害→stage | 2/20 | 2/20 | 0/20 | 1/20 |
| 桃花→count | 15/20 | 13/20 | 17/20 | 13/20 |
| 桃花→element | 12/20 | 18/20 | 11/20 | 7/20 |
| 桃花→positions | 17/20 | 14/20 | 16/20 | 15/20 |
| 破→god | 11/20 | 4/20 | 11/20 | 9/20 |
| 破→stage | 1/20 | 1/20 | 0/20 | 1/20 |
| 驿马→god | 8/20 | 11/20 | 9/20 | 14/20 |
| 驿马→positions | 16/20 | 15/20 | 19/20 | 14/20 |

| Matched-chart diagnostic | G | T | B | A |
| --- | --- | --- | --- | --- |
| Both wordings correct | 137/240 | 130/240 | 136/240 | 126/240 |
| Both source choices correct when gold changes | 12/73 | 8/73 | 11/73 | 9/73 |
| Isolated downstream correct, full question wrong | 18/80 | 29/80 | 34/80 | 52/80 |
| Full question correct, isolated downstream wrong | 10/80 | 9/80 | 2/80 | 4/80 |
| Novel wording form 1 | 92/120 (76.7%) | 81/120 (67.5%) | 87/120 (72.5%) | 85/120 (70.8%) |
| Novel wording form 2 | 59/120 (49.2%) | 64/120 (53.3%) | 63/120 (52.5%) | 59/120 (49.2%) |

A shortcut that applies the downstream operation directly to the natal branch scored 30/240 (12.5%). The different-answer source switches test whether the model changes its answer when the chart stays fixed. The isolated probe supplies the correct derived branch. A high isolated score alongside a much lower full score suggests a gap in using the operation inside a chain.

A gained 26.3 percentage points over B on isolated downstream probes (paired chart 95% interval +13.8 to +38.8). On full questions that used those same operands, its gain was 6.3 points (−3.8 to +16.2). Across the eight other families, A lost 6.9 points relative to B. The gain on the earlier Phase 7 primary (+11 answers over G) and the familiar wording of the new panel (+7) did not carry over to the new primary wording (−7).

A post-hoc output check found that A answered only 2/40 new life-stage questions correctly. Among the 38 misses, six final answers named the wrong life stage, 14 named a Ten God and 18 returned some other value or failed to parse. The same model could often finish the lookup when given the branch. On transfer-03-00, A correctly derived 申, then answered 比肩, a Ten God. Given the same 申 directly, it used day stem 辛 to count four places backward from 子 and correctly answered 帝旺. On transfer-05-01, A derived 酉 correctly but reversed the elemental relationship in its Ten-God calculation. Its isolated answer correctly identified 正财. These paired replies show two different failure modes, choosing the requested operation and executing it within the longer question.

## Preservation of rule and calendar capabilities

U8 scored 399/400 (99.8%) on the new rule questions and 364/400 (91.0%) on calendar questions. Retention intervals resample whole event groups within task type. All contrasts below are percentage points.

| Track | Contrast | Δ | Paired 95% interval |
| --- | --- | --- | --- |
| rules | A-G | +0.00 | [-1.0, +1.0] |
| rules | A-U8 | -1.25 | [-2.3, -0.2] |
| rules | B-G | -0.50 | [-2.0, +0.8] |
| rules | T-U8 | -1.50 | [-2.8, -0.5] |
| calendar | A-G | +0.00 | [-1.5, +1.5] |
| calendar | A-U8 | -0.75 | [-3.7, +2.0] |
| calendar | B-G | +0.25 | [-1.2, +1.8] |
| calendar | T-U8 | -1.25 | [-4.0, +1.5] |

| Calendar behavior | U8 | G | T | B | A |
| --- | --- | --- | --- | --- | --- |
| Refused an answerable chart | 13/248 | 34/248 | 34/248 | 30/248 | 30/248 |
| Gave a definite chart on a nonanswerable case | 19/152 | 2/152 | 3/152 | 6/152 | 6/152 |

## Decision and limits

The screen requires A−B of at least +4 points with its chart-paired interval above zero, improvement over G, and point losses no larger than 1.5 familiar-rule points or 2 calendar points against both G and U8. A met all four retention point criteria. It missed the required primary advantage over B, the chart interval crossed zero, and its primary score fell below G.

The 12 families are a finite set of synthetic two-step rules. The two formulations vary the language while sharing the chart header, system prompt and answer schema. Broad real-user wording, deeper chains, base-model pretraining exposure and seed-to-seed variation remain unmeasured. These reserves have now been evaluated. Any subsequent confirmation needs new families or a new task construction.

## Model choice and next experiment

Keep Phase 7 G as the generalization reference. It led the primary panel by one answer over B and seven over A, and no primary pairwise interval excluded zero. U8 remains the retention reference. Despite G and A having the same aggregate calendar score, A issued six definite charts on nonanswerable cases against G’s two. The preservation thresholds passed on point estimates for A, while the differences versus U8 remain uncertain at the specified margins.

The next experiment should test selection of the downstream operation in a complete question. Using training-only combinations, pair the same chart, source and derived branch with two different requests, such as life stage versus Ten God. Also pair the same requested operation with different source branches. Vary the wording independently of the operation and require the normal final answer. Compare with an equally sized ordinary worked-answer continuation from G at the same token budget. Hold out entire source–operation pairs and use independently written questions plus the paired isolated probes. A gain on composed questions while retaining isolated accuracy would directly address the failure observed here.

A second seed would check training variability after a method clears a fresh end-to-end screen. Reusing this reserve panel for future tuning would turn it into development data. No follow-on training or deployment was launched.

## Integrity and reproducibility

Before any transfer inference, input-only checks found chart tuples that repeated across date blocks and calendar dates or prompts shared with the archive. We regenerated the panels from their original seeds with explicit exclusions. The retention audit checks chart identities, exact prompts and calendar input dates. The evaluator stored raw replies and input and adapter hashes. The analysis re-scored every reply, checked identities and paired chart groups, and reused the archived Phase 7 G/T replies only on their original development inputs.

[Frozen protocol](phase8_protocol.md) · [Training inputs and hashes](../data/phase8/training_manifest.json) · [Transfer panel](../data/phase8/transfer_manifest.json) · [Retention audit](../results/phase8/retention_audit.json) · [Full numerical comparison and provenance](../results/phase8/comparison.json) · [Raw replies, timings and training logs](../results/phase8)
