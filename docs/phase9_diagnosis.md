# Phase 9 archive diagnosis and methods review

Phase 8 points to at least three distinct difficulties. Rephrasing a complete question can change how G reads the source, which operation it performs, and whether it returns a scorable answer. With familiar wording, G can identify the correct branch and return an answer from the requested category while computing the wrong value. Finally, its graph training covered some downstream operands much more thoroughly than others. A comparison that teaches only the operation name has limited leverage over the latter two problems.

## Re-scoring the archived replies

The two question forms below share the chart, source, requested operation, correct branch and answer schema. The new wording was written by a separate evaluation agent. The first-step check recognizes the conventional “可得” and explicit “所得地支=” markers. Unrecognized first steps remain unclassified. “Correct first branch” describes the answer text. Internal computation cannot be recovered from that text.

| Phase 8 model and wording | Wrong final answers | First branch recognized | Recognized and correct | Unparseable strict final answer |
| --- | ---: | ---: | ---: | ---: |
| G, familiar | 54 | 54 | 54 | 0 |
| G, new | 89 | 72 | 57 | 15 |
| A, familiar | 47 | 47 | 47 | 0 |
| A, new | 96 | 74 | 62 | 3 |

All 54 familiar-wording G errors returned a valid member of the requested answer category. Of those, 21 were wrong life stages and 33 were wrong Ten Gods. A showed the same pattern, with 16 and 31 errors respectively. On the new wording, only 13 of G’s 37 life-stage misses returned a life-stage name. For A the count fell to six of 38. Its 52 charts correct on the isolated lookup and wrong on the composed question establish a large gap between the two conditions. Removing the source lookup also shortens and changes the question, so the gap alone cannot isolate its cause. [Archived experiment and examples](phase8_transfer_experiment.md)

To test a specific shortcut account, I computed every other available downstream operation on the *correct derived branch* for every primary miss. Only five of G’s 89 wrong final answers and six of A’s 96 exactly equaled such an alternative. This narrow test leaves open wrong operands, wrong execution after choosing another operation, another operation outside the eight tested, and format errors. It does establish that replacing the selected operation name in an otherwise perfect executor cannot explain most observed answers. [Reproducible archive audit](../scripts/phase9/diagnosis.py)

## Exposure to downstream operands

I reconstructed the day stem and derived branch in G’s actual graph-training replies. Its 438 life-stage examples covered 109 of 120 possible pairs. Its 292 Ten-God examples covered 73. The two trained Ten-God producers (桃花 and 华盖) can reach just eight of the 12 branches. This exposure measure applies to the graph-training rows. The base model and earlier replay could also have learned these values.

| Operation on Phase 8’s four targeted families | Charts whose day-stem/branch pair appeared in G graph training | G familiar, pair seen | G familiar, pair absent | G new wording, pair seen | A isolated, pair seen |
| --- | ---: | ---: | ---: | ---: | ---: |
| Life stage | 38/40 | 19/38 | 0/2 | 3/38 | 38/38 |
| Ten God | 9/40 | 5/9 | 2/31 | 5/9 | 7/9 |

The sparse Ten-God operand coverage is a plausible contributor to G’s familiar-wording failures. The source and operand distributions are correlated, and these small observational strata cannot identify a causal effect. For life stage, 38 of 40 held-out operands already appeared in graph training. A’s perfect isolated score on those 38 contrasts with 2/38 on their complete, independently worded questions. Repeating isolated lookup examples has weak empirical justification as the next primary intervention. [Training data](../data/phase7/g/train.jsonl) · [Phase 8 evaluation inputs](../data/phase8/transfer.jsonl)

## Experimental alternatives

**Vary the training distribution.** Diversity of semantic structures, operands, roles and instructions can reduce opportunities to infer an operation from a recurring source or a single question phrase. Yao and Koller found that uniform sampling of meaning representations improved compositional semantic parsing over sampling from the training distribution. Zhou, Jiang and Bansal found benefits from greater example diversity across multiple synthetic tasks, with results sensitive to task and difficulty mix. For BaZi, this motivates balancing operations across training producers, checking reachable operand cells and measuring effects on novel chart–source–operation combinations. These studies do not supply an expected effect size for this model. [Yao & Koller, NAACL 2024](https://aclanthology.org/2024.naacl-long.25/) · [Zhou et al., EMNLP 2023](https://aclanthology.org/2023.emnlp-main.898/)

**Teach the actual computation inside the complete question.** A life-stage derivation can expose the day stem, start, direction, branch, signed distance and resulting stage. A Ten-God derivation can expose the correct main hidden stem, element relationship and polarity. Brief operation-first planning is a competing addition to this training, particularly for cross-wording errors. Nye et al. found that supervised intermediate computations improved algorithmic performance on arithmetic and program execution. The Phase 7 T treatment already named the branch but did not consistently spell out the unobserved calculation. A matched ordinary-derivation arm can test the value of any extra supervision. [Nye et al., 2021](https://arxiv.org/abs/2112.00114)

**Direct allocation to a verifier or a stronger teacher.** Step-supervised verification produced gains on MATH in Lightman et al.’s setting. BaZi already provides an exact generator and tables, so the first efficient experiment can supervise the known intermediate values directly. A verifier that chooses among multiple completions changes inference cost and the evaluated system. It deserves a separate same-budget comparison if single-response training continues to plateau. A frontier-model baseline could measure an actual capability gap if the same privately held prompts and decoding rules are applied. Phase 7–8 provide no measured frontier comparison. [Lightman et al., 2023](https://arxiv.org/abs/2305.20050)

**Measure controlled transfer.** Familiar and newly worded cases, isolated and composed cases, and same-chart changes in requested operation probe different failures. Fresh combinations and separately authored language matter because Phase 8 used all 12 reserve edges. Hold out entire source–operation combinations, count distinct charts as statistical units, and identify changes in component exposure. Keysers et al. formalized the distinction between shifts in compound distributions and shifts in elementary components. Our simple coverage audit implements one descriptive check on that distinction. [Keysers et al., ICLR 2020](https://arxiv.org/abs/1912.09713)

The strongest focused comparison holds the starting weights, optimization budget and retention replay fixed, tests broad training coverage against a matched sparse distribution, then tests explicit planning against the same broad inputs. Because full coverage changes which examples are possible, the corresponding treatment estimates a training-data policy as a whole. Complement accuracy with matched-chart correctness when the requested operation changes and with day-dependent class and operand breakdowns. Generalization evidence comes from the held-out multi-step answers.
