# BaZiReason: independent model capability audit and Phase 6 research plan

14 November 2026 · written after Phase 5 (commit `5e9961d`), before any Phase 6 work.
No code was changed, no model was run and nothing was trained for this document. Every number was recomputed from saved records under `results/` and `data/` with read-only scripts, or quoted from a generated results page. Section 5 lists what could not be checked without the GPU.

The companion file [`phase6_execution_prompt.md`](phase6_execution_prompt.md) is the brief for the implementation agent.

---

# Part I. What the models can do today

## 1. Who produced the 98.6%

Phase 5's 98.6% is a system result. The same 498 validation questions were answered by G1 under every condition, so the contributions can be separated on identical items:

| Condition (validation, 498 rule questions; the four pillars are always given) | Correct | Who did the work |
|---|---:|---|
| Untuned base, alone | 64 (12.9%) | the base model |
| G1 alone | 341 (68.5%) | the trained weights |
| A1 alone (best model alone) | 347 (69.7%) | the trained weights |
| G1 with the full fact sheet | 366 (73.5%) | weights plus reading: the sheet already states most conclusions |
| G1 with the code's conclusion in the prompt | 461 (92.6%) | the code computed the answer; the model copied it |
| The same, after the code's repair turn and fallback | 491 (98.6%) | the code |

On the sealed split the same pattern holds: A1 alone 688 of 1,005 (68.5%); G1 with the conclusion supplied 918 (91.3%); after repair and fallback 991 (98.6%). Of the 73 repaired answers, 68 were rewritten by the model after it was handed the correct values and 5 are the engine's own text.

**So the independent reasoning accuracy of the best trained model is about 69%, not 98.6%.** About 23 points of the system's score come from copying a computed answer, and about 7 from the code correcting the model.

## 2. What each phase's training changed in the weights

| Phase | What the weights gained | Evidence |
|---|---|---|
| 2 | A complete worked calendar algorithm | Exact four pillars without tools: 0/84 → 63/84. Its replies write out every intermediate number (year remainder, leap days, day count, mod 60) |
| 3 | Rule procedures in the training wordings | Rule rows in seen wordings: P2 93/514 → R3 283/514. On independent questions with third-party labels R3 is no better than P2 (162 vs 166 of 280) |
| 4 (A1) | Transfer to new wordings; a different refusal prior | Independent questions 162 → 201 of 280; unseen wordings 136 → 161 of 274. Plain chart requests fell from 88 to 42 of 99 on the Phase 5 sealed calendar set |
| 5 (G1) | Copying a supplied conclusion | Conclusion supplied: A1 813 → G1 918 of 1,005. Alone: A1 347 → G1 341 of 498. No gain in independent reasoning |

Training did move the model a long way from the base (12.9% → 69.7% alone). But the last two phases added almost nothing to what the model can work out by itself.

## 3. The limitations, with causes

### 3.1 The model has the tables; it fails on steps that compute

I ran the claim checker over all 1,005 sealed replies of A1 alone and grouped its 4,004 checkable statements by kind.

| Statement kind | Wrong / checked |
|---|---:|
| Day master, hidden stems, pillars, spouse star (pure look-ups) | 0 / 1,884 |
| Flow-year items | 22 / 570 (3.9%) |
| Ten Gods | 9 / 140 (6.4%) |
| Branch relations | 70 / 715 (9.8%) |
| Strength | 10 / 101 (9.9%) |
| Stars (神煞) | 26 / 164 (15.9%) |
| Five-element counts | 24 / 73 (32.9%) |
| Pattern (格局) | 55 / 141 (39.0%) |
| 旬空 | 35 / 51 (68.6%) |

Look-ups are solved. Errors sit in the steps that need arithmetic, counting or matching, and the raw replies show why:

- **旬空.** The trained procedure is a formula done in one line: "序数 = (6×6 − 5×10) mod 60 = −14 mod 60 = 46，旬首为甲寅". The arithmetic is right here and the next jump (46 → 甲辰) is wrong. In other replies the branch index is dropped or the subtraction is wrong. A1 alone gets 15 of 40 on independent 旬空 questions and 2 of 16 in blind wording.
- **Element counts.** The reply lists every character's element correctly, then writes the wrong totals in one jump ("火3" where its own list shows two).
- **Phase 2 is the counter-example.** The day pillar needs far more arithmetic, but there every intermediate value is written out, and the model reaches 75–80% exact pillars.

**Cause: the rule procedures were written with steps too large for a 4B model to do in its head.** This is a data design problem, and it is cheap to change because the rule code writes every derivation.

### 3.2 Composite questions fail at the last step, and drift between procedures

Of A1's 317 wrong sealed answers, 195 contain no wrong checkable statement, 117 contain one, 5 have no readable conclusion, and none is a loop. Most errors are therefore in the final matching or aggregation, where the checker has nothing to check.

Example (blind wording, flow year): the model computes 辛亥 correctly and writes the branch card "亥：六合寅，冲巳…". It then lists "月柱癸巳——无". The fact it needed is on the previous line. The closing lines also switch to the layout of a different procedure (listing hidden stems, as in the domain questions).

| Question type (A1 alone, sealed) | Correct |
|---|---:|
| Single facts | 74.4% |
| Strength / pattern / 用神 by school | 81.0% / 72.4% / 74.1% |
| Relations | 54.2% |
| Flow years | 47.7% |
| Overview | 42.9% |
| Areas of life | 24.2% |

### 3.3 Generalization to natural questions is weak

- **Seen against unseen template wording** (sealed template rules, A1 alone): 158/204 (77.5%) against 114/207 (55.1%).
- **Template against blind wording, same question type** (A1 alone, sealed):

  | Type | Template wording | Blind wording |
  |---|---:|---:|
  | Flow years | 41/70 | 0/16 |
  | Ten Gods | 45/56 | 6/16 |
  | Pattern | 37/42 | 5/16 |
  | Strength | 38/42 | 9/16 |
  | Areas of life | 14/50 | 2/16 |

- **Cause.** All Phase 3–4 rule training used between 2 and 27 distinct question wordings per family (flow 12, pattern 10, domain 10, overview 2), written as formal instructions ("请推出流年干支、流年天干的十神…"). Each composite family had 380–1,070 training examples in total, each seen once.

### 3.4 The model does not use evidence unless the evidence is the answer

- **A1 with the full fact sheet** was no better than A1 alone on development data (506 vs 511 of 769), and worse on template rules (165 vs 191 of 294).
- **G1 was trained on 400 fact-sheet rows** and gains 5.0 points from the sheet (341 → 366 of 498). It gets worse on flow years (25 → 16), Ten Gods (29 → 25), areas of life (12 → 7) and ambiguous questions (17 → 4).
- **The sheet is not "raw facts".** It states strength, pattern, 用神 by school, relations and stars outright. So "derive the answer from verified raw facts" has never been measured in this project; only "find the answer in the sheet" and "copy the conclusion" have.

### 3.5 Self-correction is close to zero

Phase 4's verify-and-repair told the model which fields were wrong, never the values. Recounted from the raw files, on first answers that were wrong:

| Model | Fixed its own error |
|---|---:|
| Phase 2 | 11 / 941 |
| R3 | 17 / 631 |
| A1 | 19 / 556 (3.4%) |

Phase 5's repair works because the code supplies the values. Whether G1 would also copy a wrong conclusion was not tested in Phase 5. The earlier evidence points that way: round 3 followed 98% of the corrupted tool answers in benchmark_v3, and no model corrected more than 21 of 150 on the Phase 4 final test.

### 3.6 "Catastrophic forgetting" is mostly a moving refusal prior

On the 818 sealed calendar requests:

| | Phase 2 | R3 | A1 | G1 |
|---|---:|---:|---:|---:|
| Answerable requests refused (of 532) | 56 | 15 | 198 | 125 |
| Definite chart where none exists (of 286) | 109 | 144 | 22 | 65 |
| Balanced decision accuracy | 74.5% | 72.7% | 73.4% | 73.4% |

Refusals swing from 3% to 37% while balanced accuracy stays within two points. On matched twins (solar-term window, 40 that need the tool and 39 that do not) the four adapters score 5/28, 1/30, 39/1 and 31/6: each answers almost every twin the same way. No adapter has learnt to tell the twins apart; each round moved the prior. The one exception is A1 on solar-term boundaries (65/66 and 31/33), the family its data targeted.

**Cause.** Seven sequential rounds, each one pass at a decaying learning rate, each with a different task mix (v1 → v2 → r1 → r2 → r3 → A1 → G1; 28,147 rows and about 21.7 million tokens in total). The decision families were never trained as balanced twins in one mixture.

### 3.7 Some scores are more generous than they look

- **Ambiguous questions.** The scorer grades only the decision (`determinate: false`, `needs`). A1 alone scores 48/49, but 41 of those replies state a wrong pattern or strength inside the answer.
- **Declines.** A reply that declines in prose but uses a different conclusion label is scored wrong; the three "能不能长寿" rows are labelled "no rule answer" while the router treats them as predictions. The label taxonomy for declines is not clean.
- **Labels.** Only the independent questions have third-party labels (lunar-python). Everything else is labelled by the same rule code that writes the training data and the verification.
- **Blind wording.** The writer is an untuned Qwen3.5-4B, a newer generation of the base model's family, and no person wrote any test question.

### 3.8 BaziQA

Every condition ever run lies between 25.5% and 32.9%; chance is 24.7%. The untuned base with a verified chart gets 31.2%, inside the range of every tuned model. The contest keys are events in real people's lives, and no rule in the engine maps a chart to them, so rule training cannot be expected to move this number. The adapters also loop more on these open questions (A1 94 loops in 1,665 replies with the chart; the base 0). BaziQA should stay a historical, out-of-domain reference. Training towards its keys would be leakage and would imply a predictive claim this project does not make.

### 3.9 Is Qwen3-4B the bottleneck?

There is no evidence yet that it is.
- Look-ups are at 100%, and Phase 2 shows the model can learn a long arithmetic algorithm.
- Loss on held-out rows of each training distribution ends at 0.007–0.02 per token.
- The adapter covers only the last 16 of 36 layers at rank 16 (14.7 million parameters) on a 4-bit base. That is a limit on the adapter, not on the model.
- The other candidates cannot be trained here today: Qwen3.5 and Gemma 4 ran out of GPU memory in back-propagation (Phase 2 bake-off), and a larger Qwen3 needs a download and a memory probe.

One measurement is missing and decides this question: accuracy on fresh samples of the exact training distribution. If that is near 100% and the test is at 69%, the problem is data coverage. If it is also far below 100%, the model is under-fitting and capacity or the base matters.

## 4. Summary of causes

| Limitation | Main cause | Fixable by |
|---|---|---|
| Computed steps wrong | procedure steps too coarse | rewriting derivations into small written steps |
| Composite answers wrong at the end | no explicit matching lines; few examples; procedure drift | finer steps, more examples, contrastive pairs |
| Natural wording fails | about 10 formal wordings per family | large wording variety, held-out wording clusters |
| Evidence ignored | never trained to derive from raw facts | evidence-use tasks without the answer |
| No self-correction | never trained on its own errors | on-policy error-and-fix data |
| Skills trade off | sequential rounds, shifting mixtures | one joint, balanced mixture |
| Refusal prior swings | unbalanced decision data | matched twins in equal numbers |

## 5. What this audit could not check

- Accuracy on the training distribution itself (needs the GPU).
- G1 alone on the sealed split (never run; only on validation).
- Whether G1 follows a wrong supplied conclusion.
- The checker's statement kinds cover fixed-form statements only; "195 errors with no wrong statement" means no wrong checkable statement.

---

# Part II. Recommended strategy

## 6. The decision

**Stop extending the adapter chain. Train one new model jointly from the untuned base on a rebuilt, balanced corpus, and judge it on what it does alone.**

- **Do not keep training G1.** Its last round taught copying, and every further round would shift the mixture again.
- **Keep as fixed baselines:** the base; Phase 2 (calendar reference); A1 (best model alone, the bar to beat); G1 (conclusion-following reference and the Phase 5 system's generator). R3, the merged adapter and earlier rounds stay archived.
- **Unified before multi-expert.** Adapter routing added under one point and interpolation failed its gate, so a single jointly trained model is the better-justified target. Two specialists on the existing adapter bank are the fallback, not the plan.
- **No new base model in Quarter 1,** unless the under-fitting test in section 3.9 says the model, not the data, is the limit.

## 7. Hypotheses and how each is isolated

| | Hypothesis | Test | Predicts |
|---|---|---|---|
| H1 | Sequential rounds cause the skill trade-off | **U0:** joint from base on the union of the existing corpora, same adapter settings and token budget as the chain | calendar no worse than Phase 2 and rules no worse than A1 in one adapter |
| H2 | Coarse procedure steps cause the computed-step errors | same rows, derivations rewritten in small steps | 旬空, counts and pattern statements above 90% right |
| H3 | Narrow wording causes the natural-question gap | add many written wordings per question type; test on held-out wording clusters and a different writer | seen/unseen gap under 5 points; blind composites rise sharply |
| H4 | The model can learn to derive from raw facts and to check itself | add evidence-use and error-and-fix tasks | raw facts help instead of hurting; self-fix rate rises from about 3% |
| H5 | Adapter capacity limits the result | **U1-hi:** the best recipe with more layers or rank | a gain over U1 at equal data; no gain means capacity is not the limit |
| H6 | The base model limits the result | only if fresh in-distribution accuracy stays low after H2–H5 | evidence for a Quarter 2 base-model study |

H2–H4 are screened as additive steps at a reduced budget (about 4 hours each), then the winning recipe is trained once at full budget (**U1**). Additive screening confounds interactions; the report must say so.

## 8. The corpus (all labels from code)

1. **Small-step procedures.** No mental arithmetic and no silent tallies. Examples: 旬空 by stepping back to the 甲 of the decade and naming the two unused branches; element counts as one list per element followed by a sum check; relations as one line per pair that quotes the card entry; flow years with one membership line per pillar.
2. **A curriculum of composition.** Single steps, then two-step chains, then full procedures, with some combinations held out for testing.
3. **Contrastive pairs.** The same chart under two procedures that get confused, and pairs of charts that differ in one character where the answer flips.
4. **Wording variety.** Many natural wordings per question type, written by a local untuned model from plain descriptions, filtered mechanically, labelled by code. Whole wording clusters and the test writer are held out.
5. **Evidence use without the answer.** Raw chart facts, or the relevant rule card, with a mechanical check that the gold answer is not in the prompt. Some prompts contain one corrupted fact that the model must notice.
6. **Self-check from its own mistakes.** Run the current model on training charts, let the code find the first wrong step, and train "find it, redo from there". Mix in right answers that must be confirmed unchanged.
7. **Decisions as matched twins,** in equal numbers, with the distinguishing reason in the target.
8. **Calendar tasks** at a fixed share throughout, so nothing is learnt first and overwritten later.

## 9. Evaluation that cannot hide weakness

Six levels, always reported separately, never averaged:

| Level | The model receives | Measures |
|---|---|---|
| L0 | the question and the four pillars (birth data for calendar tasks) | independent knowledge and reasoning |
| L1 | plus verified raw facts, with no derived conclusion | deriving from evidence |
| L2 | plus the relevant rule text, with no chart-specific conclusion | using a knowledge base |
| L3 | plus the code's conclusion (and a variant with a wrong conclusion) | following, and blind following |
| L4 | its own first answer and a request to check it (no hint; "there is an error"; "this field is wrong") | self-correction: fix rate and break rate |
| L5 | the full system with repair and fallback | application reliability |

- **Primary result: L0 on a new sealed set.** L5 is reported last and labelled as the system.
- **New test material:** new charts in a new year block; questions from a writer not used for training, plus the user's own questions if provided; held-out combinations of rules; counterfactual edits; dialogues whose later turns need new derivations; calendar twins.
- **Scoring fixes:** grade the content of ambiguous answers; one clean label scheme for declines; report statement-level accuracy and the first wrong step.
- **Already-seen sets** (every earlier split, the Phase 5 sealed split, BaziQA) are development or historical material.

## 10. Success, risks and stopping

**Success** (new sealed set, pre-registered before scoring):
- L0 rule accuracy of the new model above A1's with a paired 95% lower bound above 0; the target is +10 points, and +10 on natural wording.
- Calendar not below Phase 2 by more than 5 points, with higher balanced decision accuracy on twins.
- L1 above L0 (evidence helps).
- Self-fix rate at least 25% with at most 2% of right answers broken.
- Replies with a wrong statement at most half of A1's rate.

**Risks.**
- **Longer derivations** cost tokens and add places to slip; per-step accuracy is measured to catch this.
- **A joint run from the base** may trail Phase 2 on calendar if the calendar share is too small. Fallback: start from Phase 2, or keep a calendar specialist.
- **Shared labels** remain: the rule code labels training and testing. More third-party-labelled items reduce this but do not remove it.

**If the plan fails.**
- **Computed steps stay wrong after small steps:** treat in-head computation as a measured limit of a 4B model and move those steps to tool calls in Quarter 2.
- **The joint model cannot hold both skills:** two specialists with routing.
- **Fresh in-distribution accuracy stays low:** a base-model study in Quarter 2.

**Stop training when** two successive full-budget changes each fail to give a paired gain on development data whose interval excludes zero, or when the last run that can still be evaluated before the report must start (27 November).

**Training cost.** About 1.5 million tokens per hour on this machine. U0 and U1 are about 14 hours each, screening runs about 4 hours each, U1-hi about 16 hours: roughly 60 GPU-hours of training plus about 30 of evaluation.

---

# Part III. Quarter 1 roadmap

The final report is due Friday 4 December 2026 (check the time on Gradescope).

| Dates | Work | Output |
|---|---|---|
| 15–16 Nov | Reproduce this audit by script. Measure fresh in-distribution accuracy. Build the six-level harness and the new development, validation and sealed sets. Run the levels for base, Phase 2, A1 and G1 on development data | baseline ladder; pre-registration |
| 16–18 Nov | Build the joint corpus generators with tests. Train **U0** (joint, existing data) | answer to H1 |
| 18–22 Nov | Screening runs: small steps, wording variety, evidence and self-check tasks | the recipe for U1 |
| 22–26 Nov | Train **U1** at full budget, then **U1-hi**. Validation; frozen selection | the candidate model |
| 26–28 Nov | Score the new sealed set once, all six levels, for base, A1, G1, U0 and the selected model | final tables |
| 29 Nov–3 Dec | Report in the course template, README, reproduction commands, Quarter 2 proposal | submission |

**Essential before submission:** the six-level evaluation, the baseline ladder, U0, one redesigned-data model (U1), and the sealed comparison. **Can move to Quarter 2:** U1-hi if time runs out, a different or larger base model, explanations distilled from a stronger teacher, a learnt router, questions and review from people who know BaZi, and the application.

**The research story for the report:** a small local model was adapted to a rule-governed domain; a hybrid system reaches 98.6% but the model alone reaches about 69%; the gap is explained by measurable causes (step size, wording coverage, evidence use, self-correction, sequential training); the next model is tested against each cause, and the limits that remain are stated. Traditional rules are labelled by school and convention, and no result is a claim about predicting anyone's life.
