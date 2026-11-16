# BaZiReason Phase 6 report: what the model can do alone

Yonghao Wang · DSC 180A/B Capstone, Domain D25 · draft, November 2026
Branch `phase6/model-capability`. Every earlier adapter, dataset and result is unchanged. Phase 6 writes only to `data/phase6/`, `configs/phase6/`, `results/phase6/`, `scripts/phase6/`, new modules in `src/bazireason/` and new adapter folders.

All numbers come from files under `results/phase6/`. The full tables are in [`results/phase6/RESULTS.md`](../results/phase6/RESULTS.md), built by `scripts/phase6/make_report6.py` from saved records; it runs no model. Every model result is real greedy inference with MLX on an Apple M5 MacBook Pro (32 GB). Every score is an exact comparison with the rule code, the calendar engine or lunar-python. Intervals are paired 95% bootstrap intervals over row groups. The design was frozen in [`phase6_preregistration.md`](phase6_preregistration.md) before any Phase 6 model was trained; its section 9 lists the later changes, each written before the result it could affect. The working log is [`phase6_log.md`](phase6_log.md).

**Every accuracy in this report names its level.** L0 is the model answering alone and is the headline. L5 is the Phase 5 hybrid system, in which rule code computes and repairs the answer; it is reported last and apart, and it is never called model accuracy.

<!-- RESULTS: summary table goes here once the sealed runs exist -->

## 2. What was built

| Piece | What exists now |
|---|---|
| Audit reproduction | `scripts/phase6/audit_repro.py` recomputes every number of the capability audit from saved records (`results/phase6/audit_repro.json`); all agree |
| New evaluation sets | `scripts/phase6/build_eval6.py`, `gen_blind6.py`: development (`dev6`), validation (`valid6`) and sealed (`sealed6`) sets in their own year blocks, with two new item types: held-out combinations of trained rules, and counterfactual pairs |
| Natural questions | Evaluation questions from an untuned Qwen3.5-4B (the Phase 5 writer); training questions from the untuned Qwen3-4B with other descriptions and personas. No exact overlap (tested) |
| Evaluation ladder | `bazireason.ladder`, `scripts/phase6/run6.py`: the same items at L0 (alone), L1 (raw facts), L2 (rule text), L3 / L3w (right / wrong conclusion), L4 (self-check, three variants); `run_l5.py` for the Phase 5 system |
| Small-step derivations | `bazireason.steps6`: every computed step written out (旬空 by walking back to the 甲, element counts per character with a sum check, pattern membership, relation look-ups, flow-year arithmetic). Checked on 7,200 derivations: every final JSON equals the rule code's, and the claim checker finds no wrong statement |
| Training data | `bazireason.train_data6`, `scripts/phase6/build_train6.py`: exact token budgets through the tokenizer, fresh training-year charts, every evaluation chart excluded. Each set rebuilds bit for bit (`--check`) |
| Decision scripts | `scripts/phase6/decide6.py`: the frozen screening rule (section 5 of the pre-registration) and selection rule (section 6) |
| Diagnostics | `diag_fit.py` (fit to the training distribution), `own_replies.py` (a model's own replies on fresh training-distribution questions), `truncation6.py`, `panel6.py` (checkpoint panel), `probe_memory.py` |
| Tests | `tests/test_phase6.py`: manifests and hashes, blind rebuild, year blocks, training leakage, the seal, no answer at L0-L2, the scorers, the derivations, combinations and counterfactual pairs |

## 3. Design

### 3.1 Questions

| Hypothesis | Tested by |
|---|---|
| H1: sequential training rounds cause the trade-off between rule and calendar skill | U0 (all earlier data, one joint run from the base) against A1 (rules) and Phase 2 (calendar) |
| H2: coarse procedure steps cause the computed-step errors | S1 (small steps) against U0 |
| H3: narrow wording causes the natural-question gap | S2 (S1 plus natural wordings) against S1 |
| H4: the model can learn to use evidence and check itself | S3 (S2 plus evidence and self-check rows) against S2 |
| H5: adapter capacity limits the result | U1-hi against U1, only if U1 under-fits |
| H6: the base model limits the result | accuracy on fresh training-distribution questions |
| H7: at equal compute, a short system prompt gives a better model | S2s (S2's recipe, seed and 6.0 M-token budget with a 36-token system prompt) against S2 |
| H8: continuing with the failing question types weighted up improves the model further | each continuation (U3-U10) against its parent |
| H9: writing the steps S1 gets wrong out further fixes them | U3v (derivation version 2) against U3: the same parent, weights, seed and budget |
| H10: the base model's size limits the result | S1-8B (S1's training file and settings on Qwen3-8B) against S1 |

H7 and H8 belong to a second round that was added on 14 November, while S1 was training and before any screening or validation result existed (pre-registration, section 10). H9 was added the same evening after S1's error analysis, before any other screening result and before any round-2 run (section 10.5). H10 was added on 15 November, after the user approved downloading Qwen3-8B and before any run on it (section 11).

**The plan was revised once after the screening** (pre-registration section 12, recorded before any of its runs existed). S3's added components gave no gain alone and lowered calendar and combination accuracy, so the runs that the final model comes from use S2's recipe; the registered U1, with S3's recipe, is still trained. H7 became a twin at 6.0 M tokens, which decides the prompt of the 15 M-token run. The same section records a failed data build on 15 November: one new training question was on an evaluation chart, the check stopped the build, and no result was affected.

### 3.2 Data

| Part | dev6 | valid6 | sealed6 | What it is |
|---|---:|---:|---:|---|
| rules | 294 | 289 | 462 | Phase 3 rule families with an exact rule-code plan; one wording per family held out of training |
| indep | 140 | 140 | 280 | labels from lunar-python, in a chart format and wordings never used in training |
| blind | 188 | 184 | 379 | natural questions from an untuned Qwen3.5-4B |
| combo | 100 | 100 | 160 | held-out combinations of trained rules: flow year plus life stage, 旬空 plus positions, revealed hidden stems, Ten-God counts |
| cf | 100 | 100 | 160 | counterfactual pairs: one character of the chart changed, and the answer changes |
| calendar | 219 | 269 | 816 | chart requests with answerable and non-answerable twins |

- **Year blocks:** dev6 uses one year of each validation triple, valid6 the other two, sealed6 the test block. Training charts come from the training years.
- **Exclusions:** every chart of an earlier evaluation split is excluded from valid6 and sealed6, and every evaluation chart is excluded from Phase 6 training.
- **Seal:** sealed6 was opened once, after `results/phase6/selection.json` existed; every access is logged in `results/phase6/sealed_access.jsonl`.

### 3.3 Levels

| Level | The model receives |
|---|---|
| L0 | the question and the four pillars |
| L1 | plus each symbol's element, polarity and hidden stems; no derived fact (checked mechanically) |
| L2 | plus generic rule text chosen by question type, the same for every chart |
| L3 / L3w | plus the rule code's conclusion / the same words with a wrong conclusion |
| L4 | its own L0 reply and a check request: (a) re-check; (b) "there is an error"; (c) "this field is wrong". No value is given |
| L5 | the Phase 5 hybrid system: router, supplied conclusion, value repair, engine fallback |

No repair, fallback, loop guard or routing touches L0-L4. Decoding is greedy, with 1,100 new tokens for rule items and 700 for calendar rows.

### 3.4 Training runs

The runs of the first round start from the untuned base with the earlier adapter settings (LoRA rank 16, scale 2, last 16 of 36 layers), learning rate 2e-4 with cosine decay to 2e-5, batch 2 with gradient accumulation 2, one pass, sequences of at most 2,048 tokens.

| Run | Budget | Data |
|---|---:|---|
| U0 | 6.0 M tokens | the union of every earlier training file, in proportion |
| S1 | 6.0 M | small-step procedures on fresh charts in template wordings; fresh calendar rows; 12% earlier other tasks |
| S2 | 6.0 M | S1 plus natural wordings (45% of procedure rows), other chart formats and contrast rows |
| S3 | 6.0 M | S2 plus evidence rows, reference checking, self-check rows from S1's own replies, matched calendar twins |
| U1 | 15 M | the recipe chosen by the frozen screening rule (S3's: nothing was dropped); trained after the chain |
| U1-hi | 15 M | U1's data with all 36 layers, only if U1 under-fits |
| S2s | 6.0 M | S2's recipe with the short system prompt: 11,845 rows against 8,161 in the same budget |
| U2 | 15 M | S2's recipe, the best of the screening, with the system prompt that the S2s twin decides |
| U3-U10 | 6.0 M each | continuations of the latest run, with the question types reweighted by its measured errors; learning rate 1e-4 |
| U3v | 6.0 M | U3 with derivation version 2 (the fork) |

**The screening is additive:** each S run adds one component to the previous run. A component's effect is measured in the presence of the earlier ones, so interactions are confounded with the order in which the components were added.

**Round 2** rests on two measurements made before it was designed.

- **Compute went to a fixed prompt.** The main system prompt (279 tokens) is 34% of the tokens of the S1 training file, and the final reply, the only part with a training loss, is 43% (`results/phase6/token_share.json`). U2 replaces it with a 36-token prompt; the prompts of the strict-format calendar tasks and of the multiple-choice task stay, because they define another output format. A model is always evaluated with the system prompt it was trained with.
- **The models under-fit.** A1 answers 80.8% of its own training rows, and U0 with 6.0 M tokens is 20 points below A1. The continuations therefore add training where the parent measurably fails: the parent answers fit sample B (100 template questions per question type on fresh charts, never training input), and each type's weight is its recipe weight times max(error, 0.05).

- **S1's errors are specific.** Its replies were compared with the rule code's derivations step by step (`scripts/phase6/first_error.py`). On training wordings, the wrong answers start at a few steps that ask for a relation between two elements, or a filtered list, without writing the table first. Derivation version 2 writes those steps out; every answer is unchanged.

The chain has at most eight levels and ends after two levels in a row without a new best on dev6. Every round-2 run is a selection candidate, and the sealed split stayed closed until the round ended.

## 4. Development results (dev6)

These are development results: dev6 was read freely to make decisions. The claims of this report rest on the sealed split (section 5). Every number is in [`results/phase6/RESULTS.md`](../results/phase6/RESULTS.md); intervals are paired 95% intervals.

### 4.1 The model alone (L0)

| Model | Training | All rule items (822) | Held-out combinations (100) | Calendar (219) |
|---|---|---:|---:|---:|
| Untuned base | none | 14.1% | 38% | 1.8% |
| Phase 2 | calendar, two rounds | 23.6% | 23% | 74.0% |
| A1 | six sequential rounds | 60.5% | 48% | 62.1% |
| U0 | the earlier data in one joint run, 6.0 M tokens | 40.3% | 40% | 62.1% |
| S1 | small steps, 6.0 M | 66.8% | 36% | 58.4% |
| **S2** | S1 plus natural wordings, 6.0 M | **79.9%** | 50% | 63.5% |
| S3 | S2 plus evidence, self-check and calendar twins, 6.0 M | 77.6% | 38% | 53.0% |
| S2s | S2's recipe with the short prompt, 6.0 M | 80.0% | 55% | 62.6% |
| S1-8B | S1's data on Qwen3-8B, 6.0 M | 66.4% | 41% | 63.0% |

Four hours of training from the untuned base on the new data (S1) passed the end of the earlier six-round chain (A1). S2, which adds natural wordings, is 19.5 points above A1 [15.8, 23.4].

### 4.2 What each change did

| Hypothesis | Comparison | Rule items, points [95%] | Reading |
|---|---|---|---|
| H1 sequential rounds | U0 minus A1 | −20.2 [−23.9, −16.7] | one joint run on the earlier data, at about a third of the tokens, reproduces neither specialist (calendar: 11.9 below Phase 2) |
| H2 small steps | S1 minus U0 | +26.5 [+22.2, +30.5] | supported: 旬空 96% right (A1 28%), element counts 98%, life stages 95% |
| H3 wording | S2 minus S1 | +13.1 [+9.7, +16.6] | supported: held-out template wording 37.3% to 71.3%, natural wordings 68.6% to 88.8% |
| H4 evidence and self-check | S3 minus S2 | −2.3 [−5.2, +0.5] | no gain alone; costs below |
| H7 short prompt | S2s minus S2 | +0.1 [−2.9, +3.2] | not supported: 45% more rows in the same budget changed nothing |
| H10 base size | S1-8B minus S1 | −0.4 [−3.7, +3.1] | not supported: the larger base learns the same data no better |

- **What S3's components bought and cost.** Raw facts in the prompt hurt S3 less (L1 minus L0 −1.9 [−4.5, +0.7], against −7.7 [−10.5, −4.9] for S2), and a plain re-check no longer breaks right answers (0 of 278, against 21 of 290 for S2). But it fixes almost none (2 of 184), and the mixture cost held-out combinations (−12.0 [−23.0, −1.0]) and calendar accuracy (−10.5 [−17.1, −3.8]).
- **The frozen screening rule dropped nothing** (it drops a component only when the run is significantly worse on all rule items), so the registered U1 has S3's recipe. The runs that lead to the final model use S2's recipe instead; the change and its reasons were recorded before any of those runs existed (pre-registration, section 12).
- **Two null results changed the plan.** More rows of the same mixture (H7) and a base twice as large (H10) both gave nothing. What separates the models is what the training text makes explicit, not how much of it there is or how large the model is.

### 4.3 Where the remaining errors start

Each reply was compared with the rule code's derivation step by step (`scripts/phase6/first_error.py`).

| Wording of the question | S1 | S2 | S2s |
|---|---:|---:|---:|
| training template wordings (244) | 203 | 200 | 196 |
| held-out template wording (150) | 56 | 107 | 104 |
| natural wordings, written by another model (188) | 129 | 167 | 171 |

- **Unseen wording was S1's largest loss** and S2 removed most of it.
- **On wordings the model knows, about one answer in five is still wrong, and more data did not change that.** The wrong answers start at a few steps of the long derivations: the hidden-stem step of area-of-life questions, the clash step of flow years, the pair and trio steps of relations.
- **These steps have one thing in common: the text must decide, while it is written, where to stop or what to leave out.** The clearest case is the life stage of the day master. S1 answered 13 of 14 counterfactual life-stage items, S2 9 and S2s 4. In the wrong replies the count through the stages is right but ends at the wrong branch. S1's replies wrote all twelve stages and read the answer off.
- **Derivation version 2** removes those decisions: it writes the whole table first and then looks the answer up (pre-registration 10.5 and 12.6; checked on 15,000 derivations, every answer unchanged). A paired fork of the first continuation (U3 against U3v) tests it.
- **Held-out combinations stay near 50%.** The replies start the right sub-procedures and fail where they join. An arm of the second continuation level (U4c) adds composition rows of other kinds; the four held-out kinds stay out of training.

### 4.4 Calendar

Every Phase 6 model is about 10 points below Phase 2 on calendar requests, and the errors are decisions, not arithmetic. S2 refuses almost no answerable request (9 of 145) but gives a definite chart in 50 of the 74 requests where none exists; it answers 13 of 40 day-boundary requests and none of the 8 that need a time-zone tool. S3's matched twins moved the error to the other side (66 answerable requests refused). The continuations therefore weight calendar requests by measured error, in cells of request family and expected status (pre-registration 12.5).

### 4.5 A failed build, and what the audit found

On 15 November the data build for U1 stopped at its leakage check: one of 4,002 new training-writer questions was on a chart that is also an evaluation chart (the same four pillars recur on another date). The queue script did not stop after the failed step and reported completion, so the GPU was idle for about two hours.

- **No result is affected.** No row of any trained set, and no row of the earlier training files, is on a Phase 6 evaluation chart. No candidate had a validation score, and the sealed split was closed.
- **Fixed:** builds leave out and count any row on a held-out chart instead of stopping; every queue step is followed by a stop; both are tested. Dry runs of the later builds then found two more ways a build could have stopped (edited charts in contrast rows, and dialogue rows that come with the calendar data), which were fixed before they could occur.

<!-- RESULTS: section 5 (sealed results) and the sections after it are written from results/phase6/RESULTS.md once the sealed runs exist -->
