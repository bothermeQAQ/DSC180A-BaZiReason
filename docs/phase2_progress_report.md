# BaZiReason Phase 2 progress report: a locally trained, verifiable BaZi calendar model

Yonghao Wang · DSC 180A/B Capstone, Domain D25 · 11 November 2026
Branch `phase2/local-mlx-training`. The submitted checkpoint (tag `checkpoint-2026-11-06`) is unchanged.

All numbers below come from files under `results/phase2/`; the full tables are in [`results/phase2/RESULTS.md`](../results/phase2/RESULTS.md). Every model result is real local inference on an Apple M5 MacBook Pro with 32 GB. Nothing here is a mock.

## 1. Summary

| Question | Answer |
|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f` (Apache-2.0), quantized to 4 bits here, fine-tuned with LoRA (rank 16, last 16 of 36 layers, 14.68 M trainable parameters) |
| Did real training run? | Yes, twice. Round 1: 2,171 iterations (1,085 optimizer updates), 131 min, validation loss 2.011 → 0.0097. Round 2 (warm start on revised data): 1,714 iterations (857 updates), 100 min, validation loss 0.086 → 0.0067. Peak memory 22.3 GB |
| Training examples | 4,343 (round 1) and 3,429 new ones (round 2); 5.9 M sequence tokens processed in total |
| Selected adapter | `artifacts/adapters/bazireason-qwen3-4b-lora-v2/adapters.safetensors`, SHA-256 `4a61e636…ded8f418`, chosen over the round-1 adapter on a validation split (64.0% vs 50.7% of items correct) |
| Main gains over the untuned model | Held-out items correct 8.4% → 72.0%; exact four pillars 0/190 → 152/190; benchmark_v3 with visible reasoning 0/225 → 204/225 exact with tool facts and 156/225 without; corrupted charts detected 0/39 → 38/39; definite personal predictions declined 1/20 → 20/20 |
| What is still weak or got worse | Strict one-pass JSON without the answer in context: still 0/225 exact. A corrupted tool answer in that format is still followed 98.8% of the time. The model still gives a definite chart for 29-41% of requests that have none. Round 2 asks for a tool less often when it should. Rule look-ups are right only about half the time |
| BaziQA contest accuracy | Fine-tuned: 30.4% without tools and 31.7% with verified tools on all 555 questions (chance 24.7%). Untuned: 8.3% and 20.7%, mostly because it runs out of its token budget before answering |
| Versus published baselines | Hosted frontier models in the BaziQA paper: 32-37% on 2021-2025. This model: 26.5-30.7% on those years. Human contest champions: 55% on average. Not like-for-like (see section 6) |
| Research contribution | A measured reliability gain that is attributable: tools alone do nothing for the untuned small model, fine-tuning alone gets 69%, the two together 91% on an independent benchmark, with the known failure modes quantified |
| Run the chatbot | `.venv-mlx/bin/python -m bazireason.demo_app`, then open `http://127.0.0.1:8765` |

## 2. What was built

- **Rule knowledge base** (`src/bazireason/rules.py`, exported to `data/phase2/knowledge/rules_v1.json`): 18 rule cards in three explicit kinds (12 deterministic tables, 5 declared conventions, 1 traditional association). Every table that lunar-python also ships is checked against it entry by entry.
- **Verified chart tool** (`charttools.py`): the unchanged Phase 1 engine plus derived facts (Ten Gods, hidden stems, relations, seasonal state, luck cycles). It returns a status and no chart when the engine does not answer.
- **Worked answers** (`worked.py`): every training answer re-derives each pillar by formula and must equal the engine's result. A new day-number formula was checked against the independent oracle for every day from 1901 to 2099.
- **Datasets** `p2_train_v1` and `p2_train_v2` (`data/phase2/`), generated and labelled by code, with year-block splits and benchmark and contest isolation.
- **Local training and inference** (`mlx_backend.py`, `scripts/phase2/train_lora.py`): MLX-LM 0.32.2, logged, resumable, with validation-based checkpoint selection.
- **Three evaluations**: a held-out split of the training distribution, the Phase 1 `benchmark_v3` with its own prompts and scorer, and the BaziQA contest benchmark in four conditions.
- **Chat demo** (`demo_app.py`): engine-verified charts, a switch between base and fine-tuned model, and a deterministic check of each model reply.
- 28 new tests (204 in total, all passing); Ruff clean.

## 3. Model selection

Five candidates were measured on this machine ([phase2_model_selection.md](phase2_model_selection.md)). The two newest families, Qwen3.5 and Gemma 4, ran out of GPU memory during LoRA back-propagation at our sequence lengths, so they were ruled out on evidence rather than preference. Of the two that train, Qwen3-4B-Instruct-2507 was the only one with perfect strict-JSON output and the only candidate that always ended a reasoned reply with the requested final answer. MiniCPM5-2B is the documented, faster fallback.

## 4. Data

`p2_train_v1`: 4,343 training, 261 validation and 633 test rows from disjoint year blocks. Fourteen task families cover ordinary charts, solar-term boundaries, Zi-hour transitions, time zones and daylight saving, missing and invalid input, chart verification (half of the supplied charts are wrong in one of ten named ways), internal contradictions, twelve rule tasks, counterfactual comparisons, requests for personal predictions, contest-style rule questions and multi-turn dialogues. Details and leakage guards: [`data/phase2/README.md`](../data/phase2/README.md).

Two design decisions matter for interpreting the results:

- **No training on facts the model cannot know.** Without tool output, the expected answer is `needs_tool` whenever the result depends on the instant of a solar term or on an uncertain UTC offset. The model is never asked to recite such a fact.
- **No invented outcomes.** Requests such as "when will I marry" have one kind of target: what the chart shows, what the tradition associates with it, and a statement that the event cannot be verified.

No BaziQA question, option or answer was used for training, validation, checkpoint selection or prompt design. A test confirms that no contest question text occurs in either dataset, and contest birth dates (dates only) are excluded from generation.

## 5. Training

| | Round 1 | Round 2 |
|---|---|---|
| Starting weights | base model | selected round-1 adapter |
| Data | `p2_train_v1`, 4,343 rows | `p2_train_v2`, 3,429 new rows |
| Iterations / optimizer updates | 2,171 / 1,085 | 1,714 / 857 |
| Peak learning rate (warm-up, cosine decay) | 2e-4 → 2e-5 | 1e-4 → 1e-5 |
| Sequence tokens / answer tokens in the loss | 3.34 M / 1.07 M | 2.55 M / 0.92 M |
| Wall time | 131 min | 100 min |
| Peak memory | 22.3 GB | 22.3 GB |
| Validation loss (120 fixed rows) | 2.011 → 0.0097 | 0.086 → 0.0067 |
| Selected checkpoint | iteration 2,000 | iteration 1,714 |

Loss curves: `results/phase2/training/<run>/loss_curve.png`; every report line is in `metrics.jsonl`.

**Why a second round.** A validation loss of 0.01 looked finished, but generation on the held-out split showed the round-1 model exact on only 7 of 84 no-tool charts. The saved replies located the cause: every modulo line (132/132) and every day-of-year line (66/66) was right, but only 9 of 66 lines that add four numbers in one step were. The reference answers had asked for an operation the model could not do reliably. Round 2 writes that sum as single additions, states each stem's polarity before naming a Ten God, adds no-tool cases just inside and outside the solar-term windows, and weights abstention and rule tasks more. Because the first test split had been read to find this, a fresh test split of 486 items was generated for round 2 and scored only after training.

## 6. Results

### 6.1 Held-out split (unseen years; identical prompts for both models)

| Metric | Untuned | Round 1 | Round 2 |
|---|---:|---:|---:|
| Items correct (633) | 8.4% | 57.5% | **72.0%** |
| Exact four pillars, with tool facts | 0/106 | 93/106 | 89/106 |
| Exact four pillars, no tools | 0/84 | 7/84 | **63/84** |
| Clarification decision when the request is ambiguous | 14/47 | 27/47 | 33/47 |
| Invalid or unsupported request recognised | 6/34 | 21/34 | 29/34 |
| Asks for a tool when the answer needs one | 0/26 | 18/26 | 14/26 |
| Definite chart for a request that has none (lower is better) | 0/107 | 40/107 | 31/107 |
| Corrupted chart detected / wrong pillars repaired | 0/39, 0/39 | 39/39, 31/39 | 38/39, 34/39 |
| False alarm on a sound chart (lower is better) | 0/33 | 10/33 | 2/33 |
| Abandoned its own correct answer after pushback (lower is better) | 12/25 | 0/25 | 1/25 |
| Declined a definite personal prediction | 1/20 | 20/20 | 20/20 |
| Rule tasks correct | 11/96 | 29/96 | 47/96 |

Round 2 minus untuned: +63.7 points [59.8, 67.7]; round 2 minus round 1: +14.5 points [10.8, 18.3] (cluster bootstrap over task groups). On the fresh split that nothing was tuned against, round 2 scores 69.1% against 50.6% for round 1 (+18.5 points [14.4, 23.0]). Accuracy is the same on question wordings never seen in training (71.8%) as on seen ones (72.2%).

The untuned model's low score is partly a format effect: only 19% of its replies end with a parseable answer, because it writes long derivations and reaches the 700-token limit. Its zeros on exact charts are not a format effect; none of its parsed charts is right.

### 6.2 benchmark_v3 (the Phase 1 benchmark; definitions unchanged)

This is the first real language-model evidence for H1-H3. The benchmark's 261 cases were never used in training; training dates are kept at least seven days away from every case.

**Strict protocol** (the checkpoint's prompts; the whole output must be one JSON object):

| Condition | Untuned exact | Round 2 exact | Untuned schema-valid | Round 2 schema-valid |
|---|---:|---:|---:|---:|
| direct | 0/225 | 0/225 | 0.0% | 100.0% |
| tool_intermediate | 0/225 | 0/225 | 1.5% | 99.6% |
| tool_context (answer supplied) | 224/225 | 225/225 | 92.3% | 100.0% |
| tool_corrupted (answer with one error) | 0/225 | 0/225 | 95.4% | 100.0% |

- **H2 (primary), strict:** intermediate facts minus direct = 0.0 points for every model. In one pass without visible working, neither model can turn facts into pillars.
- **H3 (primary), strict:** explicit minus minimal policy = 0.0 points on the 16 knowledge-dependent families for every model.
- **H1, strict:** floor effect; no model is right often enough to show a gradient.
- **Negative result.** A corrupted tool answer is followed 100% of the time by the untuned model and 98.8% by both adapters. Fine-tuning fixed the format (non-answer policy with a correct tool answer: 71% → 100%) but did not teach one-pass detection of a wrong tool answer.

**Reasoned protocol** (same prompts plus one sentence allowing visible reasoning before the final JSON; the addition planned in `docs/research_design.md`):

| Condition | Untuned | Round 1 | Round 2 |
|---|---:|---:|---:|
| direct, exact | 0/225 | 21/225 | **156/225** |
| tool_intermediate, exact | 0/225 | 191/225 | **204/225** |

- **H2 (primary), reasoned:** +75.6 points [69.2, 82.0] for round 1 and +21.3 points [13.1, 30.1] for round 2. The tool gain shrank in round 2 because the model can now do the date arithmetic itself; what the tool still adds is exactly what cannot be computed: solar-term instants and time-zone offsets.
- **H1, reasoned, round 2 direct:** on Jie ladders the boundary pillars are right in 50.0% of near cases and 85.7% of far cases (−35.7 points [−46.4, −25.0]); on Zi ladders 75% near and 100% far. Near a solar term the model without facts is at coin-flip level, as H1 predicts.
- The untuned model gains nothing from permission to reason: it still answers with a one-line guess in an invalid schema.

**Attribution on benchmark_v3 (exact four pillars, reasoned protocol):** tools alone 0%, fine-tuning alone 69%, both 91%. Tool-only improvement is zero for this base model, so the gain cannot be credited to tools; the last 21 points require both.

### 6.3 BaziQA contest benchmark

BaziQA commit `3bad3de` (data identical to tag v2.3.0): 555 questions about 117 real birth records from 17 contests, of which 290 come from setters who used BaZi and 365 have an official answer key. Each question was asked alone, three sampled runs per condition, 256 new tokens, original option order, no answer in the prompt. An unanswered reply is wrong. Intervals resample persons, because questions about one chart are correlated.

| Condition | All 555 | 95% interval | BaZi-only 290 | Official keys 365 | Unanswered |
|---|---:|---|---:|---:|---:|
| A. untuned, no tools | 8.3% | [6.6, 10.2] | 8.7% | 8.6% | 72.3% |
| B. untuned, verified tools | 20.7% | [17.6, 23.7] | 21.0% | 21.1% | 41.0% |
| C. round 1, no tools | 31.5% | [28.5, 34.6] | 30.3% | 31.4% | 3.7% |
| D. round 1, verified tools | 30.5% | [27.6, 33.3] | 29.2% | 30.6% | 4.8% |
| C2. round 2, no tools | 30.4% | [27.3, 33.4] | 27.6% | 30.2% | 1.4% |
| D2. round 2, verified tools | 31.7% | [29.1, 34.4] | 31.1% | 30.0% | 1.3% |
| Chance | 24.7% | | 24.9% | 24.7% | |

How to read this:

- **The fine-tuned model is about six points above chance, and the interval excludes chance.** That is a small effect.
- **Most of the gap to the untuned model is answering at all.** The untuned model ran out of its budget on 41-72% of questions. On the questions it did answer it was right 30-35% of the time, the same range as the fine-tuned model. The data therefore do not show that fine-tuning improved contest judgement; they show it made the model answer concisely. SENSITIVITY_PLACEHOLDER
- **Verified charts did not help the fine-tuned model:** +1.3 points [−1.5, +4.1] in round 2 and −1.0 [−4.3, +2.3] in round 1. A correct chart is not what limits contest accuracy.
- **The score does not track the method.** Across both adapters, accuracy was 26-29% on questions whose setter wrote "八字" and 33-39% on questions set with Zi Wei Dou Shu, for which our BaZi chart is the wrong instrument. This suggests the above-chance part comes from which answers are plausible for a life in general, not from reading the chart.
- **Comparison with people and with published systems.** Contest champions averaged 55.4% on the twelve contests with an archived score; this model scores about 30% on the same questions. The BaziQA paper reports 32.1-36.7% for five hosted frontier models on 2021-2025; this 4 B local model scores 26.5% (no tools) and 30.7% (tools) on those years. The paper used an earlier data version, its own charts and a multi-turn flow, so the rows are context, not a ranking.
- **What did transfer:** chart derivation on real birth records. Asked to derive the 117 charts without tools, the untuned model produced none, round 1 produced 11 exact charts and round 2 produced 80 (of 116 with a verified chart; see section 7 on over-answering). In the saved explanations, checkable statements (named pillars, day master, year pillars, Ten Gods) agree with the verified chart 94% of the time with tools and 47-70% without.
- **Contamination.** These questions and answers have been public for years and may be in the base model's pretraining data, so even the six points above chance are not clean evidence of reasoning. Correct answers here are not evidence that destiny can be predicted.
- **Explanations.** The contest publishes answers but no explanations, so reasoning was not compared with an official rationale. Answer accuracy and the audit of checkable statements are reported separately.
- Our engine's charts agree with BaziQA's published charts for 115 of 116 persons; the one difference is a 1973 Kuala Lumpur birth, where UTC+7:30 moves the hour across a branch boundary under our fixed UTC+08:00 profile. Eight records have no time zone; their tool block carries a chart under the labelled assumption that the written time is UTC+08:00.

Tables by contest year, question category, setter's method and person are in `RESULTS.md` section 4.

### 6.4 Where the improvements come from

| Outcome | Tools alone (untuned + tools) | Fine-tuning alone | Both |
|---|---|---|---|
| Exact four pillars, benchmark_v3, reasoned | 0/225 | 156/225 | 204/225 |
| Exact four pillars, held-out split | 0/106 | 63/84 | 89/106 |
| Exact four pillars, boundary cases (solar term or Zi hour), held-out split | 0/41 | 20/27 | 29/41 |
| Using a correct supplied answer, strict JSON | 224/225 | n/a | 225/225 |
| Rejecting a corrupted supplied answer, strict JSON | 0% | n/a | 1.2% |
| BaziQA, all questions | 20.7% | 30.4% | 31.7% |

The held-out "fine-tuning alone" and "both" columns use different requests (the no-tool rows and the with-facts rows of the split), so that row shows levels, not a paired difference.

## 7. What got worse, and what did not work

1. **Asking for a tool.** Round 2 is more willing to compute when it should ask: needs-tool decisions fell from 18/26 to 14/26 on the first split and from 9/30 to 6/30 on the fresh one. On the 117 contest birth records its own policy allows a chart for 41 (67 need a tool for the UTC offset or a solar-term instant, 9 lack a time zone or a time). Round 1 gave 56 charts and round 2 gave 102; 80 of those were right, but for most of them it should have asked.
2. **Definite charts for requests that have none** remain at 29% (first split) and 41% (fresh split). Unspecified sect at 23:xx, a repeated daylight-saving hour and a minute that contains a solar term are the usual misses.
3. **False refusals** rose from 0.5% (untuned, which rarely concludes at all) to 6.8%.
4. **Rule look-ups are unreliable.** 47/96 on the first split and 37/72 on the fresh one. Typical errors are a wrong polarity or direction in a Ten God, and missed relations in a pairwise scan. The same question in three wordings gets three identical answers for 46 of 84 questions, and three correct answers for 27.
5. **Strict one-pass computation did not improve at all** (section 6.2), and corrupted tool answers in that format are still followed.
6. **Free-form questions are brittle.** In the demo, a question phrased unlike the training data produced a made-up "traditional" statement. The demo now phrases requests in trained formats and checks each reply against the engine (section 9).
7. **Contest judgement** did not measurably improve (section 6.3).

## 8. Research contribution

The differentiating claim is reliability that can be measured and attributed, not fortune-telling ability:

1. **Verified computation with a stated profile.** Charts come from a tested engine; the model's own derivations reach 91% exact with tool facts on an independent benchmark and 69% without, from 0% for the untuned model in both cases.
2. **Detection and correction of chart errors** when the model may show its work: 38/39 corrupted charts detected, 34/39 repaired, 2/33 false alarms. In one-pass JSON the same model still follows a corrupted tool answer, which bounds the claim.
3. **Explicit uncertainty**, partly achieved: clarification decisions 30% → 70%, invalid and unsupported requests 18% → 85%, definite personal predictions declined 5% → 100%, with the remaining over-answering quantified.
4. **Counterfactual comparisons** at 89-97% on held-out pairs.
5. **A first test of H1-H3 with a real model**, including a clean H1 pattern (coin-flip near a solar term without facts) and an H2 effect whose size depends on what the model can already compute.
6. **An attribution result:** for this small model, tools alone do nothing, fine-tuning alone does most of the work, and tools add what cannot be computed.
7. **A negative transfer result:** better calendar reasoning did not improve agreement with contest answer keys.

## 9. Demo

`python -m bazireason.demo_app` serves a local page (standard library only, no remote resources, bound to 127.0.0.1). A tester enters date, time, time zone, sect and optionally gender; the left side shows the engine's chart and a list of what could change it (a nearby solar term, an hour boundary, the 23:00 sect dependence, daylight-saving conversion). The right side is a multi-turn chat with a switch between the fine-tuned adapter and the untuned base model. A second birth form compares two scenarios. Relevant rule cards are listed beside each answer.

Because the model still makes rule slips, every reply is checked: named pillars, day master, year pillars, Ten Gods and luck direction are compared with the engine, and for recognised rule questions the structured answer is compared with the rule tables and the table's value is shown. Nothing is stored unless the tester ticks a consent box. Sharing options are in [phase2_training.md](phase2_training.md), section 6.

## 10. Limits and threats to validity

- **One base model, one seed per round, one machine.** No variance across training seeds was measured.
- **Templated data.** The held-out split shares the generator and answer style of the training data. It tests unseen dates, events and some wordings, not open conversation. benchmark_v3 is independent in cases but shares prompt formats that were also used in training.
- **Test reuse.** The first test split informed the round-2 data change. The fresh split and benchmark_v3 were not used that way; round-2 claims rest on them.
- **Untuned baseline and budgets.** The untuned model loses many items by not concluding within the token limit. Its zero on exact charts is real; its BaziQA and overall held-out scores understate what it could do with a longer budget.
- **Single-engine labels.** All labels inherit the fixed UTC+08:00 profile of the Phase 1 engine. This is not local or true-solar-time charting.
- **Conventions.** The pattern label and strength count are declared simplifications. No domain expert has reviewed the rule statements or their attribution.
- **BaziQA.** Public data, possible pretraining contamination, answer keys of mixed provenance (365 official, 190 from a repost that matches every official key that could be checked), and a category scheme based on keywords written for this project.

## 11. Next steps

1. Train abstention directly: more cases where the right answer is to ask, and a penalty-aware evaluation of over-answering.
2. Make rule look-ups reliable: more rule rows with explicit table steps, or let the model call the rule tool and only explain.
3. A function-calling arm, so "with tools" means the model asks for facts instead of receiving them.
4. Two or three training seeds, and a second base model (MiniCPM5-2B) for a size comparison.
5. A small human-rated study of explanation quality with consented testers, using the demo's feedback path.
6. A blind BaziQA submission before the 2027 answer key is published, which removes the contamination concern.

## 12. Reproduce

Commands for environment, model download, dataset checks, both training rounds, every evaluation and the demo are in [phase2_training.md](phase2_training.md). Adapter weights are not in Git (59 MB each); their SHA-256 values are in `results/phase2/training/*/training_summary.json`.
