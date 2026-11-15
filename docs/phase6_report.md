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

All runs start from the untuned base with the earlier adapter settings (LoRA rank 16, scale 2, last 16 of 36 layers), learning rate 2e-4 with cosine decay to 2e-5, batch 2 with gradient accumulation 2, one pass, sequences of at most 2,048 tokens.

| Run | Budget | Data |
|---|---:|---|
| U0 | 6.0 M tokens | the union of every earlier training file, in proportion |
| S1 | 6.0 M | small-step procedures on fresh charts in template wordings; fresh calendar rows; 12% earlier other tasks |
| S2 | 6.0 M | S1 plus natural wordings (45% of procedure rows), other chart formats and contrast rows |
| S3 | 6.0 M | S2 plus evidence rows, reference checking, self-check rows from S1's own replies, matched calendar twins |
| U1 | 15 M | the recipe chosen by the frozen screening rule |
| U1-hi | 15 M | U1's data with all 36 layers, only if U1 under-fits |

**The screening is additive:** each S run adds one component to the previous run. A component's effect is measured in the presence of the earlier ones, so interactions are confounded with the order in which the components were added.

<!-- RESULTS: sections 4 onwards are written from results/phase6/RESULTS.md once the runs exist -->
