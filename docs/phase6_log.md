# Phase 6 working log

Decisions, surprises and negative results, in order. Plan: [`phase6_execution_prompt.md`](phase6_execution_prompt.md). Audit: [`phase6_capability_audit.md`](phase6_capability_audit.md). Frozen design: [`phase6_preregistration.md`](phase6_preregistration.md).

## 14 November 2026

- **Branch** `phase6/model-capability`, from `phase5/grounded-system` at `6e0676f`.
- **Audit reproduced by script.** `scripts/phase6/audit_repro.py` recomputes every number of the audit from saved records; all agree (`results/phase6/audit_repro.json`).
- **New evaluation sets** (`scripts/phase6/build_eval6.py`, `gen_blind6.py`):
  - **Year blocks:** dev6 and valid6 split the validation block (one year of each triple against the other two); sealed6 is the test block.
  - **Exclusions:** earlier evaluation charts are excluded from valid6 and sealed6.
  - **New item types:** held-out combinations of trained rules, and counterfactual pairs.
  - **Natural questions:** written by Qwen3.5-4B for evaluation (188, 184 and 379) and by the untuned Qwen3-4B for training (3,286), with different descriptions and personas; no exact overlap.
  - **Design note:** the counterfactual pairs for strength, pattern and relations reuse a training template wording on purpose; they test a one-character change, not wording. The leakage test checks the blind, combination and counterfactual-fact questions.
- **Day-master questions dropped from the blind sets.** The L0 head line names the day master ("（日主丙）"), so they test nothing.
- **Small-step derivations** (`bazireason.steps6`), checked on 7,200 derivations:
  - every final JSON equals the rule code's;
  - the claim checker finds no wrong statement in 37,000 checked statements;
  - two false alarms of the checker were worded around: a rule sentence that named patterns not chosen, and "流年干是…" read as "年干是…".
- **Evaluation ladder** (`bazireason.ladder`, `scripts/phase6/run6.py`):
  - L1 sheets contain no derived word (checked for every item);
  - L2 text depends only on the question type;
  - wrong conclusions for L3w are never scored right against the truth.
- **Training sets** are not in Git (23-24 MB each). They are rebuilt bit for bit, and `build_train6.py --check` compares hashes.
- **Fit diagnostic** (`scripts/phase6/diag_fit.py`, A1 alone, same scorer):

  | Sample | Correct |
  |---|---:|
  | A1's own training rows (literal) | 194/240 (80.8%) |
  | Fresh rows from A1's generator, training years | 183/240 (76.2%) |
  | dev6 rule rows | 157/294 (53.4%) |

  - **A1 under-fits even its own training rows** on multi-step questions: areas of life 8/25, relations 33/48 and flow years 36/43 on the literal rows. Single-step questions are near 100%.
  - **The audit's training losses (0.007-0.02 per token) hid this:** the average is dominated by template text, while errors sit in a few computed tokens.
  - **The dev6 gap has a second cause, wording.** On held-out wordings A1 gets 45.3% against 61.8% on training wordings. All 20 dev6 overview questions use the held-out wording "帮我系统地讲一遍这个八字在传统规则下的结构…", and A1 answers them as areas-of-life questions (0/20; checked in the raw replies).
- **Schedule change:** the development ladder for the existing models was cut to L0. Their L1-L4 runs on dev would have taken about eight GPU hours before the first training run; they run on valid6 and sealed6 instead, as pre-registered, and on dev6 for A1 and U0 at the end of the screening queue.
- **dev6 baselines, model alone (L0), all 822 rule items:** base 14.1%, Phase 2 23.6%, A1 60.5%, G1 54.9% (A1 minus G1 5.6 points [2.3, 8.8]). Calendar: Phase 2 74.0%, A1 62.1%, G1 65.3%.
  - A1 still states 旬空 wrongly in 59 of 80 statements and pattern in 45 of 93: the computed steps the small-step data targets.
  - Combinations of trained rules: A1 48%. Counterfactual pairs right on both charts: A1 25/50.
- **U0 started** (13:50): peak memory 21.2 GB, about 0.27 iterations per second, loss 1.23 at step 40 and 0.51 at step 120.
- **Pre-registration section 9** records the decision script's name, the second set of training wordings for U1 (`train6b`), U1's self-check input, U1-hi's memory fallback and the valid6 runs, all before any screening run was trained.
- **Queues:** `queue_u1.sh` waits for the screening queue, applies the frozen screening rule (`decide6.py screen`), writes `train6b`, builds and trains U1, runs its fit check, dev6 L0 and the valid6 runs the selection rule reads. It stops if small steps lose to U0, since the rule then leaves U1's recipe undefined.
- **Truncation** (`scripts/phase6/truncation6.py`): on dev6 the untuned base reaches the 1,100-token cap on 412 of 822 rule items and the 700-token cap on 214 of 219 calendar rows; every cut-off reply scores wrong. The base's low L0 score is partly a format and length failure. The tuned models are cut off on under 1.1%.
- **Retention panel** (`scripts/phase6/panel6.py`): 80 dev6 rule items (4 per question type) and 40 calendar rows, scored at saved checkpoints. Diagnostic only.
