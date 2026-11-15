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
- **U0 finished** (17:51; 3,905 iterations in 4.0 hours; peak memory 23.3 GB; validation loss 0.0157 at the last checkpoint, which the trainer selected).
- **U0 on dev6, model alone (L0)** — a development result; the pre-registered H1 test is on sealed6:
  - **Rule items:** 331/822 (40.3%), 20.2 points below A1 [16.7, 23.9] and 26.2 above the base.
  - **Calendar:** 136/219 (62.1%), 11.9 points below Phase 2 [4.7, 18.7], level with A1.
  - **So one joint run on the earlier data reproduces neither specialist** at this budget. The budgets differ: U0 saw 6.0 M tokens sampled from the earlier files, while A1 is the end of six sequential rounds (the seven rounds up to G1 used about 21.7 M tokens). The comparison that the screening uses, S1 against U0, has equal budgets.
  - **Where U0 falls short of A1:** the multi-step types (flow years 10% against 47%, Ten Gods 12% against 54%, pattern 31% against 72%) and the template wordings (29.2% against 61.8%).
  - **Computed statements:** wrong in 20.9% of the checked statements (A1 15.1%), including 旬空 in 35 of 43.
- **S1 started** (18:17; 7,922 rows, 5,992,693 sequence tokens, 3,961 iterations).
- **Report script:** the training table now skips a run until its first validation loss exists.
- **Round 2 added at the user's request** (20:47; pre-registration section 10), recorded while S1 was training: no screening result and no valid6 or sealed6 score existed.
  - **Why:**
    - the main system prompt (279 tokens, in 89% of the rows) is 34% of S1's training tokens, and the final reply, the only part with a loss, is 43% (`results/phase6/token_share.json`);
    - the models under-fit: A1 answers 80.8% of its own training rows, and U0 with 6.0 M tokens is 20 points below A1.
  - **U2:** U1's recipe with a 36-token system prompt at the same 15 M-token budget, which holds about 1.4 times as many rows (1.45 in a dry run at a small budget).
  - **Continuations U3-U5:** from the best parent on dev6, with the question types reweighted by the parent's errors on a new fit sample B (100 questions per type on fresh charts, never training input). Each has 6.0 M tokens at half the learning rate; the loop ends when a continuation does not beat its parent on dev6.
  - **Selection and sealed runs:** the candidates now include U2 and the continuations. sealed6 stays closed until the round ends; L0 runs are added for U1, U2 and the ancestors of the selected model.
  - **Corrected figure:** a first quick check on 1,500 rows counted every system prompt together and gave 42% and a 1.6-times gain. Over the whole file the main prompt is 34% and the gain about 1.4 times. The other 7.5% are the long prompts of the strict-format calendar tasks, which define their output format and stay.
- **Fit sample A is not all fresh for S2, S3 and U1** (found while designing round 2). About 45% of it are training-writer questions that these runs train on, and S1's replies to its first part are the input of the self-check rows. The U1-hi rule now reads the share on the questions whose chart is not in U1's training file (`scripts/phase6/fit_clean.py`; pre-registration section 9). The U1-hi queue had only been waiting and was restarted with this change.
- **Tokenization checked:** each stem and branch is one token, and none of the 60 stem-branch pairs is merged.
- **Round-2 code:**
  - `bazireason.prompts6`: a model always receives the system prompt it was trained with. The swap happens in the reply cache, so it covers every level, the calendar rows and L5.
  - `train_data6` and `build_train6.py`: the short prompt, type weights, sample-B charts blocked, continuation configurations.
  - `own_replies.py --sample b`, `decide6.py structure / next / sealed-extra`, `queue_r2.sh`.
  - The earlier training sets still rebuild bit for bit and every cached reply still matches its prompt (checked). Five new tests.
