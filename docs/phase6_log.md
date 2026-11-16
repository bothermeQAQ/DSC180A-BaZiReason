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
- **S1 finished** (22:15; 3,961 iterations in 3.98 hours; peak memory 22.3 GB) and was scored on dev6, model alone (L0):
  - **Rule items:** 549/822 (66.8%), 6.3 points above A1 [2.2, 10.5] and 26.5 above U0 [22.2, 30.5]. Four hours of training from the base on small steps beat the end of the earlier chain.
  - **Where it gained:** 旬空 96% (A1 28%), element counts 98%, life stages 95%, areas of life 52% (A1 15%), overview 55% (A1 0%), counterfactual pairs right on both charts 42/50 (A1 25/50).
  - **Where it lost against A1:** relations 30% (A1 50%), flow years 40% (47%), strength 65% (76%), held-out combinations 36% (48%).
  - **Calendar:** 128/219 (58.4%), 15.5 points below Phase 2 [6.4, 24.1].
  - **Cut-off replies:** 30 of 822 (3.6%; mean reply 307 tokens). They are loops, not long derivations: the longest reference derivation on dev6 has 951 tokens.
- **Where S1's wrong answers start** (`scripts/phase6/first_error.py`, which compares each reply with the rule code's derivation step by step; `results/phase6/first_error/dev6_p6s1.json`):

  | Wording of the question | Right |
  |---|---:|
  | training template wordings | 203/244 (83.2%) |
  | held-out template wording | 56/150 (37.3%) |
  | natural (blind) wordings | 129/188 (68.6%) |

  - **Wording is the largest cause.** With a wording it has not seen, S1 often starts another procedure or improvises one (relations 1/25 on the held-out wording and 0/10 on blind wordings; pattern 1/10 on blind wordings, answered as strength questions). Most loops start there. S1 was trained on template wordings only; S2 tests the remedy.
  - **On training wordings the errors are few and specific** (41 of 244): the season and support steps of strength (得令 7, 得势 3), the hidden-stem step of area-of-life questions (4), the clash step of flow years (天克地冲 4), and the pair, trio and summary steps of relations (11). Each of these asks for a relation between two elements, or a filtered list, without writing the table first.
  - **Held-out combinations:** S1 starts the right sub-procedures and fails at the step that joins them, often in a loop.
- **Changes to round 2 after this analysis** (23:19; pre-registration 10.5, before any other screening result, any valid6 score and any round-2 run):
  - **Derivation version 2** writes the three failing steps out further (element positions before 得令 and 得势; all hidden stems with their Ten Gods before the relevant ones; the palace card and the 相克 order before the look-ups). Every answer is unchanged and the claim checker finds no wrong statement in 9,000 version-2 derivations (`scripts/phase6/steps_check.py`, `results/phase6/steps_v2_check.json`). Replies stay under the 1,100-token cap: the longest overview has 1,063 tokens (1,000 in version 1).
  - **Fork:** U3 and U3v differ only in the derivation version (H9); the chain continues from the better one.
  - **Longer loop:** up to eight levels, ending after two levels in a row without a new best on dev6.

## 15 November 2026

- **S2 finished** (00:04-04:03; 4,080 iterations in 3.99 hours; peak memory 21.0 GB) and was scored on dev6.
  - **Model alone (L0), all rule items:** 657/822 (79.9%), 19.5 points above A1 [15.8, 23.4] and 13.1 above S1 [9.7, 16.6].
  - **Wording, the gap S2 targets:** held-out template wording 107/150 (71.3%; S1 37.3%), natural wordings 167/188 (88.8%; S1 68.6%, +20.2 [12.8, 27.1]). On training wordings S2 is level with S1 (76.4% against 79.9%).
  - **Combinations of trained rules:** 50/100 (S1 36/100, A1 48/100). Counterfactual pairs right on both charts: 41/50.
  - **Computed statements wrong:** 10.9% (S1 36.7%, A1 15.1%). Cut-off replies: 11 of 822 (1.3%).
  - **Calendar:** 139/219 (63.5%), 10.5 points below Phase 2 [3.7, 16.6]. S2 almost never refuses an answerable request (9/145), but gives a definite chart in 50 of 74 requests where none exists.
  - **Raw facts hurt S2:** L1 minus L0 is −7.7 points [−10.5, −4.9].
  - **Self-check hurts S2:** asked to re-check (L4a), it fixes 8 of 165 wrong answers and breaks 21 of 290 right ones.
  - S3 adds the evidence rows, self-check rows and calendar twins aimed at these three weaknesses.
- **Where S2's wrong answers start** (`first_error.py`): on training wordings, mostly at the hidden-stem step of area-of-life questions (7) and the 天克地冲 step of flow years (6). Both are written out in derivation version 2, which was recorded before this result. Life-stage counting on training wordings fell from 13/14 (S1) to 9/14.
- **Qwen3-8B downloaded** (04:00-04:13, approved by the user; `Qwen/Qwen3-8B-MLX-4bit` at revision 383413e9, 4.4 GB, Apache-2.0) into `models/qwen3-8b-mlx-4bit` (not in Git). It has the same architecture family, 36 layers and the same tokenizer as the 4B base: every stem and branch is the same single token.
  - **Prompt rendering:** this is the hybrid-thinking Qwen3 release, so with thinking off its chat template ends the prompt with an empty `<think></think>` block. Training (`bazireason.finetune`) and evaluation (`MLXChatModel.prompt_ids`) render prompts with the same call and options, so the model sees the same prompt in both.
  - **Smoke test:** load, LoRA on the last 16 layers (19.4 M trainable parameters), one training step at 6.4 GB peak. It ran for five seconds on the GPU while S2 was being scored. Greedy replies do not depend on timing, so no result is affected.
- **Larger-base pilot pre-registered** (04:13; pre-registration section 11) and queued: S1's training file and settings on Qwen3-8B ("S1-8B", H10), after the screening queue and before U1.
  - **Batch size:** set by a memory probe, in this order: batch 2; batch 1 with gradient accumulation 4 (the same updates); the same with gradient checkpointing.
  - **Code:** `scripts/phase6/p8_config.py`, `queue_p8.sh`, `probe_memory.py --model --batch`. `run6.Bank` loads an adapter on the base named in its configuration. `queue_u1.sh` was restarted (it was only waiting) so that it waits for the pilot.
  - If the pilot stops for a technical reason, the queue goes on to U1.
- **S3 finished** (06:01-09:57 training, scored by 11:25). Model alone on dev6: 638/822 rule items (77.6%), 2.3 points below S2 [−5.2, +0.5]; combinations 38/100 (S2 50), counterfactual 81/100 (S2 90); calendar 116/219 (53.0%), 10.5 points below S2 [−17.1, −3.8]. Raw facts hurt it less (L1 minus L0 −1.9 [−4.5, +0.7]; S2 −7.7), and a re-check no longer breaks right answers (0 of 278; S2 21 of 290) but fixes almost none (2 of 184).
- **Screening decision** (20:38, `results/phase6/screening.json`, the frozen rule of section 5): nothing dropped (small steps +26.5 [22.2, 30.5], wording +13.1 [9.7, 16.6], evidence and self-check −2.3 [−5.2, +0.5]); the calendar share stays; the best S run is S2.
- **Larger-base pilot S1-8B** (14:00-20:37): the memory probe rejected batch 2 (peak above the limit) and accepted batch 1 (17.3 GB in training); 7,922 iterations in 5.94 hours (282 tokens per second, 67% of the 4B's speed). dev6 L0: 546/822 against S1's 549, −0.4 points [−3.7, +3.1]; calendar 138/219, +4.6 [−1.0, +10.1]. **H10 is not supported**: at this data size the larger base learns the same data no better. By section 11 no further run uses it.
- **The U1 queue failed and reported success** (20:52-21:02). The build stopped at the leakage check: one of the 4,002 `train6b` questions is on a chart that is also an evaluation chart. `queue_u1.sh` had no stop after that step, so the training, fit and scoring steps each failed in seconds and it printed `QUEUE_U1_DONE`; U1-hi failed at its first step; the round-2 queue waited for a marker that never came. The GPU was idle for 1 hour 55 minutes. Opus 5.5 found this in a status check and changed nothing.
- **Audit by the new lead agent** (22:53): no row of any trained set and no row of the earlier training files is on a Phase 6 evaluation chart, for every family; no candidate has a valid6 score; sealed6 is closed. No result is affected.
- **Recovery and revised plan** (pre-registration section 12, committed at 22:55 before any of its runs existed):
  - the build leaves out a training-writer question on an evaluation chart; every queue step is followed by a stop, and a test checks it;
  - the main line uses S2's recipe; H7 becomes a 6 M-token twin (S2s against S2); U2 is S2's recipe at 15 M tokens with the prompt the twin decides; the chain follows as registered; U1 as registered runs after the chain; U1-hi only by hand.
- **S2s started** (22:57): 11,845 rows in the same 6.0 M-token budget as S2's 8,161. `queue_main.sh` waits behind it. Continuation instructions: `docs/phase6_handoff.md`.
- **Calendar weights for the continuations** (pre-registration 12.5, committed at 23:03 before any continuation existed): the parent answers fit sample C (780 fresh calendar requests in 21 cells of family and expected status; the reference replies all score right), and the continuation draws its calendar rows by cell weight from a pool twice as large. In a dry run with made-up errors, requests that need a time-zone tool went from 14 to 61 rows and ambiguous day-boundary requests from 13 to 67.
- **Two more ways a build could have stopped the queue, found by a dry run of the U2 and continuation builds and fixed** (23:10-23:20):
  - a contrast row edits one character of a chart, and the edited chart was not checked against the held-out charts; the edit of a real chart is often another real chart, so it can be a chart of fit sample B;
  - the dialogue rows that come with the calendar data carry charts and were not filtered.
  - Both generators now avoid held-out charts, and the build leaves out and counts any row that still lands on one, instead of stopping. Over six seeds (7,138 rows) no row is on a held-out chart. S1, S2, S3 and S2s still rebuild bit for bit.
