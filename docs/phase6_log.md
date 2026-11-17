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

## 16 November 2026

- **S2s finished** (22:57-03:11 training, scored by 03:39; 5,922 iterations, peak memory 20.0 GB). Model alone on dev6: 658/822 rule items (80.0%) against S2's 657; **H7: +0.1 points [−2.9, +3.2]**, calendar 137/219 against 139 (−0.9 [−7.6, +6.0]). The short prompt held 45% more rows in the same budget and gave no gain. It is not worse, so U2 uses it (`decide6.py prompt`).
  - **By wording** (`first_error.py`): training wordings 196/244 (S2 200), held-out template wording 104/150 (S2 107), natural wordings 171/188 (S2 167). Which question types fail on the held-out wording changes from run to run (flow years 9/24 here, 17/24 for S2).
  - **A slip that grew with training on mixed tasks:** on the counterfactual life-stage items S1 answers 13 of 14, S2 9 and S2s 4. The walk through the twelve stages is right but ends at the wrong branch. S1 had written all twelve stages and read the answer off.
  - **Reading:** more rows of the same mixture no longer help at this size. What is left are slips at steps that must decide where to stop or what to leave out, unseen wordings, held-out combinations and calendar decisions.
- **U2's first build under-filled its calendar part** (20% of the tokens instead of 30%): the calendar generator's pool has a fixed size, and with the short prompt the budget needs more rows than it holds. Training was stopped after six minutes (100 iterations, no result), the set and its files were removed, and `calendar_rows` now draws from a pool twice as large when the first cannot fill the budget. The rebuilt U2 has 29,775 rows and 15.0 M tokens with the planned shares; it has trained since 04:04. Every earlier set still rebuilds bit for bit.
- **The continuation build was dry-run end to end** (U2, the fork with both kinds of weights, a later level): no row on a held-out chart, the fork arms share the same calendar requests, the derivation version is inherited, and a rebuild is identical.
- **Derivation version 2 extended** (pre-registration 12.6, committed before the fork existed): the life-stage walk lists all twelve stages, the pattern step writes the 比劫 check for every stem that appears, and the relations step lists all eight groups of three. Checked on 15,000 derivations (`results/phase6/steps_v2_check.json`): every answer unchanged, no wrong statement, longest reply 1,072 tokens of the 1,100 cap.
- **Composition rows as an arm of the second chain level** (pre-registration 12.7, committed before any continuation existed). `bazireason.compose6` writes six kinds of question that carry the result of one trained procedure into another (for example the branch that clashes with a given branch and the Ten Gods of its hidden stems). None is one of the four held-out combinations or ends with their last operation on the same inputs. 2,400 derivations checked: every final JSON equals the rule code's value and the claim checker finds no wrong statement. U4c gets 10% of its tokens as composition rows, taken from the procedure share; its twin U4 differs in nothing else (H11).
- **Dry run of the whole chain** (U2, U3v, U4 and U4c, U5): the arms of a level share the same calendar requests, U4c's procedure questions are a subset of U4's, later levels inherit the derivation version and the composition share, and no row is on a held-out chart. All six existing training sets, the U2 in training included, rebuild bit for bit with the final code.
- **The two fit-sample measurements were run end to end on the CPU with a stand-in model** (it answers with the rule code's reply except every seventh question): `own_replies.py --sample b` wrote 2,000 scored replies in 20 question types and `--sample c` 780 in 21 calendar cells, the short system prompt was sent for U2, and both kinds of weights were computed from the files. Nothing of this run was kept in the results folder.
- **Held-out template wording** (150 dev6 items): S2's 43 wrong answers almost all start at a step slip, as on the training wordings; S2s adds 21 replies that start another procedure (15 flow-year questions answered with a shortened year arithmetic, 6 overviews answered as area-of-life questions). Which run does this varies, so no separate remedy is added: the chain's candidates are compared on valid6, which has held-out wordings too.
- **Status at 10:25:** U2 at iteration 8,920 of 14,887, no error; validation loss 0.0089 at iteration 8,500 on its own 120 validation rows (S2 ended at 0.0097 and S2s at 0.0116, each on its own rows, so the three are not one comparison). Peak memory 22.3 GB. Training should end about 14:35 and the dev6 scores exist about 15:05.
- **valid6 kept out of sight until the selection.** From this afternoon the queue scores each candidate on valid6. Two places would have shown those scores before the selection: the queue log (one line with the count per part) and `RESULTS.md` (a full section per split). `run6.py` now prints the valid6 line without the count, and `make_report6.py` writes the valid6 tables only when `selection.json` exists. Added before any candidate had a valid6 score; the section held only Phase 2's calendar row (176/269, already in the pre-registration). The records on disk and the selection rule are unchanged. Two tests cover it.
- **Report draft:** H11 and U4c are in the tables of section 3.
- **U2 finished** (04:04-14:38 training, 10.6 h, 14,887 iterations, final validation loss 0.0047, peak memory 22.6 GB; dev6 scored by 15:06). Model alone on dev6:
  - **Rule items 733/822 (89.2%)**: template rules 261/294, independent 140/140, blind wording 187/188, combinations 50/100, counterfactual 95/100 (45 of 50 pairs right on both charts). Held-out template wording 133/150 (88.7%; S2 107, S2s 104).
  - **Calendar 182/219 (83.1%)**, against Phase 2's 162: +9.1 points [+3.4, +15.2]. Balanced decision accuracy 85.7%; answerable requests refused 14/145 (S2 9, S2s 51); definite charts where none exists 13/74 (S2 50, S2s 14).
  - **U2 minus S2s** (same recipe and prompt; 15 M against 6.0 M tokens, another seed, two more wording files): +9.1 [+6.1, +12.2] on the rule items, +20.5 [+12.7, +27.8] on calendar. **U2 minus S2:** +9.2 [+6.2, +12.3] and +19.6 [+12.1, +27.1]. U2 minus A1: +28.7 [+24.8, +32.5].
  - **This corrects the reading of 16 November, 04:00.** S2s against S2 showed that more rows in the same token budget do not help. It did not show that more training does not help: 2.5 times the tokens gave 9 points on the rule items and 20 on calendar. The 6.0 M runs were under-trained.
  - **The held-out combinations did not move:** 50/100 (S2 50, S2s 55). They are now 50 of U2's 89 wrong answers: flow-year life stage 7/25, 旬空 positions 10/25, revealed hidden stems 15/25, Ten-God counts 18/25.
  - **Area-of-life items are the weakest trained type** (31/46; the 15 wrong are the largest group after the combinations). By `first_error.py`, 9 of the 11 errors on training wordings start at the hidden-stem step.
  - **A step got less truthful with more training.** In 35 of U2's 46 area-of-life replies the hidden-stem step names a stem that the branch does not hide, for example "寅藏壬正财" (寅 hides 甲丙戊); S1 14, S2 14, S2s 22. U2 minus S2s: +28.3 points [+13.0, +43.5]. The answer is right in 24 of the 35, because the final JSON does not carry that step. Over all rule items the share of replies with a wrong statement is 88/822 (S2 86, S2s 106). Version 1 writes that step as a filtered list; version 2 writes all four branches first. The fork tests it, and a second reading for H9 on the statements was registered before the fork existed (pre-registration 12.9).
  - **Calendar by kind of request** (U2 / S2 / Phase 2): 子时 changes with a definite chart 25/13/21 of 32, plain charts 23/25/22 of 28, solar-term windows that need the tool 8/3/0 of 10, time zones that need the tool 7/0/7 of 8, unsupported requests 8/1/7 of 8. Weakest now: solar-term windows with a definite chart 5/10, 子时 ambiguous 5/8, Shanghai daylight-saving requests that need the tool 1/4.
  - **What the held-out combination replies do:** they run the nearest trained procedure instead of chaining two. Flow-year life stage: the walk through the stages is right, but the year's branch is not worked out and one is read off the chart. Ten-God count: the reply works out one stem and answers 1. Revealed hidden stems: the reply runs the pattern procedure over all four branches. 旬空 positions vary from run to run (S2 13/25, S2s 20, U2 10); 2 of U2's 15 wrong answers name the right pillars as 年支 or 时支 instead of 年柱 or 时柱. The composition rows of 12.7 aim at this; nothing is added to them.
- **valid6 for U2** is being scored by the queue (not read). `truncation6.py` now also leaves valid6 out until the selection.
- **The fork started at 15:37** (`round2.json`: level 3, U3 and U3v from U2, 733 right on dev6). valid6 for U2 was scored before it (15:06-15:37) and stays unread. The queue is measuring U2 on fit sample B.
- **A restart could have dropped an arm of a pair.** `decide6.py next` counted a level as reached once one arm had a dev6 result, so a queue restarted between U3 and U3v (or U4 and U4c) would have skipped the second arm. Now such an arm is asked for once more with the same parent before the chain goes on (pre-registration 12.10; tested with the stub loop and against the real files without recording). Changed after the queue's level-3 call, so the running level is not affected; the next call is after both arms.
- **U3 is training** (built 16:34 from U2's fit samples, 10,942 rows, 6.0 M tokens, start from U2's adapter; 0.38 iterations per second). U2's fit samples: B 1,960 of 2,000 right (life stage 89, overview 91, relations 94 of 100; most types 96-100), C 638 of 780. So U3's question-type weights are nearly flat; its calendar weights favour Shanghai daylight-saving requests that need the tool (22 of 32 wrong) and solar-term windows with a definite chart (25 of 40 wrong).
- **Every wrong flow-year and area-of-life answer of U2 is on a year the training never asked about.** Phase 6 training rows ask only about 2024-2049; the evaluation asks about years of the chart's life. U2: flow years 16/16 inside the window, 39/44 outside; area of life 13/13 inside, 18/33 outside (S2 18/33 outside too: more training raised the inside to 100% and left the outside where it was). From level 4 on, the template rows of these two types ask about years of the chart's life, as Phase 3 did (pre-registration 12.11, `--years life`). Checked: 6,000 derivations over 1910-2160 right and without wrong statements; every existing set rebuilds bit for bit; U3v builds identically; a level-4 dry run is clean and its arms stay paired. `RESULTS.md` has a new table of these items inside and outside the window.
- **When the model counts as well trained** (21:07, pre-registration 12.12, written before U3's dev6 result was read). The user accepted the timeline, so U1 and the diagnostics stay before the selection as registered, and asked to be told when the model is well trained. The chain's stop rule says when more of the same stops paying, not whether a gap in the data is left (the year window was one), so four conditions are written down: the chain has ended; every trained question type with at least 30 dev6 items is at 90% or more; years outside 2024-2049 are within 10 points of years inside; calendar has levelled off. The held-out combinations are not a condition. If the chain ends with a condition unmet, one targeted level is registered and run if it can start by 27 November. The user is told either way, before the selection. U2 does not meet 2 (area of life 31/46) or 3 (57/77 outside, 29/29 inside).
