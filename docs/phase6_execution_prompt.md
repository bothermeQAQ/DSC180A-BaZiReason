You are the implementation and experiment agent for Phase 6 of my UCSD DSC 180A/B Capstone, BaZiReason. The repository is the current working directory. An independent audit of Phases 1–5 has been done; its findings and the plan are below and in `docs/phase6_capability_audit.md`. Your job is to carry the plan out for real: build, train, measure, fix, report. Work autonomously through the whole plan. Do not stop after each step to ask whether to continue.

# 1. The goal, and what counts as success

Phase 5 reported 98.6% on 1,005 questions. That is the accuracy of a system in which rule code computes the answer, the model copies it, and the code repairs or replaces the model's reply. The best model answering alone gets about 69%. I do not want the two confused again.

**Phase 6 is about the language model itself.** The goal is a model that knows more, reasons more reliably, generalizes to natural questions, uses evidence, and catches its own mistakes, together with an evaluation that proves how much of that is true.

**Success is a demonstrated gain in what the model does alone** (level L0 below) over the untuned base and over the best existing adapter, on new sealed data, with paired intervals. A higher score obtained because code supplied or substituted the answer does not count as success, however high it is. A carefully measured negative result is acceptable; an inflated one is not.

The chatbot and its interface are Quarter 2 work. Keep the existing code and tests working, and do not spend effort on the interface.

# 2. Where the project stands (verify, do not trust)

Read `docs/phase6_capability_audit.md` in full. Its main findings, all recomputed from saved records:

- **Who did the work** (validation, 498 questions, the four pillars always given): base alone 12.9%; A1 alone 69.7%; G1 alone 68.5%; G1 with the code's conclusion in the prompt 92.6%; after the code's repair and fallback 98.6%. Phase 5's training (G1) improved copying, not independent reasoning.
- **Look-ups are solved; computed steps are not.** In A1's replies alone, 0 of 1,884 look-up statements are wrong, against 68.6% of 旬空 statements, 39.0% of pattern statements and 32.9% of element counts. The trained procedures do arithmetic, tallies and matching in single jumps. Phase 2's calendar algorithm, which writes every intermediate value, reached 75–80% exact pillars.
- **Composite answers fail at the last step** (flow years 47.7%, relations 54.2%, overview 42.9%, areas of life 24.2% for A1 alone), often with every look-up right, and drift into another procedure's layout on unfamiliar wording.
- **Natural wording collapses:** on the same question types, template against blind wording is 41/70 against 0/16 for flow years and 37/42 against 5/16 for pattern. Rule training used 2–27 formal wordings per family, and 380–1,070 examples per composite family, each seen once.
- **Evidence is not used unless it is the answer.** The Phase 5 fact sheet states most conclusions outright; even so it helps G1 by only 5 points and hurts several question types. Deriving from raw facts has never been measured.
- **Self-correction is about 3%** when the model is told which field is wrong but not the value (A1 fixed 19 of 556).
- **The calendar "forgetting" is largely a moving refusal prior:** balanced decision accuracy is 72.7–74.5% for all four adapters while refusals swing from 3% to 37%, and no adapter separates matched twins.
- **Training so far** is a chain of seven one-pass rounds (28,147 rows, about 21.7 million tokens), each with a different task mix, on the last 16 of 36 layers at rank 16.
- **Scoring gaps:** ambiguous answers are graded on the decision only (41 of A1's 48 "right" ones contain a wrong derived statement), and the labels for declines are not clean.
- **BaziQA** has never moved off chance for any condition and is not a target.

Start by reproducing these numbers with a committed script that reads saved files only, and record any that differ. Then read the code you will reuse and the raw replies behind each finding. If a finding does not hold, say so and adjust the plan.

# 3. Rules that do not bend

- **Preserve everything.** Never overwrite or delete an adapter, dataset, result file, report or the submitted checkpoint. Phase 6 writes to new folders and new adapter names. Corrections to old numbers are addenda.
- **Honest levels.** Every table and every sentence that reports an accuracy names its level (section 4). System accuracy is never described as model accuracy. The headline of the report is L0.
- **No fabricated or estimated results.** Every number comes from a saved file through a script. If a run fails, show the error. If you find a bug after scoring, report both results.
- **A new sealed test, opened once.** Every earlier split, including the Phase 5 sealed split, and BaziQA are development or historical material now. Build a new sealed set, keep it closed until a selection file exists, log every access, and score it once.
- **Freeze before you score.** Write a pre-registration (questions, levels, primary comparisons, selection rule, stopping rule, gates) and commit it before any validation or sealed scoring. Later changes are listed with reasons.
- **Labels come from code or a third-party library,** never from a language model. A model may write questions or explanations; code must check them, and nothing unchecked enters training.
- **No leakage.** No evaluation chart, wording cluster or question may appear in training; test this. The writer of sealed questions must not be the writer of training paraphrases. BaziQA never enters training.
- **Scope of claims.** Rules that depend on a school or convention are labelled as such. Nothing is presented as validated prediction of anyone's life.
- **Machine.** One GPU job at a time. Chain long jobs in background scripts and wait for them. No downloads without my approval; if you conclude that a different base model or writer model is needed, finish everything else first and ask once, with the evidence.
- **Git.** Work on a new branch from the current one. Commit locally as work completes. Do not push, merge or open a pull request. Leave my untracked notes alone. Adapter weights and caches stay out of Git.
- **Privacy.** Use made-up birth data in any test.

# 4. The evaluation ladder

Build one harness that runs the same items at six levels and reports them separately, never averaged together:

| Level | The model receives | It measures |
|---|---|---|
| L0 | the question and the four pillars (birth data for calendar tasks) | independent knowledge and reasoning |
| L1 | plus verified raw facts, with no derived conclusion | deriving from evidence |
| L2 | plus the relevant rule text, with no chart-specific conclusion | using a knowledge base |
| L3 | plus the code's conclusion; and a variant where that conclusion is wrong | following, and blind following |
| L4 | its own first answer and a request to check it | self-correction |
| L5 | the full Phase 5 system with repair and fallback | application reliability |

Requirements:
- **No answer in the prompt at L0–L2.** Enforce this mechanically for every item and test it. The Phase 5 fact sheet fails this requirement; design the raw-fact level so that nothing in it is the output of the procedure being asked about.
- **L4 gives no values.** Run three variants: a plain request to re-check; "there is an error"; "this field is wrong". Report the fix rate on wrong first answers and the break rate on right ones.
- **L3 with a wrong conclusion** measures whether the model notices or follows.
- **No deterministic fallback, repair with values, loop guard or routing may touch L0–L4 scores.**
- **Metrics at every level:** first-pass accuracy; statement-level accuracy and the first wrong step; consistency between a reply's steps and its conclusion; false refusals and definite answers where none exists, on matched twins; loops and truncation; tokens and seconds.
- **Greedy decoding is the primary setting.** Inference-time techniques (voting over samples, a second derivation compared with the first, a repetition guard) may be measured, but as separate rows that are never merged into L0.
- **Fix the scorers the audit flagged:** grade the content of ambiguous answers, and give declines one clean label scheme. Rescore old records as addenda where it matters.

Test material, in development, validation and sealed versions built from separate chart years and seeds:
- rule questions across every question type, with more items labelled by the third-party library;
- natural questions from a writer not used for training, kept in held-out wording clusters; and my own questions if I have put them in the folder you create for them (create it with a short README, and proceed without waiting);
- held-out combinations of rules, and counterfactual edits where one character changes the answer;
- dialogues whose later turns need a new derivation, not only a kept answer;
- calendar requests with matched answerable and non-answerable twins.

Size the sealed set from a power calculation for the primary comparisons and state it.

Before training anything, run the ladder for the base, Phase 2, A1 and G1 on development data. This baseline ladder is itself a result.

One more diagnostic decides how to read everything else: score A1 alone on fresh samples of its own training distribution, and on a sample of its literal training rows. Near-perfect there with 69% on the test means a data-coverage problem; well below perfect means under-fitting.

# 5. The training plan

The recommendation is to stop extending the adapter chain and train one model jointly from the untuned base on a rebuilt, balanced corpus. Existing adapters are fixed baselines: base, Phase 2 (calendar reference), A1 (the bar to beat alone), G1 (conclusion-following reference). Check this recommendation against what you find; if the evidence points elsewhere, follow the evidence and explain.

Each run isolates one question:

1. **U0, joint against sequential.** A fresh adapter from the base on the union of the existing training corpora, in one shuffled mixture, with the same adapter settings as the chain and a comparable token budget. Does one adapter then hold the calendar skill and the rule skill together?
2. **Screening runs at a reduced budget,** added one at a time on top of the U0 recipe and compared on development data:
   - **small-step procedures:** rewrite the derivations so that no step needs arithmetic, counting or matching in the model's head (the audit gives examples for 旬空, element counts, relations and flow years);
   - **wording variety:** many natural wordings per question type, written by a local untuned model from plain descriptions, filtered by rules fixed in advance, labelled by code;
   - **composition and contrast:** single steps, then chains, then full procedures; the same chart under procedures that get confused; chart pairs that differ in one character;
   - **evidence use and self-check:** L1 and L2 style tasks without the answer, some with one corrupted fact to notice; and error-and-fix rows built from the current model's own wrong replies on training charts, with the first wrong step located by code, mixed with right answers that must be confirmed unchanged;
   - **decisions as matched twins** in equal numbers, with the distinguishing reason in the target.
3. **U1, the full-budget run** with the components that helped.
4. **U1-hi, capacity.** The same recipe with more layers or a higher rank, after a short memory probe. A gain means the adapter was the limit; no gain means it was not.

Guidance:
- **Measured cost on this machine:** about 1.5 million training tokens per hour, with a peak of 20–24 GB at 1,920-token sequences. Plan budgets from that.
- **Generate fresh charts** instead of repeating examples.
- **Keep every skill in the mixture from the first step to the last,** at fixed shares. That is the guard against forgetting. If a later continuation is ever needed, replay the full mixture.
- **Watch a small retention panel** that covers every skill at regular checkpoints, and keep the checkpoint history.
- **Keep the conclusion-following tasks a small share,** so the model does not learn to wait for an answer.
- **Additive screening confounds interactions.** Say so in the report.
- **Select on validation by L0** under a frozen rule that includes a calendar gate. Never select on L3 or L5.

# 6. What to pre-register

- **Primary:** the selected model's L0 accuracy minus A1's on sealed rule questions, and the same on natural wording (success: lower bound above 0; target +10 points).
- **Calendar:** not below Phase 2 by more than 5 points, and higher balanced decision accuracy on twins.
- **Evidence:** L1 minus L0 for the selected model (success: above 0), next to the same for A1 and G1.
- **Self-correction:** fix rate and break rate against A1 (target: fix at least 25%, break at most 2%).
- **Faithfulness:** replies with a wrong checkable statement at most half of A1's rate.
- **Joint against sequential:** U0 against the chain on both skills.
- Report every comparison whatever its sign.

# 7. Stopping, fallbacks and time

- **Stop training** when two successive full-budget changes each fail to produce a paired development gain whose interval excludes zero, or when a run could no longer be evaluated in time.
- **Deadline.** The Quarter 1 report is due Friday 4 December 2026. The last training run must start by 27 November. Keep 29 November onwards for writing.
- **If time runs short,** the essentials in order are: the ladder harness and baseline ladder; U0; one redesigned-data model; the sealed comparison; the report. U1-hi can move to Quarter 2.
- **Fallbacks, each reported as a result:**
  - **Small steps do not fix computed statements:** record it as a limit of in-head computation for this model size.
  - **The joint model cannot hold both skills:** evaluate two specialists on the existing adapter bank.
  - **Accuracy on the training distribution stays low:** write up the case for a base-model study and ask me about a download.

# 8. Deliverables

- New data, generators, configurations and scripts with tests: split hashes and regeneration, leakage, the seal, the no-answer check for L0–L2, the scorers, and the new derivations checked against the rule code.
- A results page and a summary file generated by script from saved records, with every table labelled by level.
- A report that answers:
  - what the model can do alone, and how far each change moved it;
  - what each experiment showed about its hypothesis, including negative results;
  - what the hybrid system adds on top, reported separately;
  - the limits that remain, with their causes;
  - exact reproduction commands;
  - a proposal for Quarter 2.
- A draft of the model-research sections for the course report in the existing LaTeX template, as a new file that leaves the submitted checkpoint report untouched.
- Updated README sections, the working log, the full test suite and lint passing, and local commits.

Finish with a concise summary: what was built, what the model gained alone, what did not improve, and what remains.

# 9. How to work

- Make decisions and keep going. Ask me only for something outside these rules: a download, a push, deleting anything, or a resource you cannot work around.
- Inspect the repository and choose folders, formats and methods yourself; the plan above states aims and constraints, not file layouts.
- When a result looks wrong (a zero, identical replies, a sudden jump), suspect the harness first and read raw outputs before interpreting it.
- Read failures before designing data for them, and quote real examples in the report.
- Pitfalls already met in this project:
  - build every prompt with the shared prompt helper, because one phase once sent no question at all;
  - cap the padded tokens per batch, because long batches have run out of GPU memory;
  - load and use an MLX model on one thread;
  - longer derivations need a larger token cap, so report truncation whenever it happens.
