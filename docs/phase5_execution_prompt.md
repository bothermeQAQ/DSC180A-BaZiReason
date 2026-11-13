You are the implementation and experiment agent for Phase 5 of my UCSD DSC 180A/B Capstone, BaZiReason. The repository is the current working directory. A research review of Phases 1–4 has already been done; its findings and the plan are below. Your job is to carry the plan out for real: build, run, measure, fix, report. Work autonomously through the whole plan. Do not stop after each step to ask whether to continue.

# 1. What BaZiReason is

BaZiReason studies whether a small local language model can do verifiable Chinese Four Pillars (BaZi) calendar and traditional-rule reasoning. It has three layers:

- **Deterministic code:** a calendar engine (lunar-python based, fixed UTC+8 clock, 立春 and 节 boundaries), rule tables and a rule layer (`src/bazireason/rules.py`, `deep_rules.py`, `charttools.py`), an independent tool check and repair workflows (`workflow.py`), and a claim checker for free text (`claim_check.py`).
- **Fine-tuned models:** LoRA adapters on `models/qwen3-4b-instruct-2507-mlx-4bit`, trained locally with MLX. Each adapter was warm-started from the one before: Phase 2 (`bazireason-qwen3-4b-lora-v2`, "P2") → Phase 3 rounds 1–3 (`-p3-r1`, `-p3-r2`, `-p3-r3`; "R3") → Phase 4 (`-p4-a1`, "A1"). All are under `artifacts/adapters/` and are not in Git.
- **A local chatbot:** `python -m bazireason.demo_app`, which shows the engine's chart, sends the question to a model and checks the reply with code.

Read these before changing anything: `README.md`, `docs/phase4_report.md`, `docs/phase4_preregistration.md`, `results/phase4/RESULTS.md`, `docs/phase3_progress_report.md`, `data/phase4/README.md`, `docs/phase2_training.md` (environment), and the code in `src/bazireason/` that you will reuse (`workflow.py`, `claim_check.py`, `demo_app.py`, `demo_deep.py`, `deep_rules.py`, `rules.py`, `mlx_backend.py`, `train_data4.py`, `indep_rules.py`, `p2_eval.py`, `p3_eval.py`) and `scripts/phase4/`. Do not rely on the summaries alone; check the code and the saved outputs when a detail matters.

# 2. What the review established (verified against raw outputs)

Treat these as the starting point. Re-check any of them cheaply if your work depends on it.

**Results that hold up**
- All 7,760 saved final-test replies rescored from raw text reproduce the stored scores exactly.
- Phase 4 final test, model alone: rule reasoning A1 67.9%, R3 62.5%, P2 14.5%, untuned base 6.2%. Independent rule questions (lunar-python labels): A1 71.8%, R3 57.9%, P2 59.3%, base 22.9%. Calendar rows: P2 71.4%, R3 70.6%, A1 67.3%.
- The code tool check flags 150/150 corrupted tool answers and 0/30 clean ones. Models told only which pillar failed still copy most errors.

**Problems the review found**
1. **No adapter is reliable at anything the code can already compute.** Best rule families are 60–80%. Standalone four pillars peak at 81%. 旬空 is at most 35% right even with lenient scoring; A1 gets the arithmetic wrong in its own derivation. Models write a correct Ten God table and then contradict it two lines later.
2. **The ask-for-a-tool decision is not learned, only shifted.** R3 never asks (0/30 when it should; 80/83 answers when it may). A1 asks almost always (29/30 when it should, but 42 of 83 answerable charts refused, with invented reasons such as "19 October is inside the 寒露 window"). Neither discriminates.
3. **Template dependence.** Several task families (strength, pattern, Ten Gods, table drills) score 86–100% on the validation split, which uses only training wordings, and 43–87% on the final test (validation counts per family are small). On the final test A1 gets 75% on seen wordings and 59% on unseen ones. Final training loss is 0.008.
4. **The chatbot does not match the benchmarks.**
   - Routing: 12 of 20 ordinary questions ("我五行缺什么", "空亡在哪", "年干是什么十神", "我命里有贵人吗") miss the keyword router in `demo_deep.plan` and go to the model in a generic format that no benchmark measures.
   - Recomputing known facts: the model receives only the four pillars and re-derives facts the engine already has, then the page reports "核验未通过".
   - Format: every turn is forced to end with a JSON conclusion, even "能讲得通俗一点吗".
   - Wrong refusal: "我什么时候换大运" is routed to the prediction refusal, although the engine computes 大运.
5. **Evaluation weaknesses.**
   - `data/phase4/final_test` has now been read during analysis; it is development material from here on. No sealed set remains.
   - The "unseen wording" templates are the Phase 3 test-only templates, which were inspected before A1 was designed (137 of 140 family/kind/template combinations).
   - The independent questions were written by the same author, at the same time, as A1's training augmentations (both use 日元, for example). Their independence is the label source and the charts, not blind authorship.
   - The Phase 4 selection rule was underpowered (168 calendar rows, ±7 points), selected P2 by construction, and left every informative comparison exploratory.
   - One scoring artifact: `indep_rules._norm` rejects a 旬空 answer given as a string ("寅卯") instead of a list. Content-lenient totals are A1 212/280 (75.7%), R3 169, R2 166, P2 167, base 64. Rankings do not change.
   - Part of every "tuned vs base" gap is format: the base model gives a parseable conclusion in only 160 of 280 independent replies and 77 of 484 calendar replies.
   - One training run and one seed per adapter. Verify-and-repair (W1) on generated rows checks with the same code that made the labels.
6. **BaziQA is near chance and unstable.** Every condition is 25–33%; always answering B scores 27.9%. Three sampled runs give the same answer on only 21–35% of questions. Fine-tuning has never helped. A post-hoc vote over nine answers from three adapters reached 36.2%; this is unconfirmed and on an inspected benchmark.

**Feasibility probes already run (scratch, not in the repo)**
- **Adapter hot-swap works.** All adapters have identical structure (rank 16, scale 2, layers 20–35, seven projection types, 14.68 M parameters). `net.load_weights(list(weights.items()), strict=False)` on one loaded base swaps an adapter in 3.5 ms, with output bit-identical to a fresh load. One base plus every adapter in memory is 2.4 GB; the chatbot currently loads a separate 2.3 GB copy per model.
- **Adapters are nested, not independent.** Cosine between R3 and A1 total updates is 0.97, P2 and A1 0.71. Each round's own step is nearly orthogonal to every other step and to P2 (|cos| < 0.03). So adding or averaging adapters double-counts shared history; only interpolation along the chain or arithmetic on the per-round steps is meaningful. Interpolating the LoRA factors directly matches interpolating the updates within 0.7%.
- **Interpolation R3↔A1 on the Phase 4 validation split** (baselines: R3 185/240 rule, 115/168 calendar, 7 false refusals, 80/140 independent; A1 193, 108, 48, 98):

  | Share of the way from R3 to A1 | Rule items (240) | Calendar rows (168) | Exact pillars, no tools (62) | Answerable charts refused | Asks for the tool when needed (23) | Independent questions (140) |
  |---|---:|---:|---:|---:|---:|---:|
  | 0 (R3) | 185 | 115 | 50 | 7 | 1 | 80 |
  | 0.25 | 192 | **121** | 42 | 17 | 9 | 84 |
  | 0.5 | 193 | 110 | 18 | 46 | 21 | 85 |
  | 0.75 | 191 | 104 | 12 | 51 | 23 | **105** |
  | 1 (A1) | 193 | 108 | 15 | 48 | 23 | 98 |

  The three behaviours arrive at different points on the path: the rule gain by 0.25, the over-refusal between 0.25 and 0.5, the transfer to independent wordings only after 0.5. No single point has all three. The point at 0.25 is the best calendar score of any adapter here and a candidate "balanced" adapter, but the split is small (±7 points on calendar rows), it is one run, and this validation split has been used before. The tool-need decision slides with the weights; it is never learned.
- **Adapter routing gains little.** A per-family router chosen on validation and applied to the final test, with the task family given, scores 66.3% against 64.2% for A1 alone; the per-family oracle is 68.2%.

# 3. The architecture to build: code computes, the model explains, code checks

The central decision of the review: **stop asking the model to compute what the code already knows.** Phases 2–4 trained and scored the model as a calculator. Phase 5 makes the system an engine-first assistant and measures the model on what only it can do.

1. **Deterministic kernel (never delegated to the model):** time-zone conversion, the four pillars, answerability status, 大运, hidden stems, Ten Gods, element counts, 旬空, twelve stages, branch and stem relations, stars, the strength count, pattern labels, 用神 under each declared school, flow-year relations, rule-card citations, validation of supplied tool answers, claim checking and loop detection.
2. **Intent router:** maps a user turn to the facts it needs and a response mode: a computed fact, an explanation of a rule result, a school-dependent question (say which school and why answers differ), a traditional association (label it as symbolic), a request for a personal prediction (decline), or unknown (give the full fact sheet and answer cautiously). Start from `demo_deep.plan`; extend it. Measure its accuracy.
3. **Grounded generation:** the model receives a verified fact sheet for the intent (rule conclusions, not only the pillars) and explains it in natural Chinese across turns. Keep a JSON conclusion for evaluation runs; do not force JSON in conversation.
4. **Verification and repair:** the claim checker compares the reply with the fact sheet. On a contradiction, one repair turn that supplies the correct values. If the reply still contradicts, show the engine's own derivation and say so. Never show an unverified deterministic claim as if it were checked.
5. **Transparency:** every statement type is labelled: computed, convention (school named), traditional association, or not verifiable.

Adapter roles to start from, to be confirmed by your dev experiments: A1 as the default generator (its weakness, computing charts without tools, is never used when the engine supplies the chart); R3, or a merged adapter near the 0.25 point if it holds up, for the research condition "model alone, no tools"; routing between adapters as an ablation, not as the core.

# 4. Rules that protect the research

- **Preserve everything.** Never overwrite or delete an existing adapter, dataset, result file or report. New work goes to `data/phase5/`, `configs/phase5/`, `results/phase5/`, `scripts/phase5/`, new adapter folders and new docs. If a historical number needs a correction, add an addendum; do not edit the old file.
- **Branch and Git.** Create a new branch from the current head of `phase3/deep-reasoning`. Commit locally in logical steps. Never push, merge or open a pull request. Do not commit adapter weights. Leave my untracked notes (`CLAUDE_FIX_SUMMARY.md`, `CLAUDE_INDEPENDENT_AUDIT.md`, `CODEX_FINAL_REVIEW.md`) alone.
- **Environments.** MLX and MLX-LM are in `.venv-mlx`; tests run with `.venv/bin/python -m pytest`. Training entry point: `scripts/phase2/train_lora.py --config ...`. One GPU job at a time. Run anything longer than about two minutes in the background with `nohup caffeinate -i`, and wait for it with a background waiter, not by polling.
- **Measured costs for planning.** Training 3.7 s per iteration at batch 2 (3,715 rows took 115 minutes, 21 GB peak). Batched evaluation about 1.7 s per row; interactive generation about 40 tokens per second.
- **Pre-register before you measure.** Write `docs/phase5_preregistration.md` and commit it before any Phase 5 candidate is scored on the sealed set: the systems to compare, at most three primary comparisons with their margins, the power calculation that sizes each split, the acceptance gates in section 6, and what is exploratory. Later changes go in a dated change log with the reason.
- **Three kinds of data.** Development (read freely, iterate on it): every earlier split including `data/phase4/final_test`, plus a new Phase 5 dev split. Selection: a new validation split. Sealed: a new final split that no script scores until a selection file exists, with every access logged, as `scripts/phase4/run_eval.py` does. Do not print or read sealed rows.
- **Build prompts with `workflow.prompt_messages(row)`,** never `messages[:-1]`: independent rows end with the question, and a past bug sent no question at all.
- **No fabricated or estimated results.** Every number in a report comes from a saved file. If a run fails, say so and show the error. If you find a bug after scoring, report both results.
- **BaziQA** is an inspected benchmark and accuracy on it is agreement with contest keys, not evidence that destiny can be predicted. Do not tune anything on its keys.
- **No downloads without my approval.** Use the models already in `models/`. If you conclude a larger base model is worth testing, stop and ask me.
- **Privacy.** The chatbot stores nothing without the consent boxes; keep it that way. Use made-up birth data for tests.

# 5. Work packages, in priority order

If time runs short, cut from the bottom: WP6 before WP5, WP5 before WP4. WP1–WP4 and WP7–WP8 are the core.

**WP0. Orient and correct the record (short).**
Confirm the facts in section 2 that you depend on. Add `results/phase5/addenda/` with a rescoring of the Phase 4 independent questions under both strict and content-lenient 旬空 scoring, generated by a script, and a note that the Phase 4 final test is now development material.

**WP1. Evaluation suite v2, then the pre-registration.**
Design it so that the gains from training, grounding, verification, routing and merging can be told apart on the same items. It needs:
- **A calendar set large enough for its margin** (compute the size; the Phase 4 set could not resolve ±5 points). Pair every "needs a tool" or ambiguous request with a matched answerable twin, and report balanced decision accuracy, not only one-sided rates.
- **Rule questions in two answer conditions:** model alone, and grounded with the fact sheet.
- **A blind-wording set.** Questions must not be written by you after looking at training templates. Use two sources and report them separately: (a) questions generated by an untuned local model of a different family (`models/qwen3.5-4b-mlx-4bit` or `minicpm5-2b-mlx-4bit`), prompted with plain-language descriptions of what to ask and never shown a project template, filtered only by mechanical rules fixed in advance; (b) `data/phase5/blind_user/questions.txt` if I have provided it (my own questions; create the folder with a short README telling me the format, and proceed without waiting).
- **A conversation set:** scripted multi-turn dialogues (three to five turns: follow-ups, "why", "say it more simply", a switch of topic, a prediction request) with code-decidable checkpoints per turn.
- **A router set:** questions with an intended intent label, including ones that should be declined and ones that should not.
- **Metrics:** answer accuracy; faithfulness (replies whose prose contradicts the fact sheet, per the claim checker, plus the checker's own false-alarm rate on a sample you read and record); false refusal, false definite answer and balanced decision accuracy; loops and titles outside the rule cards; router accuracy; tokens, seconds per reply at interactive single-stream speed (median and 90th percentile) and peak memory. Report format compliance separately from correctness so that the untuned base is compared fairly.

**WP2. One base model, many adapters.**
Refactor inference so a single loaded base serves every adapter by hot-swapping LoRA weights, including interpolated weights, with a test that a swapped adapter reproduces a freshly loaded one. Move the chatbot's `Models` class onto it. Log latency and memory.

**WP3. Cheap architecture experiments on development data only.**
- Interpolation: re-measure the R3↔A1 path on the larger Phase 5 dev split, with a finer grid around 0.25 and around 0.75, and, if you see a reason, arithmetic on the per-round steps. Decide with the gates in section 6 whether any merged adapter is a candidate. The 0.25 point is the one to confirm or refute.
- Adapter routing: choose a per-intent adapter table on dev and measure it on validation.
- Record the outcome even when negative. Do not spend more than a small share of the effort here; the review expects modest gains.

**WP4. The grounded system (core of the phase).**
Build the fact-sheet builder per intent from the existing rule code, the two prompt modes (evaluation with JSON, conversation without), the verifier with value-supplied repair and engine fallback, and the extended router. Then measure, on dev and validation:
- base, R3 and A1 each **alone** versus **grounded**. This 2 × 3 separates what training adds from what grounding adds. In particular: does fine-tuning still matter once the facts are supplied?
- grounded with and without verification and repair.
- how often the model contradicts or ignores a supplied fact, by intent.
- whether A1 over-refuses or asks for tools when the fact sheet is present.
Read the failures. Fix prompt and pipeline problems on dev.

**WP5. Training, only if the evidence asks for it.**
Do not train by default. Train if WP4 shows a specific, measured failure that prompts and code cannot fix: for example unfaithful or stilted grounded explanations, refusals despite supplied facts, or poor multi-turn behaviour. If you train:
- State the failure, the data that addresses it, and the expected effect before the run.
- Prefer **joint training on a balanced mixture** over another sequential round: the review's hypothesis is that sequential rounds in a fixed rank-16 budget cause the see-saw between skills. Compare one joint run from the base (or from P2) with one corrective continuation from A1 if budget allows. At most two or three runs in total.
- Data comes only from code-verified generators. For conversational style you may add model-written replies that pass every check against the fact sheet (verified self-distillation); nothing unverified enters training.
- Vary wording with templates kept separate from every evaluation wording, and hold out whole wording families.
- Every candidate must pass the regression gates in section 6 on validation before it can be selected. A candidate that fails is reported, not selected.
- Consider rank or layer changes only if a joint run still shows the see-saw.

**WP6. BaziQA as an outside reference (optional, last).**
One pre-registered replication of answer voting with new seeds, for one or two systems. Report it next to the constant-answer baseline. Do nothing else on BaziQA.

**WP7. Chatbot.**
Integrate only what the experiments supported: grounded answers, the router, verification with repair and fallback, statement labels, natural multi-turn replies without forced JSON, the single-base backend. Keep the model alone as a selectable comparison mode. Test it in a real browser with made-up births: ordinary questions, follow-ups, a prediction request, a school-dependent question, a question the router does not know. Fix what you find.

**WP8. Sealed evaluation, report, quality.**
Freeze the selection, score the sealed set once for every system in the pre-registration, and write:
- `results/phase5/RESULTS.md` and `summary.json`, generated by a script from saved files.
- `docs/phase5_report.md`: what was built; each comparison with paired intervals; what improved, what regressed, what remains unsolved; training versus grounding versus verification versus routing versus merging; latency and memory; limitations; exact commands to reproduce.
- Updated `README.md`, `results/README.md` and a `data/phase5/README.md`.
Run the full test suite and Ruff, add tests for new code (splits and hashes, leakage, the seal, hot-swap equivalence, fact-sheet correctness against the rule code, router cases, verifier behaviour), and commit locally.

# 6. Acceptance gates (write them into the pre-registration; adjust numbers only with a stated power argument, before scoring)

- **System claim:** on the sealed blind-wording and rule sets, the grounded and verified system's rate of replies that contradict a verified fact is far below the best model alone (target: under 3% of replies against roughly 16% in Phase 4), with the checker's false-alarm rate reported from a read sample.
- **Training-versus-grounding claim:** report tuned-grounded minus base-grounded with its interval, whatever the sign.
- **Any new or merged adapter** must be non-inferior to P2 on the calendar set (margin and sample size from your power calculation), must not fall below A1 on rule reasoning beyond a stated margin, and must improve balanced decision accuracy on the tool-need twins over P2, R3 and A1. If none passes, say so and keep the existing adapters.
- **Router:** at least 90% intent accuracy on the blind router set, and no computable question refused as a prediction.
- **Runtime:** report it; flag any design that more than doubles the median reply time.

What happens when a gate fails is part of the result: report it and fall back (grounded pipeline with existing adapters; adapter routing if a unified model cannot hold both skills; engine derivation shown directly where the model stays unfaithful).

# 7. How to work

- Make decisions and keep going. Ask me only for something outside these rules: a download, a push, deleting anything, or a missing resource you cannot work around.
- Diagnose failures yourself. When a result looks wrong (a zero, identical replies, a sudden jump), suspect the harness first and inspect raw outputs before interpreting it.
- Keep a short running log in `docs/phase5_log.md` of decisions, surprises and negative results.
- Do not promise or imply perfect accuracy, or better contest prediction, without evidence.

# 8. Final answer

When everything is done, tell me plainly, with numbers from the saved files:
1. Which system is strongest overall, and on what evidence.
2. What grounding, verification, training, routing and merging each contributed.
3. Whether fine-tuning still matters once facts are supplied.
4. What regressed and what remains unsolved.
5. Latency and memory of the final system.
6. What changed in the chatbot, and what a user will notice.
7. Which claims are confirmatory, which are exploratory, and the main threats to validity.
8. What you would do next.
