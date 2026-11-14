# Phase 5 working log

Decisions, surprises and negative results, in order. The plan is `docs/phase5_execution_prompt.md`; the frozen design is `docs/phase5_preregistration.md`; the report is `docs/phase5_report.md`.

## 13 November 2026

- **Branch.** `phase5/grounded-system`, from `phase3/deep-reasoning` at `c113003`. The brief named `docs/PHASE5_OPUS_PROMPT.md`; the file in the repository is `docs/phase5_execution_prompt.md` (same content).
- **Verified before building.** The review's facts were rechecked from saved files:
  - all 7,760 Phase 4 final-test replies rescore to their stored scores;
  - swapping adapters in place reproduces a fresh load exactly;
  - the R3↔A1 interpolation probe numbers are in the brief.
- **Single-base backend** (`src/bazireason/adapter_bank.py`). One base model; adapters swapped with `load_weights` (milliseconds); "base" is every `lora_b` set to zero; `mix:<a>:<b>:<alpha>` interpolates.
- **Fact sheet** (`src/bazireason/facts.py`). All verified facts of a chart as labelled lines: 确定 / 约定 / 分流派. The claim checker finds no failure in a sheet (checked).
- **Router** (`src/bazireason/intent.py`). Keyword rules in a fixed order; `plan_for(intent)` builds the oracle plan with the same code, so the oracle and the routed prompt are identical whenever the router is right.
  - Every oracle plan reproduces its row's graded label: 294 Phase 4 rule rows (families with an exact plan) and 280 independent questions.
  - Excluded ambiguity kinds: `yongshen_named` and `gender_needed` use a question variant drawn at random during generation, and the `hour_*` kinds have three-pillar charts.
- **Bug found and fixed while checking:** `indep_item` took the last four pillar tokens of a question; 旬空 questions repeat the day pillar after the chart, so 19 of 280 rows got the wrong chart. It now takes the first four (280/280 agree).
- **Blind questions, writer model.** MiniCPM5-2B, the other model family available, produced unusable questions. Examples:
  - a prompt echo;
  - "关于空亡的问题。";
  - questions asking about "你的八字" (the wrong speaker).

  **Deviation from the brief:** Qwen3.5-4B (a newer Qwen generation, untuned) writes the questions. It is not a different family, which limits the independence claim. Filters were fixed in the script before generation.
- **Blind sets:** dev 195, valid 156, sealed 314 questions. Only dev questions have been read.
- **Router tuning, on development data only.** Fixes:
  - ordering: flow and overview before single facts;
  - "对象" in "合冲刑害破对象" was matching the marriage domain;
  - stem versus branch positions;
  - a quoted branch before 藏;
  - judgement and prediction phrases.

  Router accuracy: dev blind 156/195 → 190/195; dev template rule rows 294/294; dev independent questions 280/280. **Frozen from here;** validation and sealed questions are routed by this version.
- **Validation split resized before use.** The first build had only 79 rule rows; the validation year block yields fewer rows per scale. It was rebuilt with scale 2.0 (202 rows) before anything was scored or read; only counts had been printed. The sealed split is unchanged.
- **Phase 4 scoring artefact, recorded as an addendum** (`results/phase5/addenda/indep_rescore.json`). Content-lenient 旬空 scoring gives A1 212/280 (strict 201) and R3 169 (162); no ranking changes.

## 13 November 2026, afternoon: first development results (A1, Phase 4 final test as dev)

- **A1 alone, rule rows:** 191/294; 44 replies contain a contradicted statement.
- **Full fact sheet (gf):** 165/294, **worse** than alone. Contradictions 28. A1 ignores the sheet and re-derives from its memorized tables; it errs differently with the long context.
- **Oracle-routed (go), where the code's conclusion for the exact question is supplied:** 197/294. A1 disagrees with the supplied conclusion on 97 rows (33%). Contradictions 11.
- **Routed + verified (grv):** 101 value-supplied repair turns; 93 adopt the code's value and 8 fall back to the engine. Final 294/294 correct (the router is 100% on these template rows), with 0 final contradictions.
- **Interim conclusion.** The pipeline fixes accuracy; the remaining failure is first-pass compliance (about a third of answers need a repair turn). Under test:
  - prompt version 2: conclusion after the question, plus an explicit "adopt this, do not re-derive" (`--prompt-version 2`);
  - the untuned base, which was never drilled to derive.

  Training (`train_data5.py`, p5_train_v1, ready but not built) only if both fail.
- **Queue state:** `queue_dev.sh` (A1) hands over to `queue_dev2.sh` (base, then R3) through `switch_after_a1.sh`. The interpolation grid in `queue_dev.sh` is skipped by that switch and has to be run separately.

## 13 November 2026, 15:00: A1 development matrix complete; base partial; scorer fix

A1 (Phase 4 final test as dev, plus 195 dev blind questions):

| Part | Alone (w0) | Full sheet (gf) | Oracle-routed (go) | Routed (gr) | Routed + verified, final |
|---|---:|---:|---:|---:|---:|
| Template rule rows (294) | 191 | 165 | 197 | 197 | 294 |
| Independent (280) | 212 | 219 | 243 | 243 | 280 |
| Blind (195) | 107 | 122 | 158 | 156 | 190 (= the 190 routed correctly) |

- **Grounding does not help where the model ignores it.** On template rules, interactions fall 25 → 10 and patterns 21 → 13 with the conclusion supplied; strength (24 → 30) and domain (9 → 33) rise. On blind questions the supplied conclusion helps strongly: 107 → 158.
- **Contradicted statements at first pass:** w0 120/769 replies; go 12/769.
- **Base, dev rule rows:** w0 1/294, gf 8/294, go 12/294. The base also re-derives at length in Markdown and writes free-form JSON values (`"是"` for true, a sentence for the label). It does not adopt the supplied conclusion either, and cannot serve as the generator. Its remaining dev runs were dropped: validation measures base under gr anyway, as pre-registered.
- **Scorer fix before validation (dev finding).** A list answer written as one string inside a list (`["申酉"]`) was still marked wrong; the fact scorer now reads list answers by content in every form. `scripts/phase5/rescore5.py` recomputes stored scores from raw replies once the runs are finished.
- **Queue:** `queue_dev3.sh` runs prompt version 2 for A1, then the interpolation grid (0.15, 0.25, 0.35), then dev calendar rows for P2, R3 and A1 (reused from the Phase 4 cache).

## 13 November 2026, 15:30: prompt version 2 chosen; G1 training justified; design frozen

- **Prompt version 2** puts the routed conclusion after the question, with "请直接采用上面这个已核验的结论作答…". A1 under go, version 1 → 2:

  | Part | Agreement with the supplied conclusion | Contradicted statements |
  |---|---:|---:|
  | Rule rows (294) | 197 → 203 | 11 → 4 |
  | Independent (280) | 243 → 260 | 0 → 0 |
  | Blind (195) | 158 → 168 | 1 → 6 |

  It is used for every validation and sealed run, the conversations and the chatbot.
- **Training G1, justified by a measured failure prompting did not fix.** Even with version 2, A1's first pass disagrees with the supplied conclusion on 91/294 rule rows; the base does not adopt it either (go 12/294, mostly free-form output). One continuation from A1 on `p5_train_v1` (2,260 code-written rows, prompt version 2, max 1,990 tokens; no evaluation wording, checked by a test). It runs after the interpolation grid (`queue_train.sh`), then G1's dev runs.
- **Pre-registration completed and committed before any validation scoring:**
  - G1 joins the candidates;
  - conversation guard;
  - sealed run list;
  - batch 32 for validation and sealed.
- **Queues:** `queue_valid5.sh` starts after `QUEUE_TRAIN_DONE`. `queue_sealed5.sh` will be started by hand after the validation outputs have been checked for harness bugs.

## 13 November 2026, 21:50: G1 trained; G1 development results; validation running

- **G1 training** (`configs/phase5/qwen3_4b_lora_p5_g1.yaml`, continued from A1): 1,130 iterations in 92 minutes, peak memory 23.6 GB, no truncation. Loss-validation fell 0.265 → 0.009 by iteration 500 and 0.007 at the end; the last checkpoint is kept. The loss is low because the targets are code-written and short, so it does not by itself show the model adopts the conclusion; the development runs below do.
- **G1 on development data** (prompt version 2), against A1 with version 2:

  | Part | G1 alone (w0) | G1 oracle-routed (go) | A1 go | G1 routed + verified, final |
  |---|---:|---:|---:|---:|
  | Template rule rows (294) | 176 (A1 191) | 256 | 203 | 294 (38 repairs, 1 fallback) |
  | Independent (280) | 199 (A1 212) | 280 | 260 | 280 (0 repairs) |
  | Blind (195) | 108 (A1 107) | 184 | 168 | 190 (gr 180; 12 repairs) |
  | Calendar (248) | 174 (A1 167, P2 —, R3 175) | | | |

  First-pass disagreement with the supplied conclusion falls from 91/294 to 38/294 on rule rows and from 20/280 to 0/280 on independent questions. Model-alone rule accuracy drops 15 rows on template rules and 13 on independent questions: G1 has learnt to lean on the supplied conclusion.
- **Interpolation (dev, w0; `results/phase5/dev_alpha.json`):** α = 0.15 has 639 correct, against 637 for 0.25 and 626 for 0.35, so M\* = `mix:p3r3:p4a1:0.15`.
- **Validation** (`queue_valid5.sh`) started at 19:51. Calendar and model-alone parts are done; the gr/grv runs and the conversations are under way. `queue_sealed5.sh` will start after these outputs have been checked.

## 14 November 2026, 00:30: validation finished; frozen selection; sealed run started

- **GPU memory failure, fixed in the harness.** The untuned base ran out of GPU memory at batch 32 on its repair turns (its first replies run to the 800-token cap). Chunks are now capped at 80,000 padded tokens and saved to the cache as they finish; the base's gr/grv runs were rerun with its stored first replies (`queue_valid5b.sh`). Pre-registration section 8, item 1.
- **Checks before opening the sealed split:**
  - every validation prompt was built with prompt version 2: 20 of 20 sampled stored prompt hashes match version 2, 0 match version 1;
  - no empty replies;
  - rescoring validation records with the final scorers changes nothing;
  - rescoring dev records changes one A1 blind item per file (the `["申酉"]` list fix);
  - the one dialogue every system fails (`P5V-convo-0017`) is a router error: "日主癸水在月令丑金上…十二长生" was routed to the day pillar, not the month.
- **Claim checker false alarms:** 30 of the 335 flagged validation replies, drawn at random, were read by hand. All 30 were real errors, one of them corrected by the model in the same sentence; none was a false alarm (`results/phase5/addenda/checker_flags_valid.json`).
- **Selection (`results/phase5/selection.json`):**

  | | Base | R3 | A1 | G1 | M\* (0.15) |
  |---|---:|---:|---:|---:|---:|
  | gr first-pass correct (of 498) | 203 | 386 | 408 | **461** | 387 |
  | gr replies with a contradicted statement | 23 | 9 | 9 | 2 | 8 |
  | grv final correct | 489 | 491 | 491 | 491 | 491 |
  | grv dialogues fully right (of 30) | 29 | 29 | 29 | 29 | 29 |
  | grv dialogues right at first pass | 1 | 11 | 10 | 25 | 12 |
  | Calendar, model alone (of 388) | | 253 | 249 | 264 | 272 (P2 254) |
  | Balanced decision accuracy, calendar | | 67.4% | 71.5% | 72.7% | 72.7% (P2 69.4%) |

  - **M\* fails gate (b):** model alone it gets 319 rule answers against A1's 347 (−5.6 points; the limit was −3). Gates (a) (calendar lower bound +0.8 ≥ −7) and (c) (balanced accuracy above P2, R3 and A1) pass. M\* is therefore not a candidate, and P-3 is not tested.
  - **G\* = G1**, by first-pass correct answers under gr.
- **Sealed run** (`queue_sealed5.sh`) started at 00:27; every access is logged in `results/phase5/sealed_access.jsonl`. `queue_valid_extra.sh` follows with the pre-registered secondary measures that were still missing on validation: base and R3 alone, for the 2 × 3 table, and gf and go for G1.

## 14 November 2026, 06:30: sealed results; corrections; chatbot

- **Sealed run** finished at 06:04 with no failed step (`artifacts/logs/p5_queue_sealed5.log`, six entries in `sealed_access.jsonl`).
  - The system (grv, G1) ends with 991/1005 correct, against 688 for A1 alone. P-1 +30.1 [+27.2, +33.1].
  - 0 final replies with a contradicted statement (first pass 2/1005).
  - Router 302/314 on the blind questions.
- **All 14 wrong final answers are routing errors.** Five prediction requests ("什么时候容易生病", "哪一年运势会好转") went to `unknown` or the wealth domain. Three "日主X生在月令Y" questions went to the day pillar (the same error as the one failed validation dialogue). Three "能不能长寿" questions went to the prediction refusal. The router stays frozen; these are reported, not fixed.
- **Gate computation corrected after the sealed results.** The refusal gate counted any non-prediction question sent to the prediction refusal. The three "能不能长寿" rows are `ambig:no_rule`, which has no rule answer, so they are not computable questions. The gate now counts computable intents only and lists the three next to it; they stay scored wrong.
- **Swap time.** `RESULTS.md` reported the upper middle of eight values as the median (4.68 ms); it now gives the median, 4.1 ms, and the range.
- **New exploratory tables in `RESULTS.md`:**
  - what each step adds, paired on the same items;
  - the plan's three adapter checks applied to G1 afterwards. On validation G1 passes all three. On sealed its calendar lower bound is −6.7, and its balanced decision accuracy (73.39%) is below Phase 2's (74.46%).
- **Chatbot:**
  - G1 joins the model list and is the default generator; R3, A1, Phase 2 and the base stay selectable. The Phase 3 test of the model list now includes it.
  - G1 opens with "依据规则代码已核验的结论作答" even on `unknown` turns, where nothing was checked (7 of 7 such replies on validation and sealed). The chatbot now removes that line when there is no checkable conclusion (`demo_app.drop_unverified_lead`).

## 14 November 2026, 07:30: secondary validation runs; browser test of the chatbot

- **Secondary validation runs** (`queue_valid_extra.sh`, finished 06:59): base alone 64/498 (154 readable conclusions), R3 alone 309, G1 with the full fact sheet 366, G1 oracle-routed 467. The 2 × 3 table and the paired steps are in `RESULTS.md`.
- **Browser test** (two made-up births; G1 grounded, base grounded, R3 alone). Facts, pushbacks, a prediction request, a school-dependent question, an unknown question, relations and flow years all behaved as designed; two repairs were seen and the badge named the wrong statement. Fixed:
  - the fate note on pushbacks ("是不是" is a keyword): in grounded mode the routed intent now decides;
  - the false "verified" opening on `unknown` turns (above).

  Not fixed: G1 answers "why" and "simpler" by restating the conclusion.
