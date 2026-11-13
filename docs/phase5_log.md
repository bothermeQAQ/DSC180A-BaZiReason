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
