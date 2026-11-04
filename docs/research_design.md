# Audited research design

Revision:2026-11-03. Original proposal/source: `audit/pre-review-a50f967`. The study measures profile-specified calendar-answer reliability. No live LLM hypothesis has been tested. Calendar diagnostics, deterministic controls and mocks are separate evidence types.

## Hypotheses and estimands

| Hypothesis | Defined comparison | Interpretation and status |
|---|---|---|
| H1 | Direct-model error on ordinary versus boundary categories | NOT RUN; current purposive strata unmatched, so descriptive only |
| H2 | Full-context minus direct exact accuracy, within same case/model/trial | NOT RUN; assisted-answer reliability, not identified internal reasoning improvement |
| H3 | Direct explicit-policy minus direct minimal-policy definitive-answer frequency on ambiguous references | NOT RUN; negative difference favorable only with correct policy handling and maintained scorable coverage |

The original full-context condition receives the entire target response. A zero-call copier obtains perfect reference agreement by construction. Thus H2 alone is a transmission task; the controls below make reliance and preprocessing experimentally separable. No behavioral contrast proves internal reasoning mechanisms.

## Named computation profiles

Input: naive ISO wall datetime with seconds, IANA zone, optional fold, selected convention. Load TZif bytes directly from pinned tzdata 2026.6; never mutate host TZPATH or reuse its constructor cache. Round-trip through UTC distinguishes gaps and repeated times; the latter requires explicit fold. Range rejection applies after normalization, and extreme date conversion overflow is controlled.

All four pillars use the civil instant converted to **fixed UTC+08:00**. This is a reference-clock task, not local birthplace chart casting. Sect 1 day advances at 23:00; sect 2 at midnight. Both preserve lunar-python 1.4.8's23:00-advanced hour stem. Year/month use its exact Lichun/Jie instants. True solar time, local civil-clock charts, lunar-year/month rules and normalized years outside 1901-2099 are unsupported. This admitted range is not an upstream accuracy guarantee. Malformed conventions return invalid with no valid selected convention; missing conventions request clarification.

## Versioned data and grouping

- **V1:**96 records,80 scorable,6 ambiguous,6 unsupported,4 invalid; preserved unchanged. Its four repeated Shanghai inputs and same-date-midnight weakness are explicitly audited.
- **V2:**152 unique input/profile records,126 scorable,16 ambiguous,6 unsupported,4 invalid. All12Jie in 2024/2025 have ±120-second pairs; Zi groups include adjacent 23:59:59/00:00:00 across dates. Timezone pairs have distinct group instants. Added 24 missing/complete cases form 10 groups:three sect,three time,three zone,one repeated-DST-fold group. Their 14 scorable counterparts guard against blanket refusal.

Seed 1802026. Group IDs keep related transition sides, profiles, equivalent instants and missing-field counterparts together. No equivalent instant/profile crosses v2 groups. All splits and uncertainty resampling must retain groups. The 22-case pilot is a fixed operational sample spanning category/status/profile conditions, not a random or powered benchmark. The original four-case prefix remains an explicitly ordinary-only smoke path.

Expected pillars still come from one engine. Every record states single-engine reference. Stored-label replay only checks integrity. V1's184 cells reduce to117 distinct reference-date/pillar queries. V2's270 cells reduce to169; 126 separately counted hour checks use explicitly matched sxtwl semantics. Date-level sxtwl comparisons exclude year/month on any solar-term date. Both libraries share ShouXing lineage.

Independent published HKO evidence checks two day labels and minute-level Lichun. A separate Skyfield 1.53/JPL DE421 calculation checks all 24Jie 2024/2025 with apparent geocentric longitude of date and bundled time-scale conversion. Lunar-python differs by -1.168 to +2.261 seconds; sxtwl by -15.783 to -13.019 seconds. This favors lunar-python's selected timestamps more closely but is not universal truth or full-chart adjudication. The cause of the library offset is unestablished. Kernel hash, model definition, versions, search tolerance and raw comparisons are recorded. No silent label changes or guessed corrections occur.

## Context and policy factors

| Condition | Added context | Purpose |
|---|---|---|
| `direct` | None | Same-model raw input/profile task |
| `tool_context` | Full engine response, including pillars/status/reasons | Assisted answer transmission |
| `tool_facts` | Normalized timestamp and provenance; no pillars/status/reasons | Preprocessing help; null timestamp can still reveal failure |
| `tool_corrupted` | Same as full context, but shift scorable hour stem by cycle index12, preserving branch | Detect following this specific plausible tool error |
| `copy_context` | No model invoked; copy fresh engine response | Tautological reference/control ceiling |

Every model condition has an `_minimal` counterpart. The explicit/minimal pair keeps user input, profile, computation rules, range rules and schema identical. Only explicit missing/unsupported/invalid handling instructions are omitted in minimal prompts. Both retain the schema's null-pillars requirement for a non-ok status. This manipulates when to choose non-ok, not calendar knowledge or formatting.

H3 primary contrast uses direct policy pairs on all 16 ambiguous cases, additionally inspecting new matched groups/category breakdowns. Full-context policy pairs are secondary: supplied nonanswer status makes them easier. Invalid/unsupported rows are secondary policy diagnostics rather than a substitute for ambiguity evidence. Reasons are checked for nonblank structure; semantic helpfulness requires a future blinded annotation rubric.

## Inference controls and logging

Same model/version, data, schema, temperature, maximum output tokens, attempts and applicable computation rules across paired arms. No few-shot demonstrations. Direct prompts read only input/convention; tests mutate expected labels and annotations without changing messages. Correct and corrupted full contexts use identical labels/instructions. Rotate method order by case/trial. One attempt, no hidden answer repair/retry. Retain truncation/finish metadata and failures.

A planned manifest precedes calls. It records data hash, selected IDs, methods, trials, prompt version, model/settings and call caps. Evaluation checks the hash and binds these declarations; conflicting overrides fail. Interrupted runs retain missing observations in denominators. Predictions flush per observation; no automatic resume exists. Choose new result paths for reruns. Legacy artifacts without manifests require explicit declarations. Record requested/served model, provider fingerprint when returned, prompt hashes, UTC times, returned usage and latency. Never persist credentials, request headers or provider error bodies. Monetary cost is null unless measured.

Four-condition H3 or H2-control pilot:88 completion invocations on22cases; H2 also has22 zero-call control records. Combined design:132 completion invocations plus22 controls. Mock invocations are synthetic and never live inference. No paid API use was made. Call/output-token caps do not guarantee a dollar cost; provider pricing/authorization must be checked before a live run. Exact commands and a Python pilot wrapper are in README.

## Metrics, denominators and safeguards

- Exact four-pillar accuracy: all four valid pairs match /126 scorable observations per full trial.
- Per-pillar accuracy: matching pair /same 126; schema-invalid/missing outputs fail.
- Joint exact accuracy: exact pillars plus exact convention metadata /same 126. Pillar-only accuracy remains separately defined.
- Valid-output and status/convention accuracy: all 152 requested observations per full trial.
- Nonanswer policy correctness: correct non-ok status, null pillars, nonblank reasons and profile /26 nonanswers; ambiguity-only subset separately reported.
- Definitive-answer frequency: an ok claim or nonempty pillar dict even if schema-invalid /nonanswer references; primary H3 subset 16 ambiguous cases.
- Scorable coverage: valid ok responses /126. Pair with correctness to detect blanket refusal.
- Erroneous-hour following: output matches the injected hour /corrupted scorable cases. This does not establish semantic reasoning.
- Narrow contradiction rate: non-ok status with non-null pillars /all observations; it is not H3's main outcome.
- Repeated consistency: equal complete valid responses /all within-case unordered trial pairs. Missing/invalid outputs are inconsistent; equal wrong answers are still wrong.

A missing response is not definitive, but is also not a correct policy response. Thus a timeout cannot count as successful uncertainty handling. JSON null marks undefined denominators. Duplicate IDs, booleans/floats as trial identities, invalid reference schemas and duplicate JSON keys are rejected.

Paired observation-weighted differences resample whole group IDs, retaining within-group cases/trials (default 1000 seeded resamples, capped 10000). Percentile 95% intervals are **descriptive conditional on this purposive suite**. They are not population guarantees. A single group receives no interval. Wilson intervals remain labeled descriptive and do not account for grouping. Report discordant counts, group counts, validity, failures and coverage alongside effects. No interval or test on deterministic replay establishes model performance or independent calendar correctness.

## Final-project decisions

By November 6, freeze/check the checkpoint and private submission details. Next run a real bounded pilot, fix provider/schema failures without tuning on test answers, add independently checked anchors and matched H1 controls, then freeze a200-300-case grouped evaluation. Repeat matched conditions and inspect boundary/policy errors before drawing claims. More models are optional; sound controls and stable reference labels take priority. Fine-tuning remains deferred. Limits include sample size/dependence, contamination, restricted conventions, incomplete chart evidence, provider drift, and inability to identify internal reasoning from output agreement alone.
