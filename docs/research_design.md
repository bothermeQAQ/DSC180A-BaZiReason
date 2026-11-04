# Research design

## Question and estimands

Can deterministic context improve same-model factual calendar agreement and convention handling? We measure implementation-profile agreement, not destiny prediction. The 96-case suite is an initial diagnostic instrument, not an estimate of natural request prevalence.

| Hypothesis | Planned estimand | Current state |
|---|---|---|
| H1 | Direct error rate on boundary vs ordinary strata | NOT RUN; current strata unmatched, so descriptive only |
| H2 | Within-case exact-match difference, tool context minus direct | NOT RUN; paired runner implemented |
| H3 | Change in unsupported definitive-answer rate after explicit convention-policy representation | NOT RUN; additional ablation required |

No live model access was detected. HTTP fixtures and mocks verify software, not hypotheses. Neither deterministic recomputation nor cross-library agreement estimates a model's performance.

## Profiles and input policy

Input is a naive ISO wall datetime `YYYY-MM-DDTHH:MM:SS`, IANA timezone, optional fold, and selected convention. Pin tzdata 2026.6 and force its use instead of system tzdata. Round-trip through UTC distinguishes nonexistent and repeated wall times. Offset-bearing strings and fractional precision are outside the admitted input syntax; incomplete strings request clarification.

Convert the civil instant to **fixed UTC+08:00 for all four pillars**. For international cases this is an intentionally defined reference-clock task, not local BaZi chart casting. Historical DST in Shanghai is removed by instant conversion. No guessed longitude/equation-of-time formula is used. True solar time, local-civil clock basis, lunar-year and lunar-month profiles are explicitly unsupported. Reference years 1901-2099 are a study admission policy; upstream's entire supported range has not been verified.

Sect 1: day advances at 23:00. Sect 2: day advances at midnight. Both use lunar-python's 23:00-advanced hour stem. We name them implementation profiles, without endorsing one school. Missing sect prompts clarification even on dates where both would coincide. `convention_dependent` cases have selected profiles and can be scored; genuinely omitted-profile cases have no expected pillars.

## Dataset and provenance

Seed: 1802026. Dataset: `data/synthetic/checkpoint.jsonl`, 96 records: 80 ok, 6 ambiguous, 6 unsupported, 4 invalid. See diagnostics for category counts. Expected labels come from lunar-python 1.4.8 only. Every record says `single_engine_reference`; no full-chart external validation is claimed.

Use group_id for clustered analysis and future splits. Jie observations share a transition group, Zi-hour conditions share a date group, timezone pairs share an instant group, DST conditions share a zone group. Ordinary noon dates are sampled in 1990-2030 on days 12-20. Two-minute Jie offsets avoid the observed cross-engine discrepancy intervals but do not prove astronomical correctness.

External checks: sxtwl day comparisons for all scorable records, year/month only on non-solar-term dates; no hour verification. Shared ShouXing lineage limits independence. HKO anchors are separately curated published facts and remain outside generation. The separate upstream EightChar regression is explicitly same-family evidence.

## Inference control

Conditions A and B share system rules, requested profile, user task, JSON schema, model, temperature, token cap, attempts and cases. A receives no computed pillars. B receives fresh engine context; it is **tool-provided context**, not autonomous function calling. Direct prompt construction reads only input and convention, never expected values or annotations. Tests mutate gold labels to verify prompt independence.

Use one attempt per call, no hidden repair/retry; retain API failures and truncation metadata. Alternate order by case and trial. Log requested/served model, fingerprint when supplied, UTC timestamp, prompt hash/version, dataset hash, token usage, latency, synthetic flag and error class. Reconstruct prompts using the pinned source and dataset. Do not log API keys, headers or error-body content. Live runner flushes each record so completed work survives interruption. Existing result files are not overwritten. Interrupted runs are not auto-resumed: score with missing observations or run a new named artifact and declared design.

The four-case CLI default is a **software/network pilot only**, and comprises ordinary dates; it cannot test H1 or H3. The full 96-case paired study requires `--limit 96 --max-calls 192` per trial. Provider pricing must be checked before increasing the call count. Call/output-token caps are not dollar caps. No paid calls were made here.

## Metrics

For each declared method/trial, all expected case IDs are counted, including missing records. Duplicate or unexpected IDs are errors.

- Exact four-pillar accuracy: all four valid sexagenary pairs equal reference / all labeled observations (80 per full trial).
- Per-pillar accuracy: matching pair / same labeled denominator. Schema-invalid outputs score zero.
- Valid-output rate: strict valid structured objects / all requested observations (96 per full trial).
- Status accuracy: expected status matched by a valid response / all observations.
- Convention accuracy: exact supplied convention metadata / all observations.
- Non-answer policy accuracy (named uncertainty_accuracy in JSON): correct non-ok status, null pillars, nonempty reasons, exact convention / 16 non-answer observations. This combines ambiguity, unsupported and invalid cases; inspect categories separately. Reasons are not semantically adjudicated.
- Detectable contradiction rate: non-ok status with non-null pillars / all observations. Other narrative contradictions are not measured.
- Repeated consistency: equal complete valid responses / all within-case unordered trial pairs. Missing/invalid outputs are inconsistent. Validity does not imply correctness.

JSON null is used for undefined denominators, latency, tokens or monetary cost. Wilson 95% intervals are descriptive only: the implemented formula ignores grouped dependence. H2 reports paired discordant counts. Future inference should bootstrap group IDs (e.g. 10,000 seeded resamples) and report paired group-bootstrap intervals. Do not run significance tests on the 184 correlated computational checks as if independent labeled examples.

## Additional experiment needed for H3

Pre-register an explicit-policy vs minimal-policy prompt factor, crossing direct/tool context. Keep actual input omissions identical, without inventing universal gold charts. Score unsupported definitive-answer frequency on omitted-profile and missing-time cases, plus status and semantic clarification utility. The current suite has few such cases: expand balanced missing-field/convention groups, predefine adjudication, and prevent a model from gaining by refusing every query (report coverage on scorable cases). This factor is planned, not implemented or tested.

## Limits and final-project work

Expand to 200-300 cases only after validating reference uncertainty and verifying the pilot. Add external term tables, second-level ephemeris validation, error inspection, model snapshot/version and repeated trials. A correct-context corruption control could test over-trust; it is not implemented. Do not begin SFT before freezing a separated evaluation set.
