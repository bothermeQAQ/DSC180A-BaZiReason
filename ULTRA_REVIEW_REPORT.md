# Independent research and engineering review

Date: **November 3, 2026** (America/Los_Angeles). Original reviewed commit: **a50f967**, preserved at **`audit/pre-review-a50f967`**. This review audits and improves the existing project; it does not claim that a new live-model study has been completed.

## 1. Overall academic assessment

BaZiReason is a defensible **prospective LLM reliability study with a computational instrument**, rather than evidence about LLM performance already obtained. The original report was generally honest about missing experiments and reference limits. Its 42 passing tests were reproducible, but insufficient to establish chart truth, diversity, or identification of the proposed effects. This audit found actual wrapper and evaluation bugs, plus experimental design gaps.

The most important scientific issue is H2: a model receiving the complete correct output can succeed by copying. The implemented deterministic copier obtains 126/126 exact pillar matches without a model. That is an executed demonstration of the design's trivial ceiling, not an LLM improvement or external validation. The revised design retains full context as an assisted-answer condition and adds preprocessing and erroneous-context controls. H3 now has an explicit/minimal policy factor with common computation rules and matched specified/omitted inputs. Those changes make a real study more informative; **H1, H2 and H3 remain NOT RUN on live models**.

The contribution is bounded reproducible measurement and error analysis, not a first BaZi benchmark, general reasoning advance, or endorsement of destiny prediction. A final project consisting only of this calculator, fixtures and prospective methods would not establish its central empirical question. The December report needs real matched model observations or an explicitly agreed scope change with the mentor.

## 2. Actual bugs discovered

| Bug | Independent evidence | Repair |
|---|---|---|
| Timezone provenance depends on host constructor cache | Original resets TZPATH but cached ZoneInfo objects persist; hostile-cache subprocess reproduces wrong potential source | Read explicit pinned tzdata resource bytes without process-wide mutation |
| Extreme valid ISO dates crash before rejection | Original year 0001/Shanghai and 9999/LosAngeles raise OverflowError | Controlled unsupported/outside-range responses, including reference conversion overflow |
| Trial identity aliases and method-list miscount | Original accepts `False` as trial 0; other numeric identities alias slots | Require genuine integer trial IDs and proper unique list/tuple methods; reject string/mapping collections |
| Output schema weaker than advertised | Empty profile objects/blank reasons can pass; duplicate JSON keys silently overwrite | Strict representation and nonblank reasons; reject duplicate keys and flag schema failures |
| Run declaration disappears on interruption | Original manifest is written after all completions | Write planned manifest before inference and bind scoring to hash/IDs/methods/trials |
| Missing H3 outcome | Narrow contradiction metric ignores ok answers on ambiguous input | Add ambiguity-only definitive claims, correct policy handling and coverage; preserve narrow contradiction definition |
| Invalid profile survives in error output | Stricter audit schema exposed malformed selected profiles in invalid responses | Return conventions=null for invalid profiles; eight representation regressions |

Locations, evidence, priorities and per-bug verification are in [AUDIT_FINDINGS.md](AUDIT_FINDINGS.md). A post-fix test pass is not substituted for before-fix evidence: the original overflow and boolean-trial reproductions are retained under `results/audit/`.

## 3. Methodological weaknesses

The original 96-case suite had93 unique input/profile pairs: four nominally separate timezone rows shared one Shanghai input. It sampled six of twelve month-start Jie, and its same-date midnight examples were not adjacent to the preceding 23:00 observations. The original 184 agreeing cells were 80 day+ 52 year+ 52 month, only 117 distinct reference-date/pillar queries. They excluded all hours and all term-date year/month comparisons. The report disclosed some exclusions, but the headline count did not quantify repeated evidence.

H1 still lacks fully matched near/far ordinary controls. Expanding boundary coverage does not remove year/time/task confounding. H2 full context still contains reference answers by design; corrupted-context following tests one reliance mechanism, not internal reasoning. H3 manipulates explicit policy instructions rather than discovery of universal conventions. Its16 ambiguous references and 10 new matched groups are small. Missing responses can lower definitive-claim frequency, so correct policy handling and scorable coverage must accompany it. Semantic clarification usefulness remains ungraded.

All generated pillars still have single-engine provenance. sxtwl is a distinct implementation with shared ShouXing ancestry. Published HKO anchors cover only two days and a minute-level term. Independent JPL/Skyfield timestamp results strengthen selected astronomical evidence but do not validate all four pillars or all years. The fixed UTC+8 profile is not birthplace-local charting. Public facts may appear in pretraining; synthetic dates do not prove contamination absent. Purposive sampling, related groups, provider drift and multiple exploratory contrasts limit generalization.

## 4. Fixes and design improvements implemented

The original Git tag/data remain available. A separately versioned **152-case v2** suite has 126 scorable computations, 16 ambiguous, 6 unsupported and 4 invalid inputs. All input/profile pairs are unique. It covers all 12Jie in 2024/2025 with 48 bracketing cases, 20 Zi/midnight profile cases, distinct equivalent-instant groups, and 24 new matched H3 records in 10 groups. Equivalent normalized inputs never cross groups. The 22-case operational pilot spans every category/status and both Zi profiles.

Implemented conditions: direct, full context, timestamp-only context, corrupted-hour context, deterministic copy; every model condition has a minimal-policy counterpart. Computation rules and schema stay fixed within policy pairs. The runner rotates order, hashes prompts/data, retains failures, writes manifests before calls, and scores against those declarations. Interruption cannot silently reduce the denominator. Exact IDs replace reliance on an ordinary-only prefix. `scripts/run_pilot.py` provides reproducible H2/H3/combined commands; mocks remain the default.

Added metrics distinguish joint convention/pillar success, scorable coverage, ambiguity/nonanswer definitive claims, policy correctness and corrupted-hour adherence. Paired whole-group resampling conditions on the synthetic suite; it is not a population confidence guarantee. Existing pillar-only exact accuracy remains a valid separately named definition, not falsely classified as an arithmetic bug.

## 5. Independently executed verification

| Executed check | Actual outcome |
|---|---|
| Fresh environment installed from original Git archive and pinned lock | Original 42 tests PASS |
| Original bug probes | Both extreme-date OverflowErrors and boolean trial acceptance reproduced |
| Revised full pytest, CPython 3.14.2 | **106 PASS, 0 FAIL** |
| Fresh README installation, CPython 3.12.14, revised full pytest | **106 PASS, 0 FAIL** |
| Ruff lint and format | PASS |
| `pip check` in both audit environments | PASS |
| v1/v2 generation through documented CLI | Byte-equal to committed artifacts |
| V1 partial sxtwl comparison |184/184 cells,117 distinct reference queries |
| V2 partial sxtwl comparison |270/270 cells,169 distinct reference queries |
| Supplemental matched-convention hour comparison |126/126; shared lineage, separate denominator |
| Independent Skyfield 1.53/JPLDE421 calculation |24 unique Jie 2024-2025; lunar-python minus Skyfield −1.168115 to +2.261093 s; sxtwl minus Skyfield −15.783111 to −13.018692 s |
| HKO checks | Two published day values and minute-resolution Lichun pass |
| Deterministic copying |152 records, zero model calls;126/126 pillar agreement tautological |
| Combined representative mock smoke |154 observations: 132 synthetic completion invocations plus 22 deterministic controls; no live inference |
| Python pilot wrapper H3 smoke |88 synthetic completion invocations; manifest-bound scoring completes |
| Report compilation, text and visual review | Seven letter-size pages; all pages inspected; no clipped content, unresolved references or placeholders |

The ephemeris computation uses apparent geocentric solar longitude in the ecliptic of date, IAU2000B nutation and bundled time-scale data. The downloaded DE421 kernel is hashed and kept outside Git. Its residuals are a distinct numerical comparison, not absolute astronomical proof or a diagnosis of the old library offset. No guessed correction or silent relabeling was made.

The revised tests cover all 12Jie in 1901/2024/2025/2099, Zhongqi non-boundaries, both sects/all hours, Lord Howe half-hour DST and Apia's skipped day, host caching, invalid paths/ranges, dataset coverage/grouping, strict schemas, duplicate JSON, malformed identities, policy-only deltas, corruption/copy controls, declared-denominator survival and clean CLI execution. Unit fixtures are still fixtures. They do not establish live LLM behavior or exhaust 1901-2099.

Evidence: `results/audit/original_pytest.txt`, `revised_pytest.txt/xml`, `clean_py312_tests.txt`, lint/format logs, original bug/coverage JSON, v2 diagnostic JSON, independent astronomy JSON, copy/mock artifacts and `verification_summary.json`.

## 6. Remaining scientific limitations and access

No configured API variables, installed local inference binaries/packages, or usable local endpoint were found in the inspected environment. No Codex subscription credential was repurposed as an API key. **No paid model calls, model-weight downloads, training, or live hypotheses were fabricated.** Public ephemeris/package downloads were performed solely for reproducible computation. Native Anthropic/Ollama protocols are not implemented; an authorized OpenAI-compatible endpoint is required for the current live runner.

A real pilot is an access blocker for empirical conclusions, not a reason to invent results. The control/ablation infrastructure and exact commands are ready. Provider pricing must be checked before live use; invocation/output-token caps are not monetary caps. No automatic retry/resume, semantic explanation grading, universal solar-time profile, broad platform certification or independent complete-chart oracle is claimed.

## 7. Checkpoint readiness

The live official Q1 page still gives **November 6** checkpoint and **December 4** final. Public content requirements are title, abstract and introduction including motivation/prior work/data, plus public code with reproducible README. The revised report contains those sections; the code/package/data/commands are present. No Gradescope submission has occurred. Exact hour, assignment identity, mentor approval/additions and private Piazza details still need student verification.

Style conformity is improved, but exact whole-template identity remains **PARTIAL**. The audit restored XCharter text/Charter math and Arial sans while retaining verified margins/title/abstract/citation spacing. The source ZIP download stalled and yielded no archive; monospaced fallback and plainnat remain disclosed. The report was successfully compiled and all seven pages inspected. Obtain the full official bibliography/style files or mentor/TA acceptance if strict unmodified-template conformance is required. No original-style hash was invented.

Overall: a reviewable checkpoint package with working audited infrastructure and explicitly preliminary computational evidence; **not a completed LLM research paper**. See [CHECKPOINT_STATUS.md](CHECKPOINT_STATUS.md) for exact files and remaining submission actions. GitHub synchronization/visibility are recorded separately after upload, not inferred from the previous publication log.

## 8. Prioritized December final-project plan

1. **November 3-6:** review this audit/report, confirm topic scope and template/private instructions, freeze checkpoint, submit correct PDF and public repository link. Do not delay this for training.
2. **November 7-13:** obtain an authorized model endpoint; run one 22-case same-model combined pilot (132 bounded completion invocations), inspect failures/truncation and costs, freeze prompts and a grouped held-out evaluation. Do not tune on scored answers.
3. **November 14-20:** add matched near/far H1 groups and independent calendar anchors, target 200-300 carefully checked cases, pre-register primary H2/H3 contrasts and coverage safeguards. More cases alone are not validity.
4. **November 21-27:** run repeated paired conditions; second model only if budget/access permits. Report group counts, missing/format failures, coverage, policy correctness, copy/adherence controls and clustered intervals. Inspect representative errors manually.
5. **November 28-December 3:** complete results/limitations, reproduce artifacts from a clean clone, use accepted final template, and reconcile every report claim with a saved observation.
6. **December 4:** verify private deadline and submit. If model access remains unavailable, discuss a scope change with Professor Shang rather than presenting unperformed evaluation as completed research.

Fine-tuning is optional and lower priority. It should proceed only after frozen labels/splits and meaningful untuned baselines, with audited licenses and distinct years/transition groups. It is not necessary to make the controlled reliability study coherent.
