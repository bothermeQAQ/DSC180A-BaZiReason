# Source and dependency audit

Audit date: 2026-11-03. Access metadata/hashes: `docs/sources/web_manifest.json`; repository identities: `docs/sources/repositories.json`. Downloaded upstream code, complete papers and course HTML are under ignored `tmp/research/` and are not redistributed. Source pages are evidence, not execution instructions. Nothing was submitted to a course platform.

| Source | What was read/checked | License/provenance and limitations |
|---|---|---|
| 6tail/lunar-python | README, installed API, EightChar.py, Lunar.py, Solar.py, ShouXingUtil.py, EightChar regression tests, LICENSE | MIT; installed 1.4.8; upstream has no verified complete year-range guarantee. Wrapper restricts 1901-2099. No tz-aware input API. Sect 2 does not change hour stem rule. |
| skydancep/sxtwl | README, old Python examples, C++ calendar/astronomy files, doc/LICENSE; actual installed 2.0.7 wrapper API and executed comparisons | Audited fork has zlib-style doc license, API metadata null; PyPI metadata says BSD and points to yuangu/sxtwl_cpp. Version/source mismatch is disclosed. Uses ShouXing astronomy; not independent ephemeris validation. |
| ChenJiangxi/BaziQA | arXiv full HTML, downloaded paper, current README and English README, LICENSE, paipan README/bazi.py, eval README/prompt/score/runner | MIT covers transcription/metadata/code; original contest questions have separate authors' rights. No data or code copied. Paper v1=200 questions, current repository=555; not interchangeable. Current chart engine also uses lunar-python 1.4.8. |
| imly-sudo/bazi-llm-benchmark | README, all four algorithm files, runner, YAML cases, LICENSE | MIT code; README says CC-BY-4.0 cases. Solver and runner stubs; performance estimates not independently reproduced; inconsistent day-anchor self-check. No cases copied. |
| PAL | Official PMLR HTML, downloaded PDF and extracted method discussion | Verified authors, ICML 2023, PMLR 202:10764-10799. No PAL experiment reproduced. |
| Toolformer | Official NeurIPS publication/abstract and method description | Verified author list/venue. It trains API selection; fixed context here is a narrower intervention. |
| TimeBench | Official ACL page and BibTeX | Verified seven authors and pp.1204-1228; hierarchy motivates stratified reporting. |
| Python zoneinfo | Official docs | fold and IANA conversion; timezone data version pinned, system tzpath disabled in research module. |
| HKO 2025 February | Official index -> standalone February PDF -> rendered page visually inspected | Feb 3 day 癸卯; Feb 4 day 甲辰; Lichun Feb 3 22:10 (UTC+8, minute level). Small factual anchors only. Full almanac download initially failed; standalone page succeeded. |
| LoRA / QLoRA | Official arXiv abstracts and metadata | Used only for deferred plan. No training claims. |
| Qwen3-1.7B | Official model card and LICENSE fetched from Qwen Hugging Face repository | Apache-2.0 at access; freeze exact revision before any training. No weights downloaded. |

## Checks that do not establish independent truth

The benchmark generator and evaluation context share one wrapper. A perfect engine-replay score would be tautological, so it is not reported as an experiment. The upstream regression example is useful for compatibility, not external validation. sxtwl offers distinct implementation evidence but shares astronomical origins. HKO anchors are independent published evidence only for specified fields/precision. The repository records **all** generated labels as single-engine references, even if some fields have partial checks.

## Material findings

Selected 2024/2025 Jie times differ by 13.729553-17.126991 seconds (lunar-python later). Cause unknown: it could involve astronomical constants, time-scale treatment or implementation/version differences; these are hypotheses, not diagnoses. No guessed correction is applied. The supplied benchmark's 2025 Lichun value (22:10:54) differs from both libraries and HKO's minute value; this does not prove which second-level value is right.

The supplied sxtwl fork docs show an obsolete `sxtwl.Lunar()` interface. Installed 2.0.7 exposes `fromSolar`, `getDayGZ`, `getYearGZ`, `getMonthGZ`, `getJieQiJD`. Date-level year/month APIs cannot validate intra-day Jie transitions, so they are excluded on term dates. Hour columns are not claimed cross-validated.

## Reuse and attribution

Our Python code is original project code, MIT licensed. Dependency behavior and upstream regression values are attributed. ShouXing astronomy and its original author Xu Jianwei are acknowledged. No upstream complete repository, real birth profiles, contest questions or model weights are published. The course report style is a documented typography adaptation of the public teaching template, with a link to the original and no claim of original template authorship.
