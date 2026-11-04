> **Historical (audit_v2).** Current label verification for benchmark_v3 (independent oracle, JPL boundary check) is summarized in [research_design.md](research_design.md#data-benchmark_v3).

# Calendar audit — 2026-11-03

## Assessment

The wrapper implements its declared **fixed UTC+08 reference clock** consistently in the inspected paths. This is a deliberately restricted computational convention; it is not a geographically local or true-solar-time BaZi oracle. The labels still come from lunar-python. Passing compatibility and metamorphic checks does not turn them into independently established four-pillar truth.

Two wrapper defects were found and repaired. The main cross-implementation count excludes year/month on every solar-term date, including all the synthetic Jie crossings. The perfect agreement count therefore does not validate the task's difficult intra-day year/month boundaries. A separate new JPL/Skyfield comparison supplies independent numerical evidence for the selected transition timestamps, with the scope described below.

## Findings and repairs

| Finding | Evidence before repair | Repair / verification |
|---|---|---|
| P2 — timezone provenance depended on import order | Original `calendar_engine.py:15-16` called `reset_tzpath(())`, then line 65 used `ZoneInfo(key)`. Python's constructor cache survives a path reset. A pre-existing system-data object could therefore override the claimed pinned tzdata source. Import also changed process-wide environment/path settings. | `calendar_engine.load_timezone` reads a TZif resource directly from the installed tzdata dependency using `ZoneInfo.from_file`. A subprocess test preloads a deliberately wrong cached UTC zone under the Los Angeles key and proves chart normalization still uses the package's actual Los Angeles data; the caller's path, environment and cached object remain unchanged. The generator uses this same loader. |
| P2 — valid ISO inputs could crash before study-range rejection | `0001-01-01T00:00:00` in `Asia/Shanghai`, and `9999-12-31T23:59:59` in `America/Los_Angeles`, raised `OverflowError` during UTC round trips. A maximum-date UTC input could also overflow when converted to UTC+08. | The relevant conversions now return `unsupported / outside_study_range` on overflow. Regression checks also establish that the admitted year is the normalized reference year, including both edges of 1901–2099. |
| P2 — the old diagnostics denominator concealed repeated evidence | The historical 184 rows were 80 day + 52 year + 52 month comparisons; there were only 117 distinct `(sxtwl reference date, pillar)` queries. All hours and all solar-term-date year/month fields were absent from that count. | Diagnostics now expose per-pillar coverage, distinct query counts, excluded case IDs and actual statuses. A separately named hour comparison has its own denominator. Label replay is explicitly identified as an integrity check using the same engine, and stale labels are detected. |
| Dataset design gap — narrow boundary coverage | The original suite covered six of the twelve Jie in two years. Its midnight examples used the same date as the preceding 23:00 examples, so those rows were not successive rollover pairs. | The revised suite adds the remaining six Jie and successive-midnight Zi comparisons. Calendar contract tests separately exercise all twelve Jie, Zhongqi non-boundaries and both sects. These changes increase coverage; they do not supply an independent label source. |

The new timezone resource loader also rejects malformed, non-normalized, directory, overlong and non-directory-prefix identifiers with a controlled invalid response. It does not silently fall back to host data.

## Convention evidence

The installed lunar-python 1.4.8 files `EightChar.py`, `Lunar.py` and `util/ShouXingUtil.py` are byte-identical to the previously downloaded source snapshot (`docs/sources/repositories.json`, commit `009870fbab8784901662e8d36571056d0a271f78`). Inspection establishes:

- `EightChar.getYear` and `getMonth` select exact Lichun/Jie methods.
- `getDay` selects the civil-date day for sect 2 and the 23:00-advanced day for sect 1.
- `Lunar.__computeTime` derives its stem from the advanced day for both sects. Changing sect does not rederive the hour from the displayed sect-2 day.

These are implementation contracts from the [pinned upstream source](https://github.com/6tail/lunar-python/blob/009870fbab8784901662e8d36571056d0a271f78/lunar_python/Lunar.py), not a claim that one Zi tradition is uniquely correct. The existing `Convention` specification describes this behavior accurately; no convention change was justified.

The current sxtwl v2 [day API source](https://github.com/yuangu/sxtwl_cpp/blob/master/src/day.cpp) corroborates that its year/month comparisons are date-based. Unlike the old skydancep fork's API, installed sxtwl 2.0.7 exposes `fromSolar`, `getDayGZ`, `getYearGZ`, `getMonthGZ` and `getHourGZ`. The current [hour implementation](https://github.com/yuangu/sxtwl_cpp/blob/master/src/sxtwl.cpp) keeps a step of 12 at 23:00 when `isZaoWanZiShi=True`. Its stem then advances modulo 10. Installed-binary probes over 60 consecutive dates × 24 hours reproduce that behavior. The explicit call `getHourGZ(hour, True)` matches this project's advanced-hour-stem rule; `False` has a different 23:00 stem. Source comments and boolean names should not be substituted for the executed check. These current upstream files were consulted as supporting source evidence, not proved to be the exact source archive that built the installed wheel.

The [official Python documentation](https://docs.python.org/3/library/zoneinfo.html#zoneinfo.reset_tzpath) explicitly describes cache persistence after `reset_tzpath`. Its `from_file` constructor makes a new object from supplied TZif bytes. This supports the repaired loader's independence from the process's existing constructor cache.

## Reproduced numerical evidence

| Diagnostic | Historical 96-case suite | Revised 152-case suite |
|---|---:|---:|
| OK / ambiguous / unsupported / invalid | 80 / 6 / 6 / 4 | 126 / 16 / 6 / 4 |
| Main day/year/month comparisons agreeing | 184 / 184 | 270 / 270 |
| Day / year / month rows in main count | 80 / 52 / 52 | 126 / 72 / 72 |
| Distinct reference-date/pillar queries in main count | 117 | 169 |
| Distinct normalized instant + sect inputs among OK cases | 73 | 122 |
| Supplementary matched-convention hour comparisons agreeing | 80 / 80 | 126 / 126 |
| Requested / compared solar-term timestamp pairs | 12 / 12 | 24 / 24 |
| lunar-python minus sxtwl timestamp range | 13.729553–17.126991 s | 13.312561–17.126991 s |

The hour rows are deliberately excluded from the historical main count, and they still share calendar lineage. Duplicate input pairs, sect variants and repeated date-level queries make these counts correlated. They are unsuitable as independent sample sizes for a confidence interval about calendar correctness.

At `2025-02-03T22:10:20+08:00`, lunar-python returns year/month `甲辰 / 丁丑`; it switches to `乙巳 / 戊寅` at `22:10:28`. The sxtwl Lichun timestamp is `22:10:13.057709`. Thus the libraries' transition instants disagree in an approximately 15-second window. The benchmark's ±120-second cases avoid that window. No seconds-level correction was inferred or applied. The HKO minute anchor cannot adjudicate the whole interval.

## Independent astronomical comparison added during the audit

`scripts/check_solar_terms.py` and `results/audit/independent_solar_terms.json` add a distinct JPL DE421 / Skyfield 1.53 numerical path for all 24 Jie in 2024–2025. I inspected both the project script and the installed `almanac_east_asia.solar_terms` implementation: the latter uses Earth-centered apparent solar longitude, the ecliptic/equinox of date and IAU2000B nutation. The script converts the returned UTC instant to fixed UTC+08 and compares like-for-like term names and years. This is consistent with [Skyfield's documented solar-term workflow](https://rhodesmill.org/skyfield/almanac.html#solar-terms); no timezone, term-index or frame-selection error was found in this review.

For these selected transitions, lunar-python minus Skyfield ranges from **−1.168115 to +2.261093 seconds**, while sxtwl minus Skyfield ranges from **−15.783111 to −13.018692 seconds**. This supports the lunar-python timestamps substantially more closely under the explicitly chosen apparent-longitude definition. It does not identify the cause of either residual. The recorded kernel hash, package versions and definition make the comparison reviewable; this audit did not independently derive the JPL ephemeris. The labels remain explicit lunar-python references. This result neither validates other years nor resolves all year/month labeling conventions, day anchors, hour stems or whole-chart correctness.

## Verification and unresolved limits

`python -m pytest tests/test_calendar.py tests/test_calendar_audit.py -q` passed **49 tests** after repair. The new tests cover all twelve Jie in 1901, 2024, 2025 and 2099; both sects and every integer hour; twelve Zhongqi in each of 2024/2025; an Australia/Lord_Howe 30-minute fold/gap; a Pacific/Apia skipped civil day; hostile host timezone caching; invalid resource paths; extreme ISO dates; and diagnostics that detect a changed stored reference and verify revised-suite coverage. Ruff passed on the changed calendar/analysis files and new tests.

These checks establish useful coverage of the declared implementation contract. They do not exhaust all dates or transitions in 1901–2099. The admitted range remains a study restriction, not an upstream accuracy guarantee. Independent evidence now consists of two HKO day anchors, one minute-resolution HKO Lichun anchor and the selected JPL/Skyfield astronomical comparison above. Intra-day Jie year/month *labels as a whole*, accuracy outside the compared transitions, geographically local chart conventions and universal Zi-hour correctness remain unvalidated. No live LLM hypothesis is established by these calendar diagnostics.
