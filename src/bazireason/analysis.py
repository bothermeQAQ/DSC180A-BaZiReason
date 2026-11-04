"""Computational diagnostics, distinct from model-performance experiments."""

from collections import Counter
from datetime import datetime, timedelta
from importlib.metadata import version

import sxtwl
from lunar_python import Solar

from bazireason.calendar_engine import compute_chart
from bazireason.conventions import BRANCHES, PILLARS, STEMS
from bazireason.schema import model_view


def gz(value):
    return STEMS[value.tg] + BRANCHES[value.dz]


def diagnostics(cases):
    checks = []
    hour_checks = []
    differences = []
    unmatched_terms = []
    skipped = 0
    actual_statuses = Counter()
    replay_disagreements = []
    exclusions = []
    unique_inputs = set()
    for case in cases:
        result = compute_chart(case["input"], case["convention"])
        actual_statuses[result["status"]] += 1
        if model_view(result) != case["expected"]:
            replay_disagreements.append(case["case_id"])
        if result["status"] != "ok":
            skipped += 1
            continue
        dt = datetime.fromisoformat(result["normalized_datetime"])
        unique_inputs.add((dt.isoformat(), case["convention"]["sect"]))
        date = dt + timedelta(days=int(dt.hour == 23 and case["convention"]["sect"] == 1))
        ref_day = sxtwl.fromSolar(date.year, date.month, date.day)
        checks.append(
            {
                "case_id": case["case_id"],
                "pillar": "day",
                "reference_date": date.date().isoformat(),
                "lunar_python": result["pillars"]["day"],
                "sxtwl": gz(ref_day.getDayGZ()),
            }
        )
        # Date-level year/month methods cannot validate an intra-day Jie boundary.
        d = sxtwl.fromSolar(dt.year, dt.month, dt.day)
        hour_checks.append(
            {
                "case_id": case["case_id"],
                "reference_date": dt.date().isoformat(),
                "reference_hour": dt.hour,
                "lunar_python": result["pillars"]["hour"],
                "sxtwl": gz(d.getHourGZ(dt.hour, True)),
            }
        )
        if not d.hasJieQi():
            for p, value in [("year", d.getYearGZ()), ("month", d.getMonthGZ())]:
                checks.append(
                    {
                        "case_id": case["case_id"],
                        "pillar": p,
                        "reference_date": dt.date().isoformat(),
                        "lunar_python": result["pillars"][p],
                        "sxtwl": gz(value),
                    }
                )
        else:
            exclusions.append(
                {
                    "case_id": case["case_id"],
                    "pillars": ["year", "month"],
                    "reason": "date_only_sxtwl_api_on_solar_term_date",
                }
            )
    for row in checks + hour_checks:
        row["agrees"] = row["lunar_python"] == row["sxtwl"]
    term_pairs = set()
    for case in cases:
        meta = case.get("boundary_metadata", {})
        if (
            case["category"] == "solar_term"
            and meta.get("term")
            and meta.get("engine_boundary_utc8")
        ):
            term_pairs.add(
                (datetime.fromisoformat(meta["engine_boundary_utc8"]).year, meta["term"])
            )
    for year in sorted({year for year, _ in term_pairs}):
        table = Solar.fromYmdHms(year, 6, 15, 12, 0, 0).getLunar().getJieQiTable()
        for term in sorted(term for y, term in term_pairs if y == year):
            lp = datetime.fromisoformat(table[term].toYmdHms())
            d = sxtwl.fromSolar(lp.year, lp.month, lp.day)
            if d.hasJieQi():
                t = sxtwl.JD2DD(d.getJieQiJD())
                sx = datetime(int(t.Y), int(t.M), int(t.D), int(t.h), int(t.m)) + timedelta(
                    seconds=t.s
                )
                differences.append(
                    {
                        "term": term,
                        "year": year,
                        "lunar_python": lp.isoformat(),
                        "sxtwl": sx.isoformat(),
                        "lunar_minus_sxtwl_seconds": (lp - sx).total_seconds(),
                    }
                )
            else:
                unmatched_terms.append(
                    {
                        "term": term,
                        "year": year,
                        "reason": "sxtwl_has_no_solar_term_on_lunar_python_reference_date",
                    }
                )
    return {
        "kind": "computational_diagnostics_not_llm_results",
        "n_cases": len(cases),
        "category_counts": dict(Counter(c["category"] for c in cases)),
        "status_counts": dict(actual_statuses),
        "expected_status_counts": dict(Counter(c["expected"]["status"] for c in cases)),
        "reference_replay": {
            "kind": "same_engine_label_integrity_check_not_validation",
            "agreements": len(cases) - len(replay_disagreements),
            "disagreement_case_ids": replay_disagreements,
        },
        "reference_versions": {"lunar-python": version("lunar-python"), "sxtwl": version("sxtwl")},
        "cross_implementation_checks": len(checks),
        "cross_implementation_agreements": sum(r["agrees"] for r in checks),
        "cross_implementation_disagreements": [r for r in checks if not r["agrees"]],
        "checks": checks,
        "supplementary_hour_comparison": {
            "kind": "convention_matched_cross_implementation_not_independent_truth",
            "api": "sxtwl.Day.getHourGZ(reference_hour, True)",
            "semantics": "Installed sxtwl 2.0.7 with True advances the hour stem at 23:00; "
            "the civil reference date is passed unchanged for both lunar-python sects.",
            "checks": len(hour_checks),
            "agreements": sum(r["agrees"] for r in hour_checks),
            "rows": hour_checks,
            "unique_reference_date_and_hour_checks": len(
                {(r["reference_date"], r["reference_hour"]) for r in hour_checks}
            ),
            "limitation": "Separate from the historical day/year/month comparison count. "
            "This checks a matched implementation rule; it does not adjudicate Zi traditions "
            "or validate the astronomical clock or the four-pillar chart independently.",
        },
        "non_ok_cases_skipped": skipped,
        "coverage": {
            "ok_cases": len(cases) - skipped,
            "unique_normalized_instant_and_sect_inputs": len(unique_inputs),
            "per_pillar_checks": {p: sum(r["pillar"] == p for r in checks) for p in PILLARS},
            "unique_reference_date_and_pillar_checks": len(
                {(r["reference_date"], r["pillar"]) for r in checks}
            ),
            "unique_reference_dates_per_pillar": {
                p: len({r["reference_date"] for r in checks if r["pillar"] == p}) for p in PILLARS
            },
            "excluded_checks": exclusions,
            "interpretation": "Main counts are correlated case-field comparisons, not independent "
            "observations or full-chart validations. Date-level duplicate comparisons share "
            "the same sxtwl reference query. All solar-term-date year/month fields are excluded; "
            "hours are reported separately under supplementary_hour_comparison.",
        },
        "solar_term_differences": differences,
        "solar_term_comparison_coverage": {
            "requested_term_year_pairs": len(term_pairs),
            "compared_pairs": len(differences),
            "unmatched_pairs": unmatched_terms,
        },
        "limitation": "Shared ShouXing astronomy; not independent astronomical validation. "
        "Main count checks only day and non-solar-term-date year/month. A separately counted "
        "hour comparison uses explicitly matched 23:00-advanced stem semantics. Intra-day "
        "Jie year/month labels are not cross-validated.",
        "hypotheses": {"H1": "NOT RUN", "H2": "NOT RUN", "H3": "NOT RUN"},
    }
