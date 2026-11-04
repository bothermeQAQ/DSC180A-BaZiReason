"""Computational diagnostics, distinct from model-performance experiments."""

from collections import Counter
from datetime import datetime, timedelta
from importlib.metadata import version

import sxtwl

from bazireason.calendar_engine import compute_chart
from bazireason.conventions import BRANCHES, STEMS


def gz(value):
    return STEMS[value.tg] + BRANCHES[value.dz]


def diagnostics(cases):
    checks = []
    differences = []
    skipped = 0
    for case in cases:
        result = compute_chart(case["input"], case["convention"])
        if result["status"] != "ok":
            skipped += 1
            continue
        dt = datetime.fromisoformat(result["normalized_datetime"])
        date = dt + timedelta(days=int(dt.hour == 23 and case["convention"]["sect"] == 1))
        ref_day = sxtwl.fromSolar(date.year, date.month, date.day)
        checks.append(
            {
                "case_id": case["case_id"],
                "pillar": "day",
                "lunar_python": result["pillars"]["day"],
                "sxtwl": gz(ref_day.getDayGZ()),
            }
        )
        # Date-level year/month methods cannot validate an intra-day Jie boundary.
        d = sxtwl.fromSolar(dt.year, dt.month, dt.day)
        if not d.hasJieQi():
            for p, value in [("year", d.getYearGZ()), ("month", d.getMonthGZ())]:
                checks.append(
                    {
                        "case_id": case["case_id"],
                        "pillar": p,
                        "lunar_python": result["pillars"][p],
                        "sxtwl": gz(value),
                    }
                )
    for row in checks:
        row["agrees"] = row["lunar_python"] == row["sxtwl"]
    for year in (2024, 2025):
        from lunar_python import Solar

        table = Solar.fromYmdHms(year, 6, 15, 12, 0, 0).getLunar().getJieQiTable()
        for term in ("立春", "惊蛰", "清明", "立夏", "白露", "大雪"):
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
    return {
        "kind": "computational_diagnostics_not_llm_results",
        "n_cases": len(cases),
        "category_counts": dict(Counter(c["category"] for c in cases)),
        "status_counts": dict(Counter(c["expected"]["status"] for c in cases)),
        "reference_versions": {"lunar-python": version("lunar-python"), "sxtwl": version("sxtwl")},
        "cross_implementation_checks": len(checks),
        "cross_implementation_agreements": sum(r["agrees"] for r in checks),
        "cross_implementation_disagreements": [r for r in checks if not r["agrees"]],
        "checks": checks,
        "non_ok_cases_skipped": skipped,
        "solar_term_differences": differences,
        "limitation": "Shared ShouXing astronomy; not independent astronomical validation. "
        "Only day, and non-solar-term-date year/month, are checked; hours are not.",
        "hypotheses": {"H1": "NOT RUN", "H2": "NOT RUN", "H3": "NOT RUN"},
    }
