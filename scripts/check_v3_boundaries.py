"""Skyfield/JPL check of every benchmark_v3 case within minutes of a Jie.

The pure-Python oracle (src/bazireason/oracle.py, tested in
tests/test_independent_oracle.py) defers year/month within about 40 minutes of
a Jie. This optional script covers exactly those cases: the +/-120 s ladder
rungs and the minute-precision families. For each Jie instant it compares
lunar-python with Skyfield's apparent geocentric solar longitude of date (JPL
DE421) and checks that every near case lies on the same side of the boundary.
The kernel is not downloaded here; see README.
"""

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path

UTC8 = timezone(timedelta(hours=8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="tmp/astronomy/de421.bsp")
    parser.add_argument("--data", default="data/synthetic/benchmark_v3.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    kernel = Path(args.kernel)
    if not kernel.is_file():
        parser.error(
            "Download the documented JPL DE421 kernel first; this script does not fetch it"
        )
    if Path(args.output).exists():
        parser.error("Output exists; choose a new path")
    from skyfield import almanac, almanac_east_asia
    from skyfield.api import Loader, load_file

    from bazireason.dataset_generator import read_jsonl
    from bazireason.dataset_v3 import jie_instant

    cases = read_jsonl(args.data)
    events = {}
    for case in cases:
        h1 = case.get("design", {}).get("h1") or {}
        if h1.get("boundary") == "jie" and h1["distance"] == "near":
            moment = datetime.fromisoformat(h1["event_utc8"])
            events.setdefault((moment.year, h1["term"]), []).append(case)
        if case.get("design", {}).get("jie_utc8"):
            moment = datetime.fromisoformat(case["design"]["jie_utc8"])
            term = next(
                t
                for t in (
                    "立春",
                    "惊蛰",
                    "清明",
                    "立夏",
                    "芒种",
                    "小暑",
                    "立秋",
                    "白露",
                    "寒露",
                    "立冬",
                    "大雪",
                    "小寒",
                )
                if jie_instant(t, moment.year) == moment.replace(tzinfo=None)
            )
            events.setdefault((moment.year, term), []).append(case)
    ts = Loader(str(kernel.parent)).timescale(builtin=True)
    eph = load_file(str(kernel))
    sky = {}
    for year in sorted({y for y, _ in events}):
        times, terms = almanac.find_discrete(
            ts.utc(year, 1, 1), ts.utc(year + 1, 1, 1), almanac_east_asia.solar_terms(eph)
        )
        for moment, index in zip(times, terms, strict=True):
            sky[(year, almanac_east_asia.SOLAR_TERMS_ZHS[index])] = moment.utc_datetime()
    eph.close()
    rows, failures = [], []
    for (year, term), members in sorted(events.items()):
        engine = jie_instant(term, year).replace(tzinfo=UTC8)
        reference = sky[(year, term)].astimezone(UTC8)
        residual = (engine - reference).total_seconds()
        for case in members:
            raw = case["input"]["datetime"]
            first = datetime.fromisoformat(raw if len(raw) == 19 else raw + ":00").replace(
                tzinfo=UTC8
            )
            last = first if len(raw) == 19 else first + timedelta(seconds=59)
            engine_side = "after" if first >= engine else "before" if last < engine else "inside"
            sky_side = "after" if first >= reference else "before" if last < reference else "inside"
            agree = engine_side == sky_side
            rows.append(
                {
                    "case_id": case["case_id"],
                    "term": term,
                    "year": year,
                    "input": raw,
                    "status": case["expected"]["status"],
                    "engine_side": engine_side,
                    "skyfield_side": sky_side,
                    "agrees": agree,
                }
            )
            if not agree:
                failures.append(case["case_id"])
        rows.append(
            {
                "term": term,
                "year": year,
                "engine_minus_skyfield_seconds": residual,
                "skyfield_jpl_utc8": reference.isoformat(),
                "lunar_python_utc8": engine.isoformat(),
            }
        )
    residuals = [
        r["engine_minus_skyfield_seconds"] for r in rows if "engine_minus_skyfield_seconds" in r
    ]
    result = {
        "kind": "independent_ephemeris_side_check_not_full_chart_validation",
        "versions": {n: version(n) for n in ("skyfield", "numpy", "jplephem", "lunar-python")},
        "kernel": {"name": kernel.name, "sha256": hashlib.sha256(kernel.read_bytes()).hexdigest()},
        "definition": "Apparent geocentric solar ecliptic longitude of date (IAU2000B nutation); "
        "Skyfield almanac_east_asia solar terms; bundled time-scale data.",
        "n_events": len(events),
        "n_cases": sum(len(v) for v in events.values()),
        "engine_minus_skyfield_seconds": {"min": min(residuals), "max": max(residuals)},
        "side_disagreements": failures,
        "rows": rows,
        "limitation": "Covers only the cases the low-precision oracle defers. Before 1972, "
        "Skyfield treats UTC with a fixed offset, so second-level residuals for those years "
        "partly reflect time-scale handling.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "n_events",
                    "n_cases",
                    "engine_minus_skyfield_seconds",
                    "side_disagreements",
                )
            },
            ensure_ascii=False,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
