"""Independent apparent-geocentric solar-term comparison using Skyfield/JPL.

Optional astronomy dependencies and a separately downloaded DE421 kernel are
required. The kernel is not redistributed; all inputs and hashes are logged.
These comparisons do not validate hour/day/year-pillar conventions.
"""

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path

JIE = (
    "小寒",
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
)
KERNEL_URL = "https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de421.bsp"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="tmp/astronomy/de421.bsp")
    parser.add_argument("--output", default="results/audit/independent_solar_terms.json")
    args = parser.parse_args()
    kernel = Path(args.kernel)
    if not kernel.is_file():
        parser.error(
            "Download the documented JPL DE421 kernel first; this script does not fetch it"
        )
    import sxtwl
    from lunar_python import Solar
    from skyfield import almanac, almanac_east_asia
    from skyfield.api import Loader, load_file

    loader = Loader(str(kernel.parent))
    ts = loader.timescale(builtin=True)
    eph = load_file(str(kernel))
    rows = []
    for year in (2024, 2025):
        times, terms = almanac.find_discrete(
            ts.utc(year, 1, 1),
            ts.utc(year + 1, 1, 1),
            almanac_east_asia.solar_terms(eph),
        )
        table = Solar.fromYmdHms(year, 6, 15, 12, 0, 0).getLunar().getJieQiTable()
        for moment, index in zip(times, terms, strict=True):
            term = almanac_east_asia.SOLAR_TERMS_ZHS[index]
            if term not in JIE:
                continue
            sky = moment.utc_datetime().astimezone(timezone(timedelta(hours=8)))
            lp = datetime.fromisoformat(table[term].toYmdHms()).replace(tzinfo=sky.tzinfo)
            d = sxtwl.fromSolar(lp.year, lp.month, lp.day)
            if not d.hasJieQi():
                raise ValueError("sxtwl term date differs: inspect rather than silently skipping")
            t = sxtwl.JD2DD(d.getJieQiJD())
            sx = datetime(int(t.Y), int(t.M), int(t.D), int(t.h), int(t.m), tzinfo=sky.tzinfo)
            sx += timedelta(seconds=t.s)
            rows.append(
                {
                    "year": year,
                    "term": term,
                    "skyfield_jpl_utc8": sky.isoformat(),
                    "lunar_python_utc8": lp.isoformat(),
                    "sxtwl_utc8": sx.isoformat(),
                    "lunar_python_minus_skyfield_seconds": (lp - sky).total_seconds(),
                    "sxtwl_minus_skyfield_seconds": (sx - sky).total_seconds(),
                }
            )
    eph.close()
    if len(rows) != 24 or len({(r["year"], r["term"]) for r in rows}) != 24:
        raise ValueError("Expected exactly24unique Jie across2024-2025")
    result = {
        "kind": "independent_ephemeris_comparison_not_full_chart_validation",
        "versions": {
            n: version(n) for n in ("skyfield", "numpy", "jplephem", "lunar-python", "sxtwl")
        },
        "kernel": {
            "name": kernel.name,
            "source_url": KERNEL_URL,
            "sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
            "bytes": kernel.stat().st_size,
        },
        "definition": "Apparent geocentric solar ecliptic longitude of date; IAU2000B nutation; "
        "15-degree sector transitions from Skyfield almanac_east_asia; TT/TDB ephemeris "
        "time converted to UTC with bundled Skyfield1.53 timescale data.",
        "source": "https://rhodesmill.org/skyfield/almanac.html#solar-terms",
        "n_jie": len(rows),
        "search_tolerance_seconds": 0.001,
        "rows": rows,
        "summary": {
            engine: {
                "min_seconds": min(r[engine + "_minus_skyfield_seconds"] for r in rows),
                "max_seconds": max(r[engine + "_minus_skyfield_seconds"] for r in rows),
            }
            for engine in ("lunar_python", "sxtwl")
        },
        "limitation": "Distinct JPL/Skyfield numerical path, not a proof of absolute accuracy. "
        "Frame/time-scale/model choices still matter. Covers24Jie2024-2025, "
        "not other years, pillars, or true solar time; labels are not silently changed.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"n_jie": len(rows), "search_tolerance_seconds": 0.001, "summary": result["summary"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
