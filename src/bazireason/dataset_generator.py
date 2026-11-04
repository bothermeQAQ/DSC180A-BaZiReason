"""Synthetic diagnostic suite. Labels are explicitly lunar-python references."""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from lunar_python import Solar

from bazireason.calendar_engine import compute_chart
from bazireason.conventions import Convention
from bazireason.schema import model_view

SEED = 1802026


def generate_dataset(seed=SEED):
    rng = random.Random(seed)
    cases = []

    def add(
        category, dt, zone="Asia/Shanghai", sect=2, group=None, meta=None, extra=None, profile=None
    ):
        inputs = {"datetime": dt, "timezone": zone}
        inputs.update(extra or {})
        conv = Convention(sect=sect).to_dict() if profile is None else profile
        result = compute_chart(inputs, conv)
        status = result["status"]
        record = {
            "case_id": f"BR-{len(cases) + 1:03d}",
            "input": inputs,
            "category": category,
            "group_id": group or f"independent-{len(cases) + 1}",
            "evaluation_class": (
                "convention_dependent"
                if status == "ok" and category == "zi_hour"
                else "fully_specified"
                if status == "ok"
                else status
            ),
            "convention": conv,
            "expected": model_view(result),
            "reference_implementation": result["provenance"],
            "validation_status": "single_engine_reference",
            "boundary_metadata": meta or {},
            "seed": seed,
        }
        cases.append(record)

    # 20 separated noon cases, away from common Jie dates and Zi boundaries.
    for _ in range(20):
        dt = datetime(rng.randint(1990, 2030), rng.randint(1, 12), rng.randint(12, 20), 12)
        add("ordinary", dt.isoformat(timespec="seconds"))
    # 12 transitions, two paired observations each. These are engine timestamps.
    terms = ("立春", "惊蛰", "清明", "立夏", "白露", "大雪")
    for year in (2024, 2025):
        table = Solar.fromYmdHms(year, 6, 15, 12, 0, 0).getLunar().getJieQiTable()
        for term in terms:
            boundary = datetime.fromisoformat(table[term].toYmdHms())
            for offset in (-120, 120):
                add(
                    "solar_term",
                    (boundary + timedelta(seconds=offset)).isoformat(),
                    group=f"jie-{year}-{term}",
                    meta={
                        "term": term,
                        "engine_boundary_utc8": boundary.isoformat(),
                        "offset_seconds": offset,
                        "precision": "engine_reference_only",
                    },
                )
    # Contrast CNY with Lichun; no claim these two noon dates are a minimal pair.
    for year, cny in [(2023, "01-22"), (2024, "02-10"), (2025, "01-29"), (2026, "02-17")]:
        add(
            "new_year",
            f"{year}-{cny}T12:00:00",
            group=f"newyear-{year}",
            meta={"event": "lunar_new_year_noon"},
        )
        add(
            "new_year",
            f"{year}-02-05T12:00:00",
            group=f"newyear-{year}",
            meta={"event": "after_lichun_noon"},
        )
    for date in ("2024-08-15", "2025-06-15"):
        for time in ("22:59:59", "23:00:00", "00:00:00"):
            for sect in (1, 2):
                add(
                    "zi_hour", f"{date}T{time}", sect=sect, group=f"zi-{date}", meta={"clock": time}
                )
    for year in (2000, 2004, 2024, 2028):
        add("leap_year", f"{year}-02-29T12:00:00")
    # Four pairs with the same civil instant, represented in different zones.
    from zoneinfo import ZoneInfo

    base = datetime(2024, 6, 15, 12, tzinfo=ZoneInfo("Asia/Shanghai"))
    for i, zone in enumerate(("UTC", "America/Los_Angeles", "Europe/London", "Asia/Kolkata")):
        add("timezone", base.replace(tzinfo=None).isoformat(), group=f"zone-{i}")
        local = base.astimezone(ZoneInfo(zone)).replace(tzinfo=None)
        add("timezone", local.isoformat(), zone=zone, group=f"zone-{i}")
    for zone, gap, repeated, _before in [
        (
            "America/Los_Angeles",
            "2024-03-10T02:30:00",
            "2024-11-03T01:30:00",
            "2024-03-10T01:59:59",
        ),
        ("Europe/London", "2024-03-31T01:30:00", "2024-10-27T01:30:00", "2024-03-31T00:59:59"),
    ]:
        add("dst", gap, zone=zone, group=f"dst-{zone}")
        add("dst", repeated, zone=zone, group=f"dst-{zone}")
        for fold in (0, 1):
            add("dst", repeated, zone=zone, extra={"fold": fold}, group=f"dst-{zone}")
    # 12 policy cases: uncertainty, invalidity, alternative unsupported conventions.
    add("underspecified", "2024-06-15")
    add("underspecified", None)
    add("underspecified", "2024-06-15T12:00:00", zone=None)
    add("underspecified", "2024-08-15T23:30:00", profile={})
    for field, value in [
        ("solar_time", "true_solar"),
        ("clock_basis", "local_civil"),
        ("year_boundary", "lunar_new_year"),
        ("month_boundary", "lunar_month"),
    ]:
        profile = Convention().to_dict()
        profile[field] = value
        add("unsupported", "2024-06-15T12:00:00", profile=profile)
    add("unsupported", "1800-06-15T12:00:00")
    add("unsupported", "2100-06-15T12:00:00")
    add("invalid", "2023-02-29T12:00:00")
    add("invalid", "2024-06-15T12:00:00", zone="Not/AZone")
    return cases


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
