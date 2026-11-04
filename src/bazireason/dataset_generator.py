"""Historical v1/v2 synthetic suites. Labels are explicitly lunar-python references.

These generators reproduce the committed checkpoint_v1 and audit_v2 files byte
for byte and are kept frozen. Their `evaluation_class` field is a known legacy
error (audit defect E6): it marks every zi_hour case convention_dependent. Use
`bazireason.dataset_v3.convention_dependence` (as benchmark_v3 does) instead.
"""

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
    from bazireason.calendar_engine import load_timezone

    base = datetime(2024, 6, 15, 12, tzinfo=load_timezone("Asia/Shanghai"))
    for i, zone in enumerate(("UTC", "America/Los_Angeles", "Europe/London", "Asia/Kolkata")):
        add("timezone", base.replace(tzinfo=None).isoformat(), group=f"zone-{i}")
        local = base.astimezone(load_timezone(zone)).replace(tzinfo=None)
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


def dataset_bytes(records):
    """Canonical JSONL bytes: sorted keys, UTF-8, one record per line."""
    lines = (json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    return "".join(lines).encode("utf-8")


def write_jsonl(path, records, overwrite=False):
    """Write canonical JSONL; refuse to replace an existing file unless asked (defect E9)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb" if overwrite else "xb") as f:
        f.write(dataset_bytes(records))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_audit_dataset(seed=SEED):
    """Version 2 diagnostics: all Jie, adjacent midnight and matched H3 inputs.

    The original 96-case artifact and Git tag remain unchanged. This function
    deliberately creates a new version rather than silently relabeling v1.
    """
    from bazireason.calendar_engine import load_timezone

    cases = generate_dataset(seed)

    def refresh(case):
        computed = compute_chart(case["input"], case["convention"])
        case["expected"] = model_view(computed)
        case["reference_implementation"] = computed["provenance"]
        case["evaluation_class"] = (
            "convention_dependent"
            if computed["status"] == "ok" and case["category"] == "zi_hour"
            else "fully_specified"
            if computed["status"] == "ok"
            else computed["status"]
        )

    def add(category, inputs, convention, group, meta):
        case = {
            "input": inputs,
            "category": category,
            "group_id": group,
            "convention": convention,
            "boundary_metadata": meta,
            "validation_status": "single_engine_reference",
            "seed": seed,
        }
        refresh(case)
        cases.append(case)

    # v1 repeated the same Shanghai input four times in nominally different groups.
    for i in range(4):
        pair = cases[68 + 2 * i : 70 + 2 * i]
        base = datetime(2024, 6, 15, 12, tzinfo=load_timezone("Asia/Shanghai"))
        base += timedelta(days=7 * i)
        for case in pair:
            zone = case["input"]["timezone"]
            local = base.astimezone(load_timezone(zone)).replace(tzinfo=None)
            case["input"]["datetime"] = local.isoformat()
            case["boundary_metadata"] = {"equivalent_instant_utc8": base.isoformat()}
            refresh(case)

    # Midnight must follow the preceding 23:00 rather than precede it by 23 hours.
    for case in cases:
        if case["category"] == "zi_hour" and case["boundary_metadata"]["clock"] == "00:00:00":
            wall = datetime.fromisoformat(case["input"]["datetime"]) + timedelta(days=1)
            case["input"]["datetime"] = wall.isoformat()
            case["boundary_metadata"]["successive_midnight"] = True
            refresh(case)
    for date in ("2024-08-15", "2025-06-15"):
        for wall in (
            datetime.fromisoformat(date + "T23:59:59"),
            datetime.fromisoformat(date + "T00:00:01") + timedelta(days=1),
        ):
            for sect in (1, 2):
                add(
                    "zi_hour",
                    {"datetime": wall.isoformat(), "timezone": "Asia/Shanghai"},
                    Convention(sect).to_dict(),
                    f"zi-{date}",
                    {
                        "clock": wall.time().isoformat(),
                        "successive_midnight": wall.day != int(date[-2:]),
                    },
                )

    # Add the other six month-start Jie, not the twelve intervening Zhongqi.
    for year in (2024, 2025):
        table = Solar.fromYmdHms(year, 6, 15, 12, 0, 0).getLunar().getJieQiTable()
        for term in ("小寒", "芒种", "小暑", "立秋", "寒露", "立冬"):
            boundary = datetime.fromisoformat(table[term].toYmdHms())
            assert boundary.year == year
            for offset in (-120, 120):
                add(
                    "solar_term",
                    {
                        "datetime": (boundary + timedelta(seconds=offset)).isoformat(),
                        "timezone": "Asia/Shanghai",
                    },
                    Convention().to_dict(),
                    f"jie-{year}-{term}",
                    {
                        "term": term,
                        "engine_boundary_utc8": boundary.isoformat(),
                        "offset_seconds": offset,
                        "precision": "engine_reference_only",
                    },
                )

    # H3 uses actual omitted fields with matched specified counterparts.
    # Prompt policy is varied separately at inference time, never in gold labels.
    for date in ("2023-07-12", "2024-09-14", "2025-11-16"):
        inputs = {"datetime": date + "T23:30:00", "timezone": "Asia/Shanghai"}
        for sect in (None, 1, 2):
            add(
                "h3_convention",
                dict(inputs),
                {} if sect is None else Convention(sect).to_dict(),
                f"h3-sect-{date}",
                {"missing_field": "sect" if sect is None else None},
            )
    for date in ("2023-06-14", "2024-08-16", "2025-10-18"):
        for missing in (False, True):
            add(
                "h3_missing_time",
                {"datetime": date if missing else date + "T12:00:00", "timezone": "Asia/Shanghai"},
                Convention().to_dict(),
                f"h3-time-{date}",
                {"missing_field": "time" if missing else None},
            )
    for date in ("2023-05-13", "2024-07-15", "2025-09-17"):
        for missing in (False, True):
            add(
                "h3_missing_zone",
                {"datetime": date + "T12:00:00", "timezone": None if missing else "Asia/Shanghai"},
                Convention().to_dict(),
                f"h3-zone-{date}",
                {"missing_field": "timezone" if missing else None},
            )
    for fold in (None, 0, 1):
        inputs = {"datetime": "2025-11-02T01:30:00", "timezone": "America/Los_Angeles"}
        if fold is not None:
            inputs["fold"] = fold
        add(
            "h3_dst_fold",
            inputs,
            Convention().to_dict(),
            "h3-fold-2025-LA",
            {"missing_field": "fold" if fold is None else None},
        )

    for i, case in enumerate(cases, 1):
        case["case_id"] = f"BRA-{i:03d}"
        case["dataset_version"] = "audit_v2"
    return cases


def representative_case_ids(cases):
    """Deterministic small pilot spanning categories, statuses and both Zi sects.

    This is an operational pilot selection, not a random population sample.
    """
    selected = []
    seen = set()
    for case in cases:
        status = case["expected"]["status"]
        # Include both sides of the first term and both selected Zi profiles.
        detail = case["convention"].get("sect") if case["category"] == "zi_hour" else None
        if case["category"] == "solar_term":
            detail = case["boundary_metadata"]["offset_seconds"]
        key = (case["category"], status, detail)
        if key not in seen:
            selected.append(case["case_id"])
            seen.add(key)
    return selected
