"""Adversarial checks of the declared wrapper contract, not independent truth."""

import copy
import subprocess
import sys
from datetime import datetime, timedelta
from importlib.resources import files

import pytest
import sxtwl
from lunar_python import Solar

from bazireason.analysis import diagnostics, gz
from bazireason.calendar_engine import compute_chart
from bazireason.conventions import BRANCHES, CYCLE, STEMS, Convention
from bazireason.dataset_generator import generate_audit_dataset, generate_dataset

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
ZHONGQI = (
    "大寒",
    "雨水",
    "春分",
    "谷雨",
    "小满",
    "夏至",
    "大暑",
    "处暑",
    "秋分",
    "霜降",
    "小雪",
    "冬至",
)


def chart(dt, zone="Asia/Shanghai", sect=2, **extra):
    return compute_chart({"datetime": dt, "timezone": zone, **extra}, Convention(sect).to_dict())


@pytest.mark.parametrize("year", [1901, 2024, 2025, 2099])
def test_all_twelve_jie_follow_exact_engine_boundary(year):
    """Cover every Jie, including January and the admitted range's endpoints."""
    table = Solar.fromYmdHms(year, 6, 15, 12, 0, 0).getLunar().getJieQiTable()
    for term in JIE:
        instant = datetime.fromisoformat(table[term].toYmdHms())
        before = chart((instant - timedelta(seconds=1)).isoformat())
        at = chart(instant.isoformat())
        after = chart((instant + timedelta(seconds=1)).isoformat())
        assert before["status"] == at["status"] == after["status"] == "ok", (year, term)
        old, new = before["pillars"], at["pillars"]
        assert new["month"] == CYCLE[(CYCLE.index(old["month"]) + 1) % 60], (year, term)
        assert (old["year"] != new["year"]) == (term == "立春"), (year, term)
        assert new == after["pillars"], (year, term)


@pytest.mark.parametrize("year", [2024, 2025])
def test_zhongqi_do_not_change_solar_month(year):
    table = Solar.fromYmdHms(year, 6, 15, 12, 0, 0).getLunar().getJieQiTable()
    for term in ZHONGQI:
        instant = datetime.fromisoformat(table[term].toYmdHms())
        before = chart((instant - timedelta(seconds=1)).isoformat())["pillars"]
        at = chart(instant.isoformat())["pillars"]
        assert (before["year"], before["month"]) == (at["year"], at["month"]), term


@pytest.mark.parametrize("sect", [1, 2])
def test_every_hour_branch_and_advanced_hour_stem(sect):
    noon = chart("2024-06-15T12:00:00", sect=sect)["pillars"]
    day_stem = STEMS.index(noon["day"][0])
    for hour in range(24):
        result = chart(f"2024-06-15T{hour:02d}:00:00", sect=sect)["pillars"]
        branch = ((hour + 1) // 2) % 12
        stem = (((day_stem + int(hour == 23)) % 10) % 5 * 2 + branch) % 10
        assert result["hour"] == STEMS[stem] + BRANCHES[branch]
        expected_day = CYCLE[(CYCLE.index(noon["day"]) + int(sect == 1 and hour == 23)) % 60]
        assert result["day"] == expected_day


def test_sxtwl_hour_argument_has_empirically_matched_semantics():
    """Explicit bool; source comments/names alone do not establish behavior."""
    for offset in range(60):
        day = datetime(2024, 6, 15) + timedelta(days=offset)
        sx = sxtwl.fromSolar(day.year, day.month, day.day)
        day_stem = sx.getDayGZ().tg
        for hour in range(24):
            branch = ((hour + 1) // 2) % 12
            advanced_stem = ((day_stem % 5 * 2) + (hour + 1) // 2) % 10
            assert gz(sx.getHourGZ(hour, True)) == STEMS[advanced_stem] + BRANCHES[branch]
        assert sx.getHourGZ(23, True).tg != sx.getHourGZ(23, False).tg
        assert sx.getHourGZ(23, False).tg == day_stem % 5 * 2


@pytest.mark.parametrize(
    "dt,zone",
    [
        ("0001-01-01T00:00:00", "Asia/Shanghai"),
        ("9999-12-31T23:59:59", "America/Los_Angeles"),
        ("9999-12-31T23:59:59", "UTC"),
    ],
)
def test_extreme_iso_dates_return_controlled_out_of_range(dt, zone):
    result = chart(dt, zone)
    assert result["status"] == "unsupported"
    assert result["reason_codes"] == ["outside_study_range"]


def test_range_is_applied_after_reference_clock_normalization():
    assert chart("1900-12-31T16:00:00", "UTC")["status"] == "ok"
    assert chart("1900-12-31T15:59:59", "UTC")["status"] == "unsupported"
    assert chart("2099-12-31T15:59:59", "UTC")["status"] == "ok"
    assert chart("2099-12-31T16:00:00", "UTC")["status"] == "unsupported"


def test_half_hour_fold_and_whole_day_gap():
    assert chart("2011-12-30T12:00:00", "Pacific/Apia")["status"] == "invalid"
    repeated = "2024-04-07T01:45:00"
    assert chart(repeated, "Australia/Lord_Howe")["status"] == "ambiguous"
    a = chart(repeated, "Australia/Lord_Howe", fold=0)
    b = chart(repeated, "Australia/Lord_Howe", fold=1)
    assert a["status"] == b["status"] == "ok"
    assert (
        datetime.fromisoformat(b["normalized_datetime"])
        - datetime.fromisoformat(a["normalized_datetime"])
    ).total_seconds() == 1800
    assert chart("2024-10-06T02:15:00", "Australia/Lord_Howe")["status"] == "invalid"


def test_timezone_loading_ignores_host_path_and_existing_cache(tmp_path):
    """A hostile host fixture proves provenance; do not mutate this test process."""
    directory = tmp_path / "America"
    directory.mkdir()
    (directory / "Los_Angeles").write_bytes(files("tzdata.zoneinfo").joinpath("UTC").read_bytes())
    program = """
import os, sys, zoneinfo
from datetime import datetime, timedelta
zoneinfo.reset_tzpath((sys.argv[1],))
cached = zoneinfo.ZoneInfo('America/Los_Angeles')
assert datetime(2024, 6, 15, tzinfo=cached).utcoffset() == timedelta(0)
path_before = zoneinfo.TZPATH
env_before = os.environ.get('PYTHONTZPATH')
from bazireason.calendar_engine import compute_chart
result = compute_chart({'datetime': '2024-06-15T12:00:00',
                        'timezone': 'America/Los_Angeles'}, {'sect': 2})
assert result['normalized_datetime'] == '2024-06-16T03:00:00+08:00', result
assert zoneinfo.TZPATH == path_before
assert os.environ.get('PYTHONTZPATH') == env_before
assert zoneinfo.ZoneInfo('America/Los_Angeles') is cached
"""
    run = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)], capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr


@pytest.mark.parametrize(
    "zone",
    ["/etc/localtime", "../UTC", "Asia/../UTC", "Asia//Shanghai", "Asia", "UTC/foo", "x" * 300],
)
def test_timezone_resource_loading_rejects_non_iana_paths(zone):
    assert chart("2024-06-15T12:00:00", zone)["status"] == "invalid"


def test_diagnostics_preserve_main_count_and_expose_correlated_coverage():
    result = diagnostics(generate_dataset())
    assert result["cross_implementation_checks"] == 184
    assert result["cross_implementation_agreements"] == 184
    coverage = result["coverage"]
    assert coverage["per_pillar_checks"] == {"year": 52, "month": 52, "day": 80, "hour": 0}
    assert coverage["unique_reference_date_and_pillar_checks"] < 184
    assert len(result["solar_term_differences"]) == 12
    hours = result["supplementary_hour_comparison"]
    assert hours["checks"] == hours["agreements"] == 80
    assert hours["unique_reference_date_and_hour_checks"] < 80
    assert result["reference_replay"]["agreements"] == 96


def test_diagnostics_detect_stale_reference_labels():
    case = copy.deepcopy(generate_dataset()[0])
    case["expected"]["status"] = "invalid"
    result = diagnostics([case])
    assert result["status_counts"] == {"ok": 1}
    assert result["expected_status_counts"] == {"invalid": 1}
    assert result["reference_replay"]["agreements"] == 0
    assert result["reference_replay"]["disagreement_case_ids"] == [case["case_id"]]


def test_revised_suite_calendar_comparison_coverage():
    result = diagnostics(generate_audit_dataset())
    assert result["n_cases"] == 152
    assert result["cross_implementation_checks"] == result["cross_implementation_agreements"] == 270
    assert result["coverage"]["per_pillar_checks"] == {
        "year": 72,
        "month": 72,
        "day": 126,
        "hour": 0,
    }
    assert result["coverage"]["unique_reference_date_and_pillar_checks"] == 169
    assert result["solar_term_comparison_coverage"] == {
        "requested_term_year_pairs": 24,
        "compared_pairs": 24,
        "unmatched_pairs": [],
    }
    hours = result["supplementary_hour_comparison"]
    assert hours["checks"] == hours["agreements"] == 126
    assert result["reference_replay"]["agreements"] == 152


@pytest.mark.parametrize(
    "profile",
    [
        {"sect": True},
        {"sect": None},
        {"sect": "2"},
        {"sect": 3},
        {"sect": 2, "clock_basis": None},
        {"sect": 2, "solar_time": []},
        {"sect": 2, "year_boundary": ""},
        {"sect": 2, "month_boundary": 42},
    ],
)
def test_invalid_profile_response_has_no_fictitious_selected_convention(profile):
    from bazireason.schema import model_view, valid_response

    result = compute_chart({"datetime": "2024-06-15T12:00:00", "timezone": "UTC"}, profile)
    assert result["status"] == "invalid"
    assert result["conventions"] is None
    assert valid_response(model_view(result))
