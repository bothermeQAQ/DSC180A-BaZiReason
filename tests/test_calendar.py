"""Regression, metamorphic, and civil-time validation tests."""

from datetime import datetime, timedelta

import pytest
from lunar_python import Solar

from bazireason.calendar_engine import compute_chart
from bazireason.conventions import Convention
from bazireason.schema import model_view, valid_response


def chart(dt, zone="Asia/Shanghai", sect=2, **extra):
    return compute_chart({"datetime": dt, "timezone": zone, **extra}, Convention(sect).to_dict())


def test_upstream_documented_regression_not_independent():
    # 6tail/lunar-python test/EightCharTest.py test_gan_zhi, MIT.
    assert chart("2005-12-23T08:37:00")["pillars"] == {
        "year": "乙酉",
        "month": "戊子",
        "day": "辛巳",
        "hour": "壬辰",
    }


def test_hko_independent_day_spot_check():
    # HKO Almanac 2025, February calendar, standalone PDF page 1, 3 Feb = 癸卯.
    # Only this civil day is checked, not four-pillar chart correctness.
    assert chart("2025-02-03T12:00:00")["pillars"]["day"] == "癸卯"


def test_zi_profile_hour_stem_is_not_rederived():
    a = chart("1988-02-15T23:30:00", sect=1)
    b = chart("1988-02-15T23:30:00", sect=2)
    assert a["pillars"]["day"] == "辛丑"
    assert b["pillars"]["day"] == "庚子"
    assert a["pillars"]["hour"] == b["pillars"]["hour"] == "戊子"


def test_60_day_period():
    a = datetime(2024, 6, 15, 12)
    assert (
        chart(a.isoformat())["pillars"]["day"]
        == chart((a + timedelta(days=60)).isoformat())["pillars"]["day"]
    )


def test_lichun_instant_engine_contract():
    term = Solar.fromYmdHms(2025, 2, 1, 12, 0, 0).getLunar().getJieQiTable()["立春"]
    t = datetime.fromisoformat(term.toYmdHms())
    before = chart((t - timedelta(seconds=1)).isoformat())["pillars"]
    at = chart(t.isoformat())["pillars"]
    assert (before["year"], before["month"]) == ("甲辰", "丁丑")
    assert (at["year"], at["month"]) == ("乙巳", "戊寅")


@pytest.mark.parametrize(
    "dt,zone,expected",
    [
        ("2023-02-29T12:00:00", "UTC", "invalid"),
        ("2024-02-29T12:00:00", "UTC", "ok"),
        ("2024-06-15T12:00:00", "Unknown/Zone", "invalid"),
        ("2024-06-15", "UTC", "ambiguous"),
        (None, "UTC", "ambiguous"),
        ("1800-01-01T12:00:00", "UTC", "unsupported"),
        ("2100-01-01T12:00:00", "UTC", "unsupported"),
        ("2024-03-10T02:30:00", "America/Los_Angeles", "invalid"),
        ("2024-11-03T01:30:00", "America/Los_Angeles", "ambiguous"),
        ("2024-03-31T01:30:00", "Europe/London", "invalid"),
    ],
)
def test_date_validation(dt, zone, expected):
    result = chart(dt, zone)
    assert result["status"] == expected
    assert valid_response(model_view(result))


def test_fold_distinct_instants():
    a = chart("2024-11-03T01:30:00", "America/Los_Angeles", fold=0)
    b = chart("2024-11-03T01:30:00", "America/Los_Angeles", fold=1)
    assert a["status"] == b["status"] == "ok"
    assert a["normalized_datetime"] != b["normalized_datetime"]
    assert chart("2024-06-15T12:00:00", fold=1)["status"] == "invalid"
    assert chart("2024-11-03T01:30:00", "America/Los_Angeles", fold=True)["status"] == "invalid"


def test_equivalent_instants_and_china_historical_dst():
    assert chart("2024-06-15T12:00:00")["pillars"] == chart("2024-06-15T04:00:00", "UTC")["pillars"]
    # Legal civil China DST is normalized rather than mistaken for standard UTC+8.
    assert chart("1990-06-15T12:00:00")["normalized_datetime"] == "1990-06-15T11:00:00+08:00"


def test_explicit_unsupported_and_missing_conventions():
    inputs = {"datetime": "2024-06-15T12:00:00", "timezone": "UTC"}
    # No sect: answerable at 20:00 UTC+08, where the sect cannot change any pillar...
    assert compute_chart(inputs)["status"] == "ok"
    assert compute_chart(inputs)["conventions"]["sect"] is None
    # ...but ambiguous at 23:30 UTC+08, where the two sects give different days.
    late = {"datetime": "2024-06-15T15:30:00", "timezone": "UTC"}
    assert compute_chart(late)["status"] == "ambiguous"
    assert compute_chart(late)["reason_codes"] == ["missing_convention"]
    assert compute_chart(inputs, {"sect": 2, "solar_time": "true_solar"})["status"] == "unsupported"
    assert compute_chart(inputs, {"sect": True})["status"] == "invalid"
    assert compute_chart(inputs, {"sect": 3})["status"] == "invalid"


def test_hko_minute_and_second_day_anchor():
    assert chart("2025-02-04T12:00:00")["pillars"]["day"] == "甲辰"
    term = Solar.fromYmdHms(2025, 2, 1, 12, 0, 0).getLunar().getJieQiTable()["立春"]
    t = datetime.fromisoformat(term.toYmdHms())
    # HKO sidebar shows 3 February 22:10, at minute publication precision.
    assert abs((t - datetime(2025, 2, 3, 22, 10)).total_seconds()) < 60


@pytest.mark.parametrize("convention", [[], 1, "sect", True])
def test_malformed_conventions(convention):
    assert compute_chart({}, convention)["status"] == "invalid"


def test_midnight_day_advancement():
    for sect in (1, 2):
        a = chart("2024-08-15T23:59:59", sect=sect)["pillars"]
        b = chart("2024-08-16T00:00:00", sect=sect)["pillars"]
        assert a["hour"] == b["hour"]
        assert (a["day"] == b["day"]) == (sect == 1)
