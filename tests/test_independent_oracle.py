"""Independent label check: JDN day cycle, five-rats hours, low-precision solar longitude.

The oracle shares no code with lunar-python. Time zones come from the host's
zoneinfo search path (falling back to the tzdata package where the host has
none), not from the engine's direct tzdata loader. Year and month near a Jie
are deferred to scripts/check_v3_boundaries.py (Skyfield/JPL).
"""

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from bazireason.calendar_engine import jie_around
from bazireason.dataset_generator import read_jsonl
from bazireason.oracle import UTC8, apparent_solar_longitude, day_index, pillars

DATASETS = ("data/synthetic/audit_v2.jsonl", "data/synthetic/benchmark_v3.jsonl")


def completions(case):
    """Admissible (instant, sect) completions, computed without the label engine."""
    inputs, convention = case["input"], case["convention"]
    raw = inputs["datetime"]
    zone = ZoneInfo(inputs["timezone"])
    walls = (
        [datetime.fromisoformat(raw + ":00"), datetime.fromisoformat(raw + ":59")]
        if len(raw) == 16
        else [datetime.fromisoformat(raw)]
    )
    instants = []
    for fold in [inputs["fold"]] if inputs.get("fold") is not None else [0, 1]:
        found = [
            w.replace(tzinfo=zone, fold=fold)
            for w in walls
            if (lambda b: b.replace(tzinfo=None) == w and b.fold == fold)(
                w.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc).astimezone(zone)
            )
        ]
        if len(found) == len(walls):
            instants += found
    sects = [convention["sect"]] if "sect" in convention else [1, 2]
    return [(i, s) for i in instants for s in sects]


def test_anchors_of_the_independent_oracle():
    assert day_index(date(2000, 1, 1)) == 54  # 戊午
    assert day_index(date(1949, 10, 1)) == 0  # 甲子
    # Published equinox and Lichun instants (UTC+08:00, minute precision).
    for moment, degrees in (
        (datetime(2024, 3, 20, 11, 6, tzinfo=UTC8), 0.0),
        (datetime(2025, 2, 3, 22, 10, tzinfo=UTC8), 315.0),
    ):
        error = (apparent_solar_longitude(moment) - degrees + 180) % 360 - 180
        assert abs(error) < 0.012


# (answerable cases, completions with month/year decided, completions deferred near a Jie)
EXPECTED_COVERAGE = {
    "data/synthetic/audit_v2.jsonl": (126, 78, 48),
    "data/synthetic/benchmark_v3.jsonl": (225, 205, 44),
}


@pytest.mark.parametrize("path", DATASETS)
def test_every_answerable_label_matches_the_independent_oracle(path):
    checked = dict.fromkeys(("year", "month", "day", "hour"), 0)
    deferred = 0
    for case in read_jsonl(path):
        if case["expected"]["status"] != "ok":
            continue
        options = completions(case)
        assert options, case["case_id"]
        for instant, sect in options:
            mine = pillars(instant, sect)
            for pillar, value in mine.items():
                if value is None:
                    # Deferral is allowed only within about 40 minutes of an engine Jie.
                    ref = instant.astimezone(UTC8)
                    (prev_t, _), (next_t, _) = jie_around(ref)
                    assert min(ref - prev_t, next_t - ref) < timedelta(minutes=40), case["case_id"]
                    deferred += pillar == "month"
                    continue
                assert value == case["expected"]["pillars"][pillar], (case["case_id"], pillar)
                checked[pillar] += 1
    assert checked["day"] == checked["hour"] == checked["month"] + deferred
    assert checked["year"] == checked["month"]
    n_ok = sum(c["expected"]["status"] == "ok" for c in read_jsonl(path))
    assert (n_ok, checked["month"], deferred) == EXPECTED_COVERAGE[path]


def test_oracle_independently_confirms_knowledge_dependent_ambiguity():
    confirmed = 0
    for case in read_jsonl("data/synthetic/benchmark_v3.jsonl"):
        reasons = case["expected"]["reason_codes"]
        if case["expected"]["status"] != "ambiguous" or not (
            {"missing_convention", "dst_fold_required"} & set(reasons)
        ):
            continue
        charts = {
            tuple((k, v) for k, v in pillars(i, s).items() if k in ("day", "hour"))
            for i, s in completions(case)
        }
        assert len(charts) > 1, case["case_id"]
        confirmed += 1
    assert confirmed == 9  # six unspecified-sect and three repeated-time families
