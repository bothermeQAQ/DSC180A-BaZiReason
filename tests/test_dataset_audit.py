"""Dataset design properties independent of reference pillar values."""

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pytest

from bazireason.calendar_engine import compute_chart
from bazireason.dataset_generator import (
    generate_audit_dataset,
    generate_dataset,
    read_jsonl,
    representative_case_ids,
    write_jsonl,
)


@pytest.fixture(scope="module")
def audited_cases():
    return generate_audit_dataset()


def test_original_snapshot_not_silently_changed():
    assert read_jsonl("data/synthetic/checkpoint.jsonl") == generate_dataset()


def test_versioned_suite_size_status_and_no_duplicate_inputs(audited_cases):
    assert len(audited_cases) == 152
    assert Counter(c["expected"]["status"] for c in audited_cases) == {
        "ok": 126,
        "ambiguous": 16,
        "unsupported": 6,
        "invalid": 4,
    }
    keys = [json.dumps([c["input"], c["convention"]], sort_keys=True) for c in audited_cases]
    assert len(keys) == len(set(keys))
    assert all(c["dataset_version"] == "audit_v2" for c in audited_cases)


def test_all_twelve_jie_have_paired_offsets(audited_cases):
    terms = defaultdict(set)
    for case in audited_cases:
        if case["category"] == "solar_term":
            meta = case["boundary_metadata"]
            terms[meta["term"]].add((meta["engine_boundary_utc8"][:4], meta["offset_seconds"]))
    assert len(terms) == 12
    assert all(
        v == {("2024", -120), ("2024", 120), ("2025", -120), ("2025", 120)} for v in terms.values()
    )


def test_minimal_midnight_pairs_are_successive(audited_cases):
    groups = defaultdict(list)
    for c in audited_cases:
        if c["category"] == "zi_hour":
            groups[(c["group_id"], c["convention"]["sect"])].append(c)
    for (_, sect), cases in groups.items():
        ordered = sorted(cases, key=lambda c: c["input"]["datetime"])
        assert len(ordered) == 5
        wall = [datetime.fromisoformat(c["input"]["datetime"]) for c in ordered]
        assert (wall[3] - wall[2]).total_seconds() == 1
        before = ordered[2]["expected"]["pillars"]
        after = ordered[3]["expected"]["pillars"]
        assert before["hour"] == after["hour"]
        assert (before["day"] == after["day"]) == (sect == 1)


def test_timezone_equivalent_instant_groups_are_disjoint(audited_cases):
    groups = defaultdict(list)
    for c in audited_cases:
        if c["category"] == "timezone":
            result = compute_chart(c["input"], c["convention"])
            groups[c["group_id"]].append(result)
    roots = []
    for values in groups.values():
        assert len(values) == 2
        assert values[0]["normalized_datetime"] == values[1]["normalized_datetime"]
        assert values[0]["pillars"] == values[1]["pillars"]
        roots.append(values[0]["normalized_datetime"])
    assert len(roots) == len(set(roots)) == 4


def test_h3_groups_contain_missing_and_specified_counterparts(audited_cases):
    groups = defaultdict(list)
    for c in audited_cases:
        if c["category"].startswith("h3_"):
            groups[c["group_id"]].append(c)
    assert len(groups) == 10
    for cases in groups.values():
        assert {c["expected"]["status"] for c in cases} == {"ok", "ambiguous"}
        assert sum(c["expected"]["status"] == "ambiguous" for c in cases) == 1
        missing = next(c for c in cases if c["expected"]["status"] == "ambiguous")
        assert missing["expected"]["pillars"] is None
        if missing["category"] == "h3_convention":
            # Ambiguity is substantive at23:30; profiles actually disagree.
            assert (
                len(
                    {
                        c["expected"]["pillars"]["day"]
                        for c in cases
                        if c["expected"]["status"] == "ok"
                    }
                )
                == 2
            )


def test_v2_determinism_and_representative_pilot(audited_cases, tmp_path):
    path = tmp_path / "v2.jsonl"
    write_jsonl(path, generate_audit_dataset())
    assert path.read_bytes() == Path("data/synthetic/audit_v2.jsonl").read_bytes()
    assert audited_cases != generate_audit_dataset(123)
    ids = representative_case_ids(audited_cases)
    assert ids == json.loads(Path("data/synthetic/audit_pilot_ids.json").read_text())
    selected = [c for c in audited_cases if c["case_id"] in ids]
    assert {c["category"] for c in selected} == {c["category"] for c in audited_cases}
    assert {c["expected"]["status"] for c in selected} == {
        "ok",
        "ambiguous",
        "unsupported",
        "invalid",
    }
    assert {c["convention"]["sect"] for c in selected if c["category"] == "zi_hour"} == {1, 2}


def test_equivalent_inputs_never_cross_group_boundaries(audited_cases):
    groups = defaultdict(set)
    for case in audited_cases:
        result = compute_chart(case["input"], case["convention"])
        if result["status"] == "ok":
            groups[(result["normalized_datetime"], case["convention"]["sect"])].add(
                case["group_id"]
            )
    assert all(len(ids) == 1 for ids in groups.values())
