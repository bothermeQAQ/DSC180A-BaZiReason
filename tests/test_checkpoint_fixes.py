"""Adversarial regressions for the independently audited defects E1-E9 and output handling."""

import copy
import json
import re
import sys
from importlib.metadata import metadata
from importlib.resources import files
from pathlib import Path
from zoneinfo import ZoneInfoNotFoundError

import pytest

from bazireason.baselines import recover_json, run_case
from bazireason.calendar_engine import compute_chart, load_timezone, tzdata_keys
from bazireason.cli import main
from bazireason.controls import corrupt_context
from bazireason.conventions import Convention
from bazireason.dataset_generator import (
    generate_audit_dataset,
    generate_dataset,
    read_jsonl,
    write_jsonl,
)
from bazireason.dataset_v3 import convention_dependence
from bazireason.evaluator import evaluate, resampling_clusters
from bazireason.llm_clients import Completion

PROFILE = Convention(2).to_dict()
V3 = "data/synthetic/benchmark_v3.jsonl"


def chart(dt, zone="Asia/Shanghai", convention=PROFILE, **extra):
    return compute_chart({"datetime": dt, "timezone": zone, **extra}, convention)


class Fixture:
    model = "FIXTURE-NOT-REAL-LLM"

    def __init__(self, text):
        self.text = text

    def complete(self, *args, **kwargs):
        return Completion(self.text, self.model, 0)


class CopyContext:
    """Copies whatever answer context it is shown; no calendar logic."""

    model = "FIXTURE-NOT-REAL-LLM"

    def complete(self, messages, **kwargs):
        return Completion(
            messages[1]["content"].split("Calendar tool context:\n")[1], self.model, 0
        )


@pytest.fixture(scope="module")
def v3():
    return read_jsonl(V3)


# ---- E1: one strict, version-independent datetime contract ----------------
@pytest.mark.parametrize(
    "raw,reason",
    [
        ("2025-02-03T24:00:00", "invalid_date_or_timezone"),  # 3.14 fromisoformat accepts this
        ("2025-02-03T23:60:00", "invalid_date_or_timezone"),
        ("2025-13-03T12:00:00", "invalid_date_or_timezone"),
        ("2025-W06-1T22:12:28", "invalid_datetime_format"),  # ISO week date
        ("20250203T22:12:28.0", "invalid_datetime_format"),  # ISO basic format
        ("2025-034T22:12:28.0", "invalid_datetime_format"),  # ordinal date
        ("２０２５-02-03T22:12:28", "invalid_datetime_format"),  # non-ASCII digits
    ],
)
def test_e1_datetime_contract_does_not_depend_on_python_iso_parser(raw, reason):
    result = chart(raw)
    assert (result["status"], result["reason_codes"]) == ("invalid", [reason])


# ---- E2: zone keys are exact tzdata keys on every filesystem ---------------
@pytest.mark.parametrize("zone", ["asia/shanghai", "AMERICA/LOS_ANGELES", "Asia/shanghai", "CST"])
def test_e2_timezone_keys_are_case_sensitive_on_every_host(zone):
    assert chart("2024-06-15T12:00:00", zone)["status"] == "invalid"
    with pytest.raises(ZoneInfoNotFoundError):
        load_timezone(zone)


def test_e2_key_list_comes_from_the_pinned_tzdata_package():
    listed = files("tzdata").joinpath("zones").read_text(encoding="utf-8").split()
    assert tzdata_keys() == frozenset(listed)
    assert {"Asia/Shanghai", "America/Los_Angeles", "UTC"} <= tzdata_keys()


# ---- E3: present-but-malformed is invalid; only absent values are missing ---
@pytest.mark.parametrize(
    "value,reason",
    [
        ("2025-02-03T14:12:28Z", "invalid_datetime_format"),
        ("2025-02-03T14:12:28+00:00", "invalid_datetime_format"),
        ("2025-02-03T22:12:28.000000", "invalid_datetime_format"),
        ("2025-02-03 22:12:28", "invalid_datetime_format"),
        (" 2025-02-03T22:12:28", "invalid_datetime_format"),
        (20250203, "datetime_must_be_string"),
        (["2025-02-03T22:12:28"], "datetime_must_be_string"),
        (0, "datetime_must_be_string"),
        (False, "datetime_must_be_string"),
    ],
)
def test_e3_complete_or_malformed_inputs_are_not_labeled_ambiguous(value, reason):
    result = chart(value)
    assert (result["status"], result["reason_codes"]) == ("invalid", [reason])


@pytest.mark.parametrize("inputs", [{}, {"datetime": None}, {"datetime": ""}])
def test_e3_absent_datetime_still_requests_clarification(inputs):
    result = compute_chart({"timezone": "Asia/Shanghai", **inputs}, PROFILE)
    assert (result["status"], result["reason_codes"]) == (
        "ambiguous",
        ["missing_datetime_or_timezone"],
    )


def test_e3_lower_precision_is_ambiguous_only_when_it_can_matter():
    # Lichun 2025 is 22:10:28 (UTC+08:00): only the minute containing it is ambiguous.
    assert chart("2025-02-03T22:10")["reason_codes"] == ["seconds_required"]
    assert chart("2025-02-03T22:09")["status"] == chart("2025-02-03T22:11")["status"] == "ok"
    assert chart("2025-02-03T22:09")["pillars"]["year"] == "甲辰"
    assert chart("2025-02-03T22:11")["pillars"]["year"] == "乙巳"
    assert chart("2025-02-03")["reason_codes"] == ["full_wall_time_required"]


# ---- E4: the study range is checked over every completion first -----------
@pytest.mark.parametrize("fold", [None, 0, 1])
def test_e4_lmt_fold_wholly_before_1901_is_unsupported(fold):
    extra = {} if fold is None else {"fold": fold}
    result = chart("1900-12-31T23:59:59", **extra)
    assert (result["status"], result["reason_codes"]) == ("unsupported", ["outside_study_range"])
    assert chart("1901-01-01T00:10:00")["status"] == "ok"


# ---- E5: the corrupted-context denominator is fixed by the design ---------
def test_e5_missing_or_self_describing_records_cannot_shrink_corruption_denominator(v3):
    cases = [c for c in v3 if c["category"] in ("jie_ladder", "zi_ladder")][:24]
    records = [run_case(c, CopyContext(), "tool_corrupted") for c in cases]
    full = evaluate(cases, records, methods=["tool_corrupted"], bootstrap_samples=0)
    overall = full["methods"]["tool_corrupted"]["overall"]
    assert overall["n_corrupted_context"] == 24 and overall["corruption_following_rate"] == 1
    partial = evaluate(cases, records[1:], methods=["tool_corrupted"], bootstrap_samples=0)
    overall = partial["methods"]["tool_corrupted"]["overall"]
    assert overall["n_corrupted_context"] == 24 and overall["n_corrupted_missing"] == 1
    assert overall["corruption_following_rate"] == 23 / 24
    lying = copy.deepcopy(records)
    for record in lying:
        record["corruption"], record["context_corrupted"] = None, False
    overall = evaluate(cases, lying, methods=["tool_corrupted"], bootstrap_samples=0)["methods"][
        "tool_corrupted"
    ]["overall"]
    assert overall["n_corrupted_context"] == 24 and overall["corruption_following_rate"] == 1


# ---- E6: evaluation_class compares the two sects directly ------------------
def test_e6_v3_evaluation_class_matches_direct_sect_comparison(v3):
    for case in v3:
        assert case["evaluation_class"] == convention_dependence(case["input"], case["convention"])
    dependent = [c for c in v3 if c["evaluation_class"] == "convention_dependent"]
    assert dependent and all(c["input"]["datetime"][11:13] == "23" for c in dependent)


def _legacy_errors(cases):
    return sorted(
        c["category"]
        for c in cases
        if c["expected"]["status"] == "ok"
        and (c["evaluation_class"] == "convention_dependent")
        != (convention_dependence(c["input"], c["convention"]) == "convention_dependent")
    )


def test_e6_legacy_field_errors_are_exactly_the_documented_ones():
    # zi_hour cases whose answer ignores the sect, plus 23:xx cases the old rule missed
    # (Daxue 2024 brackets at 23:15:03/23:19:03 and the h3 23:30 counterparts).
    assert _legacy_errors(generate_audit_dataset()) == (
        ["h3_convention"] * 6 + ["solar_term"] * 2 + ["zi_hour"] * 12
    )
    assert _legacy_errors(generate_dataset()) == ["solar_term"] * 2 + ["zi_hour"] * 8


# ---- E7: matched variants resample together ---------------------------------
def test_e7_v3_variants_of_one_civil_date_share_a_group(v3):
    owner = {}
    for case in v3:
        raw = case["input"].get("datetime")
        if isinstance(raw, str):
            assert owner.setdefault(raw[:10], case["group_id"]) == case["group_id"]
    clusters = resampling_clusters(v3)
    assert len(set(clusters.values())) == len({c["group_id"] for c in v3})


def test_e7_v2_noon_family_is_merged_for_resampling():
    cases = generate_audit_dataset()
    family = {
        c["case_id"] for c in cases if str(c["input"].get("datetime", ""))[:10] == "2024-06-15"
    }
    assert {"BRA-069", "BRA-085", "BRA-087", "BRA-089", "BRA-092", "BRA-096"} <= family
    clusters = resampling_clusters(cases)
    assert len({clusters[i] for i in family}) == 1
    assert len({c["group_id"] for c in cases if c["case_id"] in family}) > 1


# ---- E8: the bibliography cites the installed sxtwl distribution -----------
def test_e8_sxtwl_citation_matches_installed_metadata():
    bib = Path("reports/latex/references.bib").read_text(encoding="utf-8")
    entry = bib[bib.index("@misc{sxtwl") :].split("\n@", 1)[0]
    home = metadata("sxtwl")["Home-page"].rstrip("/")
    assert home == "https://github.com/yuangu/sxtwl_cpp"
    assert re.search(r"url=\{" + re.escape(home) + r"\}", entry)
    assert "skydancep" not in entry


# ---- E9: current defaults and write protection -----------------------------
def test_e9_generate_refuses_to_overwrite_and_protects_history(tmp_path, monkeypatch, capsys):
    for argv in (
        ["bazireason", "generate"],  # current file exists
        ["bazireason", "generate", "--dataset-version", "audit_v2", "--force"],
        ["bazireason", "generate", "--dataset-version", "checkpoint_v1", "--force"],
    ):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
    for version, path in (
        ("checkpoint_v1", "data/synthetic/checkpoint.jsonl"),
        ("audit_v2", "data/synthetic/audit_v2.jsonl"),
        ("benchmark_v3", V3),
    ):
        monkeypatch.setattr(
            sys, "argv", ["bazireason", "generate", "--dataset-version", version, "--check", path]
        )
        main()
        assert capsys.readouterr().out.startswith("PASS")
    target = tmp_path / "x.jsonl"
    write_jsonl(target, [{"a": 1}])
    with pytest.raises(FileExistsError):
        write_jsonl(target, [{"a": 2}])


def test_e9_run_defaults_to_current_suite_and_fails_closed(tmp_path, monkeypatch):
    out = tmp_path / "run.jsonl"
    monkeypatch.setattr(sys, "argv", ["bazireason", "run", "--output", str(out)])
    with pytest.raises(SystemExit) as exc:  # 261 cases x 2 methods exceeds the default cap
        main()
    assert exc.value.code == 2 and not out.exists()
    monkeypatch.setattr(
        sys,
        "argv",
        ["bazireason", "run", "--output", str(out), "--limit", "3", "--methods", "copy_context"],
    )
    main()
    manifest = json.loads((tmp_path / "run.jsonl.manifest.json").read_text())
    assert manifest["case_ids"] == ["B3-001", "B3-002", "B3-003"]


# ---- Structured outputs: strict primary scoring, separate recovery ---------
def test_fenced_json_fails_strictly_but_is_recovered_separately(v3):
    case = next(c for c in v3 if c["expected"]["status"] == "ok")
    text = "```json\n" + json.dumps(case["expected"], ensure_ascii=False) + "\n```"
    record = run_case(case, Fixture(text), "direct")
    assert record["error"] == "invalid_json" and record["response"] is None
    assert record["recovery"] == "code_fence" and record["recovered_response"] == case["expected"]
    overall = evaluate([case], [record], methods=["direct"], bootstrap_samples=0)["methods"][
        "direct"
    ]["overall"]
    assert overall["exact_accuracy"] == 0 and overall["format_failures"]["invalid_json"] == 1
    assert overall["lenient_parse_secondary"]["exact_accuracy"] == 1


@pytest.mark.parametrize(
    "text,method",
    [
        ('Here it is: {"status": "ok"} done', "outer_braces"),
        ('```\n{"status":"ok","status":"invalid"}\n```', None),  # duplicate keys stay rejected
        (None, None),  # provider returned null content
        ("no json", None),
    ],
)
def test_recovery_rules(text, method):
    value, how = recover_json(text)
    assert how == method and (value is None) == (method is None)


def test_null_provider_content_is_a_recorded_failure(v3):
    record = run_case(v3[0], Fixture(None), "direct")
    assert record["error"] == "invalid_json" and record["recovered_response"] is None


# ---- Manifest and prediction integrity -------------------------------------
def _run(monkeypatch, tmp_path, name, *extra):
    out = tmp_path / name
    argv = ["bazireason", "run", "--output", str(out), "--limit", "2", "--max-calls", "4"]
    monkeypatch.setattr(sys, "argv", argv + list(extra) + ["--methods", "direct", "copy_context"])
    main()
    return out


def _evaluate(monkeypatch, out, scores):
    monkeypatch.setattr(
        sys,
        "argv",
        ["bazireason", "evaluate", "--predictions", str(out), "--output", str(scores)],
    )
    main()
    return json.loads(scores.read_text())


def test_records_from_another_run_are_rejected(tmp_path, monkeypatch):
    out = _run(monkeypatch, tmp_path, "run.jsonl")
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    model_row = next(r for r in rows if r["method"] == "direct")
    for field, value in (
        ("prompt_version", "v0-other"),
        ("requested_model", "x"),
        ("temperature", 1),
    ):
        tampered = [dict(r, **{field: value}) if r is model_row else r for r in rows]
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in tampered))
        with pytest.raises(SystemExit) as exc:
            _evaluate(monkeypatch, out, tmp_path / f"scores-{field}.json")
        assert exc.value.code == 2


def test_truncated_final_line_is_reported_and_stays_missing(tmp_path, monkeypatch):
    out = _run(monkeypatch, tmp_path, "run.jsonl")
    text = out.read_text()
    out.write_text(text[: len(text) - 25])  # interrupted mid-write: no final newline
    result = _evaluate(monkeypatch, out, tmp_path / "scores.json")
    assert result["truncated_final_line_dropped"] is True
    assert result["missing_predictions"] == 1 and result["manifest_status"] == "complete"
    lines = text.splitlines()
    out.write_text(lines[0] + "\n{broken\n" + "\n".join(lines[1:]) + "\n")
    with pytest.raises(SystemExit) as exc:
        _evaluate(monkeypatch, out, tmp_path / "scores-broken.json")
    assert exc.value.code == 2


def test_evaluation_refuses_to_overwrite_scores(tmp_path, monkeypatch):
    out = _run(monkeypatch, tmp_path, "run.jsonl")
    _evaluate(monkeypatch, out, tmp_path / "scores.json")
    with pytest.raises(SystemExit) as exc:
        _evaluate(monkeypatch, out, tmp_path / "scores.json")
    assert exc.value.code == 2


def test_tool_corrupted_records_match_design_corruption(v3):
    case = next(c for c in v3 if (c["design"].get("h1") or {}).get("distance") == "near")
    record = run_case(case, CopyContext(), "tool_corrupted")
    expected = corrupt_context(case, compute_chart(case["input"], case["convention"]))[1]
    assert record["corruption"] == expected and expected["type"] == "jie_side"
