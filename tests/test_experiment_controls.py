"""Scientific controls and regression tests; no network or real-model outcomes."""

import copy
import json
import sys

import pytest

from bazireason.baselines import run_case
from bazireason.calendar_engine import compute_chart
from bazireason.cli import main
from bazireason.dataset_generator import generate_dataset, read_jsonl, write_jsonl
from bazireason.evaluator import evaluate, paired_estimate
from bazireason.llm_clients import Completion, SyntheticMockClient
from bazireason.metrics import aggregate, score_case
from bazireason.prompts import CALENDAR_SPEC, EXPLICIT_POLICY, build_messages
from bazireason.schema import valid_response


@pytest.fixture(scope="module")
def cases():
    return generate_dataset()


def test_policy_factor_changes_only_nonanswer_instruction(cases):
    for method in ("direct", "tool_context", "tool_facts", "tool_corrupted"):
        c = cases[0]
        context = compute_chart(c["input"], c["convention"])
        explicit = build_messages(c, method, context)
        minimal = build_messages(c, method + "_minimal", context)
        assert explicit[1] == minimal[1]
        assert explicit[0]["content"] == minimal[0]["content"] + EXPLICIT_POLICY
        assert CALENDAR_SPEC in minimal[0]["content"]


def test_controls_do_not_read_gold_or_mutate_reference(cases):
    c = copy.deepcopy(cases[0])
    reference = compute_chart(c["input"], c["convention"])
    original = copy.deepcopy(reference)
    facts = build_messages(c, "tool_facts", reference)
    corrupt = build_messages(c, "tool_corrupted", reference)
    c["expected"] = {"SENTINEL_LABEL": 999}
    assert facts == build_messages(c, "tool_facts", reference)
    assert "SENTINEL" not in json.dumps(facts)
    supplied_facts = json.loads(facts[1]["content"].split("Normalized timestamp context:\n")[1])
    assert set(supplied_facts) == {"normalized_datetime", "provenance"}
    assert "pillars" not in supplied_facts and "status" not in supplied_facts
    supplied_corrupt = json.loads(corrupt[1]["content"].split("Calendar tool context:\n")[1])
    assert supplied_corrupt["pillars"]["hour"] != reference["pillars"]["hour"]
    assert all(
        supplied_corrupt["pillars"][p] == reference["pillars"][p] for p in ("year", "month", "day")
    )
    assert reference == original


def test_copy_control_never_calls_model_or_reads_expected(cases):
    class ForbiddenClient:
        def complete(self, *args, **kwargs):
            pytest.fail("Deterministic control invoked an LLM")

    c = copy.deepcopy(cases[0])
    expected = c.pop("expected")
    record = run_case(c, ForbiddenClient(), "copy_context")
    assert record["response"] == expected
    assert record["record_kind"] == "deterministic_control"
    assert record["is_llm"] is False and record["synthetic"] is False
    assert record["attempts"] == 0 and record["prompt_sha256"] is None


def test_h2_corruption_adherence_is_distinct_from_accuracy(cases):
    class CopySuppliedContext:
        model = "FIXTURE-NOT-REAL-LLM"

        def complete(self, messages, **kwargs):
            text = messages[1]["content"].split("Calendar tool context:\n")[1]
            return Completion(text, self.model, 0)

    record = run_case(cases[0], CopySuppliedContext(), "tool_corrupted")
    result = aggregate([score_case(cases[0], record)])
    assert result["exact_accuracy"] == 0
    assert result["corrupted_hour_following_rate"] == 1
    assert result["per_pillar_accuracy"]["day"] == 1


def test_nonanswer_definitive_metric_includes_invalid_schema(cases):
    c = next(c for c in cases if c["expected"]["status"] == "ambiguous")
    # This already fails schema checks, but still makes an unsupported definitive claim.
    wrong = {"status": "ok", "pillars": {"hour": "甲子"}}
    result = aggregate([score_case(c, {"response": wrong}), score_case(c, None)])
    assert result["unsupported_definitive_rate"] == 0.5
    assert result["ambiguous_definitive_rate"] == 0.5
    assert result["valid_output_rate"] == 0
    assert result["n_nonanswer"] == 2


def test_accuracy_joint_profile_and_refusal_coverage_are_separate(cases):
    c = cases[0]
    wrong = copy.deepcopy(c["expected"])
    wrong["conventions"]["sect"] = 1 if wrong["conventions"]["sect"] == 2 else 2
    result = aggregate([score_case(c, {"response": wrong})])
    assert result["exact_accuracy"] == 1 and result["joint_exact_accuracy"] == 0
    assert result["scorable_coverage"] == 1
    refusal = run_case(c, SyntheticMockClient(), "direct")
    assert aggregate([score_case(c, refusal)])["scorable_coverage"] == 0


@pytest.mark.parametrize("trial", [True, False, 0.0, "0", None])
def test_prediction_identity_rejects_noninteger_trials(cases, trial):
    with pytest.raises(ValueError, match="identity"):
        evaluate(
            cases[:1],
            [
                {
                    "case_id": cases[0]["case_id"],
                    "method": "direct",
                    "trial": trial,
                    "response": cases[0]["expected"],
                }
            ],
        )


@pytest.mark.parametrize("methods", [[], ["direct", "direct"], [""], [None], "direct"])
def test_declared_methods_must_have_unique_nonempty_identifiers(cases, methods):
    with pytest.raises(ValueError, match="Methods"):
        evaluate(cases, [], methods=methods)


def test_stricter_schema_rejects_blank_reasons_and_malformed_profile(cases):
    value = copy.deepcopy(cases[0]["expected"])
    value["conventions"] = {}
    assert not valid_response(value)
    value = copy.deepcopy(cases[0]["expected"])
    value["conventions"]["sect"] = True
    assert not valid_response(value)
    value = {"status": "ambiguous", "pillars": None, "reason_codes": [" "], "conventions": None}
    assert not valid_response(value)


def test_parse_rejects_duplicate_keys_and_records_schema_failure(cases):
    class Fixture:
        model = "FIXTURE-NOT-REAL-LLM"

        def __init__(self, text):
            self.text = text

        def complete(self, *args, **kwargs):
            return Completion(self.text, self.model, 0)

    assert (
        run_case(cases[0], Fixture('{"status":"ok","status":"invalid"}'), "direct")["error"]
        == "invalid_json"
    )
    assert run_case(cases[0], Fixture("{}"), "direct")["error"] == "invalid_schema"


def test_grouped_estimates_and_h3_pairing(cases):
    rows = [("shared", 1), ("shared", 1), ("other", -1)]
    a = paired_estimate(rows, resamples=100, seed=7)
    assert a == paired_estimate(rows, resamples=100, seed=7)
    assert a["n_groups"] == 2 and a["mean_difference"] == 1 / 3
    assert paired_estimate([("one", 0)])["group_bootstrap95_descriptive"] is None
    c = next(c for c in cases if c["expected"]["status"] == "ambiguous")
    explicit = {"case_id": c["case_id"], "method": "direct", "trial": 0, "response": c["expected"]}
    minimal = {**explicit, "method": "direct_minimal", "response": cases[0]["expected"]}
    result = evaluate([c], [explicit, minimal], methods=["direct", "direct_minimal"])
    comparison = result["comparisons"]["direct_minus_direct_minimal"]
    assert comparison["ambiguous_definitive"]["mean_difference"] == -1
    assert comparison["ambiguous_policy_correct"]["mean_difference"] == 1
    assert comparison["coverage"]["mean_difference"] is None


def test_cli_exact_selection_controls_and_early_manifest(cases, tmp_path, monkeypatch):
    out = tmp_path / "copy.jsonl"
    ids = ",".join([cases[25]["case_id"], cases[85]["case_id"]])
    monkeypatch.setattr(
        sys,
        "argv",
        ["bazireason", "run", "--output", str(out), "--case-ids", ids, "--methods", "copy_context"],
    )
    main()
    records = read_jsonl(out)
    assert [r["case_id"] for r in records] == ids.split(",")
    manifest = json.loads((tmp_path / "copy.jsonl.manifest.json").read_text())
    assert manifest["n_calls"] == 0 and manifest["n_observations"] == 2
    assert manifest["backend"] == "deterministic_control" and manifest["status"] == "complete"
    auto_scores = tmp_path / "auto_scores.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["bazireason", "evaluate", "--predictions", str(out), "--output", str(auto_scores)],
    )
    main()
    auto = json.loads(auto_scores.read_text())
    assert auto["evaluation_design_source"] == "planned_manifest"
    assert auto["methods"]["copy_context"]["overall"]["n"] == 2

    class InterruptedClient:
        model = "FIXTURE-NOT-REAL-LLM"

        def complete(self, *args, **kwargs):
            raise KeyboardInterrupt()

    monkeypatch.setattr("bazireason.cli.SyntheticMockClient", InterruptedClient)
    interrupted = tmp_path / "interrupted.jsonl"
    monkeypatch.setattr(sys, "argv", ["bazireason", "run", "--output", str(interrupted)])
    with pytest.raises(KeyboardInterrupt):
        main()
    declared = json.loads((tmp_path / "interrupted.jsonl.manifest.json").read_text())
    assert declared["status"] == "planned" and declared["n_observations"] == 8


def test_manifest_prevents_silently_reducing_interrupted_denominator(cases, tmp_path, monkeypatch):
    data = tmp_path / "data.jsonl"
    out = tmp_path / "partial.jsonl"
    scores = tmp_path / "scores.json"
    write_jsonl(data, cases[:2])
    monkeypatch.setattr(
        sys, "argv", ["bazireason", "run", "--data", str(data), "--output", str(out)]
    )
    main()
    out.write_text(out.read_text().splitlines()[0] + "\n")
    base = [
        "bazireason",
        "evaluate",
        "--data",
        str(data),
        "--predictions",
        str(out),
        "--output",
        str(scores),
    ]
    monkeypatch.setattr(sys, "argv", base)
    main()
    assert json.loads(scores.read_text())["missing_predictions"] == 3
    for flags in (["--limit", "1"], ["--methods", "direct"], ["--trials", "2"]):
        monkeypatch.setattr(sys, "argv", base + flags)
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
    data.write_text(data.read_text() + "\n")
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


@pytest.mark.parametrize("methods", ["direct", {"direct": "unvalidated"}])
def test_method_collection_must_not_alias_string_or_mapping(cases, methods):
    with pytest.raises(ValueError, match="Methods"):
        evaluate(cases, [], methods=methods)
