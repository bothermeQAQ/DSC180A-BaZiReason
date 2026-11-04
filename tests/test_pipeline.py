"""Dataset reproducibility, prompt isolation, schema, metrics and network adapter tests."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from bazireason.baselines import run_case
from bazireason.calendar_engine import compute_chart
from bazireason.dataset_generator import generate_dataset, read_jsonl, write_jsonl
from bazireason.evaluator import evaluate
from bazireason.llm_clients import Completion, OpenAICompatibleClient, SyntheticMockClient
from bazireason.metrics import aggregate, score_case, wilson
from bazireason.prompts import build_messages
from bazireason.schema import valid_response


@pytest.fixture(scope="module")
def cases():
    return generate_dataset()


def test_dataset_determinism(cases, tmp_path):
    assert len(cases) == 96 and len({c["case_id"] for c in cases}) == 96
    assert cases == generate_dataset()
    assert cases != generate_dataset(8)
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    write_jsonl(a, cases)
    write_jsonl(b, generate_dataset())
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes() == Path("data/synthetic/checkpoint.jsonl").read_bytes()
    assert all(valid_response(c["expected"]) for c in cases)
    assert all(c["validation_status"] == "single_engine_reference" for c in cases)


def test_prompt_isolation(cases):
    c = copy.deepcopy(cases[0])
    a = build_messages(c, "direct")
    c["expected"] = {"SENTINEL_SECRET_ANSWER": 123}
    c["boundary_metadata"] = {"SENTINEL_BOUNDARY": 42}
    assert a == build_messages(c, "direct")
    assert "SENTINEL" not in json.dumps(a)
    context = compute_chart(c["input"], c["convention"])
    assert build_messages(c, "tool_context", context) != a
    with pytest.raises(ValueError):
        build_messages(c, "tool_context")


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"status": []},
        {"status": [], "pillars": None, "conventions": None, "reason_codes": []},
        {"status": "ok", "pillars": None, "conventions": {}, "reason_codes": []},
        {"status": "ambiguous", "pillars": {}, "conventions": {}, "reason_codes": ["missing"]},
    ],
)
def test_bad_schemas(value):
    assert not valid_response(value)


def test_metrics_failures_and_denominators(cases):
    c = cases[0]
    correct = {"response": c["expected"]}
    broken = copy.deepcopy(correct)
    broken["response"]["pillars"]["hour"] = "甲子"
    if c["expected"]["pillars"]["hour"] == "甲子":
        broken["response"]["pillars"]["hour"] = "乙丑"
    scores = [score_case(c, correct), score_case(c, broken), score_case(c, None)]
    result = aggregate(scores)
    assert result["exact_accuracy"] == 1 / 3
    assert result["per_pillar_accuracy"]["year"] == 2 / 3
    assert result["per_pillar_accuracy"]["hour"] == 1 / 3
    assert result["valid_output_rate"] == 2 / 3
    assert wilson(0, 0) is None
    assert aggregate([])["exact_accuracy"] is None


def test_uncertainty_and_contradictions(cases):
    c = next(c for c in cases if c["expected"]["status"] == "ambiguous")
    assert score_case(c, {"response": c["expected"]})["uncertainty_correct"]
    wrong = copy.deepcopy(c["expected"])
    wrong["pillars"] = cases[0]["expected"]["pillars"]
    score = score_case(c, {"response": wrong})
    assert score["contradiction"] and not score["uncertainty_correct"]


def test_complete_evaluation_and_missing(cases):
    output = evaluate(cases, [])
    assert output["missing_predictions"] == 192
    assert output["methods"]["direct"]["overall"]["exact_accuracy"] == 0
    assert output["paired"]["n"] == 80
    pred = {
        "case_id": cases[0]["case_id"],
        "method": "direct",
        "trial": 0,
        "response": cases[0]["expected"],
    }
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate(cases, [pred, pred])
    with pytest.raises(ValueError, match="Unexpected"):
        evaluate(cases, [{**pred, "case_id": "unknown"}])


def test_repeated_consistency(cases):
    predictions = [
        {"case_id": cases[0]["case_id"], "method": m, "trial": t, "response": cases[0]["expected"]}
        for m in ["direct", "tool_context"]
        for t in range(2)
    ]
    result = evaluate(cases[:1], predictions, trials=2)
    assert result["methods"]["direct"]["pairwise_response_consistency"] == 1


def test_api_adapter_without_network_or_secrets(cases):
    def handler(request):
        body = json.loads(request.content)
        assert body["temperature"] == 0 and body["max_tokens"] == 512
        return httpx.Response(
            200,
            json={
                "model": "fixture-model",
                "choices": [
                    {
                        "message": {"content": json.dumps(cases[0]["expected"])},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 18},
            },
        )

    client = OpenAICompatibleClient(
        "requested-model",
        "https://example.invalid/v1",
        api_key="TEST-DUMMY",
        transport=httpx.MockTransport(handler),
    )
    record = run_case(cases[0], client, "direct")
    assert valid_response(record["response"])
    assert record["served_model"] == "fixture-model" and record["usage"]["total_tokens"] == 18
    assert "TEST-DUMMY" not in json.dumps(record)


def test_invalid_json_and_http_failure(cases):
    class BadJSON:
        model = "fixture"

        def complete(self, *args, **kwargs):
            return Completion("not json", "fixture", 0)

    assert run_case(cases[0], BadJSON(), "direct")["error"] == "invalid_json"
    client = OpenAICompatibleClient(
        "fixture",
        "https://example.invalid",
        api_key="TEST-DUMMY",
        transport=httpx.MockTransport(lambda req: httpx.Response(401, text="sensitive body")),
    )
    record = run_case(cases[0], client, "direct")
    assert record["error"] == "RuntimeError"
    assert "sensitive" not in json.dumps(record)


def test_mock_is_explicit(cases):
    prediction = run_case(cases[0], SyntheticMockClient(), "tool_context")
    assert prediction["synthetic"] and prediction["served_model"].startswith("SYNTHETIC")


def test_cli_end_to_end_and_budget(tmp_path):
    data = tmp_path / "data.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    scores = tmp_path / "scores.json"

    def command(*args):
        return subprocess.run(
            [sys.executable, "-c", "from bazireason.cli import main;main()", *map(str, args)],
            capture_output=True,
            text=True,
        )

    assert command("generate", "--output", data).returncode == 0
    assert (
        command(
            "run", "--data", data, "--output", predictions, "--limit", 2, "--max-calls", 3
        ).returncode
        != 0
    )
    assert not predictions.exists()
    assert (
        command(
            "run", "--data", data, "--output", predictions, "--limit", 2, "--max-calls", 4
        ).returncode
        == 0
    )
    assert len(read_jsonl(predictions)) == 4
    assert (
        command(
            "evaluate",
            "--data",
            data,
            "--predictions",
            predictions,
            "--output",
            scores,
            "--limit",
            2,
        ).returncode
        == 0
    )
    assert json.loads(scores.read_text())["synthetic"]


def test_evaluation_rejects_bad_design(cases):
    with pytest.raises(ValueError, match="Trials"):
        evaluate(cases, [], trials=0)
    with pytest.raises(ValueError, match="Duplicate dataset"):
        evaluate([cases[0], cases[0]], [])
