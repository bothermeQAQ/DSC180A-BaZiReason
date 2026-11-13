"""Phase 4 tests: data integrity, leakage guards, approach B checks. No model is run.

The final test split is only hashed here; its rows are not inspected.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/phase4"


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def user_prompts(path):
    return {next(m["content"] for m in r["messages"] if m["role"] == "user") for r in rows(path)}


def test_phase4_files_match_their_manifests():
    for folder in ("valid", "final_test", "train_v1"):
        manifest = json.loads((DATA / folder / "manifest.json").read_text())
        for name, info in manifest["files"].items():
            digest = hashlib.sha256((DATA / folder / name).read_bytes()).hexdigest()
            assert digest == info["sha256"], (folder, name)


def test_validation_split_regenerates_byte_for_byte():
    sys.path.insert(0, str(ROOT / "scripts/phase4"))
    import build_splits

    from bazireason import indep_rules as I
    from bazireason import train_data as TD

    parts = build_splits.build("valid")
    for part, built in parts.items():
        payload = TD.dataset_bytes(built) if part != "indep" else I.dumps(built).encode()
        assert payload == (DATA / "valid" / f"{part}.jsonl").read_bytes(), part


def test_no_inspected_or_training_prompt_reaches_the_final_test():
    final = set()
    for part in ("p3", "p2", "indep"):
        final |= user_prompts(DATA / "final_test" / f"{part}.jsonl")
    inspected = set()
    for path in (
        "data/phase3/train_v1/test.jsonl",
        "data/phase2/train_v1/test.jsonl",
        "data/phase2/train_v2/test.jsonl",
        "data/phase3/p2check_v1/test.jsonl",
    ):
        inspected |= user_prompts(ROOT / path)
    assert not final & inspected
    # Only finite-space table drills can coincide with a training prompt.
    assert len(final & user_prompts(DATA / "train_v1/train.jsonl")) <= 5


def test_independent_labels_agree_with_the_projects_rule_code_on_validation():
    from bazireason import workflow as W
    from bazireason.indep_rules import _norm

    for row in rows(DATA / "valid/indep.jsonl"):
        kind = row["meta"]["kind"]
        assert _norm(kind, W.indep_engine(row)) == _norm(kind, row["meta"]["expected"]["answer"])


def test_tool_check_flags_corruptions_and_passes_clean_answers():
    from bazireason import workflow as W

    seen = {}
    for row in rows(ROOT / "data/phase3/train_v1/test.jsonl"):
        if row["family"] != "strict_shift":
            continue
        meta = row["meta"]
        failed = W.tool_check(meta["inputs"], meta["convention"], meta["supplied"])
        seen.setdefault(meta["kind"].startswith("clean"), []).append(bool(failed))
    assert not any(seen[True]) and all(seen[False])


def test_every_prompt_ends_with_the_question_and_never_with_the_answer():
    from bazireason import workflow as W

    for part in ("p3", "p2", "indep"):
        for row in rows(DATA / "valid" / f"{part}.jsonl"):
            prompt = W.prompt_messages(row)
            assert prompt[-1]["role"] == "user", (part, row["id"])
            assert len(prompt) >= len(row["messages"]) - 1
    row = rows(DATA / "valid/indep.jsonl")[0]
    repair = W.repair_turn(row, "结论JSON: {}", ["answer"])
    assert repair[1] == row["messages"][1] and "结论JSON" in repair[-1]["content"]


def test_repair_note_names_fields_but_never_values():
    from bazireason import workflow as W

    row = next(r for r in rows(DATA / "valid/p2.jsonl") if r["task"] == "chart")
    expected = row["meta"]["expected"]
    if expected.get("status") != "ok":
        pytest.skip("first chart row is not answerable")
    wrong = dict(expected, pillars=dict(expected["pillars"], day="甲子"))
    fields = W.differing_fields(expected, wrong)
    assert fields == ["日柱"] or expected["pillars"]["day"] == "甲子"
    note = W.repair_turn(row, "结论JSON: {}", fields)[-1]["content"]
    assert all(value not in note for value in expected["pillars"].values())


def test_final_test_is_sealed_without_a_selection(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts/phase4"))
    import run_eval

    monkeypatch.setattr(run_eval, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--label", "x", "--split", "final_test"])
    with pytest.raises(SystemExit):
        run_eval.main()
    assert not (tmp_path / "final_test").exists()


def test_training_data_has_procedure_headers_and_no_book_titles():
    from bazireason import train_data4 as D

    train = rows(DATA / "train_v1/train.jsonl")
    assert not any("《" in r["messages"][-1]["content"] for r in train)
    for row in train:
        if row["task"] in D.TASK_PROCEDURE:
            assert row["messages"][-1]["content"].startswith("【规则】"), row["task"]
    augmented = [r for r in train if r.get("augmented")]
    assert augmented and all("结论JSON" in r["messages"][-2]["content"] for r in augmented)


def test_claim_checker_flags_errors_but_not_partial_lists_or_rule_explanations():
    from bazireason.claim_check import check_claims

    pillars = {"year": "壬申", "month": "丁未", "day": "己未", "hour": "甲戌"}

    def verdicts(text, kind):
        return [r["ok"] for r in check_claims(text, pillars) if r["kind"] == kind]

    # Main qi only, or each stem followed by its Ten God: not wrong.
    assert verdicts("年支申藏庚伤官、壬正财、戊劫财。", "hidden") == [True]
    assert verdicts("月支未藏己。", "hidden") == [True]
    # A stem that is not hidden in the branch: wrong.
    assert verdicts("月支未藏己、丁、壬。", "hidden") == [False]
    # Naming rule explained, no claim about this chart.
    assert verdicts("劫财在阳干记阳刃格、阴干记月劫格。", "pattern") == []
    assert verdicts("按月令透干取格记作伤官格。", "pattern") == [False]
