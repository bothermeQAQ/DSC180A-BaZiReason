"""Phase 6, preregistration 12.15: level 10 waits for the user's choice.

No model is run. After level 9, "decide6.py next" waits until "decide6.py user 10 start|stop"
has recorded the user's choice; "start" runs level 10 as registered, "stop" ends the chain.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/phase5"))
sys.path.insert(0, str(ROOT / "scripts/phase6"))

RIGHT = {"p6u2": 733, "p6u3": 722, "p6u3v": 730, "p6u4": 740, "p6u4c": 739, "p6u5": 740,
         "p6u6": 747, "p6u7": 739, "p6u8": 754, "p6u9": 750}  # fmt: skip
PARENTS = {"u3": "u2", "u3v": "u2", "u4": "u3v", "u4c": "u3v", "u5": "u4", "u6": "u5",
           "u7": "u6", "u8": "u7", "u9": "u8", "u10": "u9"}  # fmt: skip


@pytest.fixture
def chain(monkeypatch):
    import decide6 as D

    right, log = dict(RIGHT), []

    def records(split, lab, level="L0"):
        if lab not in right:
            return None
        return [{"score": {"correct": i < right[lab]}} for i in range(822)]

    monkeypatch.setattr(D.M, "rule_records", records)
    monkeypatch.setattr(D, "parameters", lambda lab: 14_680_064)
    monkeypatch.setattr(D, "record", log.append)
    monkeypatch.setattr(D, "recorded", lambda: log)
    monkeypatch.setattr(D.P6, "manifest", lambda recipe: {"parent": PARENTS.get(recipe)})
    return D, right, log


def choice(level, what):
    return {"decision": "user", "level": level, "choice": what, "said": "words",
            "time": "2026-11-18 21:40"}  # fmt: skip


def test_level_9_and_earlier_levels_do_not_wait(chain, monkeypatch, capsys):
    D, right, log = chain
    del right["p6u9"]

    def sleep(seconds):
        raise AssertionError("waited before level 9")

    monkeypatch.setattr(D.time, "sleep", sleep)
    D.nxt()
    out = capsys.readouterr()
    assert out.out.strip() == "u9 u8" and "HOLD" not in out.err
    assert log[-1]["level"] == 9 and log[-1]["runs"] == ["u9"]


def test_level_10_waits_then_starts_when_the_user_says_start(chain, monkeypatch, capsys):
    D, right, log = chain
    waits = []

    def sleep(seconds):  # the user answers while "next" waits
        waits.append(seconds)
        if len(waits) == 2:
            log.append(choice(10, "start"))

    monkeypatch.setattr(D.time, "sleep", sleep)
    D.nxt()
    out = capsys.readouterr()
    assert waits == [60, 60]
    assert out.err.count("HOLD level 10") == 1  # said once, not every minute
    assert out.out.strip() == "u10 u9"
    assert log[-1]["decision"] == "next" and log[-1]["level"] == 10
    assert log[-1]["runs"] == ["u10"] and log[-1]["parent"] == "u9"
    assert "ended_by" not in log[-1]


def test_stop_ends_the_chain_after_level_9_and_ud_follows(chain, monkeypatch, capsys):
    D, right, log = chain
    log.append(choice(10, "stop"))
    monkeypatch.setattr(D.time, "sleep", lambda s: pytest.fail("waited with a choice recorded"))
    D.nxt()
    assert capsys.readouterr().out.strip() == "stop"
    assert log[-1]["decision"] == "next" and log[-1]["runs"] == [] and log[-1]["parent"] is None
    assert log[-1]["ended_by"] == "user" and log[-1]["levels_without_a_new_best"] == 1
    D.ud()  # the chain has ended: UD from its best run (U8), with U9 as the twin
    assert capsys.readouterr().out.strip() == "ud u8 --steps 3 --seed-from u9"
    log[:] = [choice(10, "stop")]
    right["p6u9"] = 760  # U9 the best: no level was built from it, so UD has no twin
    D.nxt()
    D.ud()
    assert capsys.readouterr().out.strip().splitlines()[-1] == "ud u9 --steps 3"


def test_a_choice_after_level_10_changes_nothing(chain, monkeypatch, capsys):
    D, right, log = chain
    right["p6u10"] = 751
    monkeypatch.setattr(D.time, "sleep", lambda s: pytest.fail("waited after level 10"))
    D.nxt()
    assert capsys.readouterr().out.strip() == "stop"  # the last level: the chain ends
    assert "ended_by" not in log[-1]


def test_the_choice_is_recorded_once_and_only_for_a_held_level(chain, monkeypatch, capsys):
    D, right, log = chain
    for argv in (["user", "9", "start", "words"], ["user", "10", "go", "words"],
                 ["user", "10", "start"], ["user", "x", "start", "words"]):  # fmt: skip
        monkeypatch.setattr(sys, "argv", ["decide6.py", *argv])
        with pytest.raises(SystemExit, match="usage"):
            D.user()
    assert log == []
    monkeypatch.setattr(sys, "argv", ["decide6.py", "user", "10", "stop", "do", "not", "start"])
    D.user()
    assert capsys.readouterr().out.strip() == "level 10: stop"
    assert log[-1]["choice"] == "stop" and log[-1]["said"] == "do not start"
    monkeypatch.setattr(sys, "argv", ["decide6.py", "user", "10", "start", "start it"])
    with pytest.raises(SystemExit, match="already recorded"):
        D.user()
    assert len(log) == 1


def test_a_half_written_decision_file_is_read_again(chain, monkeypatch):
    D, right, log = chain
    reads = iter([json.JSONDecodeError("half written", "[", 1), [choice(10, "start")]])

    def recorded():
        r = next(reads)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(D, "recorded", recorded)
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    assert D.users_choice(10) == "start"


def test_the_report_lists_the_choice_the_end_and_ud(tmp_path, monkeypatch):
    import make_report6 as R

    entries = [
        {"decision": "next", "level": 9, "runs": ["u9"], "parent": "u8",
         "dev6_correct": {"p6u2": 733, "p6u8": 754}},
        choice(10, "stop"),
        {"decision": "next", "level": 10, "runs": [], "parent": None, "ended_by": "user",
         "dev6_correct": {"p6u2": 733, "p6u8": 754, "p6u9": 750}},
        {"decision": "ud", "parent": "u8", "twin": "u9", "dev6_correct": {"p6u8": 754}},
    ]  # fmt: skip
    (tmp_path / "round2.json").write_text(json.dumps(entries, ensure_ascii=False))
    monkeypatch.setattr(R, "ROOT", tmp_path)
    lines, summary = [], {}
    R.round2_section(lines, summary)
    text = "\n".join(lines)
    assert "Level 10, the user's choice (preregistration 12.15, 2026-11-18 21:40):" in text
    assert "end the chain after the level before it (\"words\")" in text
    assert "the loop ends by the user's choice (preregistration 12.15)" in text
    assert "UD (preregistration 12.14):** from the chain's best run, U8" in text
    assert "its twin U9" in text
