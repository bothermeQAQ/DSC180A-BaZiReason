"""Phase 6, preregistration 12.14: derivation version 3 (each decision after its check) and UD.

No model is run. Version 3 changes training answers only: every final JSON keeps the rule code's
answer, the calendar text decides exactly as ``worked.no_tool_expected``, and versions 1 and 2 are
unchanged (the rebuild checks of every existing set cover the bytes).
"""

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/phase5"))
sys.path.insert(0, str(ROOT / "scripts/phase6"))

ZONES = (
    "Asia/Shanghai",
    "Asia/Shanghai",
    "Asia/Shanghai",
    "UTC",
    "Asia/Tokyo",
    "Asia/Kolkata",
    "Asia/Hong_Kong",
    "Asia/Singapore",
    "America/New_York",
    "America/Los_Angeles",
    "Europe/London",
    "Australia/Sydney",
    "Europe/Paris",
    "America/Chicago",
)


def test_set_steps_3_turns_the_calendar_checks_on_and_back_off():
    from bazireason import steps6 as S6
    from bazireason import worked as W

    assert not W.check_first()
    try:
        S6.set_steps(3)
        assert W.check_first() and S6.steps_version() == 3
        S6.set_steps(2)
        assert not W.check_first()
    finally:
        S6.set_steps(1)
    assert not W.check_first() and S6.steps_version() == 1


def test_derivation_version_3_keeps_every_answer_and_passes_the_claim_checker():
    from bazireason import splits6 as S
    from bazireason import steps6 as S6
    from bazireason.grounded import claim_failures
    from bazireason.p2_eval import _same
    from bazireason.p2_prompts import extract_final_json

    rng = random.Random(20261118)
    intents = ("strength", "flow:2037", "overview", "yongshen:扶抑", "interact", "pattern", "gods",
               "fact:xun_kong", "fact:life_stage:month")  # fmt: skip
    changed = set()
    for _ in range(60):
        pillars, gender = S.random_pillars(rng), rng.choice(("male", "female"))
        domain = rng.choice(("marriage", "career", "wealth", "study", "children"))
        year = rng.randrange(1915, 2150)
        for intent in (*intents, f"domain:{domain}:{year}"):
            S6.set_steps(2)
            try:
                second, reference = S6.derive(intent, pillars, gender)
                S6.set_steps(3)
                third, same = S6.derive(intent, pillars, gender)
            finally:
                S6.set_steps(1)
            assert same == reference, (intent, pillars)
            shown = extract_final_json(third)
            assert set(shown) == set(reference), (intent, pillars)
            for key in reference:  # relation lists in the order they were found; scored as sets
                if intent == "interact":
                    assert sorted(shown[key]) == sorted(reference[key]), (intent, key, pillars)
                else:
                    assert _same(shown[key], reference[key]), (intent, key, pillars)
            assert claim_failures(third, pillars, gender) == [], (intent, pillars)
            if third != second:
                changed.add(intent.split(":")[0])
    assert changed == {"interact", "domain"}


def test_version_3_relations_copy_the_keys_in_the_order_they_were_found():
    from bazireason import steps6 as S6

    pillars = {"year": "庚午", "month": "辛巳", "day": "甲子", "hour": "丙寅"}
    S6.set_steps(3)
    try:
        text, _ = S6.derive("interact", pillars)
    finally:
        S6.set_steps(1)
    assert "甲庚在相冲表 → 庚甲相冲 → 记作 干冲:甲庚" in text
    assert "寅在巳卡的害、刑 → 寅巳相刑、寅巳相害 → 记作 刑:寅巳、害:寅巳" in text
    assert "全部关系（上面记作的，按出现顺序，重复的只记一次）：干冲:甲庚、干合:丙辛、冲:子午、半合:寅午、刑:寅巳、害:寅巳。" in text  # fmt: skip
    assert text.endswith('{"answer":["干冲:甲庚","干合:丙辛","冲:子午","半合:寅午","刑:寅巳","害:寅巳"],"adjacent":[]}')  # fmt: skip


def test_version_3_area_of_life_writes_both_sexes_and_a_verdict_per_year():
    from bazireason import steps6 as S6

    pillars = {"year": "癸酉", "month": "甲子", "day": "丙午", "hour": "戊戌"}
    S6.set_steps(3)
    try:
        text, reference = S6.derive("domain:children:2031", pillars, "female")
    finally:
        S6.set_steps(1)
    assert "男命看正官、七杀，女命看食神、伤官；本命为女命 → 食神、伤官" in text
    assert "午在戌卡的半合 → 午戌半合火局，半合不算；计入" in text  # 星现 that year, not 宫动
    assert "丑在戌卡的刑 → 丑戌相刑，刑不算；不计" in text
    assert "卯在戌卡的六合 → 卯戌六合（合化火），六合 → 宫动；计入" in text
    assert reference["trigger_years"] == [2035, 2036, 2038]


def _random_request(rng):
    zone = rng.choice(ZONES)
    year = rng.randrange(1901, 2100)
    wall = datetime(year, 1, 1) + timedelta(minutes=rng.randrange(366 * 24 * 60))
    if rng.random() < 0.15:
        wall = wall.replace(hour=23)
    text = wall.strftime("%Y-%m-%dT%H:%M:%S" if rng.random() < 0.7 else "%Y-%m-%dT%H:%M")
    convention = {"sect": rng.choice((1, 2))} if rng.random() < 0.8 else {}
    return {"datetime": text, "timezone": zone}, convention


def test_version_3_calendar_answers_decide_as_no_tool_expected():
    from bazireason import worked as W
    from bazireason.calendar_engine import parse_wall_time

    rng = random.Random(2026111814)
    seen = set()
    for _ in range(4000):
        inputs, convention = _random_request(rng)
        before = W.explain_request(inputs, convention, "none")
        W.set_check_first(True)
        try:
            after = W.explain_request(inputs, convention, "none")
        finally:
            W.set_check_first(False)
        expected = W.no_tool_expected(inputs, convention)
        assert after["response"] == before["response"] == expected, inputs
        steps = after["steps"]
        if not steps[0].startswith("时区："):
            assert steps == before["steps"], inputs  # decided before the offset: unchanged
            seen.add("unchanged")
            continue
        wall = parse_wall_time(inputs["datetime"])[1]
        offset = W.known_offset(inputs["timezone"], wall)
        assert ("→ 偏移需工具。" in steps[0]) == (offset is None), inputs
        if offset is None:
            assert (
                expected["reason_codes"] == ["utc_offset_required"]
                and steps[-1] == W.OFFSET_REFUSAL
            )
            seen.add("offset")
            continue
        reference = wall + timedelta(minutes=480 - offset)
        windows = [s for s in steps if s.startswith("交节：")]
        if windows:
            inside = W.jie_window_of(reference.date())
            assert ("窗口内，需工具" in windows[0]) == (inside is not None), inputs
            assert (expected["reason_codes"] == ["jie_instant_required"]) == (inside is not None)
            seen.add("window" if inside else "outside")
        if expected["status"] == "ok":  # year, day and hour as before; the month step points
            year, month, day, hour = steps[-4:]  # to the 交节 step instead of restating windows
            old_year, old_month, old_day, old_hour = before["steps"][-4:]
            assert [year, day, hour] == [old_year, old_day, old_hour], inputs
            term = W.previous_jie_by_date(reference.date())
            assert month.startswith(f"月柱：上一个节是{term}（见交节），"), inputs
            assert month.split("（见交节），")[1] == old_month.split("之前，")[1], inputs
            seen.add("ok")
        elif expected["status"] == W.NEEDS_TOOL:
            assert steps[-1].startswith("没有该年"), inputs
        else:
            assert steps[-1] == before["steps"][-1], inputs  # the non-answer text as before
            seen.add(expected["status"])
    assert {"unchanged", "offset", "window", "outside", "ok", "ambiguous"} <= seen


def test_version_3_zone_sentences_agree_with_known_offset_month_by_month():
    from bazireason import worked as W

    for zone in ZONES:
        for year in (1901, 1919, 1945, 1951, 1952, 1960, 1975, 1976, 1980, 1985, 1986, 1990, 1991, 1992, 2024):  # fmt: skip
            for month in range(1, 13):
                wall = datetime(year, month, 15, 12)
                text, offset = W.zone_step(zone, wall)
                assert offset == W.known_offset(zone, wall), (zone, wall)
                assert text.startswith("时区：") and text.endswith("。")


def test_ud_continues_the_chains_best_run_with_its_twins_seed(monkeypatch, capsys):
    import decide6 as D

    right = {"p6u2": 733, "p6u3": 722, "p6u3v": 730, "p6u4": 740, "p6u4c": 739, "p6u5": 740,
             "p6u6": 747, "p6u7": 745, "p6u8": 746}  # fmt: skip
    parents = {"u3": "u2", "u3v": "u2", "u4": "u3v", "u4c": "u3v", "u5": "u4", "u6": "u5",
               "u7": "u6", "u8": "u7"}  # fmt: skip
    log = [{"decision": "next", "level": 10, "runs": [], "parent": None}]
    monkeypatch.setattr(D.M, "rule_records", lambda split, lab, level="L0": [
        {"score": {"correct": i < right[lab]}} for i in range(822)] if lab in right else None)  # fmt: skip
    monkeypatch.setattr(D, "parameters", lambda lab: 14_680_064)
    monkeypatch.setattr(D, "record", log.append)
    monkeypatch.setattr(D, "recorded", lambda: log)
    monkeypatch.setattr(D.P6, "manifest", lambda recipe: {"parent": parents.get(recipe)})
    D.ud()
    assert capsys.readouterr().out.strip() == "ud u6 --steps 3 --seed-from u7"
    assert log[-1]["decision"] == "ud" and log[-1]["parent"] == "u6" and log[-1]["twin"] == "u7"
    right["p6u8"] = 760  # a later result changes nothing: the decision is recorded once
    D.ud()
    assert capsys.readouterr().out.strip() == "ud u6 --steps 3 --seed-from u7"
    assert "p6ud" in D.ORDER and "p6ud" not in D.CONTINUATIONS
    log.clear()  # the chain has not ended: no decision
    log.append({"decision": "next", "level": 9, "runs": ["u9"], "parent": "u8"})
    try:
        D.ud()
    except SystemExit as stop:
        assert "not ended" in str(stop)
    else:
        raise AssertionError("ud() ran before the chain ended")
