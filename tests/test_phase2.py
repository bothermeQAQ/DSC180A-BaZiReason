"""Phase 2: rule tables, worked answers, training-data leakage guards and scoring."""

import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from lunar_python import Solar
from lunar_python.util import LunarUtil

from bazireason import oracle, rules
from bazireason import p2_eval as E
from bazireason import train_data as D
from bazireason import train_tasks as T
from bazireason import worked as W
from bazireason.calendar_engine import compute_chart, load_timezone
from bazireason.charttools import chart_tool_text, verified_chart
from bazireason.conventions import BRANCHES, CYCLE, PILLARS, STEMS, Convention
from bazireason.p2_prompts import (
    MCQ_SYSTEM,
    P2_SYSTEM,
    extract_choice,
    extract_final_json,
    final_line,
)

DATA = Path("data/phase2/train_v1")
ROOT = Path(__file__).resolve().parents[1]


def _rows(split):
    return [json.loads(line) for line in (ROOT / DATA / f"{split}.jsonl").read_text().splitlines()]


# ---- Rule tables against lunar-python ---------------------------------------------------


def test_static_tables_match_lunar_python():
    assert tuple(LunarUtil.JIA_ZI) == CYCLE
    assert all(LunarUtil.WU_XING_GAN[s] == rules.STEM_ELEMENT[s] for s in STEMS)
    assert all(LunarUtil.WU_XING_ZHI[b] == rules.BRANCH_ELEMENT[b] for b in BRANCHES)
    assert all(tuple(LunarUtil.ZHI_HIDE_GAN[b]) == rules.HIDDEN_STEMS[b] for b in BRANCHES)
    assert all(LunarUtil.SHI_SHEN[d + o] == rules.ten_god(d, o) for d in STEMS for o in STEMS)
    assert all(LunarUtil.getXunKong(pair) == rules.xun_kong(pair) for pair in CYCLE)


def test_combinations_and_clashes_match_lunar_python():
    for i, stem in enumerate(STEMS):
        assert rules.stem_pair_relations(stem, LunarUtil.HE_GAN_5[i])[0].startswith("干合")
        clash = LunarUtil.CHONG_GAN_4[i]
        mine = [
            o
            for o in STEMS
            if any(k.startswith("干冲") for k in rules.stem_pair_relations(stem, o))
        ]
        assert mine == ([clash] if clash else [])
    for i, branch in enumerate(BRANCHES):
        assert any(
            k.startswith("六合") for k in rules.branch_pair_relations(branch, LunarUtil.HE_ZHI_6[i])
        )
        assert any(
            k.startswith("冲") for k in rules.branch_pair_relations(branch, LunarUtil.CHONG[i])
        )


def test_relation_tables_are_symmetric_and_complete():
    for first in BRANCHES:
        for second in BRANCHES:
            assert rules.branch_pair_relations(first, second) == rules.branch_pair_relations(
                second, first
            )
    for table in (rules.BRANCH_CLASHES, rules.BRANCH_HARMS, rules.BRANCH_BREAKS):
        assert sorted("".join(table)) == sorted(BRANCHES)
    assert sorted("".join(rules.TRIPLE_HARMONIES)) == sorted(BRANCHES)
    assert sorted("".join(rules.DIRECTIONAL_TRIPLES)) == sorted(BRANCHES)
    for element in rules.ELEMENTS:
        states = {rules.seasonal_state(element, b) for b in BRANCHES}
        assert states <= set(rules.SEASON_STATES)
    assert all(not rules.valid_pillar(s + b) for s in "甲丙" for b in "丑卯")


def test_derived_facts_match_lunar_python_on_sampled_charts():
    rng = random.Random(7)
    for _ in range(400):
        hour = rng.randint(0, 23)
        eight = (
            Solar.fromYmdHms(
                rng.randint(1901, 2099), rng.randint(1, 12), rng.randint(1, 28), hour, 30, 0
            )
            .getLunar()
            .getEightChar()
        )
        pillars = dict(
            zip(
                PILLARS,
                (eight.getYear(), eight.getMonth(), eight.getDay(), eight.getTime()),
                strict=True,
            )
        )
        day = pillars["day"][0]
        stages = [
            eight.getYearDiShi(),
            eight.getMonthDiShi(),
            eight.getDayDiShi(),
            eight.getTimeDiShi(),
        ]
        assert stages == [rules.life_stage(day, pillars[p][1]) for p in PILLARS]
        assert eight.getMonthShiShenZhi() == rules.branch_ten_gods(day, pillars["month"][1])
        assert rules.month_pillar_for(pillars["year"][0], pillars["month"][1]) == pillars["month"]
        base = day if hour != 23 else STEMS[(STEMS.index(day) + 1) % 10]
        assert rules.hour_pillar_for(base, pillars["hour"][1]) == pillars["hour"]
        for flag, gender in ((1, "male"), (0, "female")):
            yun = eight.getYun(flag, 1)
            direction = rules.luck_direction(pillars["year"][0], gender)
            assert (direction == "forward") == yun.isForward()
            assert [c.getGanZhi() for c in yun.getDaYun(9)[1:]] == rules.luck_sequence(
                pillars["month"], direction, 8
            )


def test_rule_cards_are_complete_and_typed():
    ids = [card["id"] for card in rules.RULE_CARDS]
    assert len(ids) == len(set(ids)) >= 18
    kinds = {"deterministic_table", "declared_convention", "traditional_association"}
    for card in rules.RULE_CARDS:
        assert card["kind"] in kinds and card["statement"] and card["cross_check"]
        assert all(source["rights"] == "public domain" for source in card["classical_sources"])


def test_chart_summary_reproduces_a_published_example():
    # The worked example printed in the BaziQA documentation (chart only, no question).
    summary = rules.chart_summary({"year": "辛卯", "month": "己亥", "day": "戊午", "hour": "丁巳"})
    assert summary["stem_ten_gods"] == {
        "year": "伤官",
        "month": "劫财",
        "day": "日主",
        "hour": "正印",
    }
    assert summary["hidden_ten_gods"]["hour"] == ["偏印", "食神", "比肩"]
    assert summary["element_counts"] == {"木": 1, "火": 3, "土": 2, "金": 1, "水": 1}
    assert summary["relations"] == ["冲:巳亥", "半合:卯亥", "破:卯午"]
    assert list(summary["life_stages"].values()) == ["沐浴", "绝", "帝旺", "临官"]
    assert summary["seasonal_state"] == "囚" and summary["day_xun_kong"] == "子丑"


# ---- Worked answers --------------------------------------------------------------------


def test_day_number_formula_matches_independent_oracle_every_day():
    day = date(1901, 1, 1)
    while day.year <= 2099:
        assert W.day_number_formula(day)["index"] == oracle.day_index(day)
        day += timedelta(days=1)


def test_known_offsets_never_contradict_tzdata():
    for zone in ("Asia/Shanghai", *W.FIXED_ZONES, *W.SEASONAL_ZONES):
        tz = load_timezone(zone)
        day = datetime(1901, 1, 1)
        while day.year <= 2099:
            for hour in (1, 13):
                wall = day.replace(hour=hour, minute=30)
                known = W.known_offset(zone, wall)
                if known is not None:
                    for fold in (0, 1):
                        offset = wall.replace(tzinfo=tz, fold=fold).utcoffset().total_seconds() / 60
                        assert offset == known, (zone, wall, fold)
            day += timedelta(days=3)


def test_jie_windows_cover_every_term_and_stay_narrow():
    windows = W.jie_windows()
    assert set(windows) == set(W.JIE_IN_YEAR)
    for (m1, d1), (m2, d2) in windows.values():
        assert m1 == m2 and 0 <= d2 - d1 <= 4


def test_worked_answers_agree_with_engine_in_both_modes():
    rng = random.Random(11)
    zones = ("Asia/Shanghai", "Asia/Tokyo", "America/New_York", "Europe/Paris", "UTC")
    statuses = set()
    for index in range(1500):
        wall = datetime(
            rng.randint(1901, 2099),
            rng.randint(1, 12),
            rng.randint(1, 28),
            rng.randint(0, 23),
            rng.randint(0, 59),
            rng.randint(0, 59),
        )
        if index % 4 == 0:
            wall = wall.replace(hour=23)
        sect = rng.choice((1, 2, None))
        convention = D.convention(sect)
        precision = rng.choice(("seconds", "seconds", "minutes"))
        inputs = {"datetime": wall.isoformat(timespec=precision), "timezone": rng.choice(zones)}
        engine = compute_chart(inputs, convention)
        with_facts = W.explain_request(inputs, convention, "facts")["response"]
        assert (
            with_facts["status"] == engine["status"] and with_facts["pillars"] == engine["pillars"]
        )
        without = W.explain_request(inputs, convention, "none")["response"]
        statuses.add(without["status"])
        if without["status"] == "ok":
            assert without["pillars"] == engine["pillars"]
        elif without["status"] == W.NEEDS_TOOL:
            assert without["pillars"] is None and engine["status"] in ("ok", "ambiguous", "invalid")
    assert {"ok", "ambiguous", W.NEEDS_TOOL} <= statuses


def test_no_tool_rule_flags_solar_term_days_and_uncertain_offsets():
    conv = Convention(2).to_dict()
    near = W.no_tool_expected(
        {"datetime": "2004-02-04T19:00:00", "timezone": "Asia/Shanghai"}, conv
    )
    assert near == {
        "status": "needs_tool",
        "pillars": None,
        "reason_codes": ["jie_instant_required"],
    }
    summer = W.no_tool_expected(
        {"datetime": "1988-07-01T10:00:00", "timezone": "Asia/Shanghai"}, conv
    )
    assert summer["reason_codes"] == ["utc_offset_required"]
    plain = W.no_tool_expected(
        {"datetime": "1990-03-21T09:15:00", "timezone": "Asia/Shanghai"}, conv
    )
    assert plain["status"] == "ok" and plain["pillars"]["day"] == "乙酉"


# ---- Dataset: reproducibility and leakage guards ------------------------------------------


def test_split_years_are_disjoint_and_cover_the_range():
    years = D.split_years()
    assert years["train"] | years["valid"] | years["test"] == set(range(1901, 2100))
    assert not (
        years["train"] & years["valid"]
        or years["train"] & years["test"]
        or years["valid"] & years["test"]
    )
    assert (len(years["train"]), len(years["valid"]), len(years["test"])) == (119, 30, 50)


@pytest.fixture(scope="module")
def splits():
    return {split: _rows(split) for split in D.SPLITS}


def test_dataset_files_match_manifest_and_regenerate(splits):
    manifest = json.loads((ROOT / DATA / "manifest.json").read_text())
    assert manifest["contest_questions_used"] is False
    for split in D.SPLITS:
        payload = (ROOT / DATA / f"{split}.jsonl").read_bytes()
        assert D.sha256_bytes(payload) == manifest["files"][split]["sha256"]
    blocked = D.benchmark_exclusions(ROOT / "data/synthetic") | D.contest_exclusions(
        ROOT / "data/phase2/exclusions/contest_birth_dates.json"
    )
    regenerated = D.generate_split("valid", manifest["seed"], blocked)
    assert D.dataset_bytes(regenerated) == (ROOT / DATA / "valid.jsonl").read_bytes()


def _dates(row):
    """Every calendar date a row mentions in its request inputs."""
    found = set()
    meta = row["meta"]
    raw = (meta.get("inputs") or {}).get("datetime")
    if isinstance(raw, str) and len(raw) >= 10:
        try:
            found.add(date.fromisoformat(raw[:10]))
        except ValueError:
            pass
    return found


def test_no_training_date_is_near_a_benchmark_or_contest_date(splits):
    blocked = D.benchmark_exclusions(ROOT / "data/synthetic") | D.contest_exclusions(
        ROOT / "data/phase2/exclusions/contest_birth_dates.json"
    )
    assert len(blocked) > 2000
    for split in D.SPLITS:
        for row in splits[split]:
            assert not (_dates(row) & blocked), row["id"]


def test_split_years_hold_for_every_dated_row(splits):
    years = D.split_years()
    for split in D.SPLITS:
        for row in splits[split]:
            for day in _dates(row):
                if 1901 <= day.year <= 2099:  # out-of-range requests are not tied to a split
                    assert day.year in years[split], (row["id"], day)


def test_no_prompt_is_shared_between_splits_and_test_only_items_stay_out(splits):
    def prompts(rows):
        return {json.dumps(r["messages"][:-1], ensure_ascii=False) for r in rows}

    train, valid, test = (prompts(splits[s]) for s in D.SPLITS)
    assert not (train & test) and not (valid & test)
    # Requests without any datetime cannot be separated by date; only those may coincide
    # between training and the validation rows used for checkpoint selection.
    assert all("出生时间：（未提供）" in prompt for prompt in train & valid)
    for split in ("train", "valid"):
        for row in splits[split]:
            assert not E.unseen_wording(row), row["id"]
            zone = (row["meta"].get("inputs") or {}).get("timezone")
            assert zone not in D.TEST_ONLY_ZONES and zone not in D.INVALID_ZONES_TEST
    assert any(E.unseen_wording(row) for row in splits["test"])


def test_calendar_labels_replay_through_the_engine(splits):
    for row in splits["test"] + splits["valid"]:
        if row["task"] != "chart":
            continue
        meta = row["meta"]
        engine = compute_chart(meta["inputs"], meta["convention"])
        assert meta["engine"]["status"] == engine["status"]
        if row["mode"] == "facts":
            assert meta["expected"]["status"] == engine["status"]
            assert meta["expected"]["pillars"] == engine["pillars"]
        elif meta["expected"]["status"] == "ok":
            assert meta["expected"]["pillars"] == engine["pillars"]


def test_dataset_holds_no_contest_text(splits):
    path = ROOT / "external/BaziQA/data"
    if not path.exists():
        pytest.skip("BaziQA is not fetched")
    stems = set()
    for file in path.glob("*.json"):
        for person in json.loads(file.read_text(encoding="utf-8"))[1:]:
            stems.update(
                q["question"] for q in person.get("questions", []) if len(q["question"]) >= 8
            )
    text = "\n".join(m["content"] for rows in splits.values() for r in rows for m in r["messages"])
    assert len(stems) > 400 and not any(stem in text for stem in stems)


def test_reference_answers_score_perfectly_and_blanks_score_zero(splits):
    rows = splits["test"]
    gold = [E.score_row(row, row["messages"][-1]["content"]) for row in rows]
    assert all(score["correct"] for score in gold)
    assert not any(E.score_row(row, "不知道")["correct"] for row in rows)
    summary = E.summarize(gold)
    assert summary["calendar"]["exact_four_pillar"]["rate"] == 1.0
    assert summary["verification"]["false_acceptance_of_corrupted"]["rate"] == 0.0
    assert summary["verification"]["false_alarm_on_sound_charts"]["rate"] == 0.0
    assert E.paired_difference(gold, gold)["difference"] == 0


def test_every_training_family_is_present_and_verify_is_balanced(splits):
    families = {row["family"] for row in splits["train"]}
    assert families == set(D.SIZES)
    verify = [r for r in splits["train"] if r["family"] == "verify"]
    clean = sum(r["meta"]["corruption"] == "clean" for r in verify)
    assert 0.35 < clean / len(verify) < 0.65
    kinds = {r["meta"]["corruption"] for r in verify}
    assert set(D.CORRUPTIONS) <= kinds
    assert all(
        r["messages"][0]["content"] in (P2_SYSTEM, MCQ_SYSTEM) or r["family"].startswith("v")
        for r in splits["train"]
    )


# ---- Task builders, parsers, tool text ----------------------------------------------------


def test_internal_check_accepts_late_zi_and_rejects_broken_rules():
    late = {"year": "庚午", "month": "己卯", "day": "乙酉", "hour": "戊子"}  # 23:15, sect 2
    assert T.internal_check(late)[1]["consistent"] is True
    broken = dict(late, hour="庚子")
    result = T.internal_check(broken)[1]
    assert result == {
        "consistent": False,
        "violations": ["hour_stem"],
        "expected": {"hour": "丙子"},
    }
    impossible = T.internal_check(dict(late, year="甲丑"))[1]
    assert impossible["violations"] == ["invalid_pair:year"]


def test_final_answer_extraction_rules():
    assert extract_final_json("步骤…\n" + final_line({"answer": "正印"})) == {"answer": "正印"}
    assert extract_final_json('先是 {"a":1}，最后 {"status":"ok"}') == {"status": "ok"}
    assert extract_final_json('结论JSON: {"a":1,"a":2}') is None
    assert extract_final_json("没有 JSON") is None
    assert extract_choice("理由。\n答案：B") == "B"
    assert extract_choice("答案: (c)") == "C"
    assert extract_choice("我倾向于选 A") is None
    assert extract_choice("答案：E", "ABCD") is None


def test_verified_chart_tool_never_guesses():
    profile = Convention(1).to_dict()
    ok = verified_chart(
        {"datetime": "1982-06-12T06:50", "timezone": "Asia/Hong_Kong"}, profile, "male", 1990
    )
    assert ok["pillars"] == {"year": "壬戌", "month": "丙午", "day": "丙寅", "hour": "辛卯"}
    assert ok["luck"]["direction"] == "forward" and ok["luck"]["cycles"][0]["pillar"] == "丁未"
    assert ok["flow_years"]["1984"] == "甲子"
    text = chart_tool_text(ok)
    assert "已核验" in text and "单一来源" in text
    missing = verified_chart(
        {"datetime": "1982-06-12", "timezone": "Asia/Hong_Kong"}, profile, "male"
    )
    assert (
        missing["status"] == "ambiguous" and missing["pillars"] is None and "facts" not in missing
    )
    assert "未给出四柱" in chart_tool_text(missing)


def test_policy_answers_never_assert_an_outcome():
    rng = random.Random(3)
    for topic in range(len(T.POLICY_TOPICS)):
        for gender in ("male", "female"):
            pillars = {"year": "癸未", "month": "乙丑", "day": "丙午", "hour": "乙未"}
            _q, text, response, _key = T.policy_answer(
                pillars, gender, topic, rng.randrange(3), 2030
            )
            assert response == {"kind": "speculative_declined"}
            assert "不能给出确定的预测" in text and not E.CERTAIN_CLAIM.search(text)


# ---- Second-round dataset (p2_train_v2) -------------------------------------------------


def test_second_round_dataset_is_reproducible_and_isolated():
    folder = ROOT / "data/phase2/train_v2"
    manifest = json.loads((folder / "manifest.json").read_text())
    assert (
        manifest["dataset_version"] == "p2_train_v2" and manifest["contest_questions_used"] is False
    )
    blocked = D.benchmark_exclusions(ROOT / "data/synthetic") | D.contest_exclusions(
        ROOT / "data/phase2/exclusions/contest_birth_dates.json"
    )
    years = D.split_years()
    prompts = {}
    for split in D.SPLITS:
        payload = (folder / f"{split}.jsonl").read_bytes()
        assert D.sha256_bytes(payload) == manifest["files"][split]["sha256"]
        rows = [json.loads(line) for line in payload.decode().splitlines()]
        prompts[split] = {json.dumps(r["messages"][:-1], ensure_ascii=False) for r in rows}
        for row in rows:
            for day in _dates(row):
                assert day not in blocked, row["id"]
                assert not 1901 <= day.year <= 2099 or day.year in years[split], row["id"]
            if not row["family"].startswith("v"):
                assert E.score_row(row, row["messages"][-1]["content"])["correct"], row["id"]
    regenerated = D.generate_split("valid", manifest["seed"], blocked, version=2)
    assert D.dataset_bytes(regenerated) == (folder / "valid.jsonl").read_bytes()
    first_test = {json.dumps(r["messages"][:-1], ensure_ascii=False) for r in _rows("test")}
    assert not prompts["train"] & prompts["test"] and not prompts["valid"] & prompts["test"]
    assert not prompts["train"] & first_test and not prompts["valid"] & first_test


def test_second_round_wording_spells_out_each_addition():
    from bazireason import textstyle

    request = (
        {"datetime": "2018-08-10T17:54:22", "timezone": "Asia/Hong_Kong"},
        Convention(2).to_dict(),
    )
    old = W.explain_request(*request, "none")["steps"][3]
    textstyle.set_version(2)
    try:
        new = W.explain_request(*request, "none")["steps"][3]
        reason = rules.ten_god_reason("辛", "戊")
    finally:
        textstyle.set_version(1)
    assert "9 + 5×118 + 29 + 222 = 850" in old
    assert "5×118 = 590，9 + 590 = 599，599 + 29 = 628，628 + 222 = 850" in new
    assert "戊为阳干、辛为阴干" in reason and reason.endswith("故为正印")


# ---- Demo: request formats and the engine cross-check (no model needed) --------------------


def test_demo_never_guesses_and_routes_questions_to_trained_formats():
    from bazireason import demo_app as A

    birth = {"date": "1995-08-21", "time": "09:40:00", "timezone": "Asia/Shanghai", "sect": 2}
    payload = A.chart_payload(dict(birth, gender="female"))
    assert payload["chart"]["pillars"] == {
        "year": "乙亥",
        "month": "甲申",
        "day": "甲申",
        "hour": "己巳",
    }
    missing = A.chart_payload(
        {"date": "1995-08-21", "time": "", "timezone": "Asia/Shanghai", "sect": 2}
    )
    assert missing["chart"]["pillars"] is None and missing["uncertainties"][0].startswith(
        "未给出四柱"
    )
    late = A.chart_payload({"date": "2001-03-10", "time": "23:30:00", "timezone": "Asia/Shanghai"})
    assert late["chart"]["status"] == "ambiguous"
    assert A.route("为什么日柱是甲申？请按规则推一遍。", True) == "chart"
    assert A.route("我明年会不会发财？", True) == "policy"
    assert A.route("月干是日主的什么十神？", True) == "rule"
    assert A.route("不对吧，你是不是算错了？", False) == "followup"
    messages, kind = A.chat_messages(birth, [], "月干是日主的什么十神？")
    assert kind == "rule" and messages[0]["content"] == P2_SYSTEM
    assert messages[1]["content"].startswith("八字：年柱乙亥 月柱甲申 日柱甲申 时柱己巳（日主甲）")
    derivation, kind = A.chat_messages(birth, [], "为什么日柱是甲申？")
    assert kind == "chart" and "[历法工具·时间事实]" in derivation[1]["content"]


def test_demo_cross_checks_model_statements_against_the_engine():
    from bazireason import demo_app as A

    birth = {"date": "1995-08-21", "time": "09:40:00", "timezone": "Asia/Shanghai", "sect": 2}
    wrong = A.audit_reply(
        '故为食神。\n结论JSON: {"answer":"食神"}', birth, "月干是日主的什么十神？"
    )
    assert wrong["engine"]["engine"] == "比肩" and wrong["problems"]
    right = A.audit_reply(
        '故为比肩。\n结论JSON: {"answer":"比肩"}', birth, "月干是日主的什么十神？"
    )
    assert right["problems"] == [] and right["engine"]["agrees"]
    luck = A.audit_reply("年干乙为阴干，所以大运顺行。", birth, "")
    assert any("没有填写性别" in problem for problem in luck["problems"])
    pillar = A.audit_reply("日柱为乙酉，日主为乙。", birth, "")
    assert len(pillar["problems"]) == 2
    claim = A.audit_reply("此命注定大富大贵。", birth, "")
    assert any("无法核验" in problem for problem in claim["problems"])
