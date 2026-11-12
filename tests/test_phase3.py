"""Phase 3: rule layer, task builders, dataset leakage guards and scoring."""

import itertools
import json
import random
import re
from datetime import date
from pathlib import Path

import pytest

from bazireason import deep_rules as D
from bazireason import deep_tasks as K
from bazireason import p3_eval as E
from bazireason import rules as R
from bazireason import train_data as TD
from bazireason import train_data3 as P
from bazireason.calendar_engine import compute_chart
from bazireason.conventions import BRANCHES, CYCLE, PILLARS, STEMS
from bazireason.p2_prompts import extract_final_json
from bazireason.prompts import build_messages
from bazireason.schema import model_view

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/phase3/train_v1"
DATE = re.compile(r"(?<![0-9])([12][0-9]{3})-([01][0-9])-([0-3][0-9])(?![0-9])")


def _rows(split, folder=DATA):
    return [json.loads(line) for line in (folder / f"{split}.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def splits():
    return {split: _rows(split) for split in P.SPLITS}


def _charts(count=400, seed=3):
    rng = random.Random(seed)
    for _ in range(count):
        year, day = rng.choice(CYCLE), rng.choice(CYCLE)
        month = R.month_pillar_for(year[0], rng.choice(BRANCHES))
        hour = R.hour_pillar_for(day[0], rng.choice(BRANCHES))
        yield {"year": year, "month": month, "day": day, "hour": hour}


# ---- Rule layer --------------------------------------------------------------------------------


def test_ten_god_classes_follow_the_five_element_cycle():
    for day in STEMS:
        for cls in D.CLASSES:
            element = D.class_element(day, cls)
            assert D.element_class(day, element) == cls
            assert R.GENERATES[element] == D.class_element(day, D.class_generates(cls))
            assert R.CONTROLS[element] == D.class_element(day, D.class_controls(cls))
        for other in STEMS:
            assert D.GOD_CLASS[R.ten_god(day, other)] == D.element_class(day, R.STEM_ELEMENT[other])


def test_branch_cards_reproduce_every_pair_relation():
    for first, second in itertools.product(BRANCHES, BRANCHES):
        card = D.branch_card(first)
        key = "".join(sorted((first, second), key=BRANCHES.index))
        found = set()
        if first == second:
            if card["刑"] == [first]:
                found.add(f"自刑:{key}")
        else:
            for kind in ("六合", "冲", "害", "破"):
                if card[kind] == second:
                    found.add(f"{kind}:{key}")
            if second in card["刑"]:
                found.add(f"刑:{key}")
            if second in card["半合"]:
                found.add(f"半合:{key}")
        assert found == set(R.branch_pair_relations(first, second)), (first, second)


def test_star_tables_agree_with_the_twelve_stage_table():
    for stem in STEMS:
        assert R.life_stage(stem, D.LU[stem]) == "临官"
    for stem, branch in D.YANG_REN.items():
        assert R.is_yang(stem) and R.life_stage(stem, branch) == "帝旺"
        assert BRANCHES.index(branch) == (BRANCHES.index(D.LU[stem]) + 1) % 12
    for triple in D.YI_MA:
        assert D.XIAN_CHI[triple] == BRANCHES[(BRANCHES.index(triple[0]) + 1) % 12]
        assert D.YI_MA[triple] == D.branch_card(triple[0])["冲"]
        assert D.HUA_GAI[triple] == triple[2]
    assert all(len(branches) == 2 for branches in D.TIAN_YI.values())
    assert set(D.TIAN_YI) == set(STEMS)


def test_revealed_pattern_follows_the_declared_priority():
    differs = 0
    for pillars in _charts(1500):
        result = D.revealed_pattern(pillars)
        day = pillars["day"][0]
        hidden = R.HIDDEN_STEMS[pillars["month"][1]]
        visible = {pillars[name][0] for name in D.OTHER}
        if result["basis"] in ("main_is_peer", "main_default", "main_revealed"):
            assert result["pattern"] == R.month_pattern(pillars)
        else:
            differs += 1
            rank = ("main_revealed", "middle_revealed", "residual_revealed").index(result["basis"])
            assert result["stem"] == hidden[rank] and result["stem"] in visible
            assert R.ten_god(day, result["stem"]) not in ("比肩", "劫财")
            # No earlier hidden stem qualified.
            assert all(
                s not in visible or R.ten_god(day, s) in ("比肩", "劫财") for s in hidden[:rank]
            )
        assert D.PATTERN_USE[result["god"]] in ("顺用", "逆用")
    assert differs > 50


def test_three_schools_are_reported_separately_and_fuyi_partitions_the_classes():
    for pillars in _charts(300):
        schools = D.yongshen_by_school(pillars)
        assert set(schools) == {"扶抑", "调候", "格局"}
        fuyi = schools["扶抑"]
        assert set(fuyi["favorable_classes"]) | set(fuyi["unfavorable_classes"]) == set(D.CLASSES)
        weak = fuyi["strength"] == "偏弱"
        assert set(fuyi["favorable_classes"]) == (
            {"印", "比劫"} if weak else {"食伤", "财", "官杀"}
        )
        month = pillars["month"][1]
        want = "火" if month in "亥子丑" else "水" if month in "巳午未" else None
        assert schools["调候"]["needs"] == want


def test_flow_year_special_configurations():
    pillars = {"year": "甲子", "month": "丙寅", "day": "庚午", "hour": "丙子"}
    same = D.flow_analysis(pillars, "庚午", luck="庚午")
    assert "伏吟:day" in same["special"] and "岁运并临" in same["special"]
    assert "冲太岁" in same["special"]  # 午 clashes the year branch 子
    clash = D.flow_analysis(pillars, "甲子")
    assert "天克地冲:day" in clash["special"] and "值太岁" in clash["special"]
    assert D.clash_with_control("甲子", "庚午") and D.clash_with_control("甲子", "戊午")
    assert not D.clash_with_control("甲子", "甲午")  # same stem: a clash without control
    assert D.luck_in_year([{"pillar": "丁卯", "start_year": 2001, "end_year": 2010}], 2011) is None


def test_trigger_years_use_only_the_declared_convention():
    pillars = {"year": "癸卯", "month": "癸亥", "day": "戊辰", "hour": "戊午"}
    found = D.trigger_years(pillars, "marriage", "male", 1935, 1946)
    by_year = {item["year"]: item["reasons"] for item in found}
    # 1942 壬午 and 1943 癸未: wealth stems; 1945 乙酉: 酉 six-combines the day branch 辰;
    # 1946 丙戌: 戌 clashes it.
    assert by_year == {1942: ["星现"], 1943: ["星现"], 1945: ["宫动"], 1946: ["宫动"]}


def test_rule_cards_have_sources_and_verified_quotations():
    kinds = {"deterministic_table", "declared_convention", "traditional_association"}
    ids = [card["id"] for card in D.RULE_CARDS]
    assert len(ids) == len(set(ids)) == 12 and not set(ids) & set(R.RULES_BY_ID)
    for card in D.RULE_CARDS:
        assert card["kind"] in kinds and card["statement"]
        assert card["quotations"] or card["works_cited_without_quotation"]
    export = json.loads((ROOT / "data/phase3/knowledge/rules_v2.json").read_text())
    quotes = [q for card in export["cards"] for q in card["quotations"]]
    assert len(quotes) >= 20 and all(q["verified"] is True for q in quotes)
    classics = ROOT / "external/classics"
    if not classics.exists():
        pytest.skip("Source pages are not downloaded")
    for quote in quotes:
        text = "".join((classics / quote["local_file"]).read_text(encoding="utf-8").split())
        assert "".join(quote["text"].split()) in text


# ---- Task builders -----------------------------------------------------------------------------


def test_every_drill_and_chart_task_ends_with_its_computed_answer():
    rng = random.Random(11)
    for kind in K.DRILLS:
        for variant in range(3):
            question, text, response = K.drill(kind, rng, variant)
            assert question and text and "answer" in response
    for pillars in _charts(60):
        text, answer = K.strength_text(pillars)
        s = R.strength_indicators(pillars)
        assert answer["summary"] == s["summary"] and s["summary"] in text
        _, text, response = K.pattern_task(pillars, rng)
        assert response["answer"] == D.revealed_pattern(pillars)["pattern"] in text
        _, text, response = K.interact_task(pillars, rng)
        assert response["answer"] == R.chart_relations(pillars)
        assert set(response["adjacent"]) <= set(response["answer"])
        _, text, response = K.flow_task(pillars, rng, year=2031)
        assert response["pillar"] == R.year_pillar(2031) == "辛亥"


def test_ambiguity_tasks_are_paired_and_never_guess():
    rng = random.Random(5)
    seen = set()
    for pillars in _charts(200):
        for kind in K.AMBIG_KINDS:
            _, question, text, response = K.ambig_task(kind, pillars, rng, rng.randrange(3))
            seen.add((kind, response["determinate"]))
            if response["determinate"]:
                assert "answer" in response and response["needs"] == []
            else:
                assert "answer" not in response
    assert {("hour_strength", True), ("hour_strength", False)} <= seen
    assert {("star_basis", True), ("star_basis", False)} <= seen
    assert ("yongshen_open", False) in seen and ("yongshen_named", True) in seen
    assert ("no_rule", False) in seen and ("hour_needed", False) in seen


def test_hour_strength_is_indeterminate_only_when_the_hour_decides():
    rng = random.Random(1)
    for pillars in _charts(300):
        _, _, _, response = K.ambig_task("hour_strength", pillars, rng)
        outcomes = set()
        for branch in BRANCHES:
            trial = dict(pillars, hour=R.hour_pillar_for(pillars["day"][0], branch))
            outcomes.add(R.strength_indicators(trial)["得势"])
        assert response["determinate"] == (len(outcomes) == 1)
        if response["determinate"]:
            assert {response["answer"]} == outcomes


def test_claims_cover_all_three_verdicts_and_speculation_is_never_confirmed():
    rng = random.Random(2)
    verdicts = set()
    for pillars in _charts(150):
        _, text, response = K.claims_task(
            pillars, rng, rng.randrange(3), rng.choice(("male", "female"))
        )
        verdicts.update(response["verdicts"])
        for index, verdict in enumerate(response["verdicts"], 1):
            assert (str(index) in response["corrections"]) == (verdict == "错误")
    assert verdicts == set(K.VERDICTS)
    for _ in range(50):
        statement, verdict, _, _ = K.make_claim("speculative", next(_charts(1)), rng)
        assert verdict == "无法验证" and statement in K.SPECULATIVE


def test_multiple_choice_keys_are_unique_and_rule_determined():
    rng = random.Random(4)
    made = set()
    for pillars in _charts(250):
        for kind in K.MCQ_KINDS:
            item = K.mcq_task(kind, pillars, rng, rng.choice(("male", "female")), 2030)
            if item is None:
                continue
            made.add(kind)
            assert len(item["options"]) == 4 and len(set(item["options"])) == 4
            assert item["answer"] in K.LETTERS
    assert made == set(K.MCQ_KINDS)


def test_no_builder_asserts_a_personal_outcome():
    certain = re.compile(r"(一定会|必定会|肯定会|注定会|必然会)")
    rng = random.Random(9)
    for pillars in _charts(80):
        for domain in D.DOMAINS:
            _, text, response = K.domain_task(pillars, rng, 0, "female", domain, 2030)
            assert response["kind"] == "traditional" and "不作预测" in text
            assert not certain.search(text)
        _, text, _ = K.overview_task(pillars, rng)
        assert "不作预测" in text and not certain.search(text)


# ---- Dataset -----------------------------------------------------------------------------------


def test_phase3_dataset_matches_manifest_and_regenerates(splits):
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert manifest["contest_material_used"].startswith("birth dates only")
    for split in P.SPLITS:
        payload = (DATA / f"{split}.jsonl").read_bytes()
        assert P.sha256_bytes(payload) == manifest["files"][f"{split}.jsonl"]["sha256"]
    blocked = TD.benchmark_exclusions(ROOT / "data/synthetic") | TD.contest_exclusions(
        ROOT / "data/phase2/exclusions/contest_birth_dates.json"
    )
    regenerated = P.generate_split("valid", blocked=blocked)
    assert P.dataset_bytes(regenerated) == (DATA / "valid.jsonl").read_bytes()


def _written_dates(row):
    """Every ISO date written in the user turns of a row (the system prompt is fixed text)."""
    found = set()
    for message in row["messages"][:-1]:
        if message["role"] != "user":
            continue
        for year, month, day in DATE.findall(message["content"]):
            try:
                found.add(date(int(year), int(month), int(day)))
            except ValueError:
                pass
    return found


def test_phase3_rows_respect_year_blocks_and_exclusions(splits):
    blocked = TD.benchmark_exclusions(ROOT / "data/synthetic") | TD.contest_exclusions(
        ROOT / "data/phase2/exclusions/contest_birth_dates.json"
    )
    years = TD.split_years()
    for split in P.SPLITS:
        for row in splits[split]:
            if row["family"].startswith("replay_"):
                continue  # covered by the Phase 2 guards and tests
            for day in _written_dates(row):
                assert day not in blocked, (row["id"], day)
                assert day.year in years[split], (row["id"], day)


def test_phase3_splits_share_no_prompt_and_keep_test_only_wordings_out(splits):
    def prompts(rows):
        return {json.dumps(r["messages"][:-1], ensure_ascii=False) for r in rows}

    train, valid, test = (prompts(splits[s]) for s in P.SPLITS)
    # Table drills have a small finite question space, so a drill may recur; nothing else may.
    drills = prompts([r for rows in splits.values() for r in rows if r["family"] == "drill"])
    assert not ((train | valid) & test) - drills
    for split in ("train", "valid"):
        for row in splits[split]:
            if not row["family"].startswith("replay_"):
                assert row["meta"].get("template") != 2, row["id"]
    assert sum(row["meta"].get("template") == 2 for row in splits["test"]) > 100


def test_phase3_holds_no_contest_text_and_no_benchmark_case(splits):
    text = "\n".join(m["content"] for rows in splits.values() for r in rows for m in r["messages"])
    path = ROOT / "external/BaziQA/data"
    if path.exists():
        stems = set()
        for file in path.glob("*.json"):
            for person in json.loads(file.read_text(encoding="utf-8"))[1:]:
                stems.update(
                    q["question"] for q in person.get("questions", []) if len(q["question"]) >= 8
                )
        assert len(stems) > 400 and not any(stem in text for stem in stems)
    for line in (ROOT / "data/synthetic/benchmark_v3.jsonl").read_text().splitlines():
        raw = json.loads(line)["input"].get("datetime")
        if isinstance(raw, str) and len(raw) >= 19:
            assert raw not in text


def test_strict_audit_rows_use_the_benchmark_prompt_and_true_answers(splits):
    case = {
        "input": {"datetime": "1993-10-21T05:34:36", "timezone": "Asia/Shanghai"},
        "convention": TD.convention(2),
    }
    context = compute_chart(case["input"], case["convention"])
    assert P.strict_messages(case, model_view(context)) == build_messages(
        case, "tool_context", context
    )
    kinds = set()
    for split in ("train", "test"):
        for row in splits[split]:
            if not row["family"].startswith("strict_"):
                continue
            meta = row["meta"]
            kinds.add((split, meta["kind"]))
            truth = model_view(compute_chart(meta["inputs"], meta["convention"]))
            assert meta["expected"] == truth == json.loads(row["messages"][-1]["content"])
            assert bool(meta["changed"]) == (meta["supplied"] != truth)
            if meta["changed"] and truth["status"] == "ok" and meta["kind"] != "default_guess":
                assert all(truth["pillars"][f] != v for f, v in meta["changed"].items())
    # The two reversed-association kinds are evaluation-only.
    assert ("test", "day_shift_midday") in kinds and ("test", "hour_stem_night") in kinds
    assert ("train", "day_shift_midday") not in kinds and ("train", "hour_stem_night") not in kinds


def test_phase3_reference_answers_score_perfectly_and_blanks_score_zero(splits):
    rows = splits["test"]
    assert all(E.score3(row, row["messages"][-1]["content"])["correct"] for row in rows)
    assert not any(E.score3(row, "不知道")["correct"] for row in rows)
    for row in rows:
        if row["family"] == "strict_shift" and row["meta"]["changed"]:
            copied = E.score3(row, json.dumps(row["meta"]["supplied"], ensure_ascii=False))
            assert copied["followed_corruption"] and not copied["correct"]
    summary = E.summarize3([E.score3(row, row["messages"][-1]["content"]) for row in rows])
    assert (
        summary["overall"]["rate"] == 1
        and summary["strict_audit"]["corrupted_followed"]["hits"] == 0
    )


def test_ambiguity_scoring_separates_overconfidence_from_overcaution(splits):
    rows = [row for row in splits["test"] if row["family"] == "ambig"]
    open_row = next(r for r in rows if not r["meta"]["expected"]["determinate"])
    closed_row = next(r for r in rows if r["meta"]["expected"]["determinate"])
    guess = '结论JSON: {"determinate":true,"needs":[],"answer":"火"}'
    refuse = '结论JSON: {"determinate":false,"needs":["hour"]}'
    assert E.score3(open_row, guess)["definite_on_indeterminate"]
    assert E.score3(closed_row, refuse)["abstained_on_determinate"]
    assert extract_final_json(open_row["messages"][-1]["content"])["determinate"] is False
    assert all(len(row["meta"]["pillars"]) == len(PILLARS) for row in rows)


# ---- Second-round wording and the demo ---------------------------------------------------------


def test_second_round_dataset_is_reproducible_explicit_and_isolated():
    folder = ROOT / "data/phase3/train_v2"
    manifest = json.loads((folder / "manifest.json").read_text())
    train, valid = _rows("train", folder), _rows("valid", folder)
    assert (
        P.sha256_bytes((folder / "train.jsonl").read_bytes())
        == (manifest["files"]["train.jsonl"]["sha256"])
    )
    blocked = TD.benchmark_exclusions(ROOT / "data/synthetic") | TD.contest_exclusions(
        ROOT / "data/phase2/exclusions/contest_birth_dates.json"
    )
    assert (
        P.dataset_bytes(P.generate_split("valid", blocked=blocked, version=2))
        == (folder / "valid.jsonl").read_bytes()
    )
    years = TD.split_years()
    for split, rows in (("train", train), ("valid", valid)):
        for row in rows:
            if row["family"].startswith("replay_"):
                continue
            for day in _written_dates(row):
                assert day not in blocked and day.year in years[split], row["id"]
    # Fresh samples: no prompt repeats a first-round row outside the finite drills.
    first = {
        json.dumps(r["messages"][:-1], ensure_ascii=False)
        for split in P.SPLITS
        for r in _rows(split)
        if r["family"] != "drill"
    }
    assert not any(
        json.dumps(r["messages"][:-1], ensure_ascii=False) in first
        for r in train
        if r["family"] != "drill"
    )
    gods = next(r for r in train if r["family"] == "gods")["messages"][-1]["content"]
    assert "十神表（日主" in gods and "逐项检查十神组合" in gods and "查表" in gods
    pairs = next(r for r in train if r["family"] == "interact")["messages"][-1]["content"]
    assert "卡的" in pairs or "不在" in pairs
    assert any(r["meta"]["kind"] == "silent_check" for r in train if r["family"] == "drill")


def test_ten_god_table_matches_the_rule_table_in_both_wordings():
    from bazireason import textstyle

    for day in STEMS:
        table = K.god_table(day)
        assert all(f"{stem}{R.ten_god(day, stem)}" in table for stem in STEMS)
    textstyle.set_version(3)
    try:
        for day in STEMS:
            for stem in STEMS:
                assert K.god_reason(day, stem).endswith("故为" + R.ten_god(day, stem))
                assert K.read(day, stem) == f"查表，{stem}为{R.ten_god(day, stem)}"
    finally:
        textstyle.set_version(1)
    assert K.god_reason("甲", "庚") == R.ten_god_reason("甲", "庚")


def test_demo_routes_questions_to_trained_formats_and_never_predicts():
    from bazireason import demo_app as A
    from bazireason import demo_deep as X

    birth = {
        "date": "1990-05-17",
        "time": "14:30:00",
        "timezone": "Asia/Shanghai",
        "sect": 2,
        "gender": "female",
    }
    chart = A.chart_payload(birth)["chart"]
    expect = {
        "请整体解读一下这个盘": "overview",
        "日主强弱怎么数？": "strength",
        "按透干取格是什么格局？": "pattern",
        "这个八字的用神是什么？": "ambig",
        "按扶抑一派喜什么？": "yongshen",
        "原局有哪些合冲刑害？": "interact",
        "2031年流年如何": "flow",
        "传统上怎么看婚姻方面？": "domain",
        "这个八字的桃花在哪一支？": "ambig",
        "这个八字好不好？": "ambig",
    }
    for question, kind in expect.items():
        planned = X.plan(question, chart, today=2026)
        assert planned and planned["kind"] == kind, question
        messages, routed = A.chat_messages(birth, [], question)
        assert routed == kind and messages[0]["content"] == A.P2_SYSTEM
        # The rule code's own derivation, used as a reply, passes the check.
        reply = (
            planned["text"] + "\n结论JSON: " + json.dumps(planned["reference"], ensure_ascii=False)
        )
        audit = A.audit_reply(reply, birth, question)
        assert audit["engine"]["agrees"] and not audit["problems"], question
        assert not A.audit_reply('结论JSON: {"answer":"x"}', birth, question)["engine"]["agrees"]
    # No unique answer: the reference itself declines.
    assert X.plan("这个八字的用神是什么？", chart)["reference"]["determinate"] is False
    assert X.plan("这个八字好不好？", chart)["reference"]["determinate"] is False
    # A request for a definite personal outcome is left to the declining route.
    for question in ("我什么时候会结婚？", "我明年会不会发财？", "2030年能不能升职"):
        assert X.plan(question, chart, today=2026) is None
        assert A.chat_messages(birth, [], question)[1] == "policy"
    # A prediction request in a later turn keeps the declining format.
    history = [
        {"role": "user", "content": "日主强弱怎么数？"},
        {"role": "assistant", "content": "…"},
    ]
    messages, kind = A.chat_messages(birth, history, "我什么时候会结婚？")
    assert kind == "policy" and messages[-1]["content"].endswith(A.OUTPUT_HINTS["policy"])
    # Without a gender the marriage stars cannot be chosen, and no chart means no plan.
    nogender = A.chart_payload(dict(birth, gender=None))["chart"]
    assert X.plan("传统上怎么看婚姻方面？", nogender)["reference"]["needs"] == ["gender"]
    undated = A.chart_payload(dict(birth, time=""))["chart"]
    assert undated["status"] != "ok" and X.plan("日主强弱怎么数？", undated) is None
    assert A.chart_payload(dict(birth, time=""))["deep"] == []
    models = A.Models(
        "models/none",
        "artifacts/adapters/none",
        "artifacts/adapters/none",
        "artifacts/adapters/none",
    )
    assert [m["id"] for m in models.available()] == [
        "p3",
        "p4",
        "p2",
        "base",
    ] and models.default() == "base"
    cards = {card["id"] for card in X.relevant_cards("按扶抑一派的用神")}
    assert "R-YS-01" in cards


def test_demo_checks_pattern_and_spouse_star_claims_in_prose():
    from bazireason import demo_app as A

    birth = {"date": "1990-05-17", "time": "14:30:00", "timezone": "Asia/Shanghai", "sect": 2}
    # 壬 day master, month 巳 with 庚 revealed: 偏印格 by 透干, 偏财格 by main qi only.
    wrong = A.audit_reply("按月令透干取格为伤官格。", birth)
    assert any("伤官格" in p and "偏印格" in p for p in wrong["problems"])
    for right in ("按月令透干取格为偏印格。", "只看本气是偏财格。", "格局为枭神格。"):
        assert not A.audit_reply(right, birth)["problems"], right
    # A spouse star for one gender is a problem only if it contradicts the input.
    female_rule = "传统上女以正官、七杀象征配偶。"
    assert any("没有填写性别" in p for p in A.audit_reply(female_rule, birth)["problems"])
    assert not A.audit_reply(female_rule, dict(birth, gender="female"))["problems"]
    assert A.audit_reply(female_rule, dict(birth, gender="male"))["problems"]
    both = "男命以财为妻星，女命以官杀为夫星。"
    assert not A.audit_reply(both, birth)["problems"]


def test_claim_checker_recomputes_fixed_form_claims_without_a_model():
    from bazireason import claim_check as C

    pillars = {"year": "庚午", "month": "辛巳", "day": "壬午", "hour": "丁未"}
    text = (
        "午未六合；丁壬相合；没有五合或相冲。月支巳藏丙、庚、戊。旬空申酉。壬临午为胎。"
        "桃花在卯。记为偏弱。合计：木0、火3、土1、金2、水1。取格为伤官格。《滴天寿》云。"
        "子午相害。"
    )
    results = {r["said"] or r["kind"]: r["ok"] for r in C.check_claims(text, pillars)}
    for said in ("午未六合", "丁壬相合", "旬空申酉", "壬临午为胎", "桃花在卯", "记为偏弱"):
        assert results[said], said
    assert not results["木0、火3、土1、金2、水1"]  # 火 is 4
    assert not results["伤官格"] and not results["《滴天寿》"] and not results["子午相害"]
    assert all(
        r["kind"] != "stem_relation" or r["said"] == "丁壬相合"
        for r in C.check_claims(text, pillars)
    )
    assert C.check_claims("此命注定大富大贵。", pillars)[-1]["kind"] == "unverifiable"
