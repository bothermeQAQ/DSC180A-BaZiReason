"""Phase 5 tests: fact sheets, router, oracle plans, scoring, grounded prompts, splits. No model.

The sealed rows are only hashed here; their contents are not inspected.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/phase5"))
CHART = {"year": "庚午", "month": "辛巳", "day": "壬午", "hour": "甲辰"}


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


def test_fact_sheet_states_nothing_the_claim_checker_rejects():
    from bazireason import facts as F
    from bazireason.grounded import claim_failures

    for pillars in (CHART, {"year": "丁丑", "month": "辛亥", "day": "甲子", "hour": "己巳"}):
        for gender in ("male", "female", None):
            assert claim_failures(F.fact_sheet(pillars, gender), pillars, gender) == []


def test_oracle_plans_reproduce_every_development_label():
    from bazireason import intent as N
    from bazireason import p5_eval as E
    from bazireason.p3_eval import GRADED, _keys_match, score_ambig

    checked = 0
    for row in rows(ROOT / "data/phase4/final_test/p3.jsonl"):
        intent = E.template_intent(row)
        if not intent:
            continue
        plan = N.plan_for(intent, row["meta"]["pillars"], row["meta"].get("gender"))
        expected = row["meta"]["expected"]
        if row["family"] == "ambig":
            assert score_ambig(expected, plan["reference"])["correct"], intent
        else:
            assert _keys_match(
                expected, plan["reference"], GRADED[row["family"]] or tuple(expected)
            ), intent
        checked += 1
    assert checked == 294
    for raw in rows(ROOT / "data/phase4/final_test/indep.jsonl"):
        item = E.indep_item(raw)
        kind = item["meta"]["intent"].split(":")[1]
        plan = N.plan_for(item["meta"]["intent"], item["meta"]["pillars"], item["meta"]["gender"])
        assert E._norm_fact(kind, plan["reference"]["answer"]) == E._norm_fact(
            kind, item["meta"]["expected"]["answer"]
        )


def test_router_on_development_data_and_common_chat_questions():
    import run5

    from bazireason import grounded as G
    from bazireason import intent as N

    for part in ("rules", "indep"):
        items = run5.load_items("dev", part)
        assert all(G.router_plan(i)["intent"] == i["meta"]["intent"] for i in items), part
    blind = rows(ROOT / "data/phase5/blind/dev.jsonl")
    assert sum(G.router_plan(i)["intent"] == i["meta"]["intent"] for i in blind) >= 185
    cases = {
        "空亡在哪": "fact:xun_kong",
        "月支藏了哪些天干": "fact:hidden:month",
        "年干对我来说是什么十神": "fact:ten_god_stem:year",
        "我哪年能结婚？": "prediction",
        "日主强弱怎么数？": "strength",
        "按扶抑一派喜什么？": "yongshen:扶抑",
        "原局有哪些合冲刑害？": "interact",
        "这个八字的桃花在哪一支？": "star:桃花",
        "这个八字好不好？": "ambig:no_rule",
        "这个八字的用神是什么？": "ambig:yongshen_open",
    }
    for question, intent in cases.items():
        assert N.route(question, CHART, "male")["intent"] == intent, question


def test_fact_scoring_accepts_a_list_written_as_a_string():
    from bazireason import p5_eval as E

    item = E.indep_item(
        next(
            r
            for r in rows(ROOT / "data/phase4/final_test/indep.jsonl")
            if r["meta"]["kind"] == "xun_kong"
        )
    )
    want = "".join(item["meta"]["expected"]["answer"])
    assert E.score5(item, f'结论JSON: {{"answer":"{want}"}}')["correct"]
    assert E.score5(item, f'结论JSON: {{"answer":{json.dumps(list(want), ensure_ascii=False)}}}')[
        "correct"
    ]
    assert not E.score5(item, '结论JSON: {"answer":"子子"}')["correct"]


def test_grounded_prompts_repair_and_engine_fallback():
    import run5

    from bazireason import facts as F
    from bazireason import grounded as G
    from bazireason import workflow as W
    from bazireason.p5_eval import score5

    item = run5.load_items("dev", "rules")[0]
    assert G.messages_for(item, "w0") == W.prompt_messages(item)
    for condition in ("gf", "go", "gr"):
        user = G.messages_for(item, condition)[-1]["content"]
        assert user.startswith(F.SHEET_HEAD) and F.GROUNDING_RULE in user
        assert (F.ANSWER_HEAD in user) == (condition != "gf")
    assert G.messages_for(item, "go") == G.messages_for(item, "gr")
    plan = G.oracle_plan(item)
    wrong = "结论JSON: {}"
    check = G.verify(plan, wrong, item["meta"]["pillars"], item["meta"].get("gender"))
    assert check["conclusion_agrees"] is False
    note = G.repair_messages([], wrong, plan, check)[-1]["content"]
    assert G.shown(plan) in note  # unlike Phase 4's W1, the repair turn gives the value
    assert score5(item, G.engine_answer(plan))["correct"]


def test_followups_keep_the_verified_answer():
    from bazireason import grounded as G
    from bazireason import intent as N

    first = N.route("空亡在哪", CHART, "male")
    follow = G.turn_plan("我看别的软件说是子丑，你是不是算错了？", CHART, "male", first)
    assert follow["intent"] == "followup"
    good = '结论JSON: {"revised": false, "answer": ["申","酉"]}'
    assert G.verify_turn(follow, good, CHART, "male")["conclusion_agrees"]
    assert not G.verify_turn(
        follow, '结论JSON: {"revised": true, "answer": "子丑"}', CHART, "male"
    )["conclusion_agrees"]


def test_phase5_splits_match_their_manifests_and_blind_items_rebuild():
    import gen_blind

    for folder in ("valid", "sealed"):
        manifest = json.loads((ROOT / "data/phase5" / folder / "manifest.json").read_text())
        for name, info in manifest["files"].items():
            digest = hashlib.sha256((ROOT / "data/phase5" / folder / name).read_bytes()).hexdigest()
            assert digest == info["sha256"], (folder, name)
    for split in ("dev", "valid", "sealed"):
        log = rows(ROOT / f"data/phase5/blind/{split}_attempts.jsonl")
        payload = "".join(
            json.dumps(i, ensure_ascii=False) + "\n" for i in gen_blind.build(split, log)
        )
        assert payload.encode() == (ROOT / f"data/phase5/blind/{split}.jsonl").read_bytes(), split


def test_no_earlier_prompt_reaches_the_phase5_splits():
    import build_eval

    used = build_eval.used_prompts()
    for split in ("valid", "sealed"):
        for part in ("rules", "indep", "calendar"):
            prompts = {
                build_eval.first_user(r) for r in rows(ROOT / f"data/phase5/{split}/{part}.jsonl")
            }
            assert not prompts & used, (split, part)


def test_sealed_split_is_closed_without_a_selection(tmp_path, monkeypatch):
    import run5

    monkeypatch.setattr(run5, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run5.py", "--split", "sealed", "--labels", "p4a1"])
    with pytest.raises(SystemExit):
        run5.main()
    assert not (tmp_path / "sealed").exists() and not (tmp_path / "sealed_access.jsonl").exists()


def test_g1_training_data_has_no_evaluation_wording():
    import convo

    from bazireason import indep_rules as I
    from bazireason import train_data5 as D

    manifest = json.loads((ROOT / "data/phase5/train_v1/manifest.json").read_text())
    for name, info in manifest["files"].items():
        assert (
            hashlib.sha256((ROOT / "data/phase5/train_v1" / name).read_bytes()).hexdigest()
            == info["sha256"]
        )
    train = rows(ROOT / "data/phase5/train_v1/train.jsonl")
    users = [m["content"] for r in train for m in r["messages"] if m["role"] == "user"]
    text = "\n".join(users)
    blind = {
        i["meta"]["question"]
        for s in ("dev", "valid", "sealed")
        for i in rows(ROOT / f"data/phase5/blind/{s}.jsonl")
    }
    assert not any(q in text for q in blind)
    for phrase in (
        *convo.WHY,
        "你是不是算错了",
        *[t.split("{")[0] for ts in I.TEMPLATES.values() for t in ts if len(t.split("{")[0]) > 6],
    ):
        assert phrase not in text, phrase
    evaluation = set()
    for split in ("valid", "sealed"):
        for part in ("rules", "indep", "calendar"):
            evaluation |= {
                next(m["content"] for m in r["messages"] if m["role"] == "user")
                for r in rows(ROOT / f"data/phase5/{split}/{part}.jsonl")
            }
    assert not evaluation & set(users)
    assert all(r["messages"][-1]["role"] == "assistant" for r in train)
    assert {r["family"] for r in train} >= {"p5_routed", "p5_sheet", "p5_followup", "p5_prediction"}
    assert D.VERSION == "p5_train_v1"


def test_chatbot_drops_the_verified_opening_when_nothing_was_checked():
    from bazireason import demo_app as A

    assert (
        A.drop_unverified_lead(A.VERIFIED_LEAD + "\n这类问题规则无法确证。")
        == "这类问题规则无法确证。"
    )
    assert A.drop_unverified_lead("空亡在申酉。") == "空亡在申酉。"
    assert A.MODEL_IDS[0] == "p5" and A.DEFAULT_ADAPTER_P5.endswith("p5-g1")


def test_chatbot_fate_note_follows_the_routed_intent():
    from bazireason import demo_app as A
    from bazireason import grounded as G
    from bazireason import intent as N

    first = N.route("空亡在哪", CHART, "male")
    pushback = G.turn_plan("我看别的软件说是子丑，你是不是算错了？", CHART, "male", first)
    assert not A.speculative("我看别的软件说是子丑，你是不是算错了？", pushback)
    assert A.speculative("我哪年能结婚？", N.route("我哪年能结婚？", CHART, "male"))
    assert A.speculative("你是不是算错了？")  # model-alone mode keeps the keyword rule
