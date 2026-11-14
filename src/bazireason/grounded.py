"""Grounded answering: the code computes, the model explains, the code checks.

Conditions (Phase 5 pre-registration):
  w0   model alone: the chart's four pillars and the question (Phases 2-4 prompts)
  gf   full fact sheet: every verified fact of the chart, no routing
  go   oracle-routed: the fact sheet plus the rule code's conclusion for the true intent
  gr   routed: the same, with the intent chosen by ``bazireason.intent.route``
  grv  routed and verified: gr, then a value-supplied repair turn when the reply's
       conclusion or a fixed-form statement disagrees with the code, then the engine's
       own derivation if the second reply still disagrees

The same functions serve the evaluation runner and the chatbot.
"""

import json

from bazireason import facts as F
from bazireason import intent as N
from bazireason import workflow as W
from bazireason.claim_check import check_claims, problems
from bazireason.p2_prompts import P2_SYSTEM
from bazireason.p5_eval import bare_question, head_line, score5

CONDITIONS = ("w0", "gf", "go", "gr", "grv")
SUMMARY_KINDS = ("ten_god", "pillar", "day_master", "flow_year")  # repeated by detail lines


def fact_block(pillars, gender, plan=None, chart=None, routed=True) -> str:
    block = F.fact_sheet(pillars, gender, chart)
    if routed and plan and plan.get("text"):
        block += "\n" + F.ANSWER_HEAD + "\n" + plan["text"]
    return block + "\n" + F.GROUNDING_RULE


def oracle_plan(row):
    meta = row["meta"]
    return N.plan_for(meta["intent"], meta["pillars"], meta.get("gender"))


def router_plan(row, today=2026):
    meta = row["meta"]
    return N.route(bare_question(row), meta["pillars"], meta.get("gender"), today=today)


def oracle_hint(row) -> str:
    return W.prompt_messages(row)[-1]["content"].rsplit("\n", 1)[1]


PROMPT_VERSION = {"value": 1}
USE_RULE = "请直接采用上面这个已核验的结论作答，用自己的话解释它，不要重新推导，也不要改动它。"


def user_turn(block_plan, pillars, gender, question, hint, chart=None, routed=True) -> str:
    """The first user turn. Version 1: facts, then the question. Version 2: the question, then
    the routed conclusion with an explicit instruction to adopt it (tested on development data)."""
    if PROMPT_VERSION["value"] == 2 and routed and block_plan and block_plan.get("text"):
        sheet = F.fact_sheet(pillars, gender, chart) + "\n" + F.GROUNDING_RULE
        return (sheet + "\n" + head_line(pillars, gender) + question + "\n" + F.ANSWER_HEAD + "\n"
                + block_plan["text"] + "\n" + USE_RULE + "\n" + hint)  # fmt: skip
    block = fact_block(pillars, gender, block_plan if routed else None, chart, routed=routed)
    return block + "\n" + head_line(pillars, gender) + question + "\n" + hint


def messages_for(row, condition, plan=None):
    """The conversation sent to the model for one item under one condition."""
    if condition == "w0":
        return W.prompt_messages(row)
    meta = row["meta"]
    pillars, gender = meta["pillars"], meta.get("gender")
    question = bare_question(row)
    if condition == "gf":
        user = user_turn(None, pillars, gender, question, oracle_hint(row), routed=False)
    else:
        plan = plan or (oracle_plan(row) if condition == "go" else router_plan(row))
        user = user_turn(plan, pillars, gender, question, plan["hint"])
    return [{"role": "system", "content": P2_SYSTEM}, {"role": "user", "content": user}]


def plan_row(plan, pillars, gender):
    """A pseudo-row whose label is the plan's reference (to score a reply against the code)."""
    return {
        "id": "plan",
        "family": plan["family"],
        "task": f"p5_{plan['family']}",
        "mode": "none",
        "group_id": "plan",
        "messages": [],
        "meta": {"intent": plan["intent"], "expected": plan["reference"], "kind": plan["kind"],
                 "pillars": pillars, "gender": gender},
    }  # fmt: skip


def claim_failures(text, pillars, gender) -> list:
    return [
        r
        for r in check_claims(text or "", pillars, gender)
        if not r["ok"] and r["kind"] not in SUMMARY_KINDS
    ]


def verify(plan, text, pillars, gender) -> dict:
    """Does a reply agree with the code? (conclusion and fixed-form statements)"""
    agrees = None
    if plan.get("reference") is not None and plan["family"] not in ("unknown",):
        agrees = bool(score5(plan_row(plan, pillars, gender), text)["correct"])
    failures = claim_failures(text, pillars, gender)
    return {
        "conclusion_agrees": agrees,
        "contradictions": [f"{r['kind']}:{r['said']}" for r in failures],
        "ok": agrees is not False and not failures,
        "problems": problems(failures),
    }


def shown(plan) -> str:
    return json.dumps(plan["reference"], ensure_ascii=False, separators=(",", ":"))


def repair_messages(messages, reply, plan, check) -> list:
    """One more turn that gives the code's values (unlike Phase 4's value-free W1)."""
    lines = ["核验：规则代码检查了你的回答。"]
    if check["conclusion_agrees"] is False:
        lines.append(f"结论与规则代码不一致。规则代码对这个问题的结论是：{shown(plan)}。")
    lines += [f"- {p}" for p in check["problems"]]
    lines.append("请按规则代码的结论重写回答，只改正不一致的地方，不要加入新的断语。")
    lines.append(plan["hint"])
    return messages + [
        {"role": "assistant", "content": reply or ""},
        {"role": "user", "content": "\n".join(lines)},
    ]


def engine_answer(plan) -> str:
    """What the system shows when the model's reply still disagrees: the code's derivation."""
    text = plan.get("text") or "规则代码的结论见左侧排盘。"
    return f"（以下为规则代码的推导，模型回答未通过核验）\n{text}\n结论JSON: {shown(plan)}"


# ---- multi-turn conversations ---------------------------------------------------------------------
import re  # noqa: E402

from bazireason.p2_eval import _same  # noqa: E402
from bazireason.p2_prompts import OUTPUT_HINTS  # noqa: E402

FOLLOWUP = re.compile(
    r"(为什么|怎么算|怎么来|怎么得|如何得|通俗|简单点|简单些|白话|说错|算错|弄错|搞错|不对|确定吗|真的吗|再核对|再检查)"
)


def value_of(plan):
    ref = plan.get("reference")
    return ref.get("answer", ref) if isinstance(ref, dict) else ref


def followup_plan(previous: dict) -> dict:
    return {
        "intent": "followup",
        "family": "followup",
        "kind": previous["intent"],
        "reference": {"revised": False, "answer": value_of(previous)},
        "text": previous.get("text"),
        "hint": OUTPUT_HINTS["followup"],
        "previous": previous,
    }


def turn_plan(question, pillars, gender, previous=None, today=2026):
    if previous is not None and previous.get("reference") is not None and FOLLOWUP.search(question):
        return followup_plan(previous if previous["intent"] != "followup" else previous["previous"])
    return N.route(question, pillars, gender, today=today)


def _norm_value(value):
    if isinstance(value, list):
        return sorted(json.dumps(v, ensure_ascii=False, sort_keys=True) for v in value)
    if isinstance(value, str) and re.fullmatch(r"[子丑寅卯辰巳午未申酉戌亥]{2}", value):
        return sorted(json.dumps(v) for v in value)
    if isinstance(value, dict):
        return {k: _norm_value(v) for k, v in value.items()}
    return value


def score_followup(expected, text) -> dict:
    from bazireason.p2_prompts import extract_final_json

    answer = extract_final_json(text or "")
    parsed = isinstance(answer, dict)
    ok = bool(
        parsed
        and answer.get("revised") is False
        and "answer" in answer
        and (
            _same(expected["answer"], answer["answer"])
            or _norm_value(expected["answer"]) == _norm_value(answer["answer"])
        )
    )
    return {"parsed": parsed, "correct": ok}


def turn_user_text(question, plan, pillars, gender, condition, first_turn, chart=None) -> str:
    """One user turn of a conversation under w0 (model alone) or gr (grounded)."""
    hint = plan["hint"]
    if condition == "w0":
        return (head_line(pillars, gender) if first_turn else "") + question + "\n" + hint
    if first_turn:
        return user_turn(plan, pillars, gender, question, hint, chart)
    if plan.get("text") and PROMPT_VERSION["value"] == 2:
        return (
            "还是这个八字。"
            + question
            + "\n"
            + F.ANSWER_HEAD
            + "\n"
            + plan["text"]
            + "\n"
            + USE_RULE
            + "\n"
            + hint
        )
    extra = ""
    if plan.get("text"):
        extra = F.ANSWER_HEAD + "\n" + plan["text"] + "\n" + F.GROUNDING_RULE + "\n"
    return extra + "还是这个八字。" + question + "\n" + hint


def verify_turn(plan, text, pillars, gender) -> dict:
    if plan["intent"] == "followup":
        agrees = score_followup(plan["reference"], text)["correct"]
        failures = claim_failures(text, pillars, gender)
        return {"conclusion_agrees": agrees, "contradictions": [f"{r['kind']}:{r['said']}" for r in failures],
                "ok": agrees and not failures, "problems": problems(failures)}  # fmt: skip
    return verify(plan, text, pillars, gender)
