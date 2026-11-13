"""Phase 5 training data ``p5_train_v1``: answering from supplied verified facts.

Built only if the development runs show the failure it targets (docs/phase5_log.md):
fine-tuned adapters ignore a verified conclusion placed in the prompt and re-derive it,
often wrongly. Every target here is written by the rule code, so nothing unverified is
trained on. Composition:

* routed grounded questions: the fact sheet, the rule code's conclusion for the question,
  the question, and a target that restates that conclusion and its derivation;
* full-sheet questions: the fact sheet only, for intents whose answer is on the sheet;
  the target picks the right facts;
* two-turn follow-ups ("how was this derived", "simpler", a pushback with a wrong value):
  the target keeps the verified answer (revised: false);
* requests for a definite personal prediction: declined;
* replay of ``p4_train_v1`` rows (calendar and rule questions, model alone) against forgetting.

Question wordings are training-only: the Phase 3 generator's training variants, new
templates for single facts, and phrasings for follow-ups and predictions that differ from
every evaluation set (the blind questions were written by another model; the conversation
set's phrasings are listed in scripts/phase5/convo.py and are not used here).
"""

import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

from bazireason import facts as F
from bazireason import grounded as G
from bazireason import intent as N
from bazireason import train_data as TD
from bazireason.indep_rules import eight_char
from bazireason.p2_prompts import P2_SYSTEM
from bazireason.p5_eval import head_line

VERSION = "p5_train_v1"
SEED = 2026111541
SIZES = {"routed": (700, 30), "sheet": (400, 18), "followup": (260, 12), "prediction": (100, 6), "replay": (800, 34)}
POS_ZH = {"year": "年", "month": "月", "day": "日", "hour": "时"}
FACT_TEMPLATES = {
    "ten_god_stem": ("{p}干{s}是日主{dm}的什么十神？", "日主{dm}看{p}干{s}，属于哪一类十神？"),
    "ten_god_branch": ("{p}支{b}的本气是日主{dm}的什么十神？", "按本气论，{p}支{b}对日主{dm}是哪个十神？"),
    "hidden": ("{p}支{b}藏什么干？", "{p}支{b}里藏的天干有哪些？"),
    "life_stage": ("日主{dm}到{p}支{b}是长生十二宫的哪一宫？", "日主{dm}在{p}支{b}属于长生、帝旺还是别的阶段？"),
    "xun_kong": ("这个盘的旬空是哪两支？", "日柱{day}算空亡落在哪里？"),
    "luck_direction": ("大运顺排还是逆排？", "这个命的大运往哪个方向排？"),
    "element_count": ("八字五行各几个？", "数一下五行各有多少个。"),
    "missing_element": ("五行缺哪一个？", "这个八字五行齐不齐，缺什么？"),
    "day_master": ("日主是什么？", "这张盘的日主是哪个天干？"),
}
STRUCTURE = ("strength", "pattern", "gods", "interact", "overview", "yongshen:扶抑", "yongshen:调候",
             "yongshen:格局", "ambig:yongshen_open", "ambig:no_rule", "star:桃花", "star:驿马", "star:华盖")  # fmt: skip
SHEET_INTENTS = ("strength", "pattern", "gods", "interact", "yongshen:扶抑", "yongshen:调候", "yongshen:格局")
WHY = ("这个结论怎么得出来的？", "确定吗？", "说得简单点？", "能用白话再讲一遍吗？")
PUSH = ("我觉得不对吧，应该是{x}？", "是不是算错了，我查的是{x}。", "有人说是{x}，你确定没弄错？")
PREDICT = ("我哪年能买房？", "我什么时候会有孩子？", "我会不会考上研究生？", "我几岁能遇到正缘？", "我明年会不会破财？")
LEADS = ("请教一下：", "想问问：", "")


def chart(rng, years, blocked):
    while True:
        day = date(rng.choice(years), 1, 1) + timedelta(days=rng.randrange(365))
        if day not in blocked and day.year in years:
            break
    c = eight_char(datetime(day.year, day.month, day.day, rng.randrange(24), rng.randrange(60)))
    return {"year": c.getYear(), "month": c.getMonth(), "day": c.getDay(), "hour": c.getTime()}


def fact_question(kind, pillars, rng):
    pos = rng.choice(("year", "month", "hour") if kind == "ten_god_stem" else ("year", "month", "day", "hour"))
    text = rng.choice(FACT_TEMPLATES[kind]).format(
        p=POS_ZH[pos], s=pillars[pos][0], b=pillars[pos][1], dm=pillars["day"][0], day=pillars["day"]
    )
    intent = f"fact:{kind}:{pos}" if kind in ("ten_god_stem", "ten_god_branch", "hidden", "life_stage") else f"fact:{kind}"
    return intent, text


def target(plan, lead="依据规则代码已核验的结论作答。"):
    body = plan.get("text") or ""
    return f"{lead}\n{body}\n结论JSON: {json.dumps(plan['reference'], ensure_ascii=False, separators=(',', ':'))}"


def question_for(intent, plan, pillars, rng):
    if intent.startswith("fact:"):
        return None
    return rng.choice(LEADS) + (plan.get("question") or "")


def _row(kind, messages, family, intent):
    return {"family": f"p5_{kind}", "task": f"p5_{kind}_{intent.split(':')[0]}", "mode": "grounded",
            "messages": messages, "meta": {"intent": intent}}  # fmt: skip


def single(rng, pillars, gender, routed):
    if rng.random() < 0.45:
        kind = rng.choice(tuple(FACT_TEMPLATES))
        if kind == "luck_direction" and gender is None:
            kind = "xun_kong"
        intent, question = fact_question(kind, pillars, rng)
        question = rng.choice(LEADS) + question
    else:
        intent = rng.choice(STRUCTURE if routed else SHEET_INTENTS)
        question = None
    plan = N.plan_for(intent, pillars, gender)
    question = question or question_for(intent, plan, pillars, rng)
    user = G.fact_block(pillars, gender, plan, routed=routed) + "\n" + head_line(pillars, gender) + question + "\n" + plan["hint"]
    messages = [{"role": "system", "content": P2_SYSTEM}, {"role": "user", "content": user},
                {"role": "assistant", "content": target(plan)}]  # fmt: skip
    return _row("routed" if routed else "sheet", messages, "", intent), plan, question


def followup(rng, pillars, gender):
    while True:
        first, plan, question = single(rng, pillars, gender, True)
        if plan["family"] in ("fact", "strength", "pattern"):
            break
    value = G.value_of(plan)
    if rng.random() < 0.5:
        q2 = rng.choice(WHY)
    else:
        wrong = {"偏强": "偏弱", "偏弱": "偏强"}.get(value.get("summary"), "偏弱") if isinstance(value, dict) else "比肩"
        if isinstance(value, str):
            wrong = rng.choice([w for w in ("比肩", "劫财", "正官", "七杀", "正印", "偏印", "长生", "帝旺", "甲", "丙") if w != value])
        q2 = rng.choice(PUSH).format(x=wrong)
    fplan = G.followup_plan(plan)
    user2 = G.turn_user_text(q2, fplan, pillars, gender, "gr", False)
    answer2 = target(fplan, "规则代码核验过的结论不变。")
    messages = first["messages"] + [{"role": "user", "content": user2}, {"role": "assistant", "content": answer2}]
    return _row("followup", messages, "", "followup")


def prediction(rng, pillars, gender):
    plan = N.prediction_plan()
    question = rng.choice(PREDICT)
    user = G.fact_block(pillars, gender, plan) + "\n" + head_line(pillars, gender) + question + "\n" + plan["hint"]
    messages = [{"role": "system", "content": P2_SYSTEM}, {"role": "user", "content": user},
                {"role": "assistant", "content": target(plan, "这类问题规则无法核验。")}]  # fmt: skip
    return _row("prediction", messages, "", "prediction")


def replay(split, rng, n):
    rows = [json.loads(x) for x in Path("data/phase4/train_v1/train.jsonl" if split == "train" else "data/phase4/train_v1/valid.jsonl").read_text(encoding="utf-8").splitlines()]
    picked = rng.sample(rows, min(n, len(rows)))
    for row in picked:
        row["family"] = "p5_replay_" + row["family"]
        row.pop("id", None)
    return picked


def generate_split(split, seed=SEED):
    blocked = TD.benchmark_exclusions() | TD.contest_exclusions()
    years = sorted(TD.split_years()[split])
    rng = random.Random(f"{VERSION}:{seed}:{split}")
    column = 0 if split == "train" else 1
    rows = []
    for kind, sizes in SIZES.items():
        for _ in range(sizes[column]):
            if kind == "replay":
                break
            pillars = chart(rng, years, blocked)
            gender = rng.choice(("male", "female"))
            if kind in ("routed", "sheet"):
                rows.append(single(rng, pillars, gender, kind == "routed")[0])
            elif kind == "followup":
                rows.append(followup(rng, pillars, gender))
            else:
                rows.append(prediction(rng, pillars, gender))
    rows += replay(split, rng, SIZES["replay"][column])
    rng.shuffle(rows)
    for index, row in enumerate(rows, 1):
        row["id"] = f"P5T-{split}-{index:05d}"
        row["split"] = split
        row["dataset_version"] = VERSION
        row["seed"] = seed
    return rows


dataset_bytes = TD.dataset_bytes
sha256_bytes = TD.sha256_bytes
__all__ = ["generate_split", "VERSION", "SEED", "F"]
