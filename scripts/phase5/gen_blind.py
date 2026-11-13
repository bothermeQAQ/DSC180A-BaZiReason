"""Blind-wording questions: written by an untuned local model, never by the project author.

    .venv-mlx/bin/python scripts/phase5/gen_blind.py            # all three splits
    .venv-mlx/bin/python scripts/phase5/gen_blind.py --check    # rebuild the items from the log

For each (chart, intent) the writer model (Qwen3.5-4B, 4-bit, untuned) gets a plain-language
description of what a user wants to know and writes one question in its own words. It is
never shown a project template, rule card or training row. Questions pass mechanical
filters fixed before any output was seen (FILTERS below); up to four attempts with new
seeds, otherwise the item is dropped. Labels come from the rule code for the intended
intent (``bazireason.intent.plan_for``). Every attempt, accepted or not, is kept in
``data/phase5/blind/<split>_attempts.jsonl``.

MiniCPM5-2B (a different model family) was tried first and produced unusable questions
(prompt echoes, fragments, the wrong speaker); see docs/phase5_log.md.
"""

import argparse
import hashlib
import json
import random
import re
from datetime import date, datetime, timedelta
from pathlib import Path

from bazireason import intent as N
from bazireason import train_data as TD
from bazireason.indep_rules import eight_char
from bazireason.p5_eval import make_item

OUT = Path("data/phase5/blind")
WRITER = "models/qwen3.5-4b-mlx-4bit"
PLAN = {"dev": ("valid", 2026111501, 10), "valid": ("valid", 2026111502, 8), "sealed": ("test", 2026111503, 16)}
POS_ZH = {"year": "年", "month": "月", "day": "日", "hour": "时"}
SCHOOLS = ("扶抑", "调候", "格局")
STARS = ("桃花", "驿马", "华盖")
DOMAINS = {"marriage": "婚姻感情", "career": "事业工作", "wealth": "财运"}
EVENTS = ("哪一年会结婚", "什么时候能发财", "这辈子会不会离婚", "多少岁能升职", "什么时候会生病", "哪一年运气会转好")
CLASSES = (
    "ten_god_stem", "ten_god_branch", "hidden", "life_stage", "xun_kong", "luck_direction",
    "element_count", "missing_element", "day_master", "strength", "pattern", "yongshen",
    "interact", "flow", "star", "gods", "yongshen_open", "no_rule", "prediction", "domain",
)  # fmt: skip
SYSTEM = "你是一位对八字感兴趣的普通用户，正在和一个八字助手聊天。"
ASK = (
    "你的八字是：{chart}，{sex}。请你用自己的话，像平时聊天一样，向助手提一个问题。"
    "你想知道的是：{desc}。只写出你要问的那一句话，不要解释，也不要回答。"
)
QUESTION_WORDS = re.compile(r"[？?吗呢]|什么|哪|几|怎么|怎样|多少|是否|如何|何|有没有|是不是|能不能|会不会|好不好")
ECHO = ("结论JSON", "只写出", "不要解释", "不要回答", "向助手提")
DOMAIN_MARKS = {"marriage": ("婚", "感情", "配偶", "姻缘", "另一半", "伴侣"), "career": ("事业", "工作", "职业"),
                "wealth": ("财", "钱")}  # fmt: skip


def describe(cls, pillars, rng):
    """(intent, plain-language description, required mention or None)."""
    dm = pillars["day"][0]
    if cls in ("ten_god_stem", "ten_god_branch", "hidden", "life_stage"):
        pos = rng.choice(("year", "month", "hour") if cls == "ten_god_stem" else ("year", "month", "day", "hour"))
        stem, branch = pillars[pos][0], pillars[pos][1]
        p = POS_ZH[pos]
        desc = {
            "ten_god_stem": f"八字里{p}柱的天干{stem}，对你（日主{dm}）来说是什么十神",
            "ten_god_branch": f"八字里{p}柱的地支{branch}，按它的本气算，对你（日主{dm}）来说是什么十神",
            "hidden": f"{p}柱的地支{branch}里面藏着哪些天干",
            "life_stage": f"你的日主{dm}在{p}柱的地支{branch}上，处在十二长生的哪个阶段",
        }[cls]
        return f"fact:{cls}:{pos}", desc, ("position", pos, stem if cls == "ten_god_stem" else branch)
    simple = {
        "xun_kong": ("fact:xun_kong", "你的八字里哪两个地支是空亡（旬空）"),
        "luck_direction": ("fact:luck_direction", "你的大运是顺着排还是倒着排"),
        "element_count": ("fact:element_count", "八个字里金木水火土各有几个"),
        "missing_element": ("fact:missing_element", "你的八字五行里缺不缺什么，缺的话缺哪个"),
        "day_master": ("fact:day_master", "你的日主（日元）是哪个天干"),
        "strength": ("strength", "你的日主是偏强还是偏弱，是怎么判断出来的"),
        "pattern": ("pattern", "你的八字是什么格局"),
        "interact": ("interact", "你的八字里天干、地支之间有哪些合、冲、刑、害"),
        "gods": ("gods", "你的八字里年干、月干、时干分别是什么十神，有没有什么十神组合"),
        "yongshen_open": ("ambig:yongshen_open", "你的喜用神是什么"),
        "no_rule": ("ambig:no_rule", "你的八字整体算不算好命"),
    }
    if cls in simple:
        return (*simple[cls], None)
    if cls == "yongshen":
        school = rng.choice(SCHOOLS)
        desc = {"扶抑": "按扶抑的方法，你的用神喜什么", "调候": "按调候的方法，你的八字需要什么五行来调候",
                "格局": "按格局派的方法，你的用神是哪个十神，顺用还是逆用"}[school]  # fmt: skip
        return f"yongshen:{school}", desc, ("text", school)
    if cls == "flow":
        year = rng.randrange(2026, 2041)
        return f"flow:{year}", f"{year}年的流年和你的八字有什么关系", ("text", str(year))
    if cls == "star":
        star = rng.choice(STARS)
        return f"star:{star}", f"你的八字里{star}在哪个地支", ("text", star)
    if cls == "prediction":
        return "prediction", "你" + rng.choice(EVENTS), None
    if cls == "domain":
        domain = rng.choice(tuple(DOMAINS))
        return (f"domain:{domain}:2026", f"传统上看八字，你的{DOMAINS[domain]}方面要看什么，你的盘里情况怎样",
                ("domain", domain))  # fmt: skip
    raise ValueError(cls)


def clean(text: str) -> str:
    text = (text or "").strip().split("\n")[0].strip()
    text = re.sub(r"^(问题|我|用户|提问)[：:]\s*", "", text)
    return text.strip("“”\"'「」『』 ")


def passes(question, need, plan):
    """Mechanical filters (fixed before generation): (ok, reason)."""
    if not 6 <= len(question) <= 120:
        return False, "length"
    if any(word in question for word in ECHO):
        return False, "echo"
    if not QUESTION_WORDS.search(question):
        return False, "not_a_question"
    answer = (plan["reference"] or {}).get("answer") if plan["family"] == "fact" else None
    if isinstance(answer, str) and answer in question:
        return False, "contains_answer"
    if isinstance(answer, list) and answer and "".join(answer) in question:
        return False, "contains_answer"
    if need:
        kind = need[0]
        if kind == "position":
            _, pos, symbol = need
            if not (re.search(POS_ZH[pos] + r"(柱|干|支)", question) or symbol in question):
                return False, "no_position"
        elif kind == "domain" and not any(w in question for w in DOMAIN_MARKS[need[1]]):
            return False, "no_domain"
        elif kind == "text" and need[1] not in question:
            return False, "no_target"
    return True, "ok"


def charts(split_years, seed, count):
    blocked = TD.benchmark_exclusions() | TD.contest_exclusions()
    rng = random.Random(f"blind-charts:{seed}")
    years = sorted(split_years)
    out = []
    while len(out) < count:
        day = date(rng.choice(years), 1, 1) + timedelta(days=rng.randrange(365))
        if day in blocked or day.year not in years:
            continue
        chart = eight_char(datetime(day.year, day.month, day.day, rng.randrange(24), rng.randrange(60)))
        pillars = {"year": chart.getYear(), "month": chart.getMonth(), "day": chart.getDay(), "hour": chart.getTime()}
        out.append((pillars, rng.choice(("male", "female"))))
    return out


def build(split, attempts_log=None):
    """Items from a saved attempts log (no model needed)."""
    items = []
    for record in attempts_log:
        if not record["accepted"]:
            continue
        plan = N.plan_for(record["intent"], record["pillars"], record["gender"])
        items.append(make_item(f"P5{split[0].upper()}-blind-{len(items) + 1:05d}", split,
                               f"blind:{Path(WRITER).name}", record["pillars"], record["gender"],
                               record["question"], record["intent"], plan, f"blind-{record['slot']}"))  # fmt: skip
    return items


def generate(split):
    from bazireason.mlx_backend import MLXChatModel

    years_key, seed, per_class = PLAN[split]
    years = TD.split_years()[years_key]
    rng = random.Random(f"blind:{seed}")
    slots = []
    for index, (pillars, gender) in enumerate(charts(years, seed, per_class * len(CLASSES))):
        cls = CLASSES[index % len(CLASSES)]
        intent, desc, need = describe(cls, pillars, rng)
        slots.append({"slot": index, "cls": cls, "intent": intent, "desc": desc, "need": need,
                      "pillars": pillars, "gender": gender, "done": False})  # fmt: skip
    writer = MLXChatModel(WRITER, None, label="writer")
    log = []
    for attempt in range(4):
        todo = [s for s in slots if not s["done"]]
        if not todo:
            break
        convs = []
        for s in todo:
            chart = " ".join(f"{POS_ZH[n]}柱{s['pillars'][n]}" for n in ("year", "month", "day", "hour"))
            sex = "男" if s["gender"] == "male" else "女"
            convs.append([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": ASK.format(chart=chart, sex=sex, desc=s["desc"])}])  # fmt: skip
        texts, _ = writer.generate_batch(convs, 80, temperature=0.8, top_p=0.9, seed=seed + attempt, batch=16)
        for s, raw in zip(todo, texts, strict=True):
            question = clean(raw)
            plan = N.plan_for(s["intent"], s["pillars"], s["gender"])
            ok, reason = passes(question, s["need"], plan)
            s["done"] = ok
            log.append({"slot": s["slot"], "attempt": attempt, "cls": s["cls"], "intent": s["intent"],
                        "pillars": s["pillars"], "gender": s["gender"], "description": s["desc"],
                        "raw": raw, "question": question, "accepted": ok, "reason": reason})  # fmt: skip
    return log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--splits", nargs="+", default=list(PLAN))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        log_path, items_path = OUT / f"{split}_attempts.jsonl", OUT / f"{split}.jsonl"
        if args.check:
            log = [json.loads(x) for x in log_path.read_text(encoding="utf-8").splitlines()]
            payload = "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in build(split, log))
            same = hashlib.sha256(payload.encode()).hexdigest() == hashlib.sha256(items_path.read_bytes()).hexdigest()
            print(f"{items_path}: {'ok' if same else 'MISMATCH'}")
            continue
        if items_path.exists():
            parser.error(f"{items_path} exists; blind splits are never rewritten")
        log = generate(split)
        log_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in log), encoding="utf-8")
        items = build(split, log)
        payload = "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items)
        items_path.write_text(payload, encoding="utf-8")
        reasons = {}
        for r in log:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        print(json.dumps({"split": split, "items": len(items), "attempts": len(log), "reasons": reasons,
                          "sha256": hashlib.sha256(payload.encode()).hexdigest()}))  # fmt: skip
        if split == "dev":
            for item in items[:30]:
                print("  ", item["meta"]["intent"], "|", item["meta"]["question"])


if __name__ == "__main__":
    main()
