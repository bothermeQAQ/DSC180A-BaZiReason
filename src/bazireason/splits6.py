"""Phase 6 year blocks, earlier evaluation charts, and two new evaluation item types.

Year blocks (``phase6_years``): Phase 6 training charts come from the training years of
``train_data.split_years``. The validation block is split in two, so development and
validation use separate years; the sealed set uses the test block. Charts that appeared in
any earlier evaluation split are excluded from the Phase 6 validation and sealed sets.

Item types added in Phase 6 (evaluation only; no training row ever uses them):

* ``combo``: held-out combinations of rules the model was trained on separately
  (flow year + life stage, 旬空 + positions, hidden stems + revealed stems,
  Ten Gods + counting). Labels from ``bazireason.rules``.
* ``cf``: counterfactual pairs. The same question on two charts that differ in one
  character, chosen so that the answer changes. A model that computes gets both right.
"""

import json
import random
import re
from contextlib import contextmanager
from pathlib import Path

from bazireason import intent as N
from bazireason import rules as R
from bazireason import train_data as TD
from bazireason.conventions import BRANCHES, CYCLE, STEMS
from bazireason.deep_rules import CLASS_GODS, GOD_CLASS
from bazireason.p2_prompts import P2_SYSTEM, extract_final_json
from bazireason.p5_eval import head_line

PILLARS = ("year", "month", "day", "hour")
POS_ZH = {"year": "年", "month": "月", "day": "日", "hour": "时"}
FOUR = re.compile(r"([甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥])")


def phase6_years() -> dict:
    base = TD.split_years()
    dev = {y for y in base["valid"] if (y - 1910) % 20 == 0}
    return {
        "train": base["train"],
        "dev6": dev,
        "valid6": base["valid"] - dev,
        "sealed6": base["test"],
    }


@contextmanager
def years_as(block: str):
    """Make the Phase 2-4 generators' "valid" and "test" pools draw from one Phase 6 block."""
    years = phase6_years()[block]
    original = TD.split_years
    TD.split_years = lambda: {"train": original()["train"], "valid": years, "test": years}
    try:
        yield
    finally:
        TD.split_years = original


def chart_key(pillars: dict) -> str:
    return "".join(pillars[p] for p in PILLARS)


def earlier_eval_charts() -> set:
    """Every four-pillar chart in an earlier validation, test, sealed, blind or dialogue split."""
    paths = [*Path("data/phase4").glob("valid/*.jsonl"), *Path("data/phase4").glob("final_test/*.jsonl"),
             *Path("data/phase5").glob("valid/*.jsonl"), *Path("data/phase5").glob("sealed/*.jsonl"),
             *Path("data/phase5").glob("blind/*.jsonl"), *Path("data/phase5").glob("convo/*.jsonl"),
             *Path("data/phase3").glob("dev_v1/*.jsonl")]  # fmt: skip
    seen = set()
    for path in paths:
        if "attempts" in path.name:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            pillars = (row.get("meta") or {}).get("pillars") or row.get("pillars")
            if isinstance(pillars, dict) and len(pillars) == 4:
                seen.add(chart_key(pillars))
                continue
            users = [m["content"] for m in row.get("messages", []) if m.get("role") == "user"]
            for text in users[:1]:
                found = FOUR.findall(text)
                if len(found) >= 4:
                    seen.add("".join(found[:4]))
    return seen


# ---- held-out combinations ----------------------------------------------------------------------

COMBO_KINDS = ("flow_stage", "kong_positions", "revealed", "god_count")
COMBO_HINTS = {
    "flow_stage": "结论JSON 需含 answer（十二长生的阶段名）。",
    "kong_positions": '结论JSON 需含 answer（出现旬空地支的柱名列表，如 ["年柱"]；都没有时为 []）。',
    "revealed": "结论JSON 需含 answer（透出的天干列表；都不透时为 []）。",
    "god_count": "结论JSON 需含 answer（个数，整数）。",
}
COMBO_WORDS = {
    "flow_stage": ("{year}年流年的地支，对日主来说处在十二长生的哪个阶段？",
                   "日主在{year}年那个流年地支上，是十二长生里的哪一个阶段？"),
    "kong_positions": ("日柱所在旬的两个旬空地支，有没有出现在年、月、时三柱的地支上？出现在哪几柱？",
                       "这个八字的空亡地支，在年柱、月柱、时柱的地支里出现了几次，分别是哪几柱？"),
    "revealed": ("{p}支{b}所藏的天干里，哪些在年、月、时三个天干上透出了？",
                 "看{p}支{b}的藏干，有哪几个也出现在年干、月干、时干上？"),
    "god_count": ("年干、月干、时干这三个天干里，属于{cls}（{gods}）的有几个？",
                  "除日干外的三个天干，有几个是{cls}（{gods}）？"),
}  # fmt: skip
COMBO_CLASSES = ("财", "官杀", "食伤")


def flow_pillar(year: int) -> str:
    return CYCLE[(year - 1984) % 60]


def combo_reference(kind: str, pillars: dict, arg) -> object:
    day = pillars["day"][0]
    if kind == "flow_stage":
        return R.life_stage(day, flow_pillar(arg)[1])
    if kind == "kong_positions":
        kong = R.xun_kong(pillars["day"])
        return [f"{POS_ZH[p]}柱" for p in ("year", "month", "hour") if pillars[p][1] in kong]
    if kind == "revealed":
        stems = {pillars[p][0] for p in ("year", "month", "hour")}
        return [s for s in R.HIDDEN_STEMS[pillars[arg][1]] if s in stems]
    if kind == "god_count":
        return sum(
            GOD_CLASS[R.ten_god(day, pillars[p][0])] == arg for p in ("year", "month", "hour")
        )
    raise ValueError(kind)


def combo_item(item_id, split, kind, pillars, gender, rng, group):
    arg, fill = None, {}
    if kind == "flow_stage":
        arg = rng.randrange(2026, 2046)
        fill = {"year": arg}
    elif kind == "revealed":
        arg = rng.choice(("year", "day", "hour"))
        fill = {"p": POS_ZH[arg], "b": pillars[arg][1]}
    elif kind == "god_count":
        arg = rng.choice(COMBO_CLASSES)
        fill = {"cls": arg, "gods": "、".join(CLASS_GODS[arg])}
    question = rng.choice(COMBO_WORDS[kind]).format(**fill)
    reference = combo_reference(kind, pillars, arg)
    user = head_line(pillars, gender) + question + "\n" + COMBO_HINTS[kind]
    return {
        "id": item_id, "family": "combo", "task": f"p6_combo_{kind}", "split": split, "mode": "none",
        "group_id": group,
        "messages": [{"role": "system", "content": P2_SYSTEM}, {"role": "user", "content": user}],
        "meta": {"source": "combo", "pillars": pillars, "gender": gender, "question": question,
                 "intent": f"combo:{kind}" + (f":{arg}" if arg is not None else ""), "kind": kind,
                 "expected": {"answer": reference}, "hint": COMBO_HINTS[kind]},
    }  # fmt: skip


def _combo_norm(kind, value):
    if kind == "god_count":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if kind == "flow_stage":
        return value.strip() if isinstance(value, str) else None
    if isinstance(value, str):
        value = re.findall(
            r"[年月日时]柱" if kind == "kong_positions" else "[甲乙丙丁戊己庚辛壬癸]", value
        )
    if kind == "kong_positions" and isinstance(value, list):
        value = [v if str(v).endswith("柱") else f"{v}柱" for v in value]
    return sorted(value) if isinstance(value, list) else None


def score_combo(item, text) -> dict:
    answer = extract_final_json(text or "")
    kind = item["meta"]["kind"]
    got = answer.get("answer") if isinstance(answer, dict) else None
    want = item["meta"]["expected"]["answer"]
    ok = got is not None and _combo_norm(kind, got) == _combo_norm(kind, want)
    return {"id": item["id"], "family": "combo", "intent": item["meta"]["intent"], "source": "combo",
            "group": f"combo:{item['group_id']}", "parsed": isinstance(answer, dict), "correct": bool(ok)}  # fmt: skip


# ---- counterfactual pairs -------------------------------------------------------------------------

CF_INTENTS = ("fact:xun_kong", "fact:element_count", "fact:ten_god_stem", "fact:life_stage",
              "strength", "pattern", "interact")  # fmt: skip
CF_WORDS = {
    "fact:xun_kong": "这个八字的旬空（空亡）是哪两个地支？",
    "fact:element_count": "只数八个字、不算藏干，这个八字的五行各有几个？",
    "fact:ten_god_stem": "{p}干对日主来说是什么十神？",
    "fact:life_stage": "日主在{p}支上处于十二长生的哪个阶段？",
}


def neighbours(pillar: str) -> list:
    """Valid pillars that differ from ``pillar`` in exactly one character."""
    out = [s + pillar[1] for s in STEMS if s != pillar[0] and s + pillar[1] in CYCLE]
    out += [pillar[0] + b for b in BRANCHES if b != pillar[1] and pillar[0] + b in CYCLE]
    return out


def cf_pair(base_id, split, intent_head, pillars, gender, rng):
    """Two items, or None when no one-character change alters the answer."""
    position = None
    intent = intent_head
    if intent_head in ("fact:ten_god_stem", "fact:life_stage"):
        position = rng.choice(("year", "month", "hour") if "stem" in intent_head else PILLARS)
        intent = f"{intent_head}:{position}"
    plan_a = N.plan_for(intent, pillars, gender)
    if plan_a is None or plan_a.get("reference") is None:
        return None
    where = ["day"] if intent_head == "fact:xun_kong" else (
        [position] if position and position != "day" else ["year", "month", "hour"])  # fmt: skip
    options = [(p, n) for p in where for n in neighbours(pillars[p])]
    rng.shuffle(options)
    for p, new in options:
        if intent_head == "fact:life_stage" and p == "day" and new[0] != pillars["day"][0]:
            continue
        other = dict(pillars, **{p: new})
        plan_b = N.plan_for(intent, other, gender)
        if plan_b and plan_b["reference"] != plan_a["reference"]:
            items = []
            for tag, chart, plan in (("a", pillars, plan_a), ("b", other, plan_b)):
                q = CF_WORDS.get(intent_head, plan.get("question") or "").format(
                    p=POS_ZH.get(position or "", "")
                )
                user = head_line(chart, gender) + q + "\n" + plan["hint"]
                items.append({
                    "id": f"{base_id}{tag}", "family": plan["family"], "task": f"p6_cf_{plan['family']}",
                    "split": split, "mode": "none", "group_id": base_id,
                    "messages": [{"role": "system", "content": P2_SYSTEM}, {"role": "user", "content": user}],
                    "meta": {"source": "counterfactual", "pillars": chart, "gender": gender, "question": q,
                             "intent": intent, "kind": plan["kind"], "expected": plan["reference"],
                             "pair": base_id, "changed": p},
                })  # fmt: skip
            return items
    return None


def random_pillars(rng) -> dict:
    """A chart with consistent pillars drawn uniformly (for combo and counterfactual items)."""
    while True:
        year, day = rng.choice(CYCLE), rng.choice(CYCLE)
        month = R.month_pillar_for(year[0], rng.choice(BRANCHES))
        hour = R.hour_pillar_for(day[0], rng.choice(BRANCHES))
        return {"year": year, "month": month, "day": day, "hour": hour}


__all__ = [
    "COMBO_KINDS",
    "CF_INTENTS",
    "phase6_years",
    "years_as",
    "earlier_eval_charts",
    "score_combo",
]
_ = random
