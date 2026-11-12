"""Phase 3 additions to the chat demo.

``plan`` maps a free Chinese question to the closest request format the Phase 3
adapter was trained on and, in the same step, computes the answer the rule code
gives. The model still writes every reply; the application chooses the wording
of the request and afterwards compares the reply's structured conclusion with
the rule code, so a wrong table lookup is shown to the user instead of trusted.

Nothing here predicts an event. Questions that ask for a definite personal
outcome are left to the Phase 2 "policy" route, which declines.
"""

import json
import random
import re
from datetime import datetime

from bazireason import deep_rules as D
from bazireason import deep_tasks as K
from bazireason import rules as R
from bazireason.charttools import chart_tool_text
from bazireason.p2_prompts import extract_final_json
from bazireason.p3_eval import GRADED, _keys_match, score_ambig
from bazireason.train_tasks import GENDER_ZH, chart_line

FORMAT_NAMES = {
    "overview": "整体结构",
    "yongshen": "用神（指定流派）",
    "ambig": "无唯一答案的提问",
    "flow": "流年分析",
    "domain": "传统看法（某一方面）",
    "strength": "强弱计数",
    "pattern": "透干取格",
    "interact": "合冲刑害",
    "gods": "十神与组合",
    "claims": "核对说法",
}
DOMAIN_WORDS = {
    "marriage": ("婚", "感情", "配偶", "恋爱", "对象", "夫妻", "姻缘"),
    "career": ("事业", "工作", "职业", "官运", "升职", "仕途"),
    "wealth": ("财运", "财富", "赚钱", "发财", "收入", "钱"),
    "study": ("学业", "考试", "读书", "学习", "升学", "考研"),
    "children": ("子女", "孩子", "生育", "儿女"),
}
PREDICT_WORDS = (
    "什么时候",
    "哪一年",
    "哪年",
    "会不会",
    "能不能",
    "几岁",
    "何时",
    "是不是",
    "有没有可能",
)
STAR_WORDS = {"桃花": "桃花", "咸池": "桃花", "驿马": "驿马", "华盖": "华盖"}
NO_RULE_WORDS = (
    "好不好",
    "富贵",
    "上等",
    "下等",
    "吉凶",
    "是吉是凶",
    "命好",
    "好命",
    "幸福",
    "长寿",
)
OVERVIEW_WORDS = (
    "整体",
    "全面",
    "完整",
    "系统地",
    "总体",
    "综合",
    "解读一下",
    "讲一遍",
    "总的来说",
)
STRENGTH_WORDS = ("强弱", "身强", "身弱", "得令", "得地", "得势", "偏强", "偏弱", "旺弱")
PATTERN_WORDS = ("格局", "什么格", "透干", "取格", "定格")
INTERACT_WORDS = ("合冲", "刑害", "相冲", "相合", "六合", "三合", "相刑", "相害", "相破", "冲合")
GODS_WORDS = ("十神", "食神制杀", "伤官见官", "枭神夺食", "官杀混杂", "官印相生", "杀印相生")
NEEDS_ZH = {
    "yongshen_school": "说明按哪一派",
    "hour": "出生时辰",
    "gender": "性别",
    "star_basis": "说明以年支还是日支查",
}
RELATIVE_YEARS = {"去年": -1, "今年": 0, "明年": 1, "后年": 2}
YEAR = re.compile(r"(?<![0-9])((?:19|20|21)[0-9]{2})(?![0-9])")
NUMBERED = re.compile(r"^\s*[0-9]+[.、．)]\s*\S", re.M)
RULE_KEYWORDS = {
    "R-SS-02": ("生克", "五类", "比劫", "食伤", "官杀"),
    "R-SS-03": ("食神制杀", "伤官见官", "枭神夺食", "组合", "官杀混杂", "相生", "十神"),
    "R-TG-01": ("透干", "通根", "有根"),
    "R-GJ-02": ("格局", "取格", "什么格", "透干"),
    "R-YS-01": ("扶抑", "用神", "喜用", "喜忌", "强弱"),
    "R-YS-02": ("调候", "用神", "喜用"),
    "R-YS-03": ("格局", "用神", "顺用", "逆用"),
    "R-SM-01": ("禄", "阳刃", "羊刃", "贵人"),
    "R-SM-02": ("驿马", "桃花", "咸池", "华盖"),
    "R-SY-01": ("流年", "伏吟", "天克地冲", "太岁", "岁运"),
    "R-LQ-02": ("配偶宫", "子女宫", "宫位", "婚", "子女"),
    "R-YD-01": ("引动", "流年", "婚", "事业", "财运", "学业", "子女"),
}


def card_view(card: dict) -> dict:
    """A Phase 3 rule card in the shape the page shows (quotations become the check line)."""
    if card["quotations"]:
        source = "原文（维基文库录文）" + "；".join(
            f"{q['work']}·{q['section']}：“{q['text']}”" for q in card["quotations"]
        )
    else:
        works = "、".join(w["work"] for w in card["works_cited_without_quotation"])
        source = f"未引原文，仅注明出处：{works}"
    return {
        "id": card["id"],
        "kind": card["kind"],
        "title": card["title"],
        "statement": card["statement"],
        "cross_check": source,
    }


def relevant_cards(text: str, limit=4) -> list:
    hits = []
    for rule_id, words in RULE_KEYWORDS.items():
        score = sum(word in text for word in words)
        if score:
            hits.append((score, rule_id))
    hits.sort(reverse=True)
    return [card_view(D.RULES_BY_ID[rule_id]) for _, rule_id in hits[:limit]]


def deep_lines(pillars: dict) -> list:
    """The Phase 3 derived facts of a chart as labelled Chinese lines (rule code only)."""
    return K.overview_task(pillars, random.Random(0))[1].split("\n")


def _year(message: str, today: int):
    found = YEAR.search(message)
    if found:
        return int(found.group(1))
    for word, offset in RELATIVE_YEARS.items():
        if word in message:
            return today + offset
    return None


def _school(message: str):
    if "扶抑" in message:
        return "扶抑"
    if "调候" in message:
        return "调候"
    if any(word in message for word in ("格局用神", "格局法", "格局一派", "按格局", "以格局")):
        return "格局"
    return None


def plan(message: str, chart: dict, today: int | None = None, first_turn: bool = True):
    """The trained Phase 3 request for a question, or None when no format fits.

    Returns ``{"kind", "task", "question", "hint", "tool", "reference", "text"}``;
    ``reference`` is the structured answer of the rule code and ``text`` its
    written derivation.
    """
    if chart.get("status") != "ok":
        return None
    today = today or datetime.now().year
    pillars, gender = chart["pillars"], chart.get("gender")
    rng = random.Random(0)

    def made(kind, task, built, hint, tool=False):
        question, text, reference = built
        return {
            "kind": kind,
            "task": task,
            "question": question,
            "hint": hint,
            "tool": tool and first_turn and gender is not None,
            "reference": reference,
            "text": text,
        }

    def ambiguous(task, variant=0):
        _, question, text, reference = K.ambig_task(task, pillars, rng, variant, gender or "male")
        return made("ambig", task, (question, text, reference), "ambig")

    if len(NUMBERED.findall(message)) >= 2:
        lead = "有人对这个八字下了下面几条判断，请逐条按规则核对，标明正确、错误还是无法验证，错误的给出改正："
        lines = "\n".join(line.strip() for line in message.splitlines() if NUMBERED.match(line))
        return made("claims", "claims", (lead + "\n" + lines, None, None), "claims")
    if any(word in message for word in OVERVIEW_WORDS) and gender is not None and first_turn:
        return made("overview", "overview", K.overview_task(pillars, rng), "overview", tool=True)
    if any(word in message for word in ("用神", "喜用", "喜忌", "调候", "扶抑")):
        school = _school(message)
        if school is None:
            return ambiguous("yongshen_open")
        return made("yongshen", school, K.yongshen_task(pillars, rng, 0, school), "yongshen")
    if any(word in message for word in NO_RULE_WORDS):
        plan_ = ambiguous("no_rule")
        plan_["question"] = message
        return plan_
    for word, star in STAR_WORDS.items():
        if word in message:
            by_year = D.star_branches(star, pillars, "year")
            by_day = D.star_branches(star, pillars, "day")
            listing = "、".join(f"{k}在{v}" for k, v in D.TRIPLE_STARS[star].items())
            year_b, day_b = pillars["year"][1], pillars["day"][1]
            text = (
                f"{star}按三合局查：{listing}。古书以年支查，后来也有以日支查的。"
                f"年支{year_b}属{D._TRIPLE_OF[year_b]}，{star}在{by_year}；"
                f"日支{day_b}属{D._TRIPLE_OF[day_b]}，{star}在{by_day}。"
            )
            if by_year == by_day:
                reference = {"determinate": True, "needs": [], "answer": by_year}
            else:
                reference = {
                    "determinate": False,
                    "needs": ["star_basis"],
                    "by_basis": {"year": by_year, "day": by_day},
                }
            return made(
                "ambig", "star_basis", (f"这个八字的{star}在哪一支？", text, reference), "ambig"
            )
    predictive = any(word in message for word in PREDICT_WORDS)
    year = _year(message, today)
    domain = next(
        (key for key, words in DOMAIN_WORDS.items() if any(word in message for word in words)),
        None,
    )
    if predictive and domain:
        return None  # a request for a definite personal outcome: the declining route answers
    if domain:
        if gender is None:
            return ambiguous("gender_needed", 1) if domain == "marriage" else None
        built = K.domain_task(pillars, rng, 0, gender, domain, year or today)
        return made("domain", domain, built, "domain")
    if year is not None and ("年" in message or "流年" in message) and not predictive:
        cycles = chart["luck"]["cycles"] if "luck" in chart and first_turn else None
        built = K.flow_task(pillars, rng, 0, year, cycles)
        return made("flow", "flow", built, "flow", tool=True)
    if any(word in message for word in STRENGTH_WORDS):
        return made("strength", "strength", K.strength_task(pillars, rng), "strength")
    if any(word in message for word in PATTERN_WORDS):
        return made("pattern", "pattern", K.pattern_task(pillars, rng), "pattern")
    if any(word in message for word in INTERACT_WORDS):
        return made("interact", "interact", K.interact_task(pillars, rng), "interact")
    if any(word in message for word in GODS_WORDS) and not re.search(
        r"[年月时]干.{0,12}十神", message
    ):
        return made("gods", "gods", K.gods_task(pillars, rng), "gods")
    return None


def turn_text(plan_: dict, chart: dict, first_turn: bool) -> str:
    """The user turn sent to the model for a planned question."""
    if not first_turn:
        head = "还是这个八字。"
    elif plan_["tool"]:
        head = chart_tool_text(chart, flow_years=False) + "\n"
        if plan_["kind"] == "claims":
            head += "以上是核验过的排盘。"
    else:
        gender = chart.get("gender")
        head = chart_line(chart["pillars"]) + (f"，{GENDER_ZH[gender]}命。" if gender else "。")
        if plan_["kind"] == "claims":
            head += "\n"
    return head + plan_["question"] + "\n" + K.HINTS[plan_["hint"]]


def _shown(plan_: dict) -> str:
    """A short readable form of the rule code's structured answer."""
    ref, kind = plan_["reference"], plan_["kind"]
    if kind == "strength":
        a = ref["answer"]
        flags = "、".join(f"{k}{'是' if a[k] else '否'}" for k in ("得令", "得地", "得势"))
        return f"{flags} → {a['summary']}"
    if kind == "pattern":
        return ref["answer"]
    if kind == "interact":
        return "；".join(R.relation_text(key) for key in ref["answer"]) or "无"
    if kind == "flow":
        special = (
            "、".join(
                f"{key.split(':')[0]}（{R.PILLAR_ZH[key.split(':')[1]]}）" if ":" in key else key
                for key in ref["special"]
            )
            or "无"
        )
        return f"{ref['pillar']}年，流年天干为{ref['ten_god']}，特殊关系：{special}"
    if kind == "domain":
        years = "、".join(map(str, ref["trigger_years"])) or "无"
        return f"所看十神 {'、'.join(ref['gods'])}；按约定被引动的年份：{years}"
    if kind == "gods":
        stems = "、".join(f"{R.POSITION_ZH[k]}干{v}" for k, v in ref["stems"].items())
        return f"{stems}；组合：{'、'.join(ref['configs']) or '无'}"
    if kind == "overview":
        return f"{ref['strength']}，{ref['pattern']}，组合：{'、'.join(ref['configs']) or '无'}"
    if kind == "ambig" and not ref["determinate"]:
        needs = "、".join(NEEDS_ZH.get(need, need) for need in ref["needs"]) or "规则不能决定"
        return f"没有唯一答案（需要：{needs}）"
    return json.dumps(ref.get("answer", ref), ensure_ascii=False)


def check(plan_: dict, reply: str):
    """Compare a reply's structured conclusion with the rule code for a planned question."""
    reference = plan_["reference"]
    if reference is None:
        return None
    answer = extract_final_json(reply)
    if plan_["kind"] == "ambig":
        agrees = score_ambig(reference, answer)["correct"]
    else:
        keys = GRADED.get(plan_["kind"]) or tuple(reference)
        agrees = _keys_match(reference, answer, keys)
    return {
        "label": FORMAT_NAMES[plan_["kind"]],
        "engine": _shown(plan_),
        "agrees": bool(agrees),
        "detail": plan_["text"],
    }
