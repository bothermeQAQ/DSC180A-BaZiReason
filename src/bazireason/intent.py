"""Intent router: which verified facts a question needs, and in which answer format.

``route(question, pillars, gender, ...)`` returns a plan:
  intent     a label such as "fact:xun_kong", "fact:hidden:month", "strength", "flow:2031",
             "yongshen:扶抑", "ambig:yongshen_open", "prediction", "unknown"
  family     the answer format (and scorer) of the intent
  reference  the rule code's structured answer, or None
  text       the rule code's derivation, or None
  hint       the line that tells the model which JSON keys to give

``plan_for(intent, pillars, gender)`` builds the same plan from a known intent, so the
oracle condition and the routed condition use identical prompts whenever the router is
right. Keyword rules only; no model is involved.
"""

import random
import re

from bazireason import deep_tasks as K
from bazireason import demo_deep
from bazireason import facts as F
from bazireason.indep_rules import HINTS as FACT_HINTS
from bazireason.p2_prompts import OUTPUT_HINTS

FACT_HINTS = dict(
    FACT_HINTS,
    day_master="结论JSON 需含 answer（日主的天干，如 甲）。",
    missing_element="结论JSON 需含 answer（八字中没有的五行列表，都有时为 []）。",
)
POSITION = re.compile(r"(年|月|日|时)(?:柱)?(?:的)?(干|支|天干|地支)")
POS_KEY = {"年": "year", "月": "month", "日": "day", "时": "hour"}
PREDICTION = (
    "什么时候", "哪一年", "哪年", "何时", "几岁", "会不会", "能不能", "有没有可能", "能否",
    "多大", "将来会", "以后会", "未来会", "注定", "命中注定", "多少岁", "哪段时间", "什么时期",
    "哪个阶段",
)  # fmt: skip
LIFE_EVENTS = (
    "结婚", "离婚", "婚", "恋爱", "对象", "发财", "有钱", "富", "升职", "考上", "考研", "怀孕",
    "生孩子", "孩子", "生病", "病", "去世", "死", "寿", "出国", "买房", "中奖", "破财", "官司",
    "成功", "财运", "转运", "好运", "运气", "桃花运",
)  # fmt: skip
COMPUTED_TIMING = ("换大运", "起运", "交运", "大运从", "大运什么时候")
TEN_GODS = (
    "比肩",
    "劫财",
    "食神",
    "伤官",
    "偏财",
    "正财",
    "七杀",
    "偏官",
    "正官",
    "偏印",
    "正印",
    "枭神",
)
JUDGEMENT = (
    "好命",
    "命好",
    "好不好",
    "算不算好",
    "好的命",
    "吉凶",
    "是吉是凶",
    "富贵",
    "上等",
    "下等",
    "幸福",
    "长寿",
)
DOMAINS = {
    "marriage": ("婚", "感情", "配偶", "恋爱", "夫妻", "姻缘", "另一半", "伴侣"),
    "career": ("事业", "工作", "职业", "官运", "升职", "仕途"),
    "wealth": ("财运", "财富", "赚钱", "发财", "收入"),
    "study": ("学业", "考试", "读书", "学习", "升学", "考研"),
    "children": ("子女", "孩子", "生育", "儿女"),
}
STRENGTH_ASK = re.compile(r"(偏强还是偏弱|强还是弱|身强|身弱|旺还是弱|强弱)")


def _position(question: str, part: str | None = None):
    """(position, "stem" | "branch") of the pillar a question names; ``part`` restricts it."""
    for match in POSITION.finditer(question):
        found = "stem" if "干" in match.group(2) else "branch"
        if part is None or found == part:
            return POS_KEY[match.group(1)], found
    # "月柱的甲木", "日柱的酉": the pillar named by its character
    for match in re.finditer(
        r"(年|月|日|时)柱(?:的|里的|上的)?(?:那个)?(?:天干|地支)?[“\"「]?([甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥])(?![甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥])",
        question,
    ):
        found = "stem" if match.group(2) in "甲乙丙丁戊己庚辛壬癸" else "branch"
        if part is None or found == part:
            return POS_KEY[match.group(1)], found
    return None, None


def _fact(kind, pillars, gender, position=None):
    built = F.fact_answer(kind, pillars, position, gender)
    if built is None:
        return None
    answer, text = built
    label = f"fact:{kind}" + (f":{position}" if position and kind not in ("xun_kong",) else "")
    return {
        "intent": label,
        "family": "fact",
        "kind": kind,
        "position": position,
        "reference": {"answer": answer},
        "text": text,
        "hint": FACT_HINTS[kind],
    }


def fact_intent(question: str, pillars: dict, gender):
    """A single computable fact, or None."""
    q = question
    if "旬空" in q or "空亡" in q:
        return _fact("xun_kong", pillars, gender)
    if "本气" in q or ("十神" in q and re.search(r"(年|月|日|时)支", q)):
        position, _ = _position(q, "branch")
        if position:
            return _fact("ten_god_branch", pillars, gender, position)
    if "藏" in q and "十神" not in q and not re.search(r"(不算|不看|不计|不含|不包括)藏", q):
        position, _ = _position(q, "branch")
        if position is None:
            match = re.search(
                r"([子丑寅卯辰巳午未申酉戌亥])[^子丑寅卯辰巳午未申酉戌亥藏]{0,12}藏", q
            )
            if match:
                position = next((n for n in F.PILLARS if pillars[n][1] == match.group(1)), None)
        if position:
            return _fact("hidden", pillars, gender, position)
    if "长生" in q or re.search(r"(帝旺|临官|沐浴|冠带|十二宫|十二运)", q):
        position, _ = _position(q, "branch")
        return _fact("life_stage", pillars, gender, position or "day")
    if re.search(r"(顺排|逆排|顺行|逆行|顺还是逆|往前数|往后数|顺数|逆数|顺着排|倒着排)", q):
        return _fact("luck_direction", pillars, gender)
    if "五行" in q or re.search(r"[金木水火土].{0,2}[金木水火土].{0,2}[金木水火土]", q):
        if re.search(r"(缺|少了|没有哪|不全|全不全|齐不齐)", q):
            return _fact("missing_element", pillars, gender)
        if re.search(r"(几个|多少|数量|个数|统计|分布|各有|数一数|数数|几次|出现)", q):
            return _fact("element_count", pillars, gender)
    asks_god = (
        "十神" in q
        or any(g in q for g in TEN_GODS)
        or re.search(r"(是我|对我|算我|对日主|对日元)的?.{0,6}(什么|哪)", q)
    )
    if asks_god:
        position, _ = _position(q, "branch")
        if position and not _position(q, "stem")[0] and any(g in q for g in TEN_GODS):
            return _fact("ten_god_branch", pillars, gender, position)
        position, _ = _position(q, "stem")
        if position and position != "day":
            return _fact("ten_god_stem", pillars, gender, position)
    if (
        re.search(r"(日主|日元|日干)", q)
        and re.search(r"(是什么|是哪个|是啥|是哪一个|是哪)", q)
        and not re.search(r"(强|弱|格|用神|喜|十神|长生)", q)
    ):
        return _fact("day_master", pillars, gender)
    return None


def is_prediction(question: str) -> bool:
    if any(word in question for word in COMPUTED_TIMING):
        return False
    asks_when = any(word in question for word in PREDICTION)
    return asks_when and any(word in question for word in LIFE_EVENTS)


def prediction_plan():
    return {
        "intent": "prediction",
        "family": "policy",
        "kind": "prediction",
        "reference": {"kind": "speculative_declined"},
        "text": (
            "规则代码只能给出确定的干支结构和传统规则的说法；某件事会不会发生、在哪一年发生，"
            "没有可以核验的依据，所以不作确定预测。可以说明传统上看哪些星和宫位。"
        ),
        "hint": OUTPUT_HINTS["policy"],
    }


def _star_plan(star: str, pillars: dict, gender):
    chart = {"status": "ok", "pillars": pillars, "gender": gender}
    planned = demo_deep.plan(f"这个八字的{star}在哪一支？", chart, first_turn=False)
    return _from_demo(planned, f"这个八字的{star}在哪一支？", 2026)


def _from_demo(planned: dict, question: str, today: int) -> dict:
    kind, task = planned["kind"], planned["task"]
    year = demo_deep._year(question, today)
    if kind == "ambig" and task == "star_basis":
        star = next(s for w, s in demo_deep.STAR_WORDS.items() if w in question)
        label = f"star:{star}"
    elif kind == "ambig":
        label = f"ambig:{task}"
    elif kind == "yongshen":
        label = f"yongshen:{task}"
    elif kind == "domain":
        label = f"domain:{task}:{year or today}"
    elif kind == "flow":
        label = f"flow:{year}"
    else:
        label = kind
    return {
        "intent": label,
        "family": kind,
        "kind": task,
        "reference": planned["reference"],
        "text": planned["text"],
        "hint": K.HINTS[planned["hint"]],
        "question": planned["question"],
    }


INTERACT = re.compile(
    r"(合.{0,2}冲|冲.{0,2}合|冲.{0,2}刑|刑.{0,2}害|干支关系|六合|三合|相冲|相刑|相害|相破|半合)"
)
SCHOOL_WORDS = {
    "扶抑": ("扶抑",),
    "调候": ("调候",),
    "格局": (
        "格局一派",
        "格局派",
        "以月令格局",
        "格局取用",
        "格局之神",
        "顺用",
        "逆用",
        "格局法",
        "按格局",
    ),
}
YONGSHEN_WORDS = ("用神", "喜用", "喜忌", "喜什么", "调候", "扶抑")


def _school(question: str):
    for school, words in SCHOOL_WORDS.items():
        if any(word in question for word in words):
            return school
    return None


def route(question: str, pillars: dict, gender=None, today: int = 2026, first_turn=True) -> dict:
    """The plan for one user question about a verified chart (keyword rules, fixed order)."""
    q = question
    numbered = len(demo_deep.NUMBERED.findall(q)) >= 2
    chart = {"status": "ok", "pillars": pillars, "gender": gender or "male"}
    if not numbered:
        if is_prediction(q):
            return prediction_plan()
        if any(word in q for word in JUDGEMENT) or re.search(r"(不错|好|差|坏|苦)的命", q):
            return plan_for("ambig:no_rule", pillars, gender)
        if any(word in q for word in demo_deep.OVERVIEW_WORDS):
            return plan_for("overview", pillars, gender)
        year = demo_deep._year(q, today)
        if year is not None and (
            "流年" in q or re.search(r"[0-9]{4}\s*年", q) or "今年" in q or "明年" in q
        ):
            if not any(w for words in demo_deep.DOMAIN_WORDS.values() for w in words if w in q):
                return plan_for(f"flow:{year}", pillars, gender)
        if STRENGTH_ASK.search(q) and not any(w in q for w in YONGSHEN_WORDS):
            return plan_for("strength", pillars, gender)
        stems_named = len(set(re.findall(r"(年|月|时)(?:柱)?(?:的)?(?:天)?干", q)))
        if stems_named >= 2 or ("十神" in q and re.search(r"(组合|分别|各是|各自)", q)):
            return plan_for("gods", pillars, gender)
        domain = next((d for d, words in DOMAINS.items() if any(w in q for w in words)), None)
        if domain and (gender or domain != "marriage"):
            return plan_for(
                f"domain:{domain}:{demo_deep._year(q, today) or today}", pillars, gender
            )
        school = _school(q)
        if school and (any(w in q for w in YONGSHEN_WORDS) or school == "格局"):
            return plan_for(f"yongshen:{school}", pillars, gender)
        if any(w in q for w in YONGSHEN_WORDS):
            return plan_for("ambig:yongshen_open", pillars, gender)
        if INTERACT.search(q):
            return plan_for("interact", pillars, gender)
        if any(word in q for word in demo_deep.PATTERN_WORDS):
            return plan_for("pattern", pillars, gender)
        fact = fact_intent(q, pillars, gender)
        if fact:
            return fact
    planned = demo_deep.plan(q, chart, today=today, first_turn=True)
    if planned and planned["kind"] != "overview":
        return _from_demo(planned, q, today)
    if any(word in q for word in demo_deep.PREDICT_WORDS) and any(
        w for words in demo_deep.DOMAIN_WORDS.values() for w in words if w in q
    ):
        return prediction_plan()
    return {"intent": "unknown", "family": "unknown", "kind": None, "reference": None,
            "text": None, "hint": OUTPUT_HINTS["rule"]}  # fmt: skip


def plan_for(intent: str, pillars: dict, gender=None) -> dict:
    """The plan of a known intent (the oracle condition)."""
    head, _, rest = intent.partition(":")
    rng = random.Random(0)
    if head == "fact":
        kind, _, position = rest.partition(":")
        return _fact(kind, pillars, gender, position or None)
    if head == "prediction":
        return prediction_plan()
    if head == "unknown":
        return route("", pillars, gender)
    if head == "star":
        return _star_plan(rest, pillars, gender)
    builders = {
        "strength": lambda: K.strength_task(pillars, rng),
        "pattern": lambda: K.pattern_task(pillars, rng),
        "interact": lambda: K.interact_task(pillars, rng),
        "gods": lambda: K.gods_task(pillars, rng),
        "overview": lambda: K.overview_task(pillars, rng, 0, gender or "male"),
        "yongshen": lambda: K.yongshen_task(pillars, rng, 0, rest),
        "flow": lambda: K.flow_task(pillars, rng, 0, int(rest)),
        "domain": lambda: K.domain_task(
            pillars, rng, 0, gender or "male", rest.split(":")[0], int(rest.split(":")[1])
        ),
    }  # fmt: skip
    if head == "ambig":
        _, question, text, reference = K.ambig_task(rest, pillars, rng, 0, gender or "male")
        hint = "ambig"
        family = "ambig"
    else:
        question, text, reference = builders[head]()
        hint = {"gods": "gods", "overview": "overview"}.get(head, head)
        family = head
    kind = rest.split(":")[0] if head == "domain" else (rest or head)
    return {
        "intent": intent,
        "family": family,
        "kind": kind,
        "reference": reference,
        "text": text,
        "hint": K.HINTS[hint],
        "question": question,
    }
