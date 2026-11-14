"""Phase 6 evaluation ladder: the same item asked at several levels of help.

  L0   the question and the four pillars (indep items keep their own chart format)
  L1   plus verified raw facts: each symbol's element, polarity and hidden stems; nothing derived
  L2   plus the relevant rule text; it depends only on the question type, never on the chart
  L3   plus the rule code's conclusion; L3w gives a wrong conclusion in the same words
  L4   the model's own L0 reply and a request to check it (a: re-check; b: "there is an
       error"; c: "this field is wrong"); no value is ever given
  L5   the Phase 5 hybrid system (routing, repair with values, engine fallback); run by
       scripts/phase5 code and always reported apart

No repair, fallback, loop guard or routing touches L0-L4. ``leaks`` and ``rule_text`` are the
mechanical guarantees that L1 and L2 contain no answer (tests/test_phase6.py).
"""

import copy
import json
import random
import re

from bazireason import deep_rules as D
from bazireason import rules as R
from bazireason import splits6 as S
from bazireason.conventions import BRANCHES, CYCLE, STEMS
from bazireason.p2_eval import score_row
from bazireason.p2_prompts import P2_SYSTEM, extract_final_json
from bazireason.p5_eval import bare_question, head_line, score5

PILLARS = ("year", "month", "day", "hour")
POS = R.POSITION_ZH
LEVELS = ("L0", "L1", "L2", "L3", "L3w")
CHECKS = {
    "L4a": "请你再仔细检查一遍上面的推导和结论。如果发现错误，就改正并给出新的结论JSON；如果没有错误，就保持原来的结论，并再写一遍结论JSON。",
    "L4b": "上面的回答里有错误。请找出错在哪里，改正后给出完整的新结论JSON。",
    "L4c": "上面结论JSON里的「{field}」不对。请重新推导这一项，并给出完整的新结论JSON。",
}
DERIVED_WORDS = re.compile(
    "比肩|劫财|食神|伤官|偏财|正财|七杀|正官|偏印|正印|十神|六合|三合|半合|三会|相冲|相刑|自刑|相害|相破|相合|"
    "旬空|空亡|偏强|偏弱|得令|得地|得势|格|长生|沐浴|冠带|临官|帝旺|衰|病|死|墓|绝|胎|养|用神|喜|"
    "桃花|驿马|华盖|贵人|阳刃|禄|顺排|逆排|太岁|伏吟|[0-9]"
)


# ---- the base prompt -------------------------------------------------------------------------------


def user_text(item) -> str:
    """The L0 user turn: head line + question + hint (indep items keep their original text)."""
    if item["family"] in ("indep",) or item["meta"].get("label_source"):
        return (
            item["messages"][-1]["content"]
            if item["messages"][-1]["role"] == "user"
            else item["messages"][-2]["content"]
        )
    meta = item["meta"]
    hint = meta.get("hint") or _hint_of(item)
    return head_line(meta["pillars"], meta.get("gender")) + bare_question(item) + "\n" + hint


def _hint_of(item) -> str:
    user = next(m["content"] for m in reversed(item["messages"]) if m["role"] == "user")
    if item["messages"][-1]["role"] == "assistant":
        user = item["messages"][-2]["content"]
    return user.rsplit("\n", 1)[1]


# ---- L1: raw facts -----------------------------------------------------------------------------


def raw_sheet(pillars: dict, gender, intent: str) -> str:
    hidden = not intent.startswith("fact:hidden")
    sex = {"male": "；男命", "female": "；女命"}.get(gender, "")
    stems = "；".join(
        f"{POS[p]}干{pillars[p][0]}（{R.STEM_ELEMENT[pillars[p][0]]}，{'阳' if R.is_yang(pillars[p][0]) else '阴'}）"
        for p in PILLARS
    )
    branches = "；".join(
        f"{POS[p]}支{pillars[p][1]}（{R.BRANCH_ELEMENT[pillars[p][1]]}，{'阳' if R.is_yang(pillars[p][1]) else '阴'}"
        + (f"；藏{'、'.join(R.HIDDEN_STEMS[pillars[p][1]])}" if hidden else "") + "）"
        for p in PILLARS
    )  # fmt: skip
    return (
        "[已核验的原始事实]（只有每个干支本身的五行、阴阳和藏干，没有任何推导结论）\n"
        f"四柱：年柱{pillars['year']} 月柱{pillars['month']} 日柱{pillars['day']} 时柱{pillars['hour']}{sex}\n"
        f"天干：{stems}\n地支：{branches}\n"
    )


def leaks(sheet: str) -> list:
    """Derived words in a raw-fact sheet (must be empty)."""
    return DERIVED_WORDS.findall(sheet.split("\n", 1)[1])


# ---- L2: rule text (depends only on the question type) ------------------------------------------------

GEN = "五行相生：木生火、火生土、土生金、金生水、水生木；相克：木克土、土克水、水克火、火克金、金克木。"
ELEMENTS = "天干五行与阴阳：甲阳木、乙阴木、丙阳火、丁阴火、戊阳土、己阴土、庚阳金、辛阴金、壬阳水、癸阴水。地支五行：寅卯木、巳午火、申酉金、亥子水、辰戌丑未土。"
TEN_GODS = ("十神：以日主为我。同我者为比劫，我生者为食伤，我克者为财，克我者为官杀，生我者为印；"
            "与日主阴阳相同者为比肩、食神、偏财、七杀、偏印，阴阳相异者为劫财、伤官、正财、正官、正印。")  # fmt: skip


def _hidden_table() -> str:
    return (
        "藏干表（本气在前）："
        + "；".join(f"{b}藏{'、'.join(R.HIDDEN_STEMS[b])}" for b in BRANCHES)
        + "。"
    )


def _stage_rule() -> str:
    starts = "、".join(f"{s}{R._STAGE_START[s]}" for s in STEMS)
    return f"十二长生次序：{'、'.join(R.LIFE_STAGES)}。阳干从长生之地顺行，阴干逆行。各干长生之地：{starts}。"


def _xun_rule() -> str:
    heads = [CYCLE[i] for i in range(0, 60, 10)]
    table = "、".join(f"{h}旬空{R.xun_kong(h)}" for h in heads)
    return f"旬空：六十甲子每十个为一旬，以甲开头；一旬十个干支用到十个地支，剩下两个为旬空，按日柱所在的旬取。{table}。"


def _relation_rule() -> str:
    six = "、".join(f"{k}合{v}" for k, v in R.BRANCH_SIX_COMBINES.items())
    return (
        "天干五合：甲己、乙庚、丙辛、丁壬、戊癸；天干相冲：甲庚、乙辛、丙壬、丁癸。"
        f"地支六合：{six}；六冲：{'、'.join(R.BRANCH_CLASHES)}；相刑：{'、'.join(R.PUNISHMENT_PAIRS)}，自刑：{'、'.join(R.SELF_PUNISHING)}；"
        f"相害：{'、'.join(R.BRANCH_HARMS)}；相破：{'、'.join(R.BRANCH_BREAKS)}；"
        f"三合局：{'、'.join(f'{k}{v}' for k, v in R.TRIPLE_HARMONIES.items())}，两支含中间一支为半合；"
        f"三会方：{'、'.join(f'{k}{v}' for k, v in R.DIRECTIONAL_TRIPLES.items())}。三合、三会要三支齐全。"
    )


RULES = {
    "gods": TEN_GODS + GEN + ELEMENTS + _hidden_table() + "十神组合：" + "；".join(f"{n}（{'或'.join(a)}＋{'或'.join(b)}）" for n, a, b, _ in D.CONFIGS) + "。",
    "ten_god": TEN_GODS + GEN + ELEMENTS,
    "hidden": _hidden_table(),
    "life_stage": _stage_rule(),
    "xun_kong": _xun_rule(),
    "luck": "大运顺逆：阳年（年干为阳）男命、阴年女命顺排；阴年男命、阳年女命逆排。" + ELEMENTS,
    "count": "五行计数的约定：只数四柱的八个字（四个天干、四个地支），不算藏干。" + ELEMENTS,
    "strength": ("强弱的计数约定：得令＝日主五行在月令（月支本气五行）为旺或相（同我为旺、月令生我为相、我生月令为休、我克月令为囚、月令克我为死）；"
                 "得地＝至少一个地支藏有与日主同五行的天干；得势＝年、月、时三干中比劫或印至少两个；三项占两项为偏强，否则偏弱。")
                + TEN_GODS + GEN + ELEMENTS + _hidden_table(),
    "pattern": ("月令透干取格的约定：月支本气为比肩取建禄，为劫财时阳干取阳刃、阴干取月劫；否则按本气、中气、余气的顺序，"
                "取第一个透在年、月、时干上且不是比劫的藏干，以其十神名格；都不透则取本气。") + TEN_GODS + GEN + ELEMENTS + _hidden_table(),
    "relations": _relation_rule() + "相邻：年月、月日、日时为相邻两柱。",
    "flow": ("流年干支：（年份 − 4）除以 60 取余数 r，r 除以 10 的余数为天干序号（甲为 0），r 除以 12 的余数为地支序号（子为 0）。"
             "伏吟：流年干支与某柱完全相同；天克地冲：地支相冲且天干五行相克；值太岁：流年支与年支相同；冲太岁：流年支与年支相冲。")
            + _relation_rule() + TEN_GODS + GEN + ELEMENTS,
    "domain": ("传统对应：婚姻男命看正财、偏财，女命看正官、七杀，宫位为日支；事业看正官、七杀；财运看正财、偏财；学业看正印、偏印；"
               "子女男命看正官、七杀，女命看食神、伤官，宫位为时支。约定：流年天干为对应十神记为星现，流年支与宫位地支相冲或六合记为宫动。只是符号对应，不是预测。")
              + TEN_GODS + GEN + ELEMENTS + _hidden_table() + _relation_rule(),
    "yongshen": ("用神分流派：扶抑——按强弱计数，偏弱喜印、比劫，偏强喜食伤、财、官杀；调候（简化）——冬月亥子丑取火，夏月巳午未取水，其余月不特别论；"
                 "格局——以月令透干所取的格局之神为准，正官、财、印、食神顺用，七杀、伤官、比劫逆用。不指定流派时没有唯一答案。"),
    "stars": ("神煞：禄、阳刃、天乙贵人以日干查；驿马、桃花、华盖按地支所属三合局查：驿马申子辰在寅、寅午戌在申、巳酉丑在亥、亥卯未在巳；"
              "桃花申子辰在酉、寅午戌在卯、巳酉丑在午、亥卯未在子；华盖申子辰在辰、寅午戌在戌、巳酉丑在丑、亥卯未在未。古书以年支查，后来也以日支查，基准不同结论不同。"),
    "decline": "规则只能确定干支结构和传统说法；吉凶、富贵、寿夭以及某件事会不会发生、何时发生，没有规则上的唯一答案，也无法核验，不作确定预测。",
}  # fmt: skip


def rule_keys(intent: str) -> list:
    head, _, rest = intent.partition(":")
    kind = rest.split(":")[0]
    if head == "fact":
        return {"ten_god_stem": ["ten_god"], "ten_god_branch": ["ten_god", "hidden"], "hidden": ["hidden"],
                "life_stage": ["life_stage"], "xun_kong": ["xun_kong"], "luck_direction": ["luck"],
                "element_count": ["count"], "missing_element": ["count"], "day_master": ["ten_god"]}[kind]  # fmt: skip
    if head == "combo":
        return {"flow_stage": ["flow", "life_stage"], "kong_positions": ["xun_kong"], "revealed": ["hidden"],
                "god_count": ["ten_god"]}[kind]  # fmt: skip
    return {"gods": ["gods"], "strength": ["strength"], "pattern": ["pattern"], "interact": ["relations"],
            "flow": ["flow"], "domain": ["domain"], "overview": ["gods", "strength", "pattern", "relations"],
            "yongshen": ["yongshen", "strength", "pattern"], "star": ["stars"], "ambig": ["yongshen", "stars", "decline"],
            "prediction": ["decline"]}[head]  # fmt: skip


def rule_text(intent: str) -> str:
    seen, parts = set(), []
    for key in rule_keys(intent):
        for sentence in RULES[key].split("。"):
            if sentence and sentence not in seen:
                seen.add(sentence)
                parts.append(sentence)
    return "[规则说明]（通用规则，不针对这个八字）" + "。".join(parts) + "。\n"


# ---- L3: a conclusion, right or wrong ---------------------------------------------------------------


def _other(options, value, rng):
    return rng.choice([o for o in options if o != value])


def wrong_reference(item, rng):
    """A plausible wrong conclusion of the same shape, or None for declines and stars."""
    intent = item["meta"]["intent"]
    ref = copy.deepcopy(item["meta"]["expected"])
    head = intent.split(":")[0]
    if head in ("ambig", "prediction", "star"):
        return None
    if head == "combo":
        kind = item["meta"]["kind"]
        a = ref["answer"]
        ref["answer"] = {"flow_stage": lambda: _other(R.LIFE_STAGES, a, rng),
                         "god_count": lambda: a + 1 if a < 3 and rng.random() < 0.5 or a == 0 else a - 1,
                         "kong_positions": lambda: a[:-1] if a else [rng.choice(("年柱", "月柱", "时柱"))],
                         "revealed": lambda: a[:-1] if a else [rng.choice(STEMS)]}[kind]()  # fmt: skip
        return ref
    if head == "fact":
        kind = intent.split(":")[1]
        a = ref["answer"]
        if kind.startswith("ten_god"):
            ref["answer"] = _other(R.TEN_GODS, a, rng)
        elif kind == "hidden":
            ref["answer"] = list(
                R.HIDDEN_STEMS[
                    _other([b for b in BRANCHES if list(R.HIDDEN_STEMS[b]) != a], None, rng)
                ]
            )
        elif kind == "life_stage":
            ref["answer"] = _other(R.LIFE_STAGES, a, rng)
        elif kind == "xun_kong":
            pairs = [[BRANCHES[i], BRANCHES[i + 1]] for i in range(0, 12, 2)]
            ref["answer"] = _other([p for p in pairs if sorted(p) != sorted(a)], None, rng)
        elif kind == "luck_direction":
            ref["answer"] = "逆" if str(a).startswith("顺") else "顺"
        elif kind == "element_count":
            give = rng.choice([e for e in a if a[e] > 0])
            take = _other(list(a), give, rng)
            ref["answer"] = dict(a, **{give: a[give] - 1, take: a[take] + 1})
        elif kind == "missing_element":
            ref["answer"] = [] if a else [rng.choice("木火土金水")]
        elif kind == "day_master":
            ref["answer"] = _other(STEMS, a, rng)
        return ref
    if head == "strength":
        flags = {k: not ref["answer"][k] for k in ("得令", "得地", "得势")}
        ref["answer"] = {**flags, "summary": "偏强" if sum(flags.values()) >= 2 else "偏弱"}
    elif head == "pattern":
        ref["answer"] = _other([g + "格" for g in R.TEN_GODS[2:]], ref["answer"], rng)
    elif head == "interact":
        if ref["answer"]:
            dropped = rng.choice(ref["answer"])
            ref["answer"] = [k for k in ref["answer"] if k != dropped]
            ref["adjacent"] = [k for k in ref["adjacent"] if k != dropped]
        else:
            pillars = item["meta"]["pillars"]
            ref["answer"] = [
                f"冲:{pillars['year'][1]}{BRANCHES[(BRANCHES.index(pillars['year'][1]) + 6) % 12]}"
            ]
    elif head == "flow":
        ref["ten_god"] = _other(R.TEN_GODS, ref["ten_god"], rng)
    elif head == "gods":
        pos = rng.choice(list(ref["stems"]))
        ref["stems"][pos] = _other(R.TEN_GODS, ref["stems"][pos], rng)
    elif head == "domain":
        years = ref["trigger_years"]
        ref["trigger_years"] = years[:-1] if years else [int(intent.split(":")[2])]
    elif head == "overview":
        ref["strength"] = "偏弱" if ref["strength"] == "偏强" else "偏强"
    elif head == "yongshen":
        school = ref.get("school")
        if school == "格局":
            ref["use"] = "逆用" if ref["use"] == "顺用" else "顺用"
        elif school == "扶抑":
            ref["strength"] = "偏弱" if ref["strength"] == "偏强" else "偏强"
            ref["favorable"] = (
                ["印", "比劫"] if "财" in ref["favorable"] else ["食伤", "财", "官杀"]
            )
            ref.pop("favorable_elements", None)
        else:
            ref["needs"] = "水" if ref.get("needs") == "火" else "火"
    return ref


def conclusion_block(reference) -> str:
    return (
        "[规则代码已核验的结论] "
        + json.dumps(reference, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    )


# ---- prompts and scoring ---------------------------------------------------------------------------


def messages(item, level: str, rng=None, wrong=None) -> list:
    meta = item["meta"]
    base = user_text(item)
    if level == "L0":
        user = base
    elif level == "L1":
        user = raw_sheet(meta["pillars"], meta.get("gender"), meta["intent"]) + base
    elif level == "L2":
        user = rule_text(meta["intent"]) + base
    elif level == "L3":
        user = conclusion_block(meta["expected"]) + base
    elif level == "L3w":
        user = conclusion_block(wrong) + base
    else:
        raise ValueError(level)
    return [{"role": "system", "content": P2_SYSTEM}, {"role": "user", "content": user}]


def check_messages(item, first_reply: str, variant: str, rng) -> list:
    text = CHECKS[variant]
    if variant == "L4c":
        text = text.format(field=wrong_field(item, first_reply, rng))
    return [
        *messages(item, "L0"),
        {"role": "assistant", "content": first_reply or ""},
        {"role": "user", "content": text},
    ]


def wrong_field(item, reply, rng) -> str:
    want = item["meta"]["expected"]
    got = extract_final_json(reply or "")
    keys = list(want) if isinstance(want, dict) else ["answer"]
    if isinstance(got, dict):
        differing = [k for k in keys if got.get(k) != want.get(k)]
        if differing and not score(item, reply)["correct"]:
            return differing[0]
    return rng.choice(keys)


def score(item, text) -> dict:
    if item["family"] == "combo":
        return S.score_combo(item, text)
    if item["task"] == "chart" or item.get("part") == "calendar":
        return score_row(item, text)
    return score5(item, text)


def with_expected(item, reference):
    other = copy.deepcopy(item)
    other["meta"]["expected"] = reference
    return other


__all__ = [
    "LEVELS",
    "CHECKS",
    "messages",
    "check_messages",
    "score",
    "wrong_reference",
    "raw_sheet",
    "rule_text",
    "leaks",
]
_ = random
