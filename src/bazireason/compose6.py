"""Phase 6 composition rows (preregistration 12.7): questions that join two or three trained
procedures, written out in small steps like every other derivation.

The four held-out combinations (``splits6.COMBO_KINDS``: flow year + life stage, 旬空 +
positions, revealed hidden stems, Ten-God counts) are never built here, and neither is their
last operation on the same inputs. The kinds below teach the general skill those items test:
carrying the result of one procedure into another and answering exactly what was asked.

``build(kind, pillars, gender, rng)`` returns ``(question, hint, text, reference)``; the text
ends with ``结论JSON: ...`` whose JSON is ``reference``. Every value comes from the rule code.
"""

from bazireason import deep_rules as D
from bazireason import deep_tasks as K
from bazireason import rules as R
from bazireason import steps6 as S6
from bazireason.conventions import BRANCHES, CYCLE, STEMS

KINDS = ("stage_all", "clash_hidden", "luck_first", "top_element", "pair_gods", "stage_season")
PILLARS = ("year", "month", "day", "hour")
POS = R.POSITION_ZH
WORDS = {
    "stage_all": ("日主在年、月、日、时四个地支上，十二长生分别是哪一宫？",
                  "请把日主在四个地支上的十二长生状态逐一列出来。"),
    "clash_hidden": ("和{p}支{b}相冲的是哪个地支？那个地支的藏干，对日主来说各是什么十神？",
                     "先找出与{p}支{b}相冲的地支，再说它所藏的天干分别是日主的哪种十神。"),
    "luck_first": ("这个八字的大运是顺排还是逆排？第一步大运是哪个干支，它的天干对日主是什么十神？",
                   "请推出大运的方向和第一步大运的干支，并说明这步大运的天干是日主的哪一种十神。"),
    "top_element": ("只数八个字（不算藏干），哪一行最多？它对日主来说属于比劫、食伤、财、官杀、印中的哪一类？",
                    "八个字里个数最多的五行是什么？这一行是日主的哪一类十神？（藏干不计）"),
    "pair_gods": ("{p}支{b}和{q}支{c}之间有没有合冲刑害破？这两支的本气各是日主的什么十神？",
                  "请查{p}支{b}与{q}支{c}的关系，并分别给出两支本气的十神。"),
    "stage_season": ("日主在月支上是十二长生的哪一宫？按旺相休囚死，日主在月令又是哪一种？",
                     "请分别用十二长生和旺相休囚死两种办法，说明日主在月支的状态。"),
}  # fmt: skip
HINTS = {
    "stage_all": "结论JSON 需含 answer（对象，键为 year、month、day、hour，值为十二长生阶段名）。",
    "clash_hidden": "结论JSON 需含 branch（相冲的地支）、gods（它的藏干十神列表，按本气、中气、余气的次序）。",
    "luck_first": '结论JSON 需含 direction（"顺" 或 "逆"）、pillar（第一步大运的干支）、god（十神名称）。',
    "top_element": "结论JSON 需含 elements（个数最多的五行列表）、classes（对应的十神类别列表）。",
    "pair_gods": "结论JSON 需含 relations（关系列表，没有时为 []）、gods（对象，键为两支的位置 year、month、day、hour，值为本气十神）。",
    "stage_season": "结论JSON 需含 stage（十二长生阶段名）、state（旺、相、休、囚、死 之一）。",
}  # fmt: skip


def full_walk(day: str) -> tuple:
    """(two lines: direction and the twelve stages from 长生, {branch: stage})."""
    forward = R.is_yang(day)
    start = R._STAGE_START[day]
    i = BRANCHES.index(start)
    order = [BRANCHES[(i + k) % 12] if forward else BRANCHES[(i - k) % 12] for k in range(12)]
    stages = dict(zip(order, R.LIFE_STAGES, strict=True))
    way = "顺" if forward else "逆"
    return [
        f"日主{day}为{S6.yy(day)}干，十二长生{way}行；{day}的长生在{start}。",
        f"从{start}起{way}数十二宫：{'、'.join(b + s for b, s in stages.items())}。",
    ], stages


def chain_line(day: str) -> str:
    mine = R.STEM_ELEMENT[day]
    return f"五行位次：从{mine}起按相生次序，{K.class_chain(day)}。"


def build(kind: str, pillars: dict, gender: str, rng) -> tuple:
    day = pillars["day"][0]
    fill = {}
    if kind == "stage_all":
        lines, stages = full_walk(day)
        found = {p: stages[pillars[p][1]] for p in PILLARS}
        lines.append(
            "在上面逐一找四支："
            + "；".join(f"{POS[p]}支{pillars[p][1]}为{found[p]}" for p in PILLARS)
            + "。"
        )
        reference = {"answer": found}
    elif kind == "clash_hidden":
        p = rng.choice(PILLARS)
        b = pillars[p][1]
        other = D.branch_card(b)["冲"]
        hidden = R.HIDDEN_STEMS[other]
        gods = [R.ten_god(day, h) for h in hidden]
        fill = {"p": POS[p], "b": b}
        lines = [
            f"{POS[p]}支是{b}。地支卡片：{D.branch_card_text(b)}。",
            f"卡片上与{b}相冲的是{other}。",
            f"查藏干表：{other}藏{'、'.join(hidden)}（"
            + "，".join(f"{S6.RANK[i]}{h}" for i, h in enumerate(hidden))
            + "）。",
            S6.god_table(day),
            "逐一查表：" + "；".join(f"{h}为{g}" for h, g in zip(hidden, gods, strict=True)) + "。",
        ]
        reference = {"branch": other, "gods": gods}
    elif kind == "luck_first":
        lines = S6.fact_text("luck_direction", pillars, gender, None)
        forward = R.luck_direction(pillars["year"][0], gender) == "forward"
        month = pillars["month"]
        first = CYCLE[(CYCLE.index(month) + (1 if forward else -1)) % 60]
        way = "顺" if forward else "逆"
        lines = [
            *lines,
            f"大运从月柱{month}起{way}推一位：天干{month[0]}→{first[0]}，地支{month[1]}→{first[1]}，第一步大运为{first}。",
            S6.god_table(day),
            f"大运天干{first[0]}：查表为{R.ten_god(day, first[0])}。",
        ]
        reference = {"direction": way, "pillar": first, "god": R.ten_god(day, first[0])}
    elif kind == "top_element":
        lines, counts = S6.count_steps(pillars)
        most = max(counts.values())
        top = [e for e in R.ELEMENTS if counts[e] == most]
        classes = [D.element_class(day, e) for e in top]
        lines = [
            *lines,
            f"个数最多的是{'、'.join(top)}（{most} 个）。",
            chain_line(day),
            "；".join(f"{e}在{c}位" for e, c in zip(top, classes, strict=True)) + "。",
        ]
        reference = {"elements": top, "classes": classes}
    elif kind == "pair_gods":
        p, q = sorted(rng.sample(PILLARS, 2), key=PILLARS.index)
        b, c = pillars[p][1], pillars[q][1]
        fill = {"p": POS[p], "b": b, "q": POS[q], "c": c}
        gods = {x: R.ten_god(day, R.HIDDEN_STEMS[pillars[x][1]][0]) for x in (p, q)}
        lines = [
            f"{POS[p]}支是{b}，{POS[q]}支是{c}。地支卡片：{D.branch_card_text(b)}。",
            f"查{b}的卡片：{S6.lookup(b, c)}。",
            f"本气：{b}的本气是{R.HIDDEN_STEMS[b][0]}，{c}的本气是{R.HIDDEN_STEMS[c][0]}。",
            S6.god_table(day),
            f"查表：{POS[p]}支本气{R.HIDDEN_STEMS[b][0]}为{gods[p]}；{POS[q]}支本气{R.HIDDEN_STEMS[c][0]}为{gods[q]}。",
        ]
        reference = {"relations": list(R.branch_pair_relations(b, c)), "gods": gods}
    elif kind == "stage_season":
        mb = pillars["month"][1]
        lines, stages = full_walk(day)
        season = R.BRANCH_ELEMENT[mb]
        position = D.element_class(day, season)
        state = R.seasonal_state(R.STEM_ELEMENT[day], mb)
        assert S6.STATE_OF_POSITION[position] == state
        lines = [
            *lines,
            f"在上面找月支{mb}：{mb}{stages[mb]}，所以十二长生为{stages[mb]}。",
            f"月支{mb}属{season}。"
            + chain_line(day)[:-1]
            + "；比劫位旺、印位相、食伤位休、财位囚、官杀位死。",
            f"月令{season}在{position}位 → {state}。两种办法看的不是一回事，结论可以不同。",
        ]
        reference = {"stage": stages[mb], "state": state}
    else:
        raise ValueError(kind)
    question = rng.choice(WORDS[kind]).format(**fill)
    return question, HINTS[kind], S6.numbered(lines) + "\n" + S6.final(reference), reference


__all__ = ["KINDS", "WORDS", "HINTS", "build"]
_ = STEMS
