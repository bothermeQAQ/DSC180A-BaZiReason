"""Phase 6 small-step derivations: every computed step is written out, so the model never has
to do arithmetic, counting or matching in its head.

``derive(intent, pillars, gender)`` returns ``(text, reference)``. The text ends with the
line ``结论JSON: ...`` whose JSON equals the rule code's answer for the intent
(``bazireason.intent.plan_for``); ``tests/test_phase6.py`` checks this for every intent on
many charts, and checks the prose with the claim checker.

Why (docs/phase6_capability_audit.md, section 3.1): in the earlier procedures 旬空 was a
one-line formula, element counts and 透干 checks were single jumps, and relation matches in
flow years were stated without the lookup. The model got exactly those steps wrong (68.6%,
32.9% and 39.0% of such statements), while plain look-ups were never wrong.

Rules that depend on a school or convention say so in the text. Nothing here predicts events.
"""

import json
import random

from bazireason import deep_rules as D
from bazireason import deep_tasks as K
from bazireason import intent as N
from bazireason import rules as R
from bazireason import textstyle
from bazireason.conventions import BRANCHES, CYCLE, STEMS

PILLARS = ("year", "month", "day", "hour")
# Derivation version. 1: the screening runs and U1. 2 (preregistration 10.5): three steps that
# S1 got wrong on training wordings are written out further; answers are unchanged.
_STEPS = [1]
STATE_OF_POSITION = {"比劫": "旺", "印": "相", "食伤": "休", "财": "囚", "官杀": "死"}
CONTROL_ORDER = "相克次序：木克土、土克水、水克火、火克金、金克木"


def set_steps(version: int) -> None:
    if version not in (1, 2):
        raise ValueError(version)
    _STEPS[0] = version


def steps_version() -> int:
    return _STEPS[0]


POS = R.POSITION_ZH
RANK = ("本气", "中气", "余气")
CLASS_OF_RELATION = {"same": "比劫", "out": "食伤", "wealth": "财", "power": "官杀", "seal": "印"}


def final(reference) -> str:
    return "结论JSON: " + json.dumps(reference, ensure_ascii=False, separators=(",", ":"))


def yy(symbol) -> str:
    return "阳" if R.is_yang(symbol) else "阴"


def attr(stem) -> str:
    return f"{stem}（{R.STEM_ELEMENT[stem]}，{yy(stem)}）"


def numbered(lines) -> str:
    return "\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1))


# ---- Ten Gods ---------------------------------------------------------------------------------


def relation_step(day: str, stem: str) -> tuple:
    """(sentence on the element relation, class) of ``stem`` seen from ``day``."""
    mine, theirs = R.STEM_ELEMENT[day], R.STEM_ELEMENT[stem]
    if mine == theirs:
        return f"同为{mine}，同我者为比劫", "比劫"
    if R.GENERATES[mine] == theirs:
        return f"{mine}生{theirs}，我生者为食伤", "食伤"
    if R.CONTROLS[mine] == theirs:
        return f"{mine}克{theirs}，我克者为财", "财"
    if R.CONTROLS[theirs] == mine:
        return f"{theirs}克{mine}，克我者为官杀", "官杀"
    return f"{theirs}生{mine}，生我者为印", "印"


def god_steps(day: str, stem: str, label: str) -> list:
    """Ten God of one stem in three small steps."""
    sentence, cls = relation_step(day, stem)
    same = R.is_yang(day) == R.is_yang(stem)
    god = R.ten_god(day, stem)
    return [
        f"日主{attr(day)}；{label}{attr(stem)}。",
        f"五行：{sentence}。",
        f"阴阳{'相同' if same else '相异'}：{cls}里{'同性' if same else '异性'}的是{god}。",
    ]


def god_table(day: str) -> str:
    return K.god_table(day)


# ---- single facts -----------------------------------------------------------------------------


def xun_kong_steps(day_pillar: str) -> list:
    stem, branch = day_pillar
    back = STEMS.index(stem)
    stems = [STEMS[STEMS.index(stem) - k] for k in range(back + 1)]
    branches = [BRANCHES[(BRANCHES.index(branch) - k) % 12] for k in range(back + 1)]
    head = "甲" + branches[-1]
    start = CYCLE.index(head)
    decade = CYCLE[start : start + 10]
    used = "".join(p[1] for p in decade)
    left = [b for b in BRANCHES if b not in used]
    first = BRANCHES.index(branches[-1])
    check = [BRANCHES[(first - 2) % 12], BRANCHES[(first - 1) % 12]]
    walk = (f"从{stem}往前数到甲：{'→'.join(stems)}，退 {back} 位；地支同样退 {back} 位：{'→'.join(branches)}"
            if back else f"{stem}本身就是甲，不用退")  # fmt: skip
    return [
        f"日柱是{day_pillar}。{walk}；旬首是{head}。",
        f"{head}旬的十个干支：{'、'.join(decade)}。",
        f"这一旬用到的地支：{used}；十二地支里没用到的是{left[0]}、{left[1]}。",
        f"核对：旬首地支{branches[-1]}往前两位是{check[1]}、{check[0]}，一致。旬空为{left[0]}、{left[1]}。",
    ]


def count_steps(pillars: dict) -> tuple:
    letters = []
    per_pillar = []
    for p in PILLARS:
        s, b = pillars[p]
        letters += [(s, R.STEM_ELEMENT[s]), (b, R.BRANCH_ELEMENT[b])]
        per_pillar.append(f"{R.PILLAR_ZH[p]}{s}（{R.STEM_ELEMENT[s]}）{b}（{R.BRANCH_ELEMENT[b]}）")
    groups = []
    counts = {}
    for e in R.ELEMENTS:
        members = [c for c, el in letters if el == e]
        counts[e] = len(members)
        groups.append(f"{e}：{'、'.join(members) if members else '无'}，共 {len(members)}")
    total = "+".join(str(counts[e]) for e in R.ELEMENTS)
    lines = [
        "逐字写出五行（只数八个字，不算藏干）：" + "；".join(per_pillar) + "。",
        "按五行归类：" + "；".join(groups) + "。",
        f"核对总数：{total} = {sum(counts.values())}，正好八个字。",
    ]
    return lines, counts


def stage_steps(day: str, branch: str, label: str) -> list:
    forward = R.is_yang(day)
    start = R._STAGE_START[day]
    i = BRANCHES.index(start)
    walk = []
    for k in range(12):
        b = BRANCHES[(i + k) % 12] if forward else BRANCHES[(i - k) % 12]
        walk.append(f"{b}{R.LIFE_STAGES[k]}")
        if b == branch:
            break
    stage = R.life_stage(day, branch)
    return [
        f"日主{day}为{yy(day)}干，十二长生{'顺' if forward else '逆'}行；{day}的长生在{start}。",
        f"从{start}起{'顺' if forward else '逆'}数：{'、'.join(walk)}。",
        f"数到{label}{branch}为{stage}。",
    ]


def fact_text(kind, pillars, gender, position):
    day = pillars["day"][0]
    if kind == "day_master":
        return [f"日主就是日柱的天干：日柱{pillars['day']}，日主为{attr(day)}。"]
    if kind == "hidden":
        b = pillars[position][1]
        hidden = R.HIDDEN_STEMS[b]
        ranks = "，".join(f"{RANK[i]}{h}" for i, h in enumerate(hidden))
        return [f"{POS[position]}支是{b}。", f"查藏干表：{b}藏{'、'.join(hidden)}（{ranks}）。"]
    if kind == "ten_god_stem":
        return god_steps(day, pillars[position][0], f"{POS[position]}干")
    if kind == "ten_god_branch":
        b = pillars[position][1]
        hidden = R.HIDDEN_STEMS[b]
        return [f"{POS[position]}支{b}藏{'、'.join(hidden)}，第一个{hidden[0]}是本气。",
                *god_steps(day, hidden[0], "本气")]  # fmt: skip
    if kind == "life_stage":
        return stage_steps(day, pillars[position][1], f"{POS[position]}支")
    if kind == "xun_kong":
        return xun_kong_steps(pillars["day"])
    if kind == "luck_direction":
        s = pillars["year"][0]
        sex = "男" if gender == "male" else "女"
        forward = R.luck_direction(s, gender) == "forward"
        return [
            f"年干{s}为{yy(s)}干；{sex}命。",
            "规则：阳年男命、阴年女命顺排；阴年男命、阳年女命逆排。",
            f"{yy(s)}年{sex}命 → 大运{'顺' if forward else '逆'}排。",
        ]
    if kind in ("element_count", "missing_element"):
        lines, counts = count_steps(pillars)
        if kind == "missing_element":
            missing = [e for e in R.ELEMENTS if counts[e] == 0]
            lines.append(
                ("个数为 0 的五行：" + "、".join(missing) + "。")
                if missing
                else "没有个数为 0 的五行，五行俱全。"
            )
            lines.append("这是只数八个字的约定；算上藏干结论可能不同。")
        return lines
    raise ValueError(kind)


# ---- strength, pattern ------------------------------------------------------------------------


def seasonal_sentence(mine: str, month_branch: str) -> str:
    season = R.BRANCH_ELEMENT[month_branch]
    state = R.seasonal_state(mine, month_branch)
    if mine == season:
        how = f"与月令同为{mine}"
    elif R.GENERATES[season] == mine:
        how = f"月令{season}生{mine}"
    elif R.GENERATES[mine] == season:
        how = f"{mine}生月令{season}"
    elif R.CONTROLS[mine] == season:
        how = f"{mine}克月令{season}"
    else:
        how = f"月令{season}克{mine}"
    return f"{how} → {state}"


def strength_lines(pillars: dict) -> tuple:
    info = R.strength_indicators(pillars)
    day = pillars["day"][0]
    mine = R.STEM_ELEMENT[day]
    mb = pillars["month"][1]
    season = R.BRANCH_ELEMENT[mb]
    lines = [
        f"日主{attr(day)}；月支{mb}属{season}。",
        f"得令：{seasonal_sentence(mine, mb)}；旺或相才得令 → {'是' if info['得令'] else '否'}。",
    ]
    if _STEPS[0] >= 2:  # the positions are written first; 得令 and 得势 are read from them
        position = D.element_class(day, season)
        lines[1:] = [
            f"五行位次：从{mine}起按相生次序，{K.class_chain(day)}；"
            "比劫位旺、印位相、食伤位休、财位囚、官杀位死。",
            f"得令：月令{season}在{position}位 → {STATE_OF_POSITION[position]}；"
            f"旺或相才得令 → {'是' if info['得令'] else '否'}。",
        ]
    roots = []
    for p in PILLARS:
        b = pillars[p][1]
        hidden = R.HIDDEN_STEMS[b]
        same = [h for h in hidden if R.STEM_ELEMENT[h] == mine]
        roots.append(
            f"{POS[p]}支{b}藏{'、'.join(hidden)}：{('有' + '、'.join(same)) if same else '没有' + mine}"
        )
    n_roots = len(info["root_pillars"])
    lines.append(
        f"得地（地支藏干里有没有{mine}）："
        + "；".join(roots)
        + f"。有 {n_roots} 支，至少一支即得地 → {'是' if info['得地'] else '否'}。"
    )
    helpers = []
    for p in ("year", "month", "hour"):
        s = pillars[p][0]
        _, cls = relation_step(day, s)
        ok = cls in ("比劫", "印")
        how = "在" + cls + "位" if _STEPS[0] >= 2 else "是" + cls
        helpers.append(f"{POS[p]}干{s}（{R.STEM_ELEMENT[s]}）{how}{'，算' if ok else '，不算'}")
    n = len(info["supporting_stem_pillars"])
    lines.append(
        (
            "得势（其余三干中在比劫位或印位的个数）："
            if _STEPS[0] >= 2
            else "得势（其余三干中比劫或印的个数）："
        )
        + "；".join(helpers)
        + f"。共 {n} 个，至少两个才算 → {'是' if info['得势'] else '否'}。"
    )
    k = sum(info[x] for x in ("得令", "得地", "得势"))
    lines.append(
        f"三项中占 {k} 项，占两项及以上为偏强 → {info['summary']}。这是计数约定，各派权重不同。"
    )
    return lines, info


def pattern_lines(pillars: dict) -> tuple:
    day = pillars["day"][0]
    mb = pillars["month"][1]
    hidden = R.HIDDEN_STEMS[mb]
    found = D.revealed_pattern(pillars)
    others = [pillars[p][0] for p in ("year", "month", "hour")]
    lines = [
        f"月支{mb}藏{'、'.join(hidden)}（"
        + "，".join(f"{RANK[i]}{h}" for i, h in enumerate(hidden))
        + "）。",
        god_table(day),
    ]
    main_god = R.ten_god(day, hidden[0])
    lines.append(f"本气{hidden[0]}查表为{main_god}。")
    if found["basis"] == "main_is_peer":
        why = (
            "比肩取建禄"
            if main_god == "比肩"
            else ("劫财且日主为阳干，取阳刃" if R.is_yang(day) else "劫财且日主为阴干，取月劫")
        )
        lines.append(f"本气是{main_god}（比劫）→ 按约定不看透干：{why} → {found['pattern']}。")
        return lines, found
    lines.append(f"年、月、时三干是{'、'.join(others)}。按本气、中气、余气的顺序看是否透出：")
    for i, h in enumerate(hidden):
        god = R.ten_god(day, h)
        present = h in others
        if present and god not in ("比肩", "劫财"):
            lines.append(
                f"{h}（{RANK[i]}，{god}）：在{'、'.join(others)}中出现 → 透出，取{h}，记作{god}格。"
            )
            break
        why = "出现，但是比劫，不取" if present else "没有出现"
        lines.append(f"{h}（{RANK[i]}，{god}）：在{'、'.join(others)}中{why}。")
    else:
        lines.append(f"都不取，回到本气{hidden[0]}，记作{main_god}格。")
    lines.append("这是“月令透干取格”的约定；多干并透、会合变格各家处理不同，格名不代表吉凶。")
    return lines, found


# ---- relations --------------------------------------------------------------------------------


def lookup(a: str, b: str) -> str:
    """Where branch ``b`` sits on branch ``a``'s card, then the relation words."""
    card = D.branch_card(a)
    keys = R.branch_pair_relations(a, b)
    if a == b:
        return f"同支，{a}{'是' if card['刑'] == [a] else '不是'}自刑 → {K.relation_words(keys)}"
    slots = [kind for kind in ("六合", "冲", "害", "破") if card[kind] == b]
    if b in card["刑"]:
        slots.append("刑")
    if b in card["半合"]:
        slots.append("半合")
    where = f"{b}在{a}卡的{'、'.join(slots)}" if slots else f"{b}不在{a}卡"
    return f"{where} → {K.relation_words(keys)}"


def stem_lookup(a: str, b: str) -> str:
    key = "".join(sorted((a, b), key=STEMS.index))
    if a == b:
        return "同干 → 无"
    if key in R.STEM_COMBINES:
        return f"{key}在五合表 → {a}{b}相合（合化{R.STEM_COMBINES[key]}）"
    if key in R.STEM_CLASHES:
        return f"{key}在相冲表 → {a}{b}相冲"
    return f"{key}不在五合表、相冲表 → 无"


STEM_TABLES = "天干五合：甲己、乙庚、丙辛、丁壬、戊癸；天干相冲：甲庚、乙辛、丙壬、丁癸。"


def interact_lines(pillars: dict) -> list:
    branches = [pillars[p][1] for p in PILLARS]
    lines = [
        "地支卡片：" + "；".join(D.branch_card_text(b) for b in dict.fromkeys(branches)) + "。",
        STEM_TABLES,
    ]
    stem_parts, branch_parts = [], []
    for i, first in enumerate(PILLARS):
        for second in PILLARS[i + 1 :]:
            near = "相邻" if PILLARS.index(second) - i == 1 else "不相邻"
            stem_parts.append(
                f"{POS[first]}{pillars[first][0]}–{POS[second]}{pillars[second][0]}：{stem_lookup(pillars[first][0], pillars[second][0])}"
            )
            branch_parts.append(
                f"{POS[first]}{pillars[first][1]}–{POS[second]}{pillars[second][1]}（{near}）：{lookup(pillars[first][1], pillars[second][1])}"
            )
    lines.append("天干两两比对：" + "；".join(stem_parts) + "。")
    lines.append("地支两两比对：" + "；".join(branch_parts) + "。")
    keys = R.chart_relations(pillars)
    present = set(branches)
    triple_checks = []
    for table in (R.TRIPLE_HARMONIES, R.DIRECTIONAL_TRIPLES):
        for t in table:
            have = [b for b in t if b in present]
            if len(have) >= 2:
                triple_checks.append(
                    f"{t}：有{''.join(have)}{'，三支齐全' if len(have) == 3 else '，不全'}"
                )
    lines.append(
        "三合、三会（三支都在才算）："
        + ("；".join(triple_checks) if triple_checks else "没有两支以上同组")
        + "。"
    )
    adjacent = sorted({item["key"] for item in D.positioned_relations(pillars) if item["adjacent"]})
    lines.append(
        "全部关系："
        + K.relation_words(keys)
        + "；相邻两柱之间的："
        + K.relation_words(adjacent)
        + "。"
    )
    return lines


# ---- flow years and areas of life ------------------------------------------------------------


def year_steps(year: int) -> tuple:
    value = year - 4
    q, r = divmod(value, 60)
    flow = CYCLE[r]
    return (
        f"{year} − 4 = {value}；60×{q} = {60 * q}，{value} − {60 * q} = {r}；"
        f"{r} = 10×{r // 10} + {r % 10} → 第 {r % 10} 干{STEMS[r % 10]}（甲为 0）；"
        f"{r} = 12×{r // 12} + {r % 12} → 第 {r % 12} 支{BRANCHES[r % 12]}（子为 0）；{year} 年为{flow}年。",
        flow,
    )


def flow_lines(pillars: dict, year: int) -> list:
    day = pillars["day"][0]
    text, flow = year_steps(year)
    analysis = D.flow_analysis(pillars, flow)
    fs, fb = flow
    lines = [
        "流年干支：" + text,
        god_table(day),
        f"流年天干{fs}：查表为{R.ten_god(day, fs)}。",
        "流年支卡片：" + D.branch_card_text(fb) + "。",
        "流年支与四柱地支逐一比对："
        + "；".join(f"{POS[p]}支{pillars[p][1]}：{lookup(fb, pillars[p][1])}" for p in PILLARS)
        + "。",
        STEM_TABLES
        + "流年干与四柱天干："
        + "；".join(f"{POS[p]}干{pillars[p][0]}：{stem_lookup(fs, pillars[p][0])}" for p in PILLARS)
        + "。",
    ]
    fuyin = [p for p in PILLARS if pillars[p] == flow]
    lines.append(
        "伏吟（流年干支与某柱完全相同）："
        + "；".join(
            f"{R.PILLAR_ZH[p]}{pillars[p]}{'相同' if p in fuyin else '不同'}" for p in PILLARS
        )
        + "。"
    )
    clash = [
        p
        for p in PILLARS
        if "".join(sorted((fb, pillars[p][1]), key=BRANCHES.index)) in R.BRANCH_CLASHES
    ]
    if _STEPS[0] >= 2:  # name the clashing pillars, write the 相克 order, then judge
        parts = []
        for p in clash:
            e1, e2 = R.STEM_ELEMENT[fs], R.STEM_ELEMENT[pillars[p][0]]
            if e1 == e2:
                how = f"同为{e1}，不相克 → 不算"
            elif R.CONTROLS[e1] == e2 or R.CONTROLS[e2] == e1:
                a, b = (e1, e2) if R.CONTROLS[e1] == e2 else (e2, e1)
                how = f"{a}克{b}在次序里，相克 → 天克地冲"
            else:
                how = f"{e1}、{e2}不在次序里，不相克 → 不算"
            assert ("天克地冲" in how) == D.clash_with_control(pillars[p], flow)
            parts.append(f"{R.PILLAR_ZH[p]}{pillars[p]}：天干{fs}{e1}与{pillars[p][0]}{e2}，{how}")
        named = "、".join(R.PILLAR_ZH[p] for p in clash) or "没有"
        lines.append(
            f"天克地冲（地支相冲且天干相克）：上面与流年支{fb}相冲的柱：{named}"
            + ("。" + CONTROL_ORDER + "。" + "；".join(parts) + "。" if clash else " → 无。")
        )
    elif clash:
        parts = []
        for p in clash:
            e1, e2 = R.STEM_ELEMENT[fs], R.STEM_ELEMENT[pillars[p][0]]
            control = D.clash_with_control(pillars[p], flow)
            parts.append(
                f"{R.PILLAR_ZH[p]}{pillars[p]}：地支相冲，天干{fs}{e1}与{pillars[p][0]}{e2}{'相克 → 天克地冲' if control else '不相克 → 不算'}"
            )
        lines.append("天克地冲（地支相冲且天干相克）：" + "；".join(parts) + "。")
    else:
        lines.append("天克地冲：流年支与四柱地支没有相冲，无。")
    yb = pillars["year"][1]
    special = analysis["special"]
    if "值太岁" in special:
        tai = f"流年支{fb}与年支{yb}相同 → 值太岁"
    elif "冲太岁" in special:
        tai = f"流年支{fb}与年支{yb}相冲 → 冲太岁"
    else:
        tai = f"流年支{fb}与年支{yb}既不相同也不相冲 → 无"
    lines.append("太岁：" + tai + "。")
    lines.append("以上是可以机械判定的关系，本身不说明这一年会发生什么。")
    return lines


def domain_lines(pillars: dict, gender: str, domain: str, first: int) -> list:
    name, gods_by_gender, palace, palace_name = D.DOMAINS[domain]
    gods = gods_by_gender[gender]
    day = pillars["day"][0]
    facts = D.domain_facts(pillars, domain, gender)
    sex = "男" if gender == "male" else "女"
    lines = [
        f"传统对应：{sex}命以{'、'.join(gods)}象征{name}"
        + (f"，以{palace_name}为宫位。" if palace else "。"),
        god_table(day),
    ]
    stems = [
        f"{POS[p]}干{pillars[p][0]}为{R.ten_god(day, pillars[p][0])}"
        for p in ("year", "month", "hour")
    ]
    lines.append(
        "三干逐一查表："
        + "；".join(stems)
        + f"。其中是{'或'.join(gods)}的：{'、'.join(POS[p] + '干' for p in facts['in_stems']) or '没有'}。"
    )
    hidden = []
    for p in PILLARS:
        inside = [
            h + R.ten_god(day, h)
            for h in R.HIDDEN_STEMS[pillars[p][1]]
            if R.ten_god(day, h) in gods
        ]
        if inside:
            hidden.append(f"{POS[p]}支{pillars[p][1]}藏{'、'.join(inside)}")
    if _STEPS[0] >= 2:  # every hidden stem with its Ten God first, then the relevant ones
        every, picked = [], []
        for p in PILLARS:
            b = pillars[p][1]
            every.append(
                f"{POS[p]}支{b}藏" + "、".join(h + R.ten_god(day, h) for h in R.HIDDEN_STEMS[b])
            )
            picked += [
                f"{POS[p]}支{b}的{h}" for h in R.HIDDEN_STEMS[b] if R.ten_god(day, h) in gods
            ]
        lines.append(
            "四支藏干逐一查表："
            + "；".join(every)
            + f"。其中是{'或'.join(gods)}的：{'、'.join(picked) or '没有'}。"
        )
    else:
        lines.append("藏干里：" + ("；".join(hidden) if hidden else "不见") + "。")
    if palace and _STEPS[0] >= 2:  # the palace branch's card before the yearly look-ups
        pb = pillars[palace][1]
        lines.append(
            f"{palace_name}为{pb}。宫位卡片：{D.branch_card_text(pb)}。"
            f"约定：流年天干是{'或'.join(gods)}为星现，流年支在{pb}卡的冲或六合里为宫动。"
        )
    elif palace:
        pb = pillars[palace][1]
        lines.append(
            f"{palace_name}为{pb}；约定：流年天干是{'或'.join(gods)}为星现，流年支与{pb}相冲或六合为宫动。"
        )
    else:
        lines.append(f"约定：流年天干是{'或'.join(gods)}为星现。")
    flow = R.year_pillar(first)
    rows, hits = [], []
    for year in range(first, first + 8):
        if year == first:
            text, flow = year_steps(year)
            step = text
        else:
            nxt = CYCLE[(CYCLE.index(flow) + 1) % 60]
            step = f"{flow[0]}→{nxt[0]}，{flow[1]}→{nxt[1]}，{year}为{nxt}"
            flow = nxt
        god = R.ten_god(day, flow[0])
        reasons = ["星现"] if god in gods else []
        row = f"{year}：{step}；{flow[0]}为{god}{'，在' + '、'.join(gods) + '中' if god in gods else ''}"
        if _STEPS[0] >= 2:  # each conclusion directly after the check it rests on
            if god in gods:
                row += " → 星现"
            if palace:
                keys = R.branch_pair_relations(flow[1], pillars[palace][1])
                row += f"；{lookup(pillars[palace][1], flow[1])}"
                if any(k.startswith(("冲:", "六合:")) for k in keys):
                    row += " → 宫动"
                    reasons.append("宫动")
            if reasons:
                hits.append(year)
            rows.append(row)
            continue
        if palace:
            keys = R.branch_pair_relations(flow[1], pillars[palace][1])
            moved = any(k.startswith(("冲:", "六合:")) for k in keys)
            row += f"；{flow[1]}与{pillars[palace][1]}：{K.relation_words(keys) if keys else '无合冲刑害破'}"
            if moved:
                reasons.append("宫动")
        if reasons:
            row += " → " + "、".join(reasons)
            hits.append(year)
        rows.append(row)
    lines.append("逐年（每年天干、地支各往后一位）：" + "；".join(rows) + "。")
    lines.append(
        f"符合约定的年份：{'、'.join(map(str, hits)) or '无'}。这只是符号上的对应，{name}在现实中是否发生、何时发生，规则无法确证，不作预测。"
    )
    return lines


# ---- composite and declining families ------------------------------------------------------------


def gods_lines(pillars: dict) -> list:
    _, text, _ = K.gods_task(pillars, random.Random(0))
    return text.split("\n")


def overview_lines(pillars: dict, gender) -> list:
    day = pillars["day"][0]
    s_lines, info = strength_lines(pillars)
    p_lines, found = pattern_lines(pillars)
    configs = D.stem_configs(pillars)
    gods = "、".join(
        f"{POS[p]}干{pillars[p][0]}为{R.ten_god(day, pillars[p][0])}"
        for p in ("year", "month", "hour")
    )
    lines = ["一、十神：" + god_table(day) + gods + "。", "十神组合：" + K.configs_text(pillars)[0]]
    lines.append("二、强弱（计数约定）：" + " ".join(s_lines))
    lines.append(
        "三、格局（月令透干取格的约定）："
        + " ".join(x for x in p_lines if not x.startswith("十神表"))
    )
    rel = R.chart_relations(pillars)
    lines.append("四、合冲刑害（确定）：" + K.relation_words(rel) + "。")
    lines.append(
        "五、不能确定的部分：以上是规则推出的符号结构；现实中意味着什么、会不会发生某件事，规则无法确证，不作预测。"
    )
    _ = (info, found, configs, gender)
    return lines


def derive(intent: str, pillars: dict, gender=None):
    """(small-step text, reference) for one intent, or None when the plan is undefined."""
    head, _, rest = intent.partition(":")
    previous = textstyle.version()
    textstyle.set_version(4)
    try:
        plan = N.plan_for(intent, pillars, gender)
        if plan is None:
            return None
        reference = plan["reference"]
        if head == "fact":
            kind, _, position = rest.partition(":")
            body = numbered(fact_text(kind, pillars, gender, position or None))
        elif head == "strength":
            body = numbered(strength_lines(pillars)[0])
        elif head == "pattern":
            body = numbered(pattern_lines(pillars)[0])
        elif head == "interact":
            body = numbered(interact_lines(pillars))
        elif head == "flow":
            body = numbered(flow_lines(pillars, int(rest)))
        elif head == "domain":
            domain, year = rest.split(":")
            body = numbered(domain_lines(pillars, gender or "male", domain, int(year)))
        elif head == "gods":
            body = numbered(gods_lines(pillars))
        elif head == "overview":
            body = numbered(overview_lines(pillars, gender))
        elif head == "yongshen" and rest == "扶抑":
            s_lines, info = strength_lines(pillars)
            fy = D.fuyi(pillars)
            body = numbered(s_lines + [
                f"扶抑：{info['summary']}{'要生扶，喜印、比劫' if info['summary'] == '偏弱' else '要泄耗克制，喜食伤、财、官杀'}。",
                "对日主" + pillars["day"][0] + "而言：" + "、".join(f"{c}（{e}）" for c, e in zip(fy["favorable_classes"], fy["favorable_elements"], strict=True)) + "。",
            ])  # fmt: skip
        elif head == "yongshen" and rest == "格局":
            p_lines, found = pattern_lines(pillars)
            use = D.PATTERN_USE[found["god"]]
            body = numbered(
                p_lines
                + [
                    f"格局之神为{found['stem']}（{found['god']}）。传统上正官、财、印、食神顺用，七杀、伤官、比劫逆用 → {found['god']}属{use}。"
                ]
            )
        else:
            body = plan["text"]  # 调候, ambiguity, stars and declines: no computation to break up
    finally:
        textstyle.set_version(previous)
    return body + "\n" + final(reference), reference


__all__ = ["derive", "final", "god_steps", "xun_kong_steps", "count_steps", "year_steps"]
