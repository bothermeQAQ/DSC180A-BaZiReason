"""Deterministic traditional BaZi rule tables with provenance (Phase 2 knowledge base).

Every function here is a table lookup or a short rule that a reader can check by
hand. Nothing in this module predicts a personal outcome. Three kinds of rule
are kept apart (see ``RULE_CARDS``):

* ``deterministic_table``: fixed symbolic tables (elements, Ten Gods, hidden
  stems, combinations and clashes). Where lunar-python ships the same table,
  ``tests/test_phase2_rules.py`` checks every entry against it.
* ``declared_convention``: a rule with known traditional variants (seasonal
  strength, twelve stages, pattern naming, strength counting, luck-cycle
  start). The variant used here is stated; it is not claimed to be the only one.
* ``traditional_association``: symbolic correspondences (for example which Ten
  God stands for which relative). They are reported as tradition, never as
  facts about a person.

The rule statements are this project's own paraphrases of long-standing
conventions. The classical works named in ``RULE_CARDS`` are in the public
domain; they are cited at the level of the work, because chapter-level
citations have not been verified by a domain expert.
"""

from bazireason.conventions import BRANCHES, CYCLE, PILLARS, STEMS

ELEMENTS = "木火土金水"
STEM_ELEMENT = {s: ELEMENTS[i // 2] for i, s in enumerate(STEMS)}
BRANCH_ELEMENT = dict(zip(BRANCHES, "水土木木土火火土金金土水", strict=True))
GENERATES = {ELEMENTS[i]: ELEMENTS[(i + 1) % 5] for i in range(5)}
CONTROLS = {ELEMENTS[i]: ELEMENTS[(i + 2) % 5] for i in range(5)}
# Hidden stems in the order main qi, middle qi, residual qi.
HIDDEN_STEMS = {
    "子": ("癸",),
    "丑": ("己", "癸", "辛"),
    "寅": ("甲", "丙", "戊"),
    "卯": ("乙",),
    "辰": ("戊", "乙", "癸"),
    "巳": ("丙", "庚", "戊"),
    "午": ("丁", "己"),
    "未": ("己", "丁", "乙"),
    "申": ("庚", "壬", "戊"),
    "酉": ("辛",),
    "戌": ("戊", "辛", "丁"),
    "亥": ("壬", "甲"),
}
TEN_GODS = ("比肩", "劫财", "食神", "伤官", "偏财", "正财", "七杀", "正官", "偏印", "正印")
STEM_COMBINES = {"甲己": "土", "乙庚": "金", "丙辛": "水", "丁壬": "木", "戊癸": "火"}
STEM_CLASHES = ("甲庚", "乙辛", "丙壬", "丁癸")
BRANCH_SIX_COMBINES = {
    "子丑": "土",
    "寅亥": "木",
    "卯戌": "火",
    "辰酉": "金",
    "巳申": "水",
    "午未": "土",
}
BRANCH_CLASHES = ("子午", "丑未", "寅申", "卯酉", "辰戌", "巳亥")
TRIPLE_HARMONIES = {"申子辰": "水", "亥卯未": "木", "寅午戌": "火", "巳酉丑": "金"}
DIRECTIONAL_TRIPLES = {"寅卯辰": "木", "巳午未": "火", "申酉戌": "金", "亥子丑": "水"}
PUNISHMENT_PAIRS = ("寅巳", "巳申", "寅申", "丑戌", "未戌", "丑未", "子卯")
SELF_PUNISHING = "辰午酉亥"
BRANCH_HARMS = ("子未", "丑午", "寅巳", "卯辰", "申亥", "酉戌")
BRANCH_BREAKS = ("子酉", "丑辰", "寅亥", "卯午", "巳申", "未戌")
LIFE_STAGES = ("长生", "沐浴", "冠带", "临官", "帝旺", "衰", "病", "死", "墓", "绝", "胎", "养")
# Branch where each stem is at 长生; yang stems then run forward, yin stems backward.
_STAGE_START = dict(zip(STEMS, "亥午寅酉寅酉巳子申卯", strict=True))
SEASON_STATES = ("旺", "相", "休", "囚", "死")
# The twelve month-start terms: (name, month branch, month number counted from 寅 = 1).
JIE_MONTHS = (
    ("立春", "寅", 1),
    ("惊蛰", "卯", 2),
    ("清明", "辰", 3),
    ("立夏", "巳", 4),
    ("芒种", "午", 5),
    ("小暑", "未", 6),
    ("立秋", "申", 7),
    ("白露", "酉", 8),
    ("寒露", "戌", 9),
    ("立冬", "亥", 10),
    ("大雪", "子", 11),
    ("小寒", "丑", 12),
)
JIE_BRANCH = {name: branch for name, branch, _ in JIE_MONTHS}
JIE_NUMBER = {name: number for name, _, number in JIE_MONTHS}
PILLAR_ZH = {"year": "年柱", "month": "月柱", "day": "日柱", "hour": "时柱"}
POSITION_ZH = {"year": "年", "month": "月", "day": "日", "hour": "时"}


def is_yang(symbol: str) -> bool:
    """Polarity of a stem or branch: even positions (甲, 丙, ...; 子, 寅, ...) are yang."""
    table = STEMS if symbol in STEMS else BRANCHES
    return table.index(symbol) % 2 == 0


def element_of(symbol: str) -> str:
    return STEM_ELEMENT[symbol] if symbol in STEM_ELEMENT else BRANCH_ELEMENT[symbol]


def valid_pillar(pair: str) -> bool:
    """A sexagenary pair exists only when its stem and branch share polarity."""
    return pair in CYCLE


def ten_god(day_stem: str, other_stem: str) -> str:
    """Ten God of ``other_stem`` as seen from the day master (rule R-SS-01)."""
    mine, theirs = STEM_ELEMENT[day_stem], STEM_ELEMENT[other_stem]
    same = is_yang(day_stem) == is_yang(other_stem)
    if theirs == mine:
        return "比肩" if same else "劫财"
    if GENERATES[mine] == theirs:
        return "食神" if same else "伤官"
    if CONTROLS[mine] == theirs:
        return "偏财" if same else "正财"
    if CONTROLS[theirs] == mine:
        return "七杀" if same else "正官"
    return "偏印" if same else "正印"


def ten_god_reason(day_stem: str, other_stem: str) -> str:
    """One Chinese sentence deriving the Ten God from element relation and polarity."""
    mine, theirs = STEM_ELEMENT[day_stem], STEM_ELEMENT[other_stem]
    if theirs == mine:
        relation = f"{other_stem}与日主同属{mine}"
    elif GENERATES[mine] == theirs:
        relation = f"日主{day_stem}{mine}生{other_stem}{theirs}（我生）"
    elif CONTROLS[mine] == theirs:
        relation = f"日主{day_stem}{mine}克{other_stem}{theirs}（我克）"
    elif CONTROLS[theirs] == mine:
        relation = f"{other_stem}{theirs}克日主{day_stem}{mine}（克我）"
    else:
        relation = f"{other_stem}{theirs}生日主{day_stem}{mine}（生我）"
    polarity = "阴阳相同" if is_yang(day_stem) == is_yang(other_stem) else "阴阳相异"
    return f"{relation}，{polarity}，故为{ten_god(day_stem, other_stem)}"


def branch_ten_gods(day_stem: str, branch: str) -> list:
    """Ten Gods of a branch's hidden stems, main qi first (rule R-CG-01 with R-SS-01)."""
    return [ten_god(day_stem, hidden) for hidden in HIDDEN_STEMS[branch]]


def seasonal_state(element: str, month_branch: str) -> str:
    """旺相休囚死 of an element in the month (rule R-WS-01, main-qi season convention)."""
    season = BRANCH_ELEMENT[month_branch]
    if element == season:
        return "旺"
    if GENERATES[season] == element:
        return "相"
    if GENERATES[element] == season:
        return "休"
    if CONTROLS[element] == season:
        return "囚"
    return "死"


def life_stage(stem: str, branch: str) -> str:
    """Twelve-stage position of a stem at a branch (rule R-CS-01: yang forward, yin back)."""
    start, at = BRANCHES.index(_STAGE_START[stem]), BRANCHES.index(branch)
    return LIFE_STAGES[(at - start) % 12 if is_yang(stem) else (start - at) % 12]


def xun_kong(pair: str) -> str:
    """The two void branches of the ten-day group containing ``pair`` (rule R-KW-01)."""
    index = CYCLE.index(pair)
    first = (index - index % 10) % 12
    return BRANCHES[(first + 10) % 12] + BRANCHES[(first + 11) % 12]


def _ordered(first: str, second: str, table: str) -> str:
    return "".join(sorted((first, second), key=table.index))


def stem_pair_relations(first: str, second: str) -> list:
    """Canonical relation keys between two stems: 干合 and 干冲."""
    key = _ordered(first, second, STEMS)
    found = []
    if key in STEM_COMBINES:
        found.append(f"干合:{key}")
    if key in STEM_CLASHES:
        found.append(f"干冲:{key}")
    return found


def branch_pair_relations(first: str, second: str) -> list:
    """Canonical relation keys between two branches (六合, 冲, 半合, 刑, 害, 破)."""
    key = _ordered(first, second, BRANCHES)
    found = []
    if first == second:
        return [f"自刑:{key}"] if first in SELF_PUNISHING else []
    if key in BRANCH_SIX_COMBINES:
        found.append(f"六合:{key}")
    if key in BRANCH_CLASHES:
        found.append(f"冲:{key}")
    for triple in TRIPLE_HARMONIES:
        # A half combination needs the middle (帝旺) branch of the triple.
        if first in triple and second in triple and triple[1] in key:
            found.append(f"半合:{key}")
    if key in PUNISHMENT_PAIRS:
        found.append(f"刑:{key}")
    if key in BRANCH_HARMS:
        found.append(f"害:{key}")
    if key in BRANCH_BREAKS:
        found.append(f"破:{key}")
    return found


def chart_relations(pillars: dict) -> list:
    """All stem and branch relations among the four pillars, as sorted canonical keys.

    Pairwise keys are listed once even if several pillar pairs show them. A full
    三合 or 三会 triple is reported when all three branches are present.
    """
    stems = [pillars[p][0] for p in PILLARS]
    branches = [pillars[p][1] for p in PILLARS]
    found = set()
    for i in range(4):
        for j in range(i + 1, 4):
            found.update(stem_pair_relations(stems[i], stems[j]))
            found.update(branch_pair_relations(branches[i], branches[j]))
    present = set(branches)
    for triple in TRIPLE_HARMONIES:
        if set(triple) <= present:
            found.add(f"三合:{triple}")
    for triple in DIRECTIONAL_TRIPLES:
        if set(triple) <= present:
            found.add(f"三会:{triple}")
    return sorted(found)


def relation_text(key: str) -> str:
    """Readable Chinese for one canonical relation key."""
    kind, members = key.split(":")
    if kind == "干合":
        return f"{members[0]}{members[1]}相合（合化{STEM_COMBINES[members]}）"
    if kind == "干冲":
        return f"{members[0]}{members[1]}相冲"
    if kind == "六合":
        return f"{members[0]}{members[1]}六合（合化{BRANCH_SIX_COMBINES[members]}）"
    if kind == "冲":
        return f"{members[0]}{members[1]}相冲"
    if kind == "半合":
        element = next(e for t, e in TRIPLE_HARMONIES.items() if set(members) <= set(t))
        return f"{members[0]}{members[1]}半合{element}局"
    if kind == "三合":
        return f"{members}三合{TRIPLE_HARMONIES[members]}局"
    if kind == "三会":
        return f"{members}三会{DIRECTIONAL_TRIPLES[members]}方"
    if kind == "刑":
        return f"{members[0]}{members[1]}相刑"
    if kind == "自刑":
        return f"{members[0]}{members[1]}自刑"
    if kind == "害":
        return f"{members[0]}{members[1]}相害"
    return f"{members[0]}{members[1]}相破"


def element_counts(pillars: dict) -> dict:
    """Count of each element over the eight visible characters (hidden stems excluded)."""
    counts = dict.fromkeys(ELEMENTS, 0)
    for name in PILLARS:
        counts[STEM_ELEMENT[pillars[name][0]]] += 1
        counts[BRANCH_ELEMENT[pillars[name][1]]] += 1
    return counts


def month_pattern(pillars: dict) -> str:
    """Pattern label from the month branch's main qi (rule R-GJ-01, a simplified convention).

    Classical practice also weighs which hidden stem is revealed among the
    heavenly stems and whether combinations change the month qi, so a
    practitioner may name a different pattern. This label is a classification
    aid, not a judgement of fortune.
    """
    day = pillars["day"][0]
    god = ten_god(day, HIDDEN_STEMS[pillars["month"][1]][0])
    if god == "比肩":
        return "建禄格"
    if god == "劫财":
        return "阳刃格" if is_yang(day) else "月劫格"
    return god + "格"


def strength_indicators(pillars: dict) -> dict:
    """Three counted indicators of day-master support (rule R-QR-01, declared convention).

    得令: the day master's element is 旺 or 相 in the month. 得地: at least one
    branch hides a stem of the day master's own element. 得势: at least two of
    the other three heavenly stems are of the same or the generating element.
    The summary uses "two of three"; schools weigh these differently, so it is
    reported as a counting convention, never as a settled verdict.
    """
    day = pillars["day"][0]
    mine = STEM_ELEMENT[day]
    state = seasonal_state(mine, pillars["month"][1])
    roots = [
        name
        for name in PILLARS
        if any(STEM_ELEMENT[h] == mine for h in HIDDEN_STEMS[pillars[name][1]])
    ]
    helpers = [
        name
        for name in ("year", "month", "hour")
        if STEM_ELEMENT[pillars[name][0]] == mine
        or GENERATES[STEM_ELEMENT[pillars[name][0]]] == mine
    ]
    flags = {"得令": state in ("旺", "相"), "得地": bool(roots), "得势": len(helpers) >= 2}
    return {
        "day_master": day,
        "element": mine,
        "seasonal_state": state,
        "root_pillars": roots,
        "supporting_stem_pillars": helpers,
        **flags,
        "summary": "偏强" if sum(flags.values()) >= 2 else "偏弱",
    }


def luck_direction(year_stem: str, gender: str) -> str:
    """Luck cycles run forward for yang-year males and yin-year females (rule R-DY-01)."""
    if gender not in ("male", "female"):
        raise ValueError("gender must be 'male' or 'female'")
    return "forward" if is_yang(year_stem) == (gender == "male") else "backward"


def luck_sequence(month_pillar: str, direction: str, count: int = 8) -> list:
    """The first ``count`` luck pillars: the month pillar stepped through the cycle."""
    step = 1 if direction == "forward" else -1
    start = CYCLE.index(month_pillar)
    return [CYCLE[(start + step * k) % 60] for k in range(1, count + 1)]


def year_pillar(bazi_year: int) -> str:
    """Pillar of the BaZi year that starts at the 立春 of ``bazi_year`` (rule R-LN-01)."""
    return CYCLE[(bazi_year - 4) % 60]


def first_month_stem(year_stem: str) -> str:
    """五虎遁: stem of the 寅 month for a year stem."""
    return STEMS[(STEMS.index(year_stem) % 5 * 2 + 2) % 10]


def month_pillar_for(year_stem: str, month_branch: str) -> str:
    number = (BRANCHES.index(month_branch) - 2) % 12
    return STEMS[(STEMS.index(first_month_stem(year_stem)) + number) % 10] + month_branch


def first_hour_stem(day_stem: str) -> str:
    """五鼠遁: stem of the 子 hour for a day stem."""
    return STEMS[STEMS.index(day_stem) % 5 * 2 % 10]


def hour_pillar_for(day_stem: str, hour_branch: str) -> str:
    return STEMS[(STEMS.index(first_hour_stem(day_stem)) + BRANCHES.index(hour_branch)) % 10] + (
        hour_branch
    )


def hour_branch_of(hour: int) -> str:
    """Two-hour branch of a clock hour: 子 covers 23:00-00:59."""
    return BRANCHES[((hour + 1) // 2) % 12]


def chart_summary(pillars: dict) -> dict:
    """Every deterministic derived fact for one chart; no interpretation of fortune."""
    day = pillars["day"][0]
    return {
        "day_master": day,
        "day_master_element": STEM_ELEMENT[day],
        "day_master_polarity": "阳" if is_yang(day) else "阴",
        "stem_ten_gods": {
            name: ("日主" if name == "day" else ten_god(day, pillars[name][0])) for name in PILLARS
        },
        "hidden_stems": {name: list(HIDDEN_STEMS[pillars[name][1]]) for name in PILLARS},
        "hidden_ten_gods": {name: branch_ten_gods(day, pillars[name][1]) for name in PILLARS},
        "element_counts": element_counts(pillars),
        "relations": chart_relations(pillars),
        "seasonal_state": seasonal_state(STEM_ELEMENT[day], pillars["month"][1]),
        "life_stages": {name: life_stage(day, pillars[name][1]) for name in PILLARS},
        "day_xun_kong": xun_kong(pillars["day"]),
        "month_pattern": month_pattern(pillars),
        "strength": strength_indicators(pillars),
    }


_CLASSICS = {
    "yuanhai": {"work": "《渊海子平》", "attribution": "宋·徐大升 编", "rights": "public domain"},
    "sanming": {"work": "《三命通会》", "attribution": "明·万民英", "rights": "public domain"},
    "zhenquan": {"work": "《子平真诠》", "attribution": "清·沈孝瞻", "rights": "public domain"},
    "ditian": {"work": "《滴天髓》", "attribution": "旧题 京图 撰", "rights": "public domain"},
    "wuxing": {"work": "《五行大义》", "attribution": "隋·萧吉", "rights": "public domain"},
}
_REVIEW = (
    "Statement is this project's paraphrase. Cited at the level of the work; chapter-level "
    "citation and expert review are pending."
)


def _card(rule_id, kind, title, statement, classics, cross_check):
    return {
        "id": rule_id,
        "kind": kind,
        "title": title,
        "statement": statement,
        "classical_sources": [_CLASSICS[key] for key in classics],
        "cross_check": cross_check,
        "review_status": _REVIEW,
    }


_LIB = "lunar-python 1.4.8 (MIT)"
RULE_CARDS = (
    _card(
        "R-WX-01",
        "deterministic_table",
        "天干地支的五行与阴阳",
        "甲乙属木、丙丁属火、戊己属土、庚辛属金、壬癸属水；寅卯属木、巳午属火、申酉属金、亥子属水、"
        "辰戌丑未属土。天干、地支按序奇数位为阳、偶数位为阴（甲丙戊庚壬、子寅辰午申戌为阳）。",
        ("wuxing", "yuanhai"),
        f"{_LIB} LunarUtil.WU_XING_GAN / WU_XING_ZHI: all 22 entries agree.",
    ),
    _card(
        "R-WX-02",
        "deterministic_table",
        "五行生克",
        "相生：木生火、火生土、土生金、金生水、水生木。相克：木克土、土克水、水克火、火克金、金克木。",
        ("wuxing",),
        "Used by R-SS-01, whose 100 outputs agree with lunar-python's Ten Gods table.",
    ),
    _card(
        "R-GZ-01",
        "deterministic_table",
        "六十甲子与干支配对",
        "天干与地支按序循环相配成六十甲子，只有阴阳相同的干支才能相配（如甲子、乙丑），"
        "甲丑、乙子之类的组合不存在。",
        ("sanming",),
        f"{_LIB} LunarUtil.JIA_ZI: 60/60 pairs agree with conventions.CYCLE.",
    ),
    _card(
        "R-SS-01",
        "deterministic_table",
        "十神",
        "以日干为“我”：同我者为比肩（阴阳同）、劫财（阴阳异）；我生者为食神（同）、伤官（异）；"
        "我克者为偏财（同）、正财（异）；克我者为七杀（同）、正官（异）；生我者为偏印（同）、正印（异）。",
        ("yuanhai", "zhenquan"),
        f"{_LIB} LunarUtil.SHI_SHEN: 100/100 day-stem and other-stem pairs agree.",
    ),
    _card(
        "R-CG-01",
        "deterministic_table",
        "地支藏干",
        "子藏癸；丑藏己癸辛；寅藏甲丙戊；卯藏乙；辰藏戊乙癸；巳藏丙庚戊；午藏丁己；未藏己丁乙；"
        "申藏庚壬戊；酉藏辛；戌藏戊辛丁；亥藏壬甲。首位为本气。地支的十神按藏干分别论。",
        ("yuanhai", "sanming"),
        f"{_LIB} LunarUtil.ZHI_HIDE_GAN: 12/12 branches agree, including order.",
    ),
    _card(
        "R-HE-01",
        "deterministic_table",
        "天干五合与相冲",
        "五合：甲己合（土）、乙庚合（金）、丙辛合（水）、丁壬合（木）、戊癸合（火）。"
        "相冲：甲庚、乙辛、丙壬、丁癸。",
        ("sanming", "yuanhai"),
        f"{_LIB} LunarUtil.HE_GAN_5 and CHONG_GAN_4: all pairs agree.",
    ),
    _card(
        "R-ZH-01",
        "deterministic_table",
        "地支六合与六冲",
        "六合：子丑、寅亥、卯戌、辰酉、巳申、午未。六冲：子午、丑未、寅申、卯酉、辰戌、巳亥。",
        ("sanming", "yuanhai"),
        f"{_LIB} LunarUtil.HE_ZHI_6 and CHONG: all pairs agree.",
    ),
    _card(
        "R-ZH-02",
        "deterministic_table",
        "三合、半合与三会",
        "三合局：申子辰合水、亥卯未合木、寅午戌合火、巳酉丑合金。三合局中含中间一支（子、卯、午、酉）"
        "的两支为半合。三会方：寅卯辰会木、巳午未会火、申酉戌会金、亥子丑会水。",
        ("sanming",),
        "Standard table; no library table exists, so it is checked by unit tests of its own "
        "symmetry and coverage only.",
    ),
    _card(
        "R-ZH-03",
        "deterministic_table",
        "刑、害、破",
        "相刑：寅巳申三刑、丑戌未三刑、子卯相刑，辰午酉亥自刑。相害：子未、丑午、寅巳、卯辰、申亥、酉戌。"
        "相破：子酉、丑辰、寅亥、卯午、巳申、未戌。",
        ("sanming",),
        "Standard table; checked by unit tests of its own symmetry and coverage only.",
    ),
    _card(
        "R-KW-01",
        "deterministic_table",
        "旬空",
        "六十甲子每十个为一旬，每旬有两个地支轮空：甲子旬空戌亥、甲戌旬空申酉、甲申旬空午未、"
        "甲午旬空辰巳、甲辰旬空寅卯、甲寅旬空子丑。",
        ("sanming",),
        f"{_LIB} LunarUtil.getXunKong: 60/60 pairs agree.",
    ),
    _card(
        "R-WS-01",
        "declared_convention",
        "旺相休囚死",
        "以月支本气五行为当令：与月令同者旺，月令所生者相，生月令者休，克月令者囚，被月令所克者死。"
        "本项目把辰戌丑未月一律按土当令处理；也有流派按季节分配，结论可能不同。",
        ("wuxing", "sanming"),
        "Declared convention; unit tests check that each month gives exactly one state per element.",
    ),
    _card(
        "R-CS-01",
        "declared_convention",
        "十二长生",
        "阳干顺行、阴干逆行：甲长生在亥、丙戊在寅、庚在巳、壬在申；乙长生在午、丁己在酉、辛在子、"
        "癸在卯。另有“阴阳同生同死”一说，本项目不采用。",
        ("sanming", "yuanhai"),
        f"{_LIB} EightChar.get*DiShi: agrees on every pillar of 2,000 sampled charts.",
    ),
    _card(
        "R-GJ-01",
        "declared_convention",
        "月令取格（简化）",
        "以月支本气相对日主的十神命名格局；本气为比肩称建禄格，为劫财时阳干称阳刃格、阴干称月劫格。"
        "古法还要看透干与会合，本项目只取本气，因此只是分类标签，不代表吉凶判断。",
        ("zhenquan",),
        "Declared simplified convention; deterministic given R-CG-01 and R-SS-01.",
    ),
    _card(
        "R-QR-01",
        "declared_convention",
        "日主强弱的三项计数",
        "得令：日主五行在月令为旺或相。得地：四支藏干中有与日主同五行者。得势：年、月、时三干中至少"
        "两个为比劫或印。三项中占两项记为偏强，否则偏弱。各派权重不同，这只是计数约定，不是定论。",
        ("ditian", "zhenquan"),
        "Declared counting convention; deterministic given the tables above.",
    ),
    _card(
        "R-DY-01",
        "deterministic_table",
        "大运顺逆与排列",
        "阳年生男、阴年生女，大运顺行；阴年生男、阳年生女，大运逆行。大运从月柱起，顺行取月柱之后的"
        "干支，逆行取之前的干支，每步十年。",
        ("yuanhai", "sanming"),
        f"{_LIB} Yun.isForward and DaYun.getGanZhi: agree on 2,000 sampled charts for both genders.",
    ),
    _card(
        "R-DY-02",
        "declared_convention",
        "起运岁数",
        "顺行数到下一个节，逆行数到上一个节，三天折一年、一天折四个月、一个时辰折十天。"
        "起运时刻直接取 lunar-python 的结果（流派 1），不同软件的折算细则略有差别。",
        ("yuanhai",),
        f"Single source: {_LIB} EightChar.getYun(gender, sect=1). Not independently verified.",
    ),
    _card(
        "R-LN-01",
        "deterministic_table",
        "流年干支",
        "以立春为界，公历 Y 年立春后的年柱序数为 (Y − 4) mod 60（甲子为 0），例如 1984 年为甲子年。",
        ("sanming",),
        "Same rule as the verified calendar engine's year pillar.",
    ),
    _card(
        "R-LQ-01",
        "traditional_association",
        "十神与六亲的传统对应",
        "传统上以正印偏印为母亲与长辈，比肩劫财为兄弟姐妹，食神伤官为子女（女命）及才艺表达，"
        "正财偏财为财与父亲、男命以财为妻，正官七杀为事业约束、女命以官杀为夫、男命以官杀为子女。"
        "这是象征性的对应，不能据此断定任何人的实际经历。",
        ("yuanhai", "zhenquan"),
        "Traditional association only; not a verifiable fact about a person.",
    ),
)
RULES_BY_ID = {card["id"]: card for card in RULE_CARDS}
SIX_RELATIVES = {
    "比肩": "兄弟姐妹、同辈",
    "劫财": "兄弟姐妹、同辈",
    "食神": "子女（女命）、才艺与表达",
    "伤官": "子女（女命）、才艺与表达",
    "偏财": "父亲、财（男命亦象妻）",
    "正财": "财、男命之妻",
    "七杀": "事业与约束、女命之夫、男命之子女",
    "正官": "事业与约束、女命之夫、男命之子女",
    "偏印": "母亲与长辈、学识",
    "正印": "母亲与长辈、学识",
}
