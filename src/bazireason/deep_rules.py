"""Phase 3 rule layer: multi-step traditional reasoning built on ``bazireason.rules``.

Everything here is deterministic. Where the tradition has competing practices
the function implements ONE declared convention and the matching rule card says
so; the three schools of 用神 are kept as three separate functions and are never
merged into one verdict.

Kinds of rule (see ``RULE_CARDS``):

* ``deterministic_table``: fixed tables (禄, 阳刃, 驿马, 咸池, 华盖, 天乙贵人) and
  relations that follow mechanically from the five-element cycle.
* ``declared_convention``: a stated choice among traditional variants (透干取格,
  扶抑喜忌, the simplified 调候 rule, which branch a star is counted from, what
  counts as a year "引动" a palace).
* ``traditional_association``: symbolic correspondences between chart positions
  and areas of life. They are reported as tradition and never as a fact or a
  prediction about a person.

Quotations in the rule cards are copied by ``scripts/phase3/build_knowledge.py``
from public-domain texts as transcribed on zh.wikisource.org; the script checks
that each quotation occurs in the downloaded page. Cards without a quotation
are this project's paraphrase and say so.
"""

from bazireason import rules as R
from bazireason.conventions import BRANCHES, CYCLE, PILLARS, STEMS

OTHER = ("year", "month", "hour")
CLASSES = ("比劫", "食伤", "财", "官杀", "印")
GOD_CLASS = {
    "比肩": "比劫",
    "劫财": "比劫",
    "食神": "食伤",
    "伤官": "食伤",
    "偏财": "财",
    "正财": "财",
    "七杀": "官杀",
    "正官": "官杀",
    "偏印": "印",
    "正印": "印",
}
CLASS_GODS = {c: tuple(g for g, k in GOD_CLASS.items() if k == c) for c in CLASSES}


def class_element(day_stem: str, cls: str) -> str:
    """Element of a Ten-God class for a day master (比劫 is the day master's own element)."""
    start = R.ELEMENTS.index(R.STEM_ELEMENT[day_stem])
    return R.ELEMENTS[(start + CLASSES.index(cls)) % 5]


def element_class(day_stem: str, element: str) -> str:
    start = R.ELEMENTS.index(R.STEM_ELEMENT[day_stem])
    return CLASSES[(R.ELEMENTS.index(element) - start) % 5]


def class_generates(cls: str) -> str:
    """比劫生食伤, 食伤生财, 财生官杀, 官杀生印, 印生比劫 (rule R-SS-02)."""
    return CLASSES[(CLASSES.index(cls) + 1) % 5]


def class_controls(cls: str) -> str:
    """比劫克财, 食伤克官杀, 财克印, 官杀克比劫, 印克食伤 (rule R-SS-02)."""
    return CLASSES[(CLASSES.index(cls) + 2) % 5]


def element_relation(first: str, second: str) -> str:
    """Relation of two elements as a short phrase, e.g. 火克金, 金生水, 同为木."""
    if first == second:
        return f"同为{first}"
    if R.GENERATES[first] == second:
        return f"{first}生{second}"
    if R.GENERATES[second] == first:
        return f"{second}生{first}"
    if R.CONTROLS[first] == second:
        return f"{first}克{second}"
    return f"{second}克{first}"


# (name, gods on one side, gods on the other, relation between the two classes)
CONFIGS = (
    ("食神制杀", ("食神",), ("七杀",), "克"),
    ("伤官见官", ("伤官",), ("正官",), "克"),
    ("枭神夺食", ("偏印",), ("食神",), "克"),
    ("财坏印", ("正财", "偏财"), ("正印", "偏印"), "克"),
    ("比劫夺财", ("比肩", "劫财"), ("正财", "偏财"), "克"),
    ("官印相生", ("正官",), ("正印", "偏印"), "生"),
    ("杀印相生", ("七杀",), ("正印", "偏印"), "生"),
    ("食伤生财", ("食神", "伤官"), ("正财", "偏财"), "生"),
    ("财生官杀", ("正财", "偏财"), ("正官", "七杀"), "生"),
    ("官杀混杂", ("正官",), ("七杀",), "并见"),
)


def stem_gods(pillars: dict) -> dict:
    day = pillars["day"][0]
    return {name: R.ten_god(day, pillars[name][0]) for name in OTHER}


def stem_configs(pillars: dict) -> list:
    """Named Ten-God pairings among the year, month and hour stems (rule R-SS-03).

    Declared detection convention: a pairing is listed when both gods are
    visible among those three stems. Hidden stems, distance and strength are
    not weighed, so this is a label, not a judgement.
    """
    seen = set(stem_gods(pillars).values())
    return [name for name, first, second, _ in CONFIGS if seen & set(first) and seen & set(second)]


# ---- Branch reference card ------------------------------------------------------------------

_TRIPLE_OF = {b: t for t in R.TRIPLE_HARMONIES for b in t}
_DIRECTION_OF = {b: t for t in R.DIRECTIONAL_TRIPLES for b in t}


def _partner(pairs, branch):
    return [p.replace(branch, "", 1) for p in pairs if branch in p]


def branch_card(branch: str) -> dict:
    """Every table partner of one branch; all pair relations follow from two cards."""
    triple = _TRIPLE_OF[branch]
    half = [b for b in triple if b != branch and (branch == triple[1] or b == triple[1])]
    return {
        "六合": _partner(R.BRANCH_SIX_COMBINES, branch)[0],
        "冲": _partner(R.BRANCH_CLASHES, branch)[0],
        "刑": [branch] if branch in R.SELF_PUNISHING else _partner(R.PUNISHMENT_PAIRS, branch),
        "害": _partner(R.BRANCH_HARMS, branch)[0],
        "破": _partner(R.BRANCH_BREAKS, branch)[0],
        "半合": half,
        "三合": triple,
        "三会": _DIRECTION_OF[branch],
    }


def branch_card_text(branch: str) -> str:
    card = branch_card(branch)
    punish = "自刑" if card["刑"] == [branch] else "刑" + "、".join(card["刑"])
    return (
        f"{branch}：六合{card['六合']}，冲{card['冲']}，{punish}，害{card['害']}，破{card['破']}，"
        f"半合{'、'.join(card['半合'])}（{card['三合']}{R.TRIPLE_HARMONIES[card['三合']]}局），"
        f"三会{card['三会']}"
    )


def positioned_relations(pillars: dict) -> list:
    """Pairwise relations with the pillars involved and whether they are adjacent."""
    found = []
    for i, first in enumerate(PILLARS):
        for second in PILLARS[i + 1 :]:
            adjacent = PILLARS.index(second) - i == 1
            for layer, index, relation_of in (
                ("stem", 0, R.stem_pair_relations),
                ("branch", 1, R.branch_pair_relations),
            ):
                for key in relation_of(pillars[first][index], pillars[second][index]):
                    found.append(
                        {
                            "key": key,
                            "layer": layer,
                            "pillars": [first, second],
                            "adjacent": adjacent,
                        }
                    )
    return found


# ---- 透干, 通根 and the month-order pattern ------------------------------------------------


def revealed_hidden(pillars: dict, branch_pillar: str = "month") -> list:
    """Hidden stems of one branch that also stand among the year, month or hour stems."""
    visible = {pillars[name][0] for name in OTHER}
    return [stem for stem in R.HIDDEN_STEMS[pillars[branch_pillar][1]] if stem in visible]


def roots(pillars: dict, stem: str) -> list:
    """(pillar, hidden stem, rank) for every hidden stem of the same element as ``stem``.

    Declared convention (rule R-TG-01): a stem is rooted in a branch that hides
    a stem of its own element; rank 0 is the branch's main qi.
    """
    element = R.STEM_ELEMENT[stem]
    return [
        (name, hidden, rank)
        for name in PILLARS
        for rank, hidden in enumerate(R.HIDDEN_STEMS[pillars[name][1]])
        if R.STEM_ELEMENT[hidden] == element
    ]


def revealed_pattern(pillars: dict) -> dict:
    """Month-order pattern with 透干 priority (rule R-GJ-02, declared convention).

    1. If the month branch's main qi is 比肩 or 劫财 of the day master, the
       chart is 建禄格, 阳刃格 (yang day master) or 月劫格 (yin day master).
    2. Otherwise take the first hidden stem of the month branch, in the order
       main, middle, residual qi, that is visible among the year, month or
       hour stems and is not 比肩 or 劫财.
    3. If none is visible, take the main qi.
    """
    day = pillars["day"][0]
    month_branch = pillars["month"][1]
    hidden = R.HIDDEN_STEMS[month_branch]
    main_god = R.ten_god(day, hidden[0])
    if main_god in ("比肩", "劫财"):
        return {
            "basis": "main_is_peer",
            "stem": hidden[0],
            "god": main_god,
            "pattern": R.month_pattern(pillars),
        }
    visible = {pillars[name][0] for name in OTHER}
    for rank, stem in enumerate(hidden):
        god = R.ten_god(day, stem)
        if stem in visible and god not in ("比肩", "劫财"):
            basis = ("main_revealed", "middle_revealed", "residual_revealed")[rank]
            return {"basis": basis, "stem": stem, "god": god, "pattern": god + "格"}
    return {"basis": "main_default", "stem": hidden[0], "god": main_god, "pattern": main_god + "格"}


# 子平真诠 groups the pattern gods into those "used with the flow" and "used against it".
PATTERN_USE = {
    "正官": "顺用",
    "正财": "顺用",
    "偏财": "顺用",
    "正印": "顺用",
    "偏印": "顺用",
    "食神": "顺用",
    "七杀": "逆用",
    "伤官": "逆用",
    "比肩": "逆用",
    "劫财": "逆用",
}


# ---- Three schools of 用神, kept apart ------------------------------------------------------


def fuyi(pillars: dict) -> dict:
    """扶抑 (rule R-YS-01): support a weak day master, restrain a strong one.

    Strength is the three-indicator count of rule R-QR-01, so the result
    inherits that convention's limits.
    """
    strength = R.strength_indicators(pillars)["summary"]
    day = pillars["day"][0]
    favorable = ("印", "比劫") if strength == "偏弱" else ("食伤", "财", "官杀")
    unfavorable = tuple(c for c in CLASSES if c not in favorable)
    return {
        "strength": strength,
        "favorable_classes": list(favorable),
        "unfavorable_classes": list(unfavorable),
        "favorable_elements": [class_element(day, c) for c in favorable],
        "unfavorable_elements": [class_element(day, c) for c in unfavorable],
    }


WINTER, SUMMER = "亥子丑", "巳午未"


def tiaohou(pillars: dict) -> dict:
    """调候, simplified (rule R-YS-02): winter months want 火, summer months want 水.

    The month-by-month prescriptions of 《穷通宝鉴》 are not reproduced; other
    months return no climate requirement under this simplified convention.
    """
    month = pillars["month"][1]
    need = "火" if month in WINTER else "水" if month in SUMMER else None
    if need is None:
        return {"season": None, "needs": None, "in_stems": [], "in_hidden": []}
    in_stems = [name for name in PILLARS if R.STEM_ELEMENT[pillars[name][0]] == need]
    in_hidden = [
        name
        for name in PILLARS
        if any(R.STEM_ELEMENT[h] == need for h in R.HIDDEN_STEMS[pillars[name][1]])
    ]
    return {
        "season": "冬" if need == "火" else "夏",
        "needs": need,
        "in_stems": in_stems,
        "in_hidden": in_hidden,
    }


def pattern_use(pillars: dict) -> dict:
    """格局用神 (rule R-YS-03): the month-order pattern god and how it is traditionally used."""
    pattern = revealed_pattern(pillars)
    return {**pattern, "use": PATTERN_USE[pattern["god"]]}


def yongshen_by_school(pillars: dict) -> dict:
    """The three schools side by side; they answer different questions and may disagree."""
    day = pillars["day"][0]
    use = pattern_use(pillars)
    return {
        "扶抑": fuyi(pillars),
        "调候": tiaohou(pillars),
        "格局": {
            "pattern": use["pattern"],
            "god": use["god"],
            "use": use["use"],
            "element": R.STEM_ELEMENT[use["stem"]],
            "class": GOD_CLASS[R.ten_god(day, use["stem"])],
        },
    }


# ---- Stars (神煞) ----------------------------------------------------------------------------

LU = dict(zip(STEMS, "寅卯巳午巳午申酉亥子", strict=True))
YANG_REN = {"甲": "卯", "丙": "午", "戊": "午", "庚": "酉", "壬": "子"}
TIAN_YI = {
    "甲": "丑未",
    "戊": "丑未",
    "庚": "丑未",
    "乙": "子申",
    "己": "子申",
    "丙": "亥酉",
    "丁": "亥酉",
    "壬": "卯巳",
    "癸": "卯巳",
    "辛": "寅午",
}
YI_MA = {"申子辰": "寅", "寅午戌": "申", "巳酉丑": "亥", "亥卯未": "巳"}
XIAN_CHI = {"申子辰": "酉", "寅午戌": "卯", "巳酉丑": "午", "亥卯未": "子"}
HUA_GAI = {"申子辰": "辰", "寅午戌": "戌", "巳酉丑": "丑", "亥卯未": "未"}
TRIPLE_STARS = {"驿马": YI_MA, "桃花": XIAN_CHI, "华盖": HUA_GAI}
STAR_NAMES = ("禄", "阳刃", "天乙贵人", "驿马", "桃花", "华盖")


def star_branches(star: str, pillars: dict, basis: str = "day") -> str:
    """Branches where a star falls. ``basis`` picks the year or day branch for triple stars.

    禄, 阳刃 and 天乙贵人 are counted from the day stem (rule R-SM-01). 驿马,
    桃花 (咸池) and 华盖 are counted from a branch's 三合 group; the classics
    count from the year, later practice also from the day, so the basis is a
    declared choice (rule R-SM-02). A yin day stem has no 阳刃.
    """
    day_stem = pillars["day"][0]
    if star == "禄":
        return LU[day_stem]
    if star == "阳刃":
        return YANG_REN.get(day_stem, "")
    if star == "天乙贵人":
        return TIAN_YI[day_stem]
    if basis not in ("year", "day"):
        raise ValueError("basis must be 'year' or 'day'")
    return TRIPLE_STARS[star][_TRIPLE_OF[pillars[basis][1]]]


def star_positions(star: str, pillars: dict, basis: str = "day") -> list:
    """Pillars of the chart whose branch carries the star."""
    targets = star_branches(star, pillars, basis)
    return [name for name in PILLARS if pillars[name][1] in targets]


# ---- Luck cycles and flow years ---------------------------------------------------------------


def clash_with_control(first: str, second: str) -> bool:
    """天克地冲: the branches clash and the stems' elements are in a control relation."""
    stems = {R.STEM_ELEMENT[first[0]], R.STEM_ELEMENT[second[0]]}
    branch_key = "".join(sorted((first[1], second[1]), key=BRANCHES.index))
    control = len(stems) == 2 and any(R.CONTROLS[a] in stems for a in stems)
    return branch_key in R.BRANCH_CLASHES and control


def flow_analysis(pillars: dict, flow: str, luck: str | None = None) -> dict:
    """Deterministic interaction of one flow-year pillar with the natal chart (rule R-SY-01)."""
    day = pillars["day"][0]
    special = []
    for name in PILLARS:
        if pillars[name] == flow:
            special.append(f"伏吟:{name}")
        if clash_with_control(pillars[name], flow):
            special.append(f"天克地冲:{name}")
    year_branch = pillars["year"][1]
    if flow[1] == year_branch:
        special.append("值太岁")
    if "".join(sorted((flow[1], year_branch), key=BRANCHES.index)) in R.BRANCH_CLASHES:
        special.append("冲太岁")
    if luck is not None and luck == flow:
        special.append("岁运并临")
    return {
        "pillar": flow,
        "stem_god": R.ten_god(day, flow[0]),
        "branch_main_god": R.ten_god(day, R.HIDDEN_STEMS[flow[1]][0]),
        "stem_relations": {
            name: R.stem_pair_relations(flow[0], pillars[name][0]) for name in PILLARS
        },
        "branch_relations": {
            name: R.branch_pair_relations(flow[1], pillars[name][1]) for name in PILLARS
        },
        "special": special,
    }


def luck_in_year(cycles: list, year: int):
    """The luck pillar whose decade contains ``year`` (None before the first cycle)."""
    for cycle in cycles:
        if cycle["start_year"] <= year <= cycle["end_year"]:
            return cycle["pillar"]
    return None


# ---- Areas of life: traditional associations -------------------------------------------------

DOMAINS = {
    # key: (Chinese name, star gods by gender, palace pillar or None, palace name)
    "marriage": (
        "婚姻",
        {"male": ("正财", "偏财"), "female": ("正官", "七杀")},
        "day",
        "配偶宫（日支）",
    ),
    "career": ("事业", {"male": ("正官", "七杀"), "female": ("正官", "七杀")}, None, None),
    "wealth": ("财运", {"male": ("正财", "偏财"), "female": ("正财", "偏财")}, None, None),
    "study": ("学业", {"male": ("正印", "偏印"), "female": ("正印", "偏印")}, None, None),
    "children": (
        "子女",
        {"male": ("正官", "七杀"), "female": ("食神", "伤官")},
        "hour",
        "子女宫（时支）",
    ),
}


def domain_facts(pillars: dict, domain: str, gender: str) -> dict:
    """Where the traditional stars and palace of one area of life sit in the chart."""
    name, gods_by_gender, palace, palace_name = DOMAINS[domain]
    gods = gods_by_gender[gender]
    day = pillars["day"][0]
    in_stems = [p for p in OTHER if R.ten_god(day, pillars[p][0]) in gods]
    in_hidden = [
        p for p in PILLARS if any(R.ten_god(day, h) in gods for h in R.HIDDEN_STEMS[pillars[p][1]])
    ]
    palace_relations = []
    if palace:
        palace_relations = [
            item["key"]
            for item in positioned_relations(pillars)
            if item["layer"] == "branch" and palace in item["pillars"]
        ]
    return {
        "domain": domain,
        "name": name,
        "gods": list(gods),
        "in_stems": in_stems,
        "in_hidden": in_hidden,
        "palace": palace,
        "palace_name": palace_name,
        "palace_branch": pillars[palace][1] if palace else None,
        "palace_relations": sorted(set(palace_relations)),
    }


def trigger_years(pillars: dict, domain: str, gender: str, first: int, last: int) -> list:
    """Years whose flow pillar "引动" an area of life (rule R-YD-01, declared convention).

    A year is listed when its stem is one of the area's star gods ("星现") or
    its branch clashes or six-combines the palace branch ("宫动"). This is a
    list of symbolic coincidences, not a forecast.
    """
    _, gods_by_gender, palace, _ = DOMAINS[domain]
    gods = gods_by_gender[gender]
    day = pillars["day"][0]
    found = []
    for year in range(first, last + 1):
        flow = R.year_pillar(year)
        reasons = []
        if R.ten_god(day, flow[0]) in gods:
            reasons.append("星现")
        if palace:
            keys = R.branch_pair_relations(flow[1], pillars[palace][1])
            if any(key.startswith(("冲:", "六合:")) for key in keys):
                reasons.append("宫动")
        if reasons:
            found.append({"year": year, "pillar": flow, "reasons": reasons})
    return found


def deep_summary(pillars: dict) -> dict:
    """All Phase 3 derived facts for one chart (no interpretation of fortune)."""
    return {
        "stem_gods": stem_gods(pillars),
        "configs": stem_configs(pillars),
        "revealed_pattern": revealed_pattern(pillars),
        "yongshen": yongshen_by_school(pillars),
        "day_master_roots": [list(item) for item in roots(pillars, pillars["day"][0])],
        "relations": positioned_relations(pillars),
        "stars": {
            star: {
                basis: {
                    "branches": star_branches(star, pillars, basis),
                    "pillars": star_positions(star, pillars, basis),
                }
                for basis in (("day", "year") if star in TRIPLE_STARS else ("day",))
            }
            for star in STAR_NAMES
        },
    }


# ---- Rule cards --------------------------------------------------------------------------------

_SOURCES = {
    "yuanhai": ("《渊海子平》", "宋·徐大升 编（明刊本题 杨淙 增校）", "淵海子平", "yuanhai.txt"),
    "sanming2": ("《三命通会》卷二", "明·万民英", "三命通會/卷二", "sanming2.txt"),
    "sanming3": ("《三命通会》卷三", "明·万民英", "三命通會/卷三", "sanming3.txt"),
    "ditian": ("《滴天髓》", "旧题 京图 撰、刘基 注", "滴天髓/11", "ditian.txt"),
    "qiongtong": ("《穷通宝鉴》", "清·余春台 编", "穷通宝鉴", "qiongtong.txt"),
    "zhenquan": ("《子平真诠》", "清·沈孝瞻", None, None),
}


def _quote(source, section, text):
    work, attribution, page, file = _SOURCES[source]
    return {
        "work": work,
        "attribution": attribution,
        "section": section,
        "text": text,
        "wikisource_page": page,
        "local_file": file,
        "rights": "public domain",
    }


def _cite(source):
    work, attribution, _, _ = _SOURCES[source]
    return {"work": work, "attribution": attribution, "rights": "public domain"}


def _card(rule_id, kind, title, statement, quotes=(), works=(), note=""):
    return {
        "id": rule_id,
        "kind": kind,
        "title": title,
        "statement": statement,
        "quotations": list(quotes),
        "works_cited_without_quotation": [_cite(key) for key in works],
        "note": note,
        "review_status": (
            "Quotations are verbatim from the zh.wikisource.org transcription and are checked "
            "by scripts/phase3/build_knowledge.py; the statement is this project's paraphrase. "
            "No domain expert has reviewed it."
        ),
    }


RULE_CARDS = (
    _card(
        "R-SS-02",
        "deterministic_table",
        "十神五类之间的生克",
        "十神按五行分为比劫、食伤、财、官杀、印五类。依五行次序：比劫生食伤、食伤生财、财生官杀、"
        "官杀生印、印生比劫；比劫克财、食伤克官杀、财克印、官杀克比劫（日主）、印克食伤。",
        quotes=(
            _quote("yuanhai", "卷首·十干", "见庚：为偏官、七杀、官鬼、将星。"),
            _quote("yuanhai", "卷首·十干", "见己：为正财、正妻，剋母，为合神。"),
        ),
        note="Follows mechanically from R-WX-02 and R-SS-01; unit tests check all 5 x 5 class pairs.",
    ),
    _card(
        "R-SS-03",
        "declared_convention",
        "天干十神组合的名称",
        "年、月、时三干同时透出两类十神时，传统上给这种搭配起了名字：食神与七杀为食神制杀，伤官与正官"
        "为伤官见官，偏印与食神为枭神夺食，财与印为财坏印，比劫与财为比劫夺财，正官与印为官印相生，"
        "七杀与印为杀印相生，食伤与财为食伤生财，财与官杀为财生官杀，正官与七杀并见为官杀混杂。"
        "本项目只按“三干是否同时出现”来标注，不衡量远近、强弱与藏干，所以它只是标签，不是吉凶判断。",
        quotes=(
            _quote("yuanhai", "诗诀", "伤官见官，为祸百端"),
            _quote("yuanhai", "诗诀", "食神制杀逢梟，不贫则夭。"),
        ),
        note="The classical verses attach outcomes to these pairings; this project reports only "
        "whether the pairing is present.",
    ),
    _card(
        "R-TG-01",
        "declared_convention",
        "透干与通根",
        "地支藏干若同时出现在年、月、时的天干上，称为透干。天干在某支的藏干中有同五行者，称为在该支通根，"
        "本气根最重，中气、余气次之。也有只认“同一个天干”才算根的做法，本项目按同五行计。",
        works=("ditian", "zhenquan"),
    ),
    _card(
        "R-GJ-02",
        "declared_convention",
        "月令透干取格",
        "月支本气为日主的比肩或劫财时，记作建禄格、阳刃格（阳干）或月劫格（阴干）。否则按本气、中气、"
        "余气的顺序，取月支藏干中第一个透于年、月、时干且不是比劫的，以它的十神名格；都不透则取本气。"
        "各家对多干并透、会合变格的处理不同，这里是一个明确写出的约定。",
        works=("zhenquan",),
        note="No verified quotation: 《子平真诠》 was not available on zh.wikisource.org when the "
        "knowledge base was built, so the work is cited without a quotation.",
    ),
    _card(
        "R-YS-01",
        "declared_convention",
        "扶抑用神",
        "扶抑一派看日主强弱：偏弱时喜印与比劫来生扶，偏强时喜食伤、财、官杀来泄耗克制。"
        "强弱沿用规则 R-QR-01 的三项计数，所以结论随那个约定而定。",
        quotes=(
            _quote("ditian", "衰旺论", "能知衰旺之真機，其於立命之奧，思過半矣。"),
            _quote("ditian", "衰旺论（注）", "旺則宜洩宜傷，衰則喜幫喜助，子平之理也"),
            _quote("ditian", "体用论", "道有體用，不可以一端論也，要在扶之抑之得其宜。"),
        ),
    ),
    _card(
        "R-YS-02",
        "declared_convention",
        "调候（简化）",
        "调候一派看出生月份的寒暖燥湿。本项目只取最粗的一条：生于亥、子、丑月（冬）取火暖局，"
        "生于巳、午、未月（夏）取水润局，其余月份不特别论调候。《穷通宝鉴》按日干逐月给出的取用"
        "远比这细，本项目没有采用那张表。",
        quotes=(
            _quote(
                "qiongtong",
                "三冬乙木",
                "喜用丙火解冻，则花木有向阳之意",
            ),
            _quote(
                "qiongtong",
                "五行总论",
                "故高者抑之使平，下者举之使崇，或益其不及，或损其太过。",
            ),
        ),
    ),
    _card(
        "R-YS-03",
        "declared_convention",
        "格局用神与三派并列",
        "格局一派以月令所取的格局之神为用神，并分顺用（正官、财、印、食神）与逆用（七杀、伤官、劫刃）。"
        "扶抑、调候、格局三派回答的不是同一个问题，结论可以不同；没有说明流派时，“用神是什么”"
        "没有唯一答案，应分别列出。",
        works=("zhenquan",),
        note="No verified quotation (see R-GJ-02).",
    ),
    _card(
        "R-SM-01",
        "deterministic_table",
        "禄、阳刃、天乙贵人（以日干查）",
        "禄：甲寅、乙卯、丙戊巳、丁己午、庚申、辛酉、壬亥、癸子。阳刃为禄前一位，只有阳干有："
        "甲卯、丙戊午、庚酉、壬子。天乙贵人：甲戊庚见丑未，乙己见子申，丙丁见亥酉，壬癸见卯巳，辛见寅午。"
        "天乙贵人另有“甲戊兼牛羊、庚辛逢马虎”的歌诀版本，本项目用上面这一版。",
        quotes=(
            _quote("sanming3", "论十干禄", "甲禄寅，乙禄卯，庚禄申，辛禄酉，壬禄亥，癸禄子"),
            _quote(
                "yuanhai",
                "论阳刃",
                "以禄前一位是也；如甲禄在寅，卯为阳刃。",
            ),
            _quote(
                "yuanhai",
                "论阳刃",
                "甲丙戊庚壬五阳有刃，乙丁己辛癸五阴无刃，故名阳刃。",
            ),
            _quote("sanming3", "论天乙贵人", "寄火贵于酉亥，寄水贵于卯巳"),
        ),
    ),
    _card(
        "R-SM-02",
        "declared_convention",
        "驿马、桃花（咸池）、华盖（以三合局查）",
        "申子辰：马在寅、桃花在酉、华盖在辰；寅午戌：马在申、桃花在卯、华盖在戌；巳酉丑：马在亥、"
        "桃花在午、华盖在丑；亥卯未：马在巳、桃花在子、华盖在未。古书以年支所在的三合局查，"
        "后来也有以日支查的；两种查法结果可能不同，回答时须说明以哪一支为准。",
        quotes=(
            _quote("sanming3", "论驿马", "申子辰人，马在寅而五阳干乘之。"),
            _quote("sanming3", "论驿马", "寅午戌生人，马在申而五阳干乘之。"),
            _quote(
                "sanming2",
                "论咸池",
                "申子辰酉、亥卯未子即长生第二位沐浴之宫是也。一名败神，一名桃花煞",
            ),
        ),
    ),
    _card(
        "R-SY-01",
        "deterministic_table",
        "流年、大运与原局的关系",
        "流年干支按 (Y − 4) mod 60 求出后：流年天干对日主论十神；流年地支与四柱地支逐一查合冲刑害破。"
        "流年与某柱干支完全相同为伏吟；地支相冲而且天干五行相克为天克地冲（反吟）；"
        "流年地支与年支相同为值太岁，相冲为冲太岁；流年与当前大运干支相同为岁运并临。"
        "这些都是可以机械判定的关系，本身不说明会发生什么事。",
        quotes=(
            _quote("yuanhai", "论大运", "子平之法，大运看支，岁君看干，交运同接木。"),
            _quote("yuanhai", "论阳刃", "忌反吟伏吟"),
        ),
    ),
    _card(
        "R-LQ-02",
        "traditional_association",
        "四柱宫位的传统对应",
        "传统上以年柱象祖上与父母、月柱象父母兄弟、日柱象自身与配偶（日支称配偶宫）、时柱象子女。"
        "这是象征性的对应，不能据此断定任何人的实际情况。",
        quotes=(
            _quote("sanming2", "论年月日时", "年为根，月为苗，日为花，时为实"),
            _quote(
                "yuanhai",
                "诗诀",
                "日上傷官，難為配偶，時上傷官，子嗣無傳。",
            ),
        ),
        note="The second quotation shows the classical day-to-spouse and hour-to-children "
        "association; its claim about outcomes is not adopted.",
    ),
    _card(
        "R-YD-01",
        "declared_convention",
        "流年“引动”的判定",
        "本项目把下面两种情形记为某一方面被流年引动：流年天干是该方面的星（如男命看婚姻的财、女命的"
        "官杀），记作“星现”；流年地支与对应宫位的地支相冲或六合，记作“宫动”。这只是把符号上的巧合"
        "列出来，传统上用它来提示“值得留意的年份”，并不是对事件的预测。",
        works=("yuanhai", "sanming2"),
        note="Declared convention of this project; practitioners use many further conditions.",
    ),
)
RULES_BY_ID = {card["id"]: card for card in RULE_CARDS}

__all__ = [
    "CLASSES",
    "CONFIGS",
    "DOMAINS",
    "GOD_CLASS",
    "RULE_CARDS",
    "STAR_NAMES",
    "branch_card",
    "branch_card_text",
    "class_controls",
    "class_element",
    "class_generates",
    "deep_summary",
    "domain_facts",
    "element_class",
    "element_relation",
    "flow_analysis",
    "fuyi",
    "luck_in_year",
    "pattern_use",
    "positioned_relations",
    "revealed_pattern",
    "roots",
    "star_branches",
    "star_positions",
    "stem_configs",
    "tiaohou",
    "trigger_years",
    "yongshen_by_school",
]
_ = CYCLE
