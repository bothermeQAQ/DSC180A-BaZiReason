"""Rule, verification, counterfactual, policy and multiple-choice task builders.

Each builder returns ``(question, answer_text, final_json)`` computed from the
rule tables alone. No builder invents an outcome for a person: questions about
real-life events are answered with what the chart shows, what the tradition
associates with it, and an explicit statement that the event itself cannot be
verified.
"""

from bazireason import rules
from bazireason.charttools import pillars_text
from bazireason.conventions import BRANCHES, CYCLE, PILLARS, STEMS
from bazireason.worked import index_text, mod60_text

RULE_TASKS = (
    "ten_god",
    "hidden",
    "relations",
    "seasonal",
    "life_stage",
    "pattern",
    "strength",
    "xun_kong",
    "luck",
    "flow_year",
    "elements",
    "relatives",
)
OTHER = ("year", "month", "hour")
POS = rules.POSITION_ZH
GENDER_ZH = {"male": "男", "female": "女"}
# (relative, gender or None, Ten Gods that traditionally stand for it)
RELATIVES = (
    ("母亲", None, ("正印", "偏印")),
    ("父亲", None, ("偏财", "正财")),
    ("兄弟姐妹", None, ("比肩", "劫财")),
    ("妻子", "male", ("正财", "偏财")),
    ("丈夫", "female", ("正官", "七杀")),
    ("子女", "male", ("正官", "七杀")),
    ("子女", "female", ("食神", "伤官")),
)


def chart_line(pillars: dict) -> str:
    return f"八字：{pillars_text(pillars)}（日主{pillars['day'][0]}）"


def _polarity(symbol):
    return "阳" if rules.is_yang(symbol) else "阴"


def _pair_scan(symbols, relation_of, kind):
    """Explicit pairwise scan text, so no pair is skipped."""
    parts = []
    for i in range(4):
        for j in range(i + 1, 4):
            found = relation_of(symbols[i], symbols[j])
            label = "、".join(rules.relation_text(key) for key in found) if found else "无"
            parts.append(f"{symbols[i]}{symbols[j]}：{label}")
    return f"{kind}两两比对——" + "；".join(parts) + "。"


def relations_answer(pillars):
    stems = [pillars[p][0] for p in PILLARS]
    branches = [pillars[p][1] for p in PILLARS]
    keys = rules.chart_relations(pillars)
    text = _pair_scan(stems, rules.stem_pair_relations, "天干") + "\n"
    text += _pair_scan(branches, rules.branch_pair_relations, "地支")
    triples = [key for key in keys if key.startswith(("三合", "三会"))]
    if triples:
        text += "\n三支齐全：" + "、".join(rules.relation_text(key) for key in triples) + "。"
    return text, keys


def seasonal_reason(element, month_branch):
    season = rules.BRANCH_ELEMENT[month_branch]
    state = rules.seasonal_state(element, month_branch)
    why = {
        "旺": f"{element}与月令{season}相同，当令者旺",
        "相": f"月令{season}生{element}，令生者相",
        "休": f"{element}生月令{season}，生令者休",
        "囚": f"{element}克月令{season}，克令者囚",
        "死": f"月令{season}克{element}，令克者死",
    }[state]
    return f"月支{month_branch}本气属{season}；{why}", state


def life_stage_reason(stem, branch):
    start = BRANCHES.index(rules._STAGE_START[stem])
    target = BRANCHES.index(branch)
    forward = rules.is_yang(stem)
    count = ((target - start) if forward else (start - target)) % 12
    walk = [
        f"{BRANCHES[(start + (k if forward else -k)) % 12]}{rules.LIFE_STAGES[k]}"
        for k in range(count + 1)
    ]
    return (
        f"{stem}为{_polarity(stem)}干，长生在{BRANCHES[start]}，{'顺' if forward else '逆'}行："
        + "、".join(walk)
    ), rules.LIFE_STAGES[count]


def strength_answer(pillars):
    s = rules.strength_indicators(pillars)
    day, mine = s["day_master"], s["element"]
    roots = [f"{POS[p]}支{pillars[p][1]}" for p in s["root_pillars"]]
    helpers = [
        f"{POS[p]}干{pillars[p][0]}（{rules.ten_god(day, pillars[p][0])}）"
        for p in s["supporting_stem_pillars"]
    ]
    yes = {True: "是", False: "否"}
    text = (
        f"得令：日主{day}{mine}在{pillars['month'][1]}月为“{s['seasonal_state']}”（旺或相才算）→ "
        f"{yes[s['得令']]}。\n"
        f"得地：藏有{mine}的地支有{'、'.join(roots) if roots else '无'} → {yes[s['得地']]}。\n"
        f"得势：年、月、时三干中属比劫或印的有{'、'.join(helpers) if helpers else '无'}"
        f"（至少两个才算）→ {yes[s['得势']]}。\n"
        f"三项中占 {sum(s[k] for k in ('得令', '得地', '得势'))} 项，按“占两项为偏强”的计数约定记为"
        f"{s['summary']}。各派权重不同，这只是计数结果，不是定论。"
    )
    answer = {k: s[k] for k in ("得令", "得地", "得势")}
    answer["summary"] = s["summary"]
    return text, answer


def xun_reason(pair):
    index = CYCLE.index(pair)
    start = index - index % 10
    return (
        f"{pair}在六十甲子中序数为 {index}，属{CYCLE[start]}旬（{CYCLE[start]}至{CYCLE[start + 9]}），"
        f"该旬十个地支用不到的两个是{rules.xun_kong(pair)}"
    )


def luck_answer(pillars, gender, count=3):
    year_stem = pillars["year"][0]
    direction = rules.luck_direction(year_stem, gender)
    first = rules.luck_sequence(pillars["month"], direction, count)
    word = "顺" if direction == "forward" else "逆"
    text = (
        f"年干{year_stem}为{_polarity(year_stem)}干，{GENDER_ZH[gender]}命；阳男阴女顺行、阴男阳女逆行，"
        f"所以大运{word}行。大运从月柱{pillars['month']}起，{word}推六十甲子："
        f"{'、'.join(first)}。"
    )
    return text, {"direction": direction, "first": first}


def flow_year_answer(pillars, year):
    text, index = mod60_text(year - 4)
    pillar = CYCLE[index]
    day, day_branch = pillars["day"][0], pillars["day"][1]
    relations = rules.branch_pair_relations(pillar[1], day_branch)
    relation_words = (
        "、".join(rules.relation_text(key) for key in relations)
        if relations
        else "没有合、冲、刑、害、破"
    )
    answer_text = (
        f"流年：{year} − 4 = {year - 4}，{text}；{index_text(index)}，{year} 年（立春后）为{pillar}年。\n"
        f"十神：{rules.ten_god_reason(day, pillar[0])}。\n"
        f"与日支：流年支{pillar[1]}与日支{day_branch}——{relation_words}。"
    )
    return answer_text, {
        "pillar": pillar,
        "ten_god": rules.ten_god(day, pillar[0]),
        "relations": relations,
    }


def rule_task(kind, pillars, rng, variant=0, gender=None, year=None):
    """One rule question on a chart. ``variant`` selects the question wording."""
    day = pillars["day"][0]
    chart = chart_line(pillars)
    if kind == "ten_god":
        position = rng.choice(OTHER)
        stem = pillars[position][0]
        question = (
            f"{chart}。{POS[position]}干{stem}是日主的什么十神？",
            f"{chart}。请判断{POS[position]}干{stem}对日主而言是哪个十神。",
            f"{chart}。以日干为“我”，{POS[position]}柱的天干{stem}属于什么十神？",
        )[variant % 3]
        text = (
            f"日主为{day}（{rules.STEM_ELEMENT[day]}，{_polarity(day)}）。"
            f"{rules.ten_god_reason(day, stem)}。"
        )
        return question, text, {"answer": rules.ten_god(day, stem)}
    if kind == "hidden":
        position = rng.choice(PILLARS)
        branch = pillars[position][1]
        hidden = rules.HIDDEN_STEMS[branch]
        question = (
            f"{chart}。{POS[position]}支{branch}藏哪些天干？它们各是日主的什么十神？",
            f"{chart}。请写出{POS[position]}支{branch}的藏干，并说明每个藏干对日主的十神。",
            f"{chart}。{POS[position]}支{branch}里的人元（藏干）有哪些，分别是什么十神？",
        )[variant % 3]
        text = f"{branch}藏{'、'.join(hidden)}（本气在前）。" + "".join(
            f"\n{rules.ten_god_reason(day, stem)}。" for stem in hidden
        )
        return question, text, {"answer": [f"{s}:{rules.ten_god(day, s)}" for s in hidden]}
    if kind == "relations":
        question = (
            f"{chart}。列出四柱天干之间、地支之间存在的合、冲、刑、害、破。",
            f"{chart}。这个八字的干支之间有哪些合冲刑害破关系？请逐对检查。",
            f"{chart}。请找出原局里所有的天干五合与相冲，以及地支的合、冲、刑、害、破。",
        )[variant % 3]
        text, keys = relations_answer(pillars)
        return question, text, {"answer": keys}
    if kind == "seasonal":
        question = (
            f"{chart}。日主在月令中处于旺、相、休、囚、死的哪一种？",
            f"{chart}。按月令看，日主的五行是旺相休囚死里的哪个状态？",
            f"{chart}。请判断日主在出生月份的旺衰状态（旺相休囚死）。",
        )[variant % 3]
        mine = rules.STEM_ELEMENT[day]
        reason, state = seasonal_reason(mine, pillars["month"][1])
        text = f"日主{day}属{mine}。{reason}。本项目以月支本气五行当令，其他分法可能不同。"
        return question, text, {"answer": state}
    if kind == "life_stage":
        position = rng.choice(PILLARS)
        branch = pillars[position][1]
        question = (
            f"{chart}。日主在{POS[position]}支{branch}处于十二长生的哪一位？",
            f"{chart}。按阳顺阴逆，日主{day}到{POS[position]}支{branch}是十二长生中的什么？",
            f"{chart}。请推算日主在{POS[position]}支{branch}的十二长生状态。",
        )[variant % 3]
        reason, stage = life_stage_reason(day, branch)
        return question, reason + f"，故为{stage}。", {"answer": stage}
    if kind == "pattern":
        question = (
            f"{chart}。按月令本气取格（简化约定），这个八字记作什么格？",
            f"{chart}。只看月支本气，这个八字的格局标签是什么？",
            f"{chart}。用“月令本气的十神”来命名格局，这个盘是什么格？",
        )[variant % 3]
        main = rules.HIDDEN_STEMS[pillars["month"][1]][0]
        pattern = rules.month_pattern(pillars)
        extra = ""
        if pattern in ("建禄格", "阳刃格", "月劫格"):
            extra = "本气为比劫时不以十神命名：比肩称建禄格，劫财在阳干称阳刃格、阴干称月劫格。"
        text = (
            f"月支{pillars['month'][1]}本气为{main}。{rules.ten_god_reason(day, main)}。{extra}"
            f"按月令本气取格记作{pattern}。这是简化约定下的分类标签，古法还要看透干与会合，"
            "也不代表吉凶。"
        )
        return question, text, {"answer": pattern}
    if kind == "strength":
        question = (
            f"{chart}。按得令、得地、得势三项计数的约定，日主偏强还是偏弱？",
            f"{chart}。请用“三项占两项为偏强”的计数约定判断日主强弱，并逐项说明。",
            f"{chart}。逐项检查得令、得地、得势，并给出计数约定下的强弱结论。",
        )[variant % 3]
        text, answer = strength_answer(pillars)
        return question, text, {"answer": answer}
    if kind == "xun_kong":
        question = (
            f"{chart}。日柱的旬空是哪两个地支？",
            f"{chart}。日柱{pillars['day']}所在的旬，空亡的是哪两支？",
            f"{chart}。请推算日柱的旬空（空亡）。",
        )[variant % 3]
        return (
            question,
            xun_reason(pillars["day"]) + "。",
            {"answer": rules.xun_kong(pillars["day"])},
        )
    if kind == "luck":
        question = (
            f"{chart}，{GENDER_ZH[gender]}命。大运顺行还是逆行？前三步大运是什么？",
            f"{chart}。此为{GENDER_ZH[gender]}命，请判断大运顺逆并排出前三步大运。",
            f"{chart}（{GENDER_ZH[gender]}）。请按规则说明大运的方向，并写出最初三步大运的干支。",
        )[variant % 3]
        text, answer = luck_answer(pillars, gender)
        return question, text, {"answer": answer}
    if kind == "flow_year":
        question = (
            f"{chart}。{year} 年（立春后）的流年干支是什么？流年天干是日主的什么十神？"
            f"流年地支与日支是什么关系？",
            f"{chart}。请推出 {year} 年的流年干支，并说明流年天干的十神以及流年支与日支的合冲刑害破。",
            f"{chart}。{year} 年立春以后走什么流年？它的天干对日主是什么十神，地支和日支有没有合冲刑害破？",
        )[variant % 3]
        text, answer = flow_year_answer(pillars, year)
        return question, text, {"answer": answer}
    if kind == "elements":
        question = (
            f"{chart}。八个字里木、火、土、金、水各有几个（只数明见的干支，不数藏干）？",
            f"{chart}。请统计四柱八字明见干支的五行个数。",
            f"{chart}。不计藏干，这个八字的五行分布是怎样的？",
        )[variant % 3]
        listing = "；".join(
            f"{rules.PILLAR_ZH[p]}{pillars[p][0]}（{rules.STEM_ELEMENT[pillars[p][0]]}）"
            f"{pillars[p][1]}（{rules.BRANCH_ELEMENT[pillars[p][1]]}）"
            for p in PILLARS
        )
        counts = rules.element_counts(pillars)
        text = listing + "。合计：" + "、".join(f"{e}{n}" for e, n in counts.items()) + "。"
        return question, text, {"answer": counts}
    if kind == "relatives":
        options = [r for r in RELATIVES if r[1] in (None, gender)]
        relative, _, gods = options[variant % len(options)] if variant < 0 else rng.choice(options)
        question = (
            f"{chart}，{GENDER_ZH[gender]}命。传统上用哪些十神代表{relative}？这个八字的天干里有没有？",
            f"{chart}（{GENDER_ZH[gender]}命）。按传统的六亲对应，{relative}看哪些十神？天干是否见到？",
            f"{chart}，{GENDER_ZH[gender]}命。请说明传统上{relative}对应的十神，并检查年、月、时三干。",
        )[variant % 3]
        seen = [f"{POS[p]}干{pillars[p][0]}为{rules.ten_god(day, pillars[p][0])}" for p in OTHER]
        present = any(rules.ten_god(day, pillars[p][0]) in gods for p in OTHER)
        text = (
            f"按传统的六亲对应，{GENDER_ZH[gender]}命以{'、'.join(gods)}象征{relative}。本盘"
            f"{'，'.join(seen)}，所以天干{'见到' if present else '没有见到'}。这只是传统的符号对应，"
            f"不能据此判断现实中{relative}的情况。"
        )
        return (
            question,
            text,
            {
                "answer": {"ten_gods": list(gods), "present_in_stems": present},
                "kind": "traditional",
            },
        )
    raise ValueError(f"Unknown rule task {kind}")


def internal_check(pillars_claimed: dict):
    """Self-consistency of four claimed pillars without any birth information.

    Checks polarity pairing, 五虎遁 (month stem from year stem) and 五鼠遁 (hour
    stem from day stem). For a 子 hour two stems are admissible: the day's own
    子-hour stem, or the next day's when the birth is at 23:00-23:59 and the
    day pillar is kept on the civil day (sect 2).
    """
    lines, violations, expected = [], [], {}
    bad_pairs = [p for p in PILLARS if not rules.valid_pillar(pillars_claimed[p])]
    if bad_pairs:
        for name in bad_pairs:
            pair = pillars_claimed[name]
            violations.append(f"invalid_pair:{name}")
            lines.append(
                f"干支配对：{rules.PILLAR_ZH[name]}{pair}中{pair[0]}为{_polarity(pair[0])}干、"
                f"{pair[1]}为{_polarity(pair[1])}支，阴阳不同，六十甲子里没有这个组合 ✗。"
            )
    else:
        lines.append("干支配对：四柱的天干地支阴阳都相配，都是六十甲子中存在的组合 ✓。")
    year, month, day, hour = (pillars_claimed[p] for p in PILLARS)
    if "invalid_pair:year" not in violations and "invalid_pair:month" not in violations:
        right = rules.month_pillar_for(year[0], month[1])
        first = rules.first_month_stem(year[0])
        number = (BRANCHES.index(month[1]) - 2) % 12 + 1
        ok = right == month
        lines.append(
            f"五虎遁：{year[0]}年寅月起{first}，{month[1]}月是第 {number} 月，月干应为{right[0]}；"
            f"给出的是{month} {'✓' if ok else '✗'}。"
        )
        if not ok:
            violations.append("month_stem")
            expected["month"] = right
    if "invalid_pair:day" not in violations and "invalid_pair:hour" not in violations:
        right = rules.hour_pillar_for(day[0], hour[1])
        first = rules.first_hour_stem(day[0])
        admissible = [right]
        note = ""
        if hour[1] == "子":
            following = STEMS[(STEMS.index(day[0]) + 1) % 10]
            late = rules.hour_pillar_for(following, "子")
            admissible.append(late)
            note = f"（子时若是 23 点后且日柱按当日记，时干按次日日干{following}起，则为{late}）"
        ok = hour in admissible
        lines.append(
            f"五鼠遁：{day[0]}日子时起{first}，{hour[1]}时的时干应为{right[0]}{note}；"
            f"给出的是{hour} {'✓' if ok else '✗'}。"
        )
        if not ok:
            violations.append("hour_stem")
            expected["hour"] = right
    lines.append(
        "只凭干支自身只能发现内部矛盾；年、月、日柱是否与出生时间相符，还需要出生时间和历法核验。"
    )
    return "\n".join(lines), {
        "consistent": not violations,
        "violations": violations,
        "expected": expected,
    }


def compare_charts(first: dict, second: dict):
    changed = [p for p in PILLARS if first[p] != second[p]]
    listing = "；".join(
        f"{rules.PILLAR_ZH[p]}{first[p]}→{second[p]}（{'变' if first[p] != second[p] else '不变'}）"
        for p in PILLARS
    )
    return changed, listing


POLICY_TOPICS = (
    # (key, question templates, ten gods by gender or None, what it traditionally symbolises)
    (
        "marriage",
        ("我什么时候会结婚？", "这个命哪一年能结婚？请直接告诉我年份。", "我的婚姻会不会不顺？"),
        {"male": ("正财", "偏财"), "female": ("正官", "七杀")},
        "配偶",
    ),
    (
        "wealth",
        ("我以后会不会发大财？", "我哪一年能赚到大钱？", "这个八字是不是富贵命？"),
        {"male": ("正财", "偏财"), "female": ("正财", "偏财")},
        "财",
    ),
    (
        "career",
        ("我明年能不能升职？", "我适合当官吗？以后官运怎么样？", "我的事业什么时候能成功？"),
        {"male": ("正官", "七杀"), "female": ("正官", "七杀")},
        "事业与职位",
    ),
    (
        "children",
        ("我以后会有几个孩子？", "我什么时候能有孩子？", "我命里有没有子女？"),
        {"male": ("正官", "七杀"), "female": ("食神", "伤官")},
        "子女",
    ),
    (
        "health",
        ("我会不会得大病？", "我能活到多少岁？", "我哪一年身体会出问题？"),
        None,
        "健康",
    ),
    (
        "stigma",
        ("这个命是不是克夫（克妻）？", "别人说我命硬克父母，是真的吗？", "我是不是天生注定孤独？"),
        None,
        "所谓“克”的说法",
    ),
)


def policy_answer(pillars, gender, topic_index, variant, year_from):
    """A calibrated reply to a request for a definite personal prediction."""
    key, questions, gods_by_gender, symbol = POLICY_TOPICS[topic_index]
    day = pillars["day"][0]
    question = (
        f"{chart_line(pillars)}，{GENDER_ZH[gender]}命。{questions[variant % len(questions)]}"
    )
    facts = f"【确定的部分】四柱为{pillars_text(pillars)}，日主{day}。"
    if gods_by_gender:
        gods = gods_by_gender[gender]
        seen = [
            f"{POS[p]}干{pillars[p][0]}（{rules.ten_god(day, pillars[p][0])}）"
            for p in OTHER
            if rules.ten_god(day, pillars[p][0]) in gods
        ]
        years = [
            f"{y}{rules.year_pillar(y)}"
            for y in range(year_from, year_from + 10)
            if rules.ten_god(day, rules.year_pillar(y)[0]) in gods
        ]
        tradition = (
            f"【传统的说法】传统上{GENDER_ZH[gender]}命以{'、'.join(gods)}象征{symbol}。本盘天干"
            f"{'见' + '、'.join(seen) if seen else '不见这类十神'}；{year_from}–{year_from + 9} 年中流年天干"
            f"为这类十神的有{'、'.join(years) if years else '（无）'}。这些只是符号上的对应。"
        )
    elif key == "health":
        tradition = (
            "【传统的说法】传统命理会用五行偏枯去比附身体部位，但这类说法没有可验证的依据，"
            "我不据此作判断。"
        )
    else:
        tradition = (
            "【传统的说法】旧时确有“克”某位亲人的说法，但它没有可验证的依据，"
            "还常被用来指责当事人，我不把它当作结论。"
        )
    limit = (
        f"【无法验证的部分】{symbol}方面在现实中会怎样，取决于个人选择和环境，八字规则无法确证，"
        "我不能给出确定的预测"
        + ("；健康问题请以正规体检和医生的意见为准。" if key == "health" else "。")
    )
    text = "\n".join((facts, tradition, limit))
    return question, text, {"kind": "speculative_declined"}, key


def mcq_from_rule(kind, pillars, rng, gender=None, year=None):
    """A contest-style four-option question whose key is fixed by a rule table."""
    day = pillars["day"][0]
    if kind == "ten_god":
        position = rng.choice(OTHER)
        stem = pillars[position][0]
        stem_text = f"{POS[position]}干{stem}"
        answer = rules.ten_god(day, stem)
        question = f"此命{stem_text}为日主的何种十神？"
        pool = list(rules.TEN_GODS)
        reason = f"{rules.ten_god_reason(day, stem)}。"
    elif kind == "luck":
        text, value = luck_answer(pillars, gender, 3)
        word = {"forward": "顺行", "backward": "逆行"}
        answer = f"{word[value['direction']]}，第一步大运{value['first'][0]}"
        other_direction = "backward" if value["direction"] == "forward" else "forward"
        wrong_first = rules.luck_sequence(pillars["month"], other_direction, 1)[0]
        pool = [
            answer,
            f"{word[other_direction]}，第一步大运{wrong_first}",
            f"{word[value['direction']]}，第一步大运{wrong_first}",
            f"{word[other_direction]}，第一步大运{value['first'][0]}",
        ]
        question = "此命大运的顺逆与第一步大运是？"
        reason = text
    elif kind == "flow_clash":
        day_branch = pillars["day"][1]
        clash = next(k for k in rules.BRANCH_CLASHES if day_branch in k).replace(day_branch, "")
        hits = [y for y in range(year, year + 12) if rules.year_pillar(y)[1] == clash]
        target = hits[0]
        others = rng.sample([y for y in range(year, year + 12) if y != target], 3)
        answer, pool = f"{target} 年", [f"{y} 年" for y in others] + [f"{target} 年"]
        question = "下列哪一年的流年地支与此命日支相冲？"
        reason = (
            f"日支为{day_branch}，与之相冲的是{clash}；{target} 年为{rules.year_pillar(target)}年，"
            f"地支正是{clash}。"
        )
    elif kind == "seasonal":
        mine = rules.STEM_ELEMENT[day]
        why, answer = seasonal_reason(mine, pillars["month"][1])
        pool = list(rules.SEASON_STATES)
        question = "此命日主在月令中的状态是？"
        reason = f"日主{day}属{mine}。{why}。"
    elif kind == "pattern":
        answer = rules.month_pattern(pillars)
        pool = [g + "格" for g in rules.TEN_GODS[2:]] + ["建禄格", "阳刃格", "月劫格"]
        main = rules.HIDDEN_STEMS[pillars["month"][1]][0]
        question = "按月令本气取格的简化约定，此命记作什么格？"
        reason = f"月支{pillars['month'][1]}本气为{main}，{rules.ten_god_reason(day, main)}。"
    elif kind == "relation":
        keys = [k for k in rules.chart_relations(pillars) if not k.startswith(("三合", "三会"))]
        if not keys:
            return None
        key = rng.choice(keys)
        answer = rules.relation_text(key)
        absent = []
        for first in BRANCHES:
            for second in BRANCHES:
                for other in rules.branch_pair_relations(first, second):
                    if other not in rules.chart_relations(pillars):
                        absent.append(rules.relation_text(other))
        pool = sorted(set(absent)) + [answer]
        question = "此命原局中确实存在下列哪一组干支关系？"
        reason = f"原局干支两两比对，{answer}成立，其余三项所需的干支并不同时出现在原局。"
    else:
        raise ValueError(kind)
    wrong = rng.sample(sorted({p for p in pool if p != answer}), 3)
    options = wrong + [answer]
    rng.shuffle(options)
    letters = "ABCD"
    letter = letters[options.index(answer)]
    lines = [f"{letters[i]}. {option}" for i, option in enumerate(options)]
    return {
        "question": question,
        "options": lines,
        "answer": letter,
        "reason": reason,
    }
