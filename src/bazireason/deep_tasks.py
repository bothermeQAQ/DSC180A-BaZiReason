"""Phase 3 task builders: table drills and multi-step traditional reasoning.

Each builder returns ``(question, answer_text, final_json)``. Labels come from
``bazireason.rules`` and ``bazireason.deep_rules`` only. The answer text writes
out the table rows and intermediate values it relies on, because the Phase 2
error analysis showed that the model had learned the form of an answer without
the tables behind it (wrong 生克 direction, wrong 刑/害 partners).

No builder states an outcome for a person. Where a question has no unique
rule-determined answer the final JSON says ``"determinate": false`` and names
what is missing.
"""

from bazireason import deep_rules as D
from bazireason import rules as R
from bazireason import textstyle
from bazireason import train_tasks as T
from bazireason.conventions import BRANCHES, CYCLE, PILLARS, STEMS
from bazireason.worked import index_text, mod60_text

POS = R.POSITION_ZH
OTHER = D.OTHER
GENDER_ZH = T.GENDER_ZH
GENERATION_ORDER = "相生次序：木→火→土→金→水→木"
CONTROL_ORDER = "相克次序：木→土→水→火→金→木"
RANK_ZH = ("本气", "中气", "余气")
HINTS = {
    "answer": "结论JSON 需含 answer。",
    "gods": "结论JSON 需含 stems、configs。",
    "strength": "结论JSON 需含 answer（得令、得地、得势、summary）。",
    "pattern": "结论JSON 需含 answer、basis。",
    "yongshen": "结论JSON 需含 school 及该派的结论。",
    "interact": "结论JSON 需含 answer、adjacent。",
    "flow": "结论JSON 需含 pillar、ten_god、branch_relations、special。",
    "domain": "结论JSON 需含 kind、gods、in_stems、trigger_years。",
    "claims": "结论JSON 需含 verdicts（正确、错误或无法验证）与 corrections。",
    "ambig": "结论JSON 需含 determinate、needs，能确定时再给 answer。",
    "overview": "结论JSON 需含 kind、strength、pattern、configs。",
    "followup": "结论JSON 需含 revised 与 answer。",
}


def polarity(symbol: str) -> str:
    return "阳" if R.is_yang(symbol) else "阴"


def explicit() -> bool:
    """Second-round wording: Ten Gods are read from a table that is written out first.

    The first round derived each Ten God in a single sentence and got the 生克
    direction wrong in about a quarter of them once several had to be found in
    one answer, although the one-table drills were right.
    """
    return textstyle.version() >= 3


def no_titles() -> bool:
    """Round-3 wording: answers never name a book (sources stay on the rule cards)."""
    return textstyle.version() >= 4


def finer_tables() -> str:
    return "更细的逐月取用表" if no_titles() else "《穷通宝鉴》逐月的取用更细"


def class_chain(day: str) -> str:
    return "→".join(f"{D.class_element(day, c)}{c}" for c in D.CLASSES)


def cycle_index_text(pair: str) -> str:
    """Index of a sexagenary pair by formula, with its 旬 and void branches."""
    s, b = STEMS.index(pair[0]), BRANCHES.index(pair[1])
    index = CYCLE.index(pair)
    start = index - index % 10
    return (
        f"{pair[0]}是第 {s} 干，{pair[1]}是第 {b} 支（都从 0 数）；序数 = (6×{s} − 5×{b}) mod 60 = "
        f"{6 * s - 5 * b} mod 60 = {index}，旬首为{CYCLE[start]}，旬空{R.xun_kong(pair)}。"
    )


def relation_tables(pillars: dict) -> str:
    """The pair tables a relation question needs: stem combinations and clashes, branch cards."""
    branches = dict.fromkeys(pillars[name][1] for name in PILLARS)
    return (
        "天干五合：甲己、乙庚、丙辛、丁壬、戊癸；天干相冲：甲庚、乙辛、丙壬、丁癸。\n地支卡片：\n"
        + "\n".join(D.branch_card_text(branch) for branch in branches)
    )


def god_table(day: str) -> str:
    """The ten stems with their Ten Gods for one day master, derived from the 相生 cycle."""
    mine = R.STEM_ELEMENT[day]
    same, other = polarity(day), "阴" if R.is_yang(day) else "阳"
    groups = []
    for cls in D.CLASSES:
        element = D.class_element(day, cls)
        stems = [s for s in STEMS if R.STEM_ELEMENT[s] == element]
        groups.append("、".join(f"{s}{R.ten_god(day, s)}" for s in stems))
    return (
        f"十神表（日主{day}，{mine}，{same}）：从{mine}起按相生次序，{class_chain(day)}；"
        f"{same}干取比肩、食神、偏财、七杀、偏印，{other}干取劫财、伤官、正财、正官、正印。"
        f"即：{'；'.join(groups)}。"
    )


def god_reason(day: str, stem: str) -> str:
    """One Ten God derived on its own (no table in the answer)."""
    if not explicit():
        return R.ten_god_reason(day, stem)
    element, cls = R.STEM_ELEMENT[stem], D.GOD_CLASS[R.ten_god(day, stem)]
    same = R.is_yang(day) == R.is_yang(stem)
    return (
        f"日主{day}属{R.STEM_ELEMENT[day]}，按相生次序{class_chain(day)}；{stem}属{element}，是{cls}；"
        f"{stem}为{polarity(stem)}干、{day}为{polarity(day)}干，阴阳{'相同' if same else '相异'}，"
        f"故为{R.ten_god(day, stem)}"
    )


def read(day: str, stem: str) -> str:
    """A Ten God read from the table written earlier in the same answer."""
    return f"查表，{stem}为{R.ten_god(day, stem)}" if explicit() else R.ten_god_reason(day, stem)


def join(items, empty="无") -> str:
    items = list(items)
    return "、".join(items) if items else empty


def relation_words(keys, empty="无") -> str:
    return join((R.relation_text(key) for key in keys), empty)


# ---- Table drills ---------------------------------------------------------------------------

DRILLS = (
    "element",
    "ten_god",
    "hidden",
    "branch_pair",
    "branch_card",
    "stem_pair",
    "cycle",
    "season",
    "stage",
    "star",
    "class",
    "luck_dir",
    "month_stem",
    "hour_stem",
    "year",
)


# Second-round additions: the Ten-God table itself, and lookups answered without any written
# derivation, which is what a one-pass check of a supplied chart needs.
DRILLS_V3 = DRILLS + ("god_table", "silent_hour", "silent_month", "silent_check", "silent_check")
SILENT = "直接给出结论JSON，不要写推导。"


def _symbol(rng):
    return rng.choice(STEMS + BRANCHES)


def drill(kind: str, rng, variant: int = 0):
    """One short question on a single table, with the table row recited in the answer."""
    if kind == "element":
        first, second = _symbol(rng), _symbol(rng)
        a, b = R.element_of(first), R.element_of(second)
        question = (
            f"{first}与{second}在五行上是什么关系？",
            f"请说明{first}和{second}的五行生克关系。",
            f"按五行生克，{first}、{second}两者谁生谁或谁克谁？",
        )[variant % 3]
        relation = D.element_relation(a, b)
        text = f"{first}属{a}，{second}属{b}。{GENERATION_ORDER}；{CONTROL_ORDER}。所以{relation}。"
        return question, text, {"answer": relation}
    if kind == "ten_god":
        day, other = rng.choice(STEMS), rng.choice(STEMS)
        question = (
            f"日主{day}，见{other}是什么十神？",
            f"以{day}为日干，{other}属于哪个十神？",
            f"{other}对日主{day}而言是什么十神？",
        )[variant % 3]
        text = f"日主为{day}（{R.STEM_ELEMENT[day]}，{polarity(day)}）。{god_reason(day, other)}。"
        return question, text, {"answer": R.ten_god(day, other)}
    if kind == "hidden":
        branch = rng.choice(BRANCHES)
        question = (
            f"地支{branch}藏哪些天干？",
            f"请写出{branch}的藏干（本气在前）。",
            f"{branch}里的人元有哪些？",
        )[variant % 3]
        hidden = R.HIDDEN_STEMS[branch]
        return question, f"{branch}藏{join(hidden)}（本气在前）。", {"answer": list(hidden)}
    if kind == "branch_pair":
        first = rng.choice(BRANCHES)
        # Half of the drills use a table partner so related and unrelated pairs are balanced.
        card = D.branch_card(first)
        partners = sorted(
            {card["六合"], card["冲"], card["害"], card["破"], *card["刑"], *card["半合"]}
        )
        second = rng.choice(partners) if rng.random() < 0.6 else rng.choice(BRANCHES)
        question = (
            f"{first}与{second}之间有哪些合、冲、刑、害、破关系？",
            f"地支{first}和{second}是什么关系（合冲刑害破）？",
            f"请查{first}、{second}两支的关系。",
        )[variant % 3]
        keys = R.branch_pair_relations(first, second)
        text = (
            D.branch_card_text(first)
            + "\n"
            + (D.branch_card_text(second) + "\n" if second != first else "")
            + f"所以{first}{second}：{relation_words(keys, '没有合冲刑害破')}。"
        )
        return question, text, {"answer": keys}
    if kind == "branch_card":
        branch = rng.choice(BRANCHES)
        question = (
            f"写出地支{branch}的六合、冲、刑、害、破、半合与三会。",
            f"{branch}与哪些地支相合、相冲、相刑、相害、相破？",
            f"请列出{branch}在各张地支关系表里的对象。",
        )[variant % 3]
        card = D.branch_card(branch)
        answer = {
            key: card[key] for key in ("六合", "冲", "刑", "害", "破", "半合", "三合", "三会")
        }
        return question, D.branch_card_text(branch) + "。", {"answer": answer}
    if kind == "stem_pair":
        first, second = rng.choice(STEMS), rng.choice(STEMS)
        question = (
            f"天干{first}与{second}是相合、相冲，还是都不是？",
            f"{first}、{second}两干之间有没有五合或相冲？",
            f"请判断{first}和{second}的天干关系。",
        )[variant % 3]
        keys = R.stem_pair_relations(first, second)
        text = (
            "天干五合：甲己、乙庚、丙辛、丁壬、戊癸；天干相冲：甲庚、乙辛、丙壬、丁癸。"
            f"{first}{second}：{relation_words(keys, '既不相合也不相冲')}。"
        )
        return question, text, {"answer": keys}
    if kind == "cycle":
        if rng.random() < 0.25:
            stem = rng.choice(STEMS)
            branch = rng.choice([b for b in BRANCHES if R.is_yang(b) != R.is_yang(stem)])
            question = (
                f"{stem}{branch}是六十甲子中的一个吗？",
                f"六十甲子里有没有{stem}{branch}？",
                f"请判断{stem}{branch}这个干支组合是否存在。",
            )[variant % 3]
            text = (
                f"{stem}为{polarity(stem)}干，{branch}为{polarity(branch)}支，阴阳不同；"
                "六十甲子只由阴阳相同的干支相配，所以不存在。"
            )
            return question, text, {"answer": {"valid": False}}
        pair = rng.choice(CYCLE)
        index = CYCLE.index(pair)
        s, b = STEMS.index(pair[0]), BRANCHES.index(pair[1])
        start = index - index % 10
        question = (
            f"{pair}在六十甲子中排第几（甲子为 0）？它属哪一旬，旬空是什么？",
            f"请求出{pair}的六十甲子序数、所在的旬和空亡。",
            f"{pair}的序数（甲子记 0）、旬首与旬空各是什么？",
        )[variant % 3]
        text = (
            f"{pair[0]}是第 {s} 干，{pair[1]}是第 {b} 支（都从 0 数）。序数 = (6×{s} − 5×{b}) mod 60 = "
            f"{6 * s - 5 * b} mod 60 = {index}。{index} − {index % 10} = {start}，旬首为{CYCLE[start]}；"
            f"该旬从{CYCLE[start]}到{CYCLE[start + 9]}，用不到的两支是{R.xun_kong(pair)}。"
        )
        return (
            question,
            text,
            {
                "answer": {
                    "valid": True,
                    "index": index,
                    "xun": CYCLE[start],
                    "kong": R.xun_kong(pair),
                }
            },
        )
    if kind == "season":
        stem, month = rng.choice(STEMS), rng.choice(BRANCHES)
        element = R.STEM_ELEMENT[stem]
        question = (
            f"{stem}生在{month}月，按旺相休囚死是哪一种？",
            f"日主{stem}，月支{month}。日主在月令是旺、相、休、囚、死的哪个？",
            f"请判断{element}在{month}月的旺衰状态。",
        )[variant % 3]
        reason, state = T.seasonal_reason(element, month)
        text = f"{stem}属{element}。{reason}。" if variant % 3 != 2 else f"{reason}。"
        return question, text, {"answer": state}
    if kind == "stage":
        stem, branch = rng.choice(STEMS), rng.choice(BRANCHES)
        question = (
            f"{stem}在{branch}处于十二长生的哪一位？",
            f"按阳顺阴逆，{stem}到{branch}是十二长生中的什么？",
            f"请推算天干{stem}在地支{branch}的十二长生。",
        )[variant % 3]
        reason, stage = T.life_stage_reason(stem, branch)
        return question, reason + f"，故为{stage}。", {"answer": stage}
    if kind == "star":
        star = rng.choice(D.STAR_NAMES)
        if star in D.TRIPLE_STARS:
            branch = rng.choice(BRANCHES)
            triple = D._TRIPLE_OF[branch]
            table = D.TRIPLE_STARS[star]
            question = (
                f"以{branch}所在的三合局查，{star}在哪一支？",
                f"{branch}年（或{branch}日）生人，{star}落在哪个地支？",
                f"请按三合局查出地支{branch}对应的{star}。",
            )[variant % 3]
            listing = "、".join(f"{key}在{value}" for key, value in table.items())
            text = f"{branch}属{triple}三合局。{star}：{listing}。所以{star}在{table[triple]}。"
            return question, text, {"answer": table[triple]}
        stem = rng.choice(STEMS)
        question = (
            f"日干{stem}的{star}在哪一支？",
            f"以日干{stem}查，{star}是哪个地支？",
            f"请查出{stem}日的{star}。",
        )[variant % 3]
        if star == "禄":
            text = (
                "禄：甲寅、乙卯、丙戊巳、丁己午、庚申、辛酉、壬亥、癸子。"
                f"{stem}的禄在{D.LU[stem]}。"
            )
            return question, text, {"answer": D.LU[stem]}
        if star == "阳刃":
            if stem in D.YANG_REN:
                text = (
                    f"阳刃是禄的前一位，只有阳干有。{stem}的禄在{D.LU[stem]}，"
                    f"前一位是{D.YANG_REN[stem]}。"
                )
            else:
                text = f"阳刃只有阳干（甲丙戊庚壬）有，{stem}是阴干，没有阳刃。"
            return question, text, {"answer": D.YANG_REN.get(stem, "")}
        text = (
            "天乙贵人：甲戊庚见丑未，乙己见子申，丙丁见亥酉，壬癸见卯巳，辛见寅午。"
            f"{stem}的天乙贵人在{join(D.TIAN_YI[stem])}。"
        )
        return question, text, {"answer": D.TIAN_YI[stem]}
    if kind == "class":
        day, cls = rng.choice(STEMS), rng.choice(D.CLASSES)
        question = (
            f"日主{day}，{cls}是什么五行？{cls}生哪一类十神、克哪一类十神？",
            f"对日主{day}来说，{cls}属哪个五行？它所生、所克的各是哪一类？",
            f"请说明日主{day}的{cls}：五行是什么，生什么、克什么。",
        )[variant % 3]
        mine = R.STEM_ELEMENT[day]
        chain = "→".join(f"{c}（{D.class_element(day, c)}）" for c in D.CLASSES)
        text = (
            f"日主{day}属{mine}。十神五类按相生次序排列：{chain}→比劫；相邻者前生后，隔一位者前克后。"
            f"所以{cls}属{D.class_element(day, cls)}，{cls}生{D.class_generates(cls)}，"
            f"{cls}克{D.class_controls(cls)}。"
        )
        return (
            question,
            text,
            {
                "answer": {
                    "element": D.class_element(day, cls),
                    "generates": D.class_generates(cls),
                    "controls": D.class_controls(cls),
                }
            },
        )
    if kind == "luck_dir":
        stem, gender = rng.choice(STEMS), rng.choice(("male", "female"))
        question = (
            f"年干为{stem}的{GENDER_ZH[gender]}命，大运顺行还是逆行？",
            f"{GENDER_ZH[gender]}命，生于{stem}年，大运的方向是？",
            f"请判断：{stem}年出生的{GENDER_ZH[gender]}性，大运顺排还是逆排？",
        )[variant % 3]
        direction = R.luck_direction(stem, gender)
        kind_of_birth = f"{polarity(stem)}年{GENDER_ZH[gender]}命"
        text = (
            f"{stem}为{polarity(stem)}干，所以是{kind_of_birth}。阳年男命、阴年女命顺行；"
            f"阴年男命、阳年女命逆行。{kind_of_birth}大运{'顺' if direction == 'forward' else '逆'}行。"
        )
        return question, text, {"answer": direction}
    if kind == "month_stem":
        stem, branch = rng.choice(STEMS), rng.choice(BRANCHES)
        number = (BRANCHES.index(branch) - 2) % 12 + 1
        pillar = R.month_pillar_for(stem, branch)
        question = (
            f"{stem}年的{branch}月，月柱是什么？",
            f"年干{stem}，月支{branch}。按五虎遁，月干是什么？",
            f"请用五虎遁求{stem}年{branch}月的月柱。",
        )[variant % 3]
        text = (
            "五虎遁：甲己年寅月起丙，乙庚年起戊，丙辛年起庚，丁壬年起壬，戊癸年起甲。"
            f"{stem}年寅月起{R.first_month_stem(stem)}，{branch}月是第 {number} 月，"
            f"顺数到第 {number} 位为{pillar[0]}，月柱{pillar}。"
        )
        return question, text, {"answer": pillar}
    if kind == "hour_stem":
        stem, branch = rng.choice(STEMS), rng.choice(BRANCHES)
        number = BRANCHES.index(branch) + 1
        pillar = R.hour_pillar_for(stem, branch)
        question = (
            f"{stem}日的{branch}时，时柱是什么？",
            f"日干{stem}，时支{branch}。按五鼠遁，时干是什么？",
            f"请用五鼠遁求{stem}日{branch}时的时柱。",
        )[variant % 3]
        text = (
            "五鼠遁：甲己日子时起甲，乙庚日起丙，丙辛日起戊，丁壬日起庚，戊癸日起壬。"
            f"{stem}日子时起{R.first_hour_stem(stem)}，{branch}时是第 {number} 位，"
            f"顺数到第 {number} 位为{pillar[0]}，时柱{pillar}。"
        )
        return question, text, {"answer": pillar}
    if kind == "year":
        year = rng.randrange(1901, 2100)
        question = (
            f"{year} 年（立春后）是什么年？",
            f"请推出 {year} 年立春以后的年柱。",
            f"{year} 年的流年干支是什么？",
        )[variant % 3]
        text, index = mod60_text(year - 4)
        answer = f"{year} − 4 = {year - 4}，{text}；{index_text(index)}，为{CYCLE[index]}年。"
        return question, answer, {"answer": CYCLE[index]}
    if kind == "god_table":
        day = rng.choice(STEMS)
        question = (
            f"写出日主{day}的十神表（十个天干各是什么十神）。",
            f"以{day}为日主，十个天干分别是什么十神？请先列表。",
            f"请列出日主{day}对应的完整十神表。",
        )[variant % 3]
        return question, god_table(day), {"answer": {s: R.ten_god(day, s) for s in STEMS}}
    if kind == "silent_hour":
        stem, branch = rng.choice(STEMS), rng.choice(BRANCHES)
        question = f"{stem}日的{branch}时，时柱是什么？{SILENT}"
        return question, "", {"answer": R.hour_pillar_for(stem, branch)}
    if kind == "silent_month":
        stem, branch = rng.choice(STEMS), rng.choice(BRANCHES)
        question = f"{stem}年的{branch}月，月柱是什么？{SILENT}"
        return question, "", {"answer": R.month_pillar_for(stem, branch)}
    if kind == "silent_check":
        upper = rng.choice(CYCLE)
        hour = rng.random() < 0.5
        branch = rng.choice(BRANCHES)
        right = (R.hour_pillar_for if hour else R.month_pillar_for)(upper[0], branch)
        shown = right
        if rng.random() < 0.5:
            shown = CYCLE[(CYCLE.index(right) + rng.choice((12, 24, 36, 48))) % 60]
        names = ("日柱", "时柱", "五鼠遁") if hour else ("年柱", "月柱", "五虎遁")
        question = f"{names[0]}{upper}、{names[1]}{shown}，符合{names[2]}吗？{SILENT}"
        return question, "", {"answer": {"consistent": shown == right, "expected": right}}
    raise ValueError(f"Unknown drill {kind}")


# ---- Ten Gods of a whole chart ----------------------------------------------------------------


def gods_text(pillars):
    day = pillars["day"][0]
    lines = [f"日主为{day}（{R.STEM_ELEMENT[day]}，{polarity(day)}）。"]
    if explicit():
        lines.append(god_table(day))
    for name in OTHER:
        lines.append(f"{POS[name]}干{pillars[name][0]}：{read(day, pillars[name][0])}。")
    hidden = []
    for name in PILLARS:
        branch = pillars[name][1]
        gods = "、".join(f"{h}{R.ten_god(day, h)}" for h in R.HIDDEN_STEMS[branch])
        hidden.append(f"{POS[name]}支{branch}藏{gods}")
    lines.append("藏干：" + "；".join(hidden) + "。")
    return lines


def configs_text(pillars):
    gods = D.stem_gods(pillars)
    found = D.stem_configs(pillars)
    seen = "、".join(f"{POS[name]}干{gods[name]}" for name in OTHER)
    if explicit():
        present = set(gods.values())
        checks = []
        for name, first, second, _ in D.CONFIGS:
            a, b = sorted(present & set(first)), sorted(present & set(second))
            need = f"{'或'.join(first)}＋{'或'.join(second)}"
            have = f"有{'、'.join(a)}" if a else "缺前者"
            have += f"、有{'、'.join(b)}" if b else "、缺后者"
            checks.append(f"{name}（{need}）：{have} {'✓' if name in found else '✗'}")
        return (
            f"三干十神：{seen}。逐项检查十神组合——" + "；".join(checks) + "。"
            f"成立的：{join(found)}。这是搭配的名称，不是吉凶判断。",
            found,
        )
    if not found:
        return f"三干十神：{seen}。它们之间没有本项目标注的十神组合。", found
    parts = []
    for name, first, second, relation in D.CONFIGS:
        if name in found:
            a = "、".join(sorted({g for g in gods.values() if g in first}))
            b = "、".join(sorted({g for g in gods.values() if g in second}))
            word = "并见" if relation == "并见" else f"相{relation}"
            parts.append(f"{a}与{b}{word}，称{name}")
    return f"三干十神：{seen}。" + "；".join(parts) + "。这是搭配的名称，不是吉凶判断。", found


def gods_task(pillars, rng, variant=0):
    question = (
        "请列出年、月、时三干与各支藏干的十神，并指出三干之间成立的十神组合。",
        "逐一推出这个八字的十神（天干与藏干），再看年月时三干构成哪些有名称的十神搭配。",
        "这个盘的十神怎么排？三干之间有没有食神制杀、伤官见官之类的组合？",
    )[variant % 3]
    lines = gods_text(pillars)
    text, found = configs_text(pillars)
    return (
        question,
        "\n".join(lines + [text]),
        {"stems": D.stem_gods(pillars), "configs": found},
    )


# ---- Strength, pattern and 用神 ----------------------------------------------------------------


def strength_text(pillars):
    """The three indicators with every hidden stem and stem written out."""
    s = R.strength_indicators(pillars)
    day, mine = s["day_master"], s["element"]
    reason, state = T.seasonal_reason(mine, pillars["month"][1])
    yes = {True: "是", False: "否"}
    hidden = "；".join(
        f"{POS[name]}支{pillars[name][1]}藏{''.join(R.HIDDEN_STEMS[pillars[name][1]])}"
        for name in PILLARS
    )
    rooted = [f"{POS[name]}支{pillars[name][1]}" for name in s["root_pillars"]]
    stems = "；".join(
        f"{POS[name]}干{pillars[name][0]}（{R.STEM_ELEMENT[pillars[name][0]]}，"
        f"{R.ten_god(day, pillars[name][0])}）"
        for name in OTHER
    )
    helpers = [f"{POS[name]}干{pillars[name][0]}" for name in s["supporting_stem_pillars"]]
    chain = f"按相生次序{class_chain(day)}。" if explicit() else ""
    count = sum(s[key] for key in ("得令", "得地", "得势"))
    text = (
        f"得令：日主{day}属{mine}。{reason}；旺或相才算得令 → {yes[s['得令']]}。\n"
        f"得地：{hidden}。其中藏有{mine}的是{join(rooted)} → {yes[s['得地']]}。\n"
        f"得势：{chain}{stems}。属比劫或印的是{join(helpers)}，共 {len(helpers)} 个，至少两个才算 → "
        f"{yes[s['得势']]}。\n"
        f"三项占 {count} 项，按“占两项为偏强”的计数约定记为{s['summary']}。"
        "各派权重不同，这是计数结果，不是定论。"
    )
    answer = {key: s[key] for key in ("得令", "得地", "得势")}
    answer["summary"] = s["summary"]
    return text, answer


def strength_task(pillars, rng, variant=0):
    question = (
        "按得令、得地、得势三项计数的约定，逐项列出依据，判断日主偏强还是偏弱。",
        "请写出每个地支的藏干和三干的十神，再按“三项占两项为偏强”判断日主强弱。",
        "日主强弱怎么数？请把得令、得地、得势各自看的是什么写清楚。",
    )[variant % 3]
    text, answer = strength_text(pillars)
    return question, text, {"answer": answer}


def pattern_text(pillars):
    day, month_branch = pillars["day"][0], pillars["month"][1]
    hidden = R.HIDDEN_STEMS[month_branch]
    result = D.revealed_pattern(pillars)
    visible = [pillars[name][0] for name in OTHER]
    lines = [
        f"月支{month_branch}藏{join(hidden)}。本气{hidden[0]}对日主{day}："
        f"{god_reason(day, hidden[0])}。"
    ]
    if explicit():
        lines = [
            god_table(day),
            f"月支{month_branch}藏{join(hidden)}。本气{hidden[0]}：{read(day, hidden[0])}。",
        ]
    if result["basis"] == "main_is_peer":
        lines.append(
            f"本气是比劫，不按透干取格：比肩记建禄格，劫财在阳干记阳刃格、阴干记月劫格。"
            f"{day}为{polarity(day)}干，记作{result['pattern']}。"
        )
    else:
        lines.append(f"年、月、时三干为{join(visible)}。按本气、中气、余气的顺序看是否透出：")
        for rank, stem in enumerate(hidden):
            god = R.ten_god(day, stem)
            shown = stem in visible
            peer = god in ("比肩", "劫财")
            note = "透" if shown else "不透"
            if shown and peer:
                note += "，但它是比劫，不取"
            lines.append(f"{stem}（{RANK_ZH[rank]}，{god}）{note}。")
            if shown and not peer:
                break
        if result["basis"] == "main_default":
            lines.append(f"都不取，回到本气{hidden[0]}，记作{result['pattern']}。")
        else:
            lines.append(f"取{result['stem']}，记作{result['pattern']}。")
    lines.append("这是“月令透干取格”的约定；多干并透、会合变格各家处理不同，格名本身不代表吉凶。")
    return "\n".join(lines), result


def pattern_task(pillars, rng, variant=0):
    question = (
        "按月令透干取格（本气、中气、余气依次看是否透于年月时干），这个八字是什么格？",
        "请用“月令透干取格”的约定定格局：先看月支本气是否为比劫，再依次看藏干有没有透出。",
        "月令的藏干里哪一个透出来了？按透干取格的约定，格局记作什么？",
    )[variant % 3]
    text, result = pattern_text(pillars)
    return question, text, {"answer": result["pattern"], "basis": result["basis"]}


SCHOOLS = ("扶抑", "调候", "格局")


def yongshen_text(pillars, school):
    day = pillars["day"][0]
    if school == "扶抑":
        text, _ = strength_text(pillars)
        result = D.fuyi(pillars)
        favorable = "、".join(
            f"{c}（{D.class_element(day, c)}）" for c in result["favorable_classes"]
        )
        rule = (
            "偏弱要生扶，喜印与比劫"
            if result["strength"] == "偏弱"
            else "偏强要泄耗克制，喜食伤、财、官杀"
        )
        chain = f"按相生次序{class_chain(day)}，" if explicit() else ""
        text += f"\n扶抑：{rule}。对日主{day}而言{chain}即{favorable}。"
        return text, {
            "school": "扶抑",
            "strength": result["strength"],
            "favorable": result["favorable_classes"],
            "favorable_elements": result["favorable_elements"],
        }
    if school == "调候":
        result = D.tiaohou(pillars)
        month = pillars["month"][1]
        if result["needs"] is None:
            text = (
                f"调候（简化约定）：冬月（亥子丑）取火，夏月（巳午未）取水。月支{month}不在其中，"
                f"按这个简化约定不特别论调候。{finer_tables()}，这里没有采用。"
            )
            return text, {"school": "调候", "needs": None, "present_in_stems": False}
        stems = [f"{POS[name]}干{pillars[name][0]}" for name in result["in_stems"]]
        hidden = [f"{POS[name]}支{pillars[name][1]}" for name in result["in_hidden"]]
        text = (
            f"调候（简化约定）：冬月（亥子丑）取火，夏月（巳午未）取水。月支{month}属{result['season']}月，"
            f"取{result['needs']}。四干中属{result['needs']}的：{join(stems)}；"
            f"藏有{result['needs']}的地支：{join(hidden)}。"
            f"{finer_tables()}，这里没有采用。"
        )
        return text, {
            "school": "调候",
            "needs": result["needs"],
            "present_in_stems": bool(result["in_stems"]),
        }
    text, result = pattern_text(pillars)
    use = D.PATTERN_USE[result["god"]]
    grouping = "正官、财、印、食神为顺用，七杀、伤官、劫刃为逆用"
    text += f"\n格局：格局之神为{result['stem']}（{result['god']}）。传统上{grouping}；{result['god']}属{use}。"
    return text, {"school": "格局", "pattern": result["pattern"], "god": result["god"], "use": use}


def yongshen_task(pillars, rng, variant=0, school=None):
    school = school or rng.choice(SCHOOLS)
    question = {
        "扶抑": (
            "按扶抑一派（强弱用三项计数），这个八字喜哪几类十神？",
            "只用扶抑法：先定日主强弱，再说喜用是哪几类十神、哪些五行。",
            "若以扶抑取用神，此盘所喜的是什么？请先数强弱。",
        ),
        "调候": (
            "按调候一派的简化约定（冬取火、夏取水），这个八字要不要调候、取什么？",
            "只看调候：出生月份寒还是热？按简化约定取哪个五行，盘里有没有？",
            "若论调候，此盘取用什么五行？天干见不见？",
        ),
        "格局": (
            "按格局一派，这个八字的格局之神是什么，属于顺用还是逆用？",
            "只用格局法：先按月令透干定格，再说明该格是顺用还是逆用。",
            "若以月令格局取用，此盘用神是哪个十神？顺用还是逆用？",
        ),
    }[school][variant % 3]
    text, response = yongshen_text(pillars, school)
    return question, text, response


# ---- Interactions ------------------------------------------------------------------------------


def interact_task(pillars, rng, variant=0):
    question = (
        "请先写出四个地支各自的关系卡片，再逐对检查天干、地支之间的合冲刑害破，并注明哪些发生在相邻两柱之间。",
        "逐对核对这个八字的干支关系：先查每个地支的合冲刑害破对象，再两两比对；相邻柱之间的关系单独列出。",
        "原局有哪些合、冲、刑、害、破？哪些是紧贴的（相邻两柱）？请按表逐项查。",
    )[variant % 3]
    branches = [pillars[name][1] for name in PILLARS]
    stems = [pillars[name][0] for name in PILLARS]
    lines = ["地支卡片："]
    for branch in dict.fromkeys(branches):
        lines.append(D.branch_card_text(branch))
    if explicit():
        lines.append("天干五合：甲己、乙庚、丙辛、丁壬、戊癸；天干相冲：甲庚、乙辛、丙壬、丁癸。")
    lines.append(T._pair_scan(stems, R.stem_pair_relations, "天干"))
    parts = []
    for i, first in enumerate(PILLARS):
        for second in PILLARS[i + 1 :]:
            a, b = pillars[first][1], pillars[second][1]
            keys = R.branch_pair_relations(a, b)
            near = "相邻" if PILLARS.index(second) - i == 1 else "不相邻"
            where = ""
            if explicit():
                # Say where the second branch appears on the first branch's card before concluding.
                card = D.branch_card(a)
                slots = [kind for kind in ("六合", "冲", "害", "破") if card[kind] == b]
                if a != b and b in card["刑"]:
                    slots.append("刑")
                if a != b and b in card["半合"]:
                    slots.append("半合")
                if a == b:
                    where = f"同支，{a}{'是' if card['刑'] == [a] else '不是'}自刑 → "
                else:
                    where = f"{b}在{a}卡的{'、'.join(slots)} → " if slots else f"{b}不在{a}卡 → "
            parts.append(
                f"{POS[first]}{a}–{POS[second]}{b}（{near}）：{where}{relation_words(keys)}"
            )
    lines.append("地支两两比对——" + "；".join(parts) + "。")
    keys = R.chart_relations(pillars)
    triples = [key for key in keys if key.startswith(("三合", "三会"))]
    if triples:
        lines.append("三支齐全：" + relation_words(triples) + "。")
    adjacent = sorted({item["key"] for item in D.positioned_relations(pillars) if item["adjacent"]})
    lines.append("相邻两柱之间的：" + relation_words(adjacent) + "。")
    return question, "\n".join(lines), {"answer": keys, "adjacent": adjacent}


# ---- Flow years and luck cycles ---------------------------------------------------------------


def clash_words(first: str, second: str) -> str:
    return "".join(sorted((first, second), key=BRANCHES.index)) + "相冲"


def special_text(pillars, flow, special, luck=None):
    parts = []
    fuyin = [key.split(":")[1] for key in special if key.startswith("伏吟:")]
    parts.append(
        "伏吟（流年与某柱干支相同）："
        + (join(R.PILLAR_ZH[name] for name in fuyin) if fuyin else "无")
    )
    clash = [key.split(":")[1] for key in special if key.startswith("天克地冲:")]
    detail = [
        f"{R.PILLAR_ZH[name]}{pillars[name]}（{clash_words(flow[1], pillars[name][1])}，"
        f"{D.element_relation(R.STEM_ELEMENT[flow[0]], R.STEM_ELEMENT[pillars[name][0]])}）"
        for name in clash
    ]
    parts.append("天克地冲（地支相冲且天干五行相克）：" + join(detail))
    year_branch = pillars["year"][1]
    if "值太岁" in special:
        parts.append(f"太岁：流年支{flow[1]}与年支{year_branch}相同，值太岁")
    elif "冲太岁" in special:
        parts.append(f"太岁：流年支{flow[1]}与年支{year_branch}相冲，冲太岁")
    else:
        parts.append(f"太岁：流年支{flow[1]}与年支{year_branch}既不相同也不相冲")
    if luck is not None:
        same = "岁运并临" in special
        parts.append(f"岁运并临（流年与大运干支相同）：{'是' if same else '否'}")
    return "；".join(parts) + "。"


def flow_task(pillars, rng, variant=0, year=2030, cycles=None):
    """One flow year against the natal chart; ``cycles`` (from the tool) adds the luck pillar."""
    question = (
        f"{year} 年（立春后）的流年与原局是什么关系？请推出流年干支、流年天干的十神、"
        "流年地支与四柱各支的合冲刑害破，并检查伏吟、天克地冲与太岁。",
        f"请分析 {year} 年流年对这个八字的作用：流年干支、十神、与四柱地支的关系，以及有没有伏吟、"
        "天克地冲、值太岁或冲太岁。",
        f"{year} 年走什么流年？它的天干是什么十神，地支和年、月、日、时支分别有什么关系？"
        "有没有伏吟、天克地冲、犯太岁？",
    )[variant % 3]
    mod_text, index = mod60_text(year - 4)
    flow = CYCLE[index]
    day = pillars["day"][0]
    luck = D.luck_in_year(cycles, year) if cycles else None
    analysis = D.flow_analysis(pillars, flow, luck)
    main = R.HIDDEN_STEMS[flow[1]][0]
    lines = [
        f"流年：{year} − 4 = {year - 4}，{mod_text}；{index_text(index)}，{year} 年为{flow}年。",
        *([god_table(day)] if explicit() else []),
        f"天干：{read(day, flow[0])}。地支{flow[1]}本气{main}，"
        + (f"{read(day, main)}。" if explicit() else f"为{R.ten_god(day, main)}。"),
        "流年支卡片：" + D.branch_card_text(flow[1]) + "。",
        "与四柱地支："
        + "；".join(
            f"{POS[name]}支{pillars[name][1]}——{relation_words(analysis['branch_relations'][name])}"
            for name in PILLARS
        )
        + "。",
    ]
    stem_hits = [
        f"{POS[name]}干{pillars[name][0]}——{relation_words(keys)}"
        for name, keys in analysis["stem_relations"].items()
        if keys
    ]
    lines.append("与四柱天干：" + (join(stem_hits, "") if stem_hits else "没有五合或相冲") + "。")
    if cycles:
        if luck is None:
            lines.append(f"大运：{year} 年在起运之前或超出所列大运，不论岁运。")
        else:
            cycle = next(c for c in cycles if c["pillar"] == luck)
            lines.append(
                f"大运：{year} 年在{luck}运（{cycle['start_year']}–{cycle['end_year']}）内。"
            )
    lines.append(special_text(pillars, flow, analysis["special"], luck))
    lines.append("以上是可以机械判定的关系，本身不说明这一年会发生什么。")
    response = {
        "pillar": flow,
        "ten_god": analysis["stem_god"],
        "branch_relations": analysis["branch_relations"],
        "special": analysis["special"],
    }
    if cycles:
        response["luck"] = luck
    return question, "\n".join(lines), response


# ---- Areas of life -----------------------------------------------------------------------------


def domain_task(pillars, rng, variant=0, gender="male", domain="marriage", first_year=2030):
    name, gods_by_gender, palace, palace_name = D.DOMAINS[domain]
    gods = gods_by_gender[gender]
    last_year = first_year + 7
    anchor = R.year_pillar(first_year)
    question = (
        f"按传统的看法，{name}方面要看哪些十神{'和宫位' if palace else ''}？它们在这个盘里的情况如何？"
        f"已知 {first_year} 年为{anchor}年，{first_year}–{last_year} 年里哪些年按“星现、宫动”的约定算被引动？",
        f"想了解传统上怎么看{name}：看哪些星{'、哪个宫' if palace else ''}，盘中在哪里；"
        f"并列出 {first_year}（{anchor}）到 {last_year} 年中符合“星现或宫动”的年份。",
        f"{name}在传统命理里对应什么十神{'与宫位' if palace else ''}？此盘有没有、在哪一柱？"
        f"{first_year}–{last_year} 年（{first_year} 年为{anchor}）哪些流年会引动它？",
    )[variant % 3]
    facts = D.domain_facts(pillars, domain, gender)
    day = pillars["day"][0]
    tradition = f"【传统的对应】{GENDER_ZH[gender]}命以{join(gods)}象征{name}"
    tradition += f"，以{palace_name}为对应的宫位。" if palace else "。"
    if explicit():
        tradition += "\n" + god_table(day)
    stems = [
        f"{POS[p]}干{pillars[p][0]}（{R.ten_god(day, pillars[p][0])}）" for p in facts["in_stems"]
    ]
    hidden = []
    for p in facts["in_hidden"]:
        inside = [h for h in R.HIDDEN_STEMS[pillars[p][1]] if R.ten_god(day, h) in gods]
        hidden.append(
            f"{POS[p]}支{pillars[p][1]}（藏{'、'.join(h + R.ten_god(day, h) for h in inside)}）"
        )
    chart = f"【盘中的事实】天干：{join(stems, '不见这类十神')}；藏干：{join(hidden, '不见')}。"
    if palace:
        chart += (
            f"{palace_name}为{pillars[palace][1]}，原局中与它有关的关系："
            f"{relation_words(facts['palace_relations'])}。"
        )
    rows, hits = [], []
    triggers = {
        t["year"]: t for t in D.trigger_years(pillars, domain, gender, first_year, last_year)
    }
    for year in range(first_year, last_year + 1):
        flow = R.year_pillar(year)
        god = R.ten_god(day, flow[0])
        row = f"{year}{flow}：{flow[0]}为{god}"
        if palace:
            keys = R.branch_pair_relations(flow[1], pillars[palace][1])
            pair = flow[1] + pillars[palace][1]
            row += "，" + (relation_words(keys) if keys else f"{pair}无合冲刑害破")
        if year in triggers:
            row += " → " + "、".join(triggers[year]["reasons"])
            hits.append(year)
        rows.append(row)
    years = "【流年】" + "；".join(rows) + f"。符合约定的年份：{join(map(str, hits))}。"
    limit = (
        f"【不能确定的部分】以上只是符号上的对应与巧合。{name}在现实中是否发生、何时发生、结果如何，"
        "八字规则无法确证，我不作预测。"
    )
    return (
        question,
        "\n".join((tradition, chart, years, limit)),
        {
            "kind": "traditional",
            "gods": list(gods),
            "in_stems": facts["in_stems"],
            "trigger_years": hits,
        },
    )


# ---- Checking statements about a chart ---------------------------------------------------------

VERDICTS = ("正确", "错误", "无法验证")
SPECULATIVE = (
    "所以命主三十五岁那年一定会发财",
    "因此命主的婚姻注定不顺",
    "可见命主将来必定官运亨通",
    "所以命主四十岁前肯定会离婚",
    "由此可知命主一生平安长寿",
    "说明命主的父亲会早逝",
    "所以命主明年必然升职",
    "因此命主一定考得上名校",
)


def _wrong(options, right, rng):
    return rng.choice([option for option in options if option != right])


def make_claim(kind, pillars, rng, gender="male", truth=True):
    """(statement, verdict, reason, correction or None) for one checkable statement."""
    day = pillars["day"][0]
    if kind == "speculative":
        return (
            rng.choice(SPECULATIVE),
            "无法验证",
            "这是对现实事件的断言，八字规则无法确证",
            None,
        )
    if kind == "ten_god":
        name = rng.choice(OTHER)
        stem = pillars[name][0]
        right = R.ten_god(day, stem)
        said = right if truth else _wrong(R.TEN_GODS, right, rng)
        return (
            f"{POS[name]}干{stem}是日主的{said}",
            "正确" if truth else "错误",
            god_reason(day, stem),
            None if truth else right,
        )
    if kind == "relation":
        i, j = sorted(rng.sample(range(4), 2))
        a, b = pillars[PILLARS[i]][1], pillars[PILLARS[j]][1]
        keys = R.branch_pair_relations(a, b)
        kinds = ("六合", "冲", "刑", "害", "破", "半合")
        present = {key.split(":")[0] for key in keys}
        if truth and present:
            said = rng.choice(sorted(present))
        elif truth:
            statement = f"{POS[PILLARS[i]]}支{a}与{POS[PILLARS[j]]}支{b}之间没有合冲刑害破"
            return statement, "正确", f"查两支卡片，{a}{b}无合冲刑害破", None
        else:
            absent = [k for k in kinds if k not in present and not (k == "刑" and a == b)]
            said = rng.choice(absent)
        word = {
            "六合": "六合",
            "冲": "相冲",
            "刑": "相刑",
            "自刑": "自刑",
            "害": "相害",
            "破": "相破",
            "半合": "半合",
        }
        return (
            f"{POS[PILLARS[i]]}支{a}与{POS[PILLARS[j]]}支{b}{word[said]}",
            "正确" if truth else "错误",
            f"查两支卡片，{a}{b}：{relation_words(keys, '无合冲刑害破')}",
            None if truth else relation_words(keys, "无合冲刑害破"),
        )
    if kind == "seasonal":
        mine = R.STEM_ELEMENT[day]
        reason, right = T.seasonal_reason(mine, pillars["month"][1])
        said = right if truth else _wrong(R.SEASON_STATES, right, rng)
        return (
            f"日主{day}在月令处于“{said}”",
            "正确" if truth else "错误",
            reason,
            None if truth else right,
        )
    if kind == "luck":
        right = R.luck_direction(pillars["year"][0], gender)
        said = right if truth else ("backward" if right == "forward" else "forward")
        word = {"forward": "顺行", "backward": "逆行"}
        stem = pillars["year"][0]
        return (
            f"此为{GENDER_ZH[gender]}命，大运{word[said]}",
            "正确" if truth else "错误",
            f"年干{stem}为{polarity(stem)}干，{polarity(stem)}年{GENDER_ZH[gender]}命大运{word[right]}",
            None if truth else word[right],
        )
    if kind == "element":
        right = R.STEM_ELEMENT[day] + polarity(day)
        options = [e + p for e in R.ELEMENTS for p in "阳阴"]
        said = right if truth else _wrong(options, right, rng)
        return (
            f"日主{day}属{said[0]}，为{said[1]}干",
            "正确" if truth else "错误",
            f"{day}属{right[0]}，为{right[1]}干",
            None if truth else f"{right[0]}、{right[1]}干",
        )
    if kind == "hidden":
        name = rng.choice(PILLARS)
        branch = pillars[name][1]
        hidden = R.HIDDEN_STEMS[branch]
        said = rng.choice(hidden) if truth else rng.choice([s for s in STEMS if s not in hidden])
        return (
            f"{POS[name]}支{branch}的藏干里有{said}",
            "正确" if truth else "错误",
            f"{branch}藏{join(hidden)}",
            None if truth else join(hidden),
        )
    if kind == "star":
        star = rng.choice(("禄", "天乙贵人", "桃花", "驿马"))
        basis_note = "（以日支查）" if star in D.TRIPLE_STARS else ""
        right = D.star_branches(star, pillars, "day")
        said = rng.choice(right) if truth else rng.choice([b for b in BRANCHES if b not in right])
        source = f"日支{pillars['day'][1]}" if star in D.TRIPLE_STARS else f"日干{day}"
        return (
            f"此命的{star}{basis_note}在{said}",
            "正确" if truth else "错误",
            f"{source}的{star}在{join(right)}",
            None if truth else join(right),
        )
    if kind == "tradition":
        _, gods_by_gender, _, _ = D.DOMAINS["marriage"]
        right = gods_by_gender[gender]
        wrong = gods_by_gender["female" if gender == "male" else "male"]
        said = right if truth else wrong
        role = "妻" if gender == "male" else "夫"
        return (
            f"传统上{GENDER_ZH[gender]}命以{join(said)}为{role}星",
            "正确" if truth else "错误",
            f"传统的六亲对应中，{GENDER_ZH[gender]}命以{join(right)}象征配偶（这是符号对应，不是事实判断）",
            None if truth else join(right),
        )
    raise ValueError(kind)


CLAIM_KINDS = ("ten_god", "relation", "seasonal", "luck", "element", "hidden", "star", "tradition")


def claims_task(pillars, rng, variant=0, gender="male", count=4):
    kinds = rng.sample(CLAIM_KINDS, count - 1) + ["speculative"]
    if rng.random() < 0.35:
        kinds[-1] = rng.choice(CLAIM_KINDS)
    rng.shuffle(kinds)
    claims = [make_claim(kind, pillars, rng, gender, rng.random() < 0.5) for kind in kinds]
    listing = "\n".join(f"{i}. {claim[0]}。" for i, claim in enumerate(claims, 1))
    lead = (
        "有人对这个八字下了下面几条判断，请逐条按规则核对，标明正确、错误还是无法验证，错误的给出改正：",
        "下面是一段解读里的几句话。哪些与规则算出的事实一致，哪些不一致，哪些是规则无法确证的？",
        "请审查这些说法是否与盘面一致（正确／错误／无法验证），并改正错误的：",
    )[variant % 3]
    lines, corrections = [], {}
    for i, (_, verdict, why, correction) in enumerate(claims, 1):
        line = f"{i}. {why} → {verdict}"
        if correction is not None:
            line += f"，应为{correction}"
            corrections[str(i)] = correction
        lines.append(line + "。")
    return (
        lead + "\n" + listing,
        "\n".join(lines),
        {"verdicts": [claim[1] for claim in claims], "corrections": corrections},
    )


# ---- Questions without a unique answer --------------------------------------------------------

AMBIG_KINDS = (
    "yongshen_open",
    "yongshen_named",
    "hour_needed",
    "hour_not_needed",
    "hour_strength",
    "gender_needed",
    "gender_given",
    "star_basis",
    "no_rule",
)
NO_RULE = (
    "这个八字好不好？",
    "这是不是富贵命？",
    "这个命算上等还是下等？",
    "这个人以后会不会幸福？",
    "此命是吉是凶？",
    "这个八字的人能不能长寿？",
)


def three_pillar_line(pillars):
    return (
        f"八字：年柱{pillars['year']} 月柱{pillars['month']} 日柱{pillars['day']}，时辰不详"
        f"（日主{pillars['day'][0]}）"
    )


def ambig_task(kind, pillars, rng, variant=0, gender="male"):
    """(header kind, question, text, response); header is 'full', 'three' or 'nogender'."""
    day = pillars["day"][0]
    if kind == "yongshen_open":
        question = (
            "这个八字的用神是什么？",
            "此盘喜用神是哪个五行？",
            "请直接告诉我这个八字取什么为用神。",
        )[variant % 3]
        schools = D.yongshen_by_school(pillars)
        fuyi, tiao, pattern = schools["扶抑"], schools["调候"], schools["格局"]
        lines = [
            "“用神”在不同流派里含义不同，没有指定流派就没有唯一答案，分别来看：",
            f"扶抑：三项计数为{fuyi['strength']}，喜{join(fuyi['favorable_classes'])}"
            f"（{join(fuyi['favorable_elements'])}）。",
            (
                f"调候（简化约定）：月支{pillars['month'][1]}属{tiao['season']}月，取{tiao['needs']}。"
                if tiao["needs"]
                else f"调候（简化约定）：月支{pillars['month'][1]}不在冬、夏月，不特别论调候。"
            ),
            f"格局：按月令透干取格为{pattern['pattern']}，格局之神属{pattern['element']}，{pattern['use']}。",
            "三派回答的不是同一个问题，结论可以不同。请说明要按哪一派，我再给出该派的推导。",
        ]
        return (
            "full",
            question,
            "\n".join(lines),
            {
                "determinate": False,
                "needs": ["yongshen_school"],
                "by_school": {
                    "扶抑": fuyi["favorable_elements"],
                    "调候": tiao["needs"],
                    "格局": pattern["god"],
                },
            },
        )
    if kind == "yongshen_named":
        school = rng.choice(SCHOOLS)
        question, text, response = yongshen_task(pillars, rng, variant, school)
        return "full", question, text, {"determinate": True, "needs": [], "answer": response}
    if kind == "hour_needed":
        question = (
            "时干是日主的什么十神？",
            "时柱与日柱之间有没有合冲？",
            "请给出时柱，并说明时支藏干的十神。",
        )[variant % 3]
        text = (
            "只知道年、月、日三柱，时辰不详。时柱由日干和出生时辰按五鼠遁决定，十二个时辰对应十二个不同的"
            f"时柱（{day}日子时起{R.first_hour_stem(day)}），不知道时辰就无法确定时干、时支，"
            "与时柱有关的十神和合冲也就无从判断。请补充出生时辰。"
        )
        return "three", question, text, {"determinate": False, "needs": ["hour"]}
    if kind == "hour_not_needed":
        choice = rng.randrange(3)
        if choice == 0:
            mine = R.STEM_ELEMENT[day]
            reason, state = T.seasonal_reason(mine, pillars["month"][1])
            question = "日主在月令处于旺相休囚死的哪一种？"
            text = f"这个问题只用到日干和月支，时辰不详不影响。日主{day}属{mine}。{reason}。"
            answer = state
        elif choice == 1:
            stem = pillars["year"][0]
            question = f"年干{stem}是日主的什么十神？"
            text = f"这个问题只用到日干和年干，时辰不详不影响。{god_reason(day, stem)}。"
            answer = R.ten_god(day, stem)
        else:
            main = R.HIDDEN_STEMS[pillars["month"][1]][0]
            question = "只看月支本气，格局标签是什么？"
            text = (
                f"这个问题只用到日干和月支，时辰不详不影响。月支{pillars['month'][1]}本气为{main}。"
                f"{god_reason(day, main)}。按月令本气取格记作{R.month_pattern(pillars)}。"
            )
            answer = R.month_pattern(pillars)
        return "three", question, text, {"determinate": True, "needs": [], "answer": answer}
    if kind == "hour_strength":
        question = (
            "按得令、得地、得势的计数约定，“得势”这一项成立吗？",
            "年、月、时三干里比劫或印够不够两个（得势）？",
            "时辰不详的情况下，能不能判断“得势”？",
        )[variant % 3]
        mine = R.STEM_ELEMENT[day]
        helpers = [
            name
            for name in ("year", "month")
            if R.STEM_ELEMENT[pillars[name][0]] == mine
            or R.GENERATES[R.STEM_ELEMENT[pillars[name][0]]] == mine
        ]
        listing = "；".join(
            f"{POS[name]}干{pillars[name][0]}（{R.ten_god(day, pillars[name][0])}）"
            for name in ("year", "month")
        )
        head = f"得势要求年、月、时三干中至少两个是比劫或印。已知{listing}，其中属比劫或印的有 {len(helpers)} 个。"
        if len(helpers) == 2:
            text = head + "已经够两个，无论时干是什么都成立。"
            return "three", question, text, {"determinate": True, "needs": [], "answer": True}
        if not helpers:
            text = head + "即使时干是比劫或印也只有一个，不够两个，无论时干是什么都不成立。"
            return "three", question, text, {"determinate": True, "needs": [], "answer": False}
        text = head + "还差一个，成立与否取决于时干，而时辰不详，所以无法确定。请补充出生时辰。"
        return "three", question, text, {"determinate": False, "needs": ["hour"]}
    if kind == "gender_needed":
        choice = variant % 2
        stem = pillars["year"][0]
        if choice == 0:
            question = "大运是顺行还是逆行？第一步大运是什么？"
            forward = R.luck_sequence(pillars["month"], "forward", 1)[0]
            backward = R.luck_sequence(pillars["month"], "backward", 1)[0]
            yang = R.is_yang(stem)
            text = (
                f"大运的方向由年干阴阳和性别共同决定。年干{stem}为{polarity(stem)}干："
                f"男命{'顺' if yang else '逆'}行，第一步{forward if yang else backward}；"
                f"女命{'逆' if yang else '顺'}行，第一步{backward if yang else forward}。"
                "没有说明性别，无法唯一确定，请补充。"
            )
            by = {
                "male": forward if yang else backward,
                "female": backward if yang else forward,
            }
        else:
            question = "传统上这个八字看婚姻的星是哪些十神？天干里有没有？"
            seen = "、".join(
                f"{POS[name]}干{pillars[name][0]}为{R.ten_god(day, pillars[name][0])}"
                for name in OTHER
            )
            text = (
                "传统的六亲对应分男女：男命以正财、偏财为妻星，女命以正官、七杀为夫星。"
                f"本盘{seen}。没有说明性别，无法确定该看哪一组，请补充。"
            )
            gods = set(D.stem_gods(pillars).values())
            by = {
                "male": bool(gods & {"正财", "偏财"}),
                "female": bool(gods & {"正官", "七杀"}),
            }
        return (
            "nogender",
            question,
            text,
            {"determinate": False, "needs": ["gender"], "by_gender": by},
        )
    if kind == "gender_given":
        text, answer = T.luck_answer(pillars, gender, 1)
        question = "大运是顺行还是逆行？第一步大运是什么？"
        return (
            "full",
            question,
            text,
            {
                "determinate": True,
                "needs": [],
                "answer": {"direction": answer["direction"], "first": answer["first"][0]},
            },
        )
    if kind == "star_basis":
        star = rng.choice(tuple(D.TRIPLE_STARS))
        question = (
            f"这个八字的{star}在哪一支？",
            f"此命{star}落在哪个地支？",
            f"请查出这个盘的{star}。",
        )[variant % 3]
        by_year = D.star_branches(star, pillars, "year")
        by_day = D.star_branches(star, pillars, "day")
        table = D.TRIPLE_STARS[star]
        listing = "、".join(f"{key}在{value}" for key, value in table.items())
        year_b, day_b = pillars["year"][1], pillars["day"][1]
        head = (
            f"{star}按三合局查：{listing}。古书以年支查，后来也有以日支查的。"
            f"年支{year_b}属{D._TRIPLE_OF[year_b]}，{star}在{by_year}；"
            f"日支{day_b}属{D._TRIPLE_OF[day_b]}，{star}在{by_day}。"
        )
        if by_year == by_day:
            text = head + "两种查法结果相同。"
            return (
                "full",
                question,
                text,
                {"determinate": True, "needs": [], "answer": by_year},
            )
        text = head + "两种查法结果不同，没有说明以哪一支为准就没有唯一答案。"
        return (
            "full",
            question,
            text,
            {
                "determinate": False,
                "needs": ["star_basis"],
                "by_basis": {"year": by_year, "day": by_day},
            },
        )
    if kind == "no_rule":
        # The last two wordings are reserved for the test split (variant 2).
        question = rng.choice(NO_RULE[4:] if variant == 2 else NO_RULE[:4])
        strength = R.strength_indicators(pillars)["summary"]
        text = (
            f"规则能确定的是结构：日主{day}，月令{pillars['month'][1]}，按三项计数约定为{strength}，"
            f"按月令透干取格为{D.revealed_pattern(pillars)['pattern']}。"
            "但“好坏”“富贵”“吉凶”“寿夭”不是任何一张规则表能算出来的结论，各派说法不一，"
            "也没有可验证的依据，所以这个问题没有规则上的唯一答案，我不下这样的断语。"
        )
        return (
            "full",
            question,
            text,
            {"determinate": False, "needs": [], "reason": "not_rule_determined"},
        )
    raise ValueError(kind)


# ---- Multiple choice with rule-determined keys ------------------------------------------------

MCQ_KINDS = (
    "flow_god",
    "palace",
    "star_year",
    "true_claim",
    "pattern2",
    "fuyi",
    "config",
    "luck_god",
    "clash_year",
)
LETTERS = "ABCD"


def _options(answer, wrong, rng):
    options = list(wrong) + [answer]
    rng.shuffle(options)
    return [f"{LETTERS[i]}. {option}" for i, option in enumerate(options)], LETTERS[
        options.index(answer)
    ]


def mcq_task(kind, pillars, rng, gender="male", year=2030):
    """A four-option question whose key follows from one or more rule applications, or None."""
    day = pillars["day"][0]
    if kind == "flow_god":
        years = list(range(year, year + 10))
        god = R.ten_god(day, R.year_pillar(rng.choice(years))[0])
        hits = [y for y in years if R.ten_god(day, R.year_pillar(y)[0]) == god]
        target = rng.choice(hits)
        others = [y for y in years if y not in hits]
        wrong = rng.sample(others, 3)
        question = f"下列哪一年的流年天干是此命日主的{god}？"
        flow = R.year_pillar(target)
        reason = (
            f"{target} 年为{flow}年。{god_reason(day, flow[0])}。其余三年的流年天干都不是这个十神。"
        )
        answer, wrong = f"{target} 年", [f"{y} 年" for y in wrong]
    elif kind == "palace":
        branch = pillars["day"][1]
        mode = rng.choice(("冲", "六合"))
        partner = D.branch_card(branch)[mode]
        years = list(range(year, year + 12))
        target = next(y for y in years if R.year_pillar(y)[1] == partner)
        wrong = rng.sample([y for y in years if y != target], 3)
        word = "相冲" if mode == "冲" else "六合"
        question = (
            f"传统上把流年地支与日支{word}看作引动配偶宫。下列哪一年的流年地支与此命日支{word}？"
        )
        reason = (
            f"日支为{branch}，与{branch}{word}的是{partner}。{target} 年为{R.year_pillar(target)}年，"
            f"地支正是{partner}。这只是符号上的引动，不代表一定发生什么。"
        )
        answer, wrong = f"{target} 年", [f"{y} 年" for y in wrong]
    elif kind == "star_year":
        star = rng.choice(("桃花", "驿马"))
        target_branch = D.star_branches(star, pillars, "day")
        years = list(range(year, year + 12))
        target = next(y for y in years if R.year_pillar(y)[1] == target_branch)
        wrong = rng.sample([y for y in years if y != target], 3)
        day_branch = pillars["day"][1]
        question = f"以日支查{star}，下列哪一年的流年地支正是此命的{star}？"
        reason = (
            f"日支{day_branch}属{D._TRIPLE_OF[day_branch]}三合局，{star}在{target_branch}。"
            f"{target} 年为{R.year_pillar(target)}年，地支是{target_branch}。"
        )
        answer, wrong = f"{target} 年", [f"{y} 年" for y in wrong]
    elif kind == "true_claim":
        kinds = rng.sample(CLAIM_KINDS, 4)
        right = make_claim(kinds[0], pillars, rng, gender, True)
        wrongs = [make_claim(k, pillars, rng, gender, False) for k in kinds[1:]]
        question = f"关于此命（{GENDER_ZH[gender]}命），下列哪一项说法符合规则？"
        reason = f"{right[2]}，所以“{right[0]}”成立；其余三项与规则算出的结果不符。"
        answer, wrong = right[0], [w[0] for w in wrongs]
    elif kind == "pattern2":
        result = D.revealed_pattern(pillars)
        answer = result["pattern"]
        pool = [g + "格" for g in R.TEN_GODS[2:]] + ["建禄格", "阳刃格", "月劫格"]
        wrong = rng.sample([p for p in pool if p != answer], 3)
        simple = R.month_pattern(pillars)
        if simple != answer and simple not in wrong:
            wrong[0] = simple  # the main-qi label is the natural distractor
        question = "按月令透干取格的约定（本气、中气、余气依次看是否透出），此命记作什么格？"
        month = pillars["month"][1]
        how = {
            "main_is_peer": "本气是比劫，不按透干取",
            "main_revealed": "本气透干",
            "middle_revealed": "本气不透而中气透干",
            "residual_revealed": "本气、中气不透而余气透干",
            "main_default": "藏干都不透，回到本气",
        }[result["basis"]]
        reason = (
            (f"日主{day}按相生次序{class_chain(day)}。" if explicit() else "")
            + f"月支{month}藏{join(R.HIDDEN_STEMS[month])}，三干为{join(pillars[n][0] for n in OTHER)}；"
            f"{how}，取{result['stem']}（{result['god']}），记作{answer}。"
        )
    elif kind == "fuyi":
        result = D.fuyi(pillars)
        strength = R.strength_indicators(pillars)
        flags = "、".join(f"{k}{'是' if strength[k] else '否'}" for k in ("得令", "得地", "得势"))
        if result["strength"] == "偏弱":
            answer = rng.choice(result["favorable_classes"])
            wrong = result["unfavorable_classes"]
            question = "按扶抑一派（强弱用得令、得地、得势三项计数），下列哪一类十神是此命所喜？"
            rule = "偏弱喜印与比劫"
        else:
            answer = rng.choice(result["unfavorable_classes"])
            wrong = result["favorable_classes"]
            question = "按扶抑一派（强弱用得令、得地、得势三项计数），下列哪一类十神是此命所忌？"
            rule = "偏强喜食伤、财、官杀，忌印与比劫"
        reason = f"三项计数：{flags}，记为{result['strength']}。{rule}，所以选{answer}。"
    elif kind == "config":
        found = D.stem_configs(pillars)
        if not found:
            return None
        answer = rng.choice(found)
        absent = [name for name, *_ in D.CONFIGS if name not in found]
        if len(absent) < 3:
            return None
        wrong = rng.sample(absent, 3)
        gods = D.stem_gods(pillars)
        question = "此命年、月、时三干之间构成下列哪一种十神组合？"
        reason = (
            (f"日主{day}按相生次序{class_chain(day)}。" if explicit() else "")
            + "三干十神为"
            + "、".join(f"{POS[n]}干{pillars[n][0] if explicit() else ''}{gods[n]}" for n in OTHER)
            + f"，构成{answer}；"
            "其余三项所需的十神没有同时出现。"
        )
    elif kind == "luck_god":
        direction = R.luck_direction(pillars["year"][0], gender)
        step = rng.choice((1, 2, 3))
        pillar = R.luck_sequence(pillars["month"], direction, step)[-1]
        answer = R.ten_god(day, pillar[0])
        wrong = rng.sample([g for g in R.TEN_GODS if g != answer], 3)
        ordinal = "一二三"[step - 1]
        question = f"此为{GENDER_ZH[gender]}命。第{ordinal}步大运的天干是日主的什么十神？"
        word = "顺" if direction == "forward" else "逆"
        stem = pillars["year"][0]
        reason = (
            f"年干{stem}为{polarity(stem)}干，{GENDER_ZH[gender]}命大运{word}行；从月柱{pillars['month']}"
            f"{word}推第{ordinal}步为{pillar}。{god_reason(day, pillar[0])}。"
        )
    elif kind == "clash_year":
        years = list(range(year, year + 30))
        hits = [y for y in years if D.clash_with_control(pillars["day"], R.year_pillar(y))]
        if not hits:
            return None
        target = hits[0]
        wrong = rng.sample([y for y in years if y not in hits], 3)
        flow = R.year_pillar(target)
        question = "下列哪一年的流年与此命日柱天克地冲（地支相冲且天干五行相克）？"
        reason = (
            f"日柱为{pillars['day']}。{target} 年为{flow}年：{clash_words(flow[1], pillars['day'][1])}，"
            f"{D.element_relation(R.STEM_ELEMENT[flow[0]], R.STEM_ELEMENT[day])}，两项都成立。"
        )
        answer, wrong = f"{target} 年", [f"{y} 年" for y in wrong]
    else:
        raise ValueError(kind)
    if len(set(wrong)) < 3 or answer in wrong:
        return None
    lines, letter = _options(answer, wrong, rng)
    if explicit() and kind in ("flow_god", "palace", "star_year", "clash_year"):
        # Give one known year and evaluate every option in the open instead of silently.
        listed = sorted(int(line[3:7]) for line in lines)
        question = f"已知 {listed[0]} 年为{R.year_pillar(listed[0])}年。" + question
        each = "、".join(f"{line[0]} {line[3:7]}{R.year_pillar(int(line[3:7]))}" for line in lines)
        reason = f"四个选项的流年：{each}。" + reason
    return {"question": question, "options": lines, "answer": letter, "reason": reason}


# ---- Whole-chart overview (tool block required) -----------------------------------------------


def overview_task(pillars, rng, variant=0, gender="male"):
    question = (
        "请根据上面的排盘，把规则能确定的结构整理成一份完整的说明，并标出哪些是约定、哪些不能确定。",
        "请整体解读这个盘：十神、强弱、格局、用神（分流派）、合冲、神煞，各自说明依据，不要预测事件。",
        "帮我系统地讲一遍这个八字在传统规则下的结构，哪些是算出来的，哪些只是传统说法。",
    )[variant % 3]
    day = pillars["day"][0]
    gods = D.stem_gods(pillars)
    config_text, found = configs_text(pillars)
    strength = R.strength_indicators(pillars)
    pattern = D.revealed_pattern(pillars)
    schools = D.yongshen_by_school(pillars)
    fuyi, tiao = schools["扶抑"], schools["调候"]
    flags = "、".join(f"{k}{'是' if strength[k] else '否'}" for k in ("得令", "得地", "得势"))
    relations = R.chart_relations(pillars)
    adjacent = sorted({i["key"] for i in D.positioned_relations(pillars) if i["adjacent"]})
    stars = []
    for star in D.STAR_NAMES:
        branches = D.star_branches(star, pillars, "day")
        if not branches:
            stars.append(f"{star}：阴干无阳刃")
            continue
        where = D.star_positions(star, pillars, "day")
        stars.append(
            f"{star}在{join(branches)}（{'见于' + join(R.PILLAR_ZH[n] for n in where) if where else '原局不见'}）"
        )
    simple = R.month_pattern(pillars)
    pattern_note = (
        f"与只看本气的简化标签{simple}相同"
        if simple == pattern["pattern"]
        else f"只看本气的简化标签是{simple}"
    )
    lines = [
        *([god_table(day)] if explicit() else []),
        f"一、日主与十神（确定）：日主{day}（{R.STEM_ELEMENT[day]}，{polarity(day)}）；"
        + "、".join(f"{POS[n]}干{pillars[n][0]}为{gods[n]}" for n in OTHER)
        + "。"
        + config_text.split("。", 1)[1],
        f"二、强弱（计数约定）：{flags}，记为{strength['summary']}。",
        f"三、格局（月令透干取格的约定）：月支{pillars['month'][1]}藏{join(R.HIDDEN_STEMS[pillars['month'][1]])}，"
        f"取{pattern['stem']}（{pattern['god']}），记作{pattern['pattern']}；{pattern_note}。",
        "四、用神（分流派，结论可以不同）：扶抑喜"
        + join(fuyi["favorable_classes"])
        + f"（{join(fuyi['favorable_elements'])}）；调候"
        + (f"取{tiao['needs']}" if tiao["needs"] else "不特别论")
        + f"；格局之神{pattern['god']}属{D.PATTERN_USE[pattern['god']]}。",
        f"五、合冲刑害（确定）：{relation_words(relations)}；其中相邻两柱之间的：{relation_words(adjacent)}。",
        "六、神煞（以日干、日支查）：" + "；".join(stars) + "。",
        "七、不能确定的部分：以上是规则推出的符号结构。它们在现实中意味着什么、会不会发生某件事，"
        "规则无法确证，我不作预测。",
    ]
    return (
        question,
        "\n".join(lines),
        {
            "kind": "overview",
            "strength": strength["summary"],
            "pattern": pattern["pattern"],
            "configs": found,
        },
    )
