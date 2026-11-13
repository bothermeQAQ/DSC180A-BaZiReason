"""Verified fact sheets: everything the rule code knows about a chart, as labelled lines.

Phase 5 stops asking the model to compute what code can compute. The model receives
these lines and explains them. Every line is produced by the calendar engine or the
project's rule code (``rules``, ``deep_rules``); nothing here interprets fortune.

Each statement type is labelled in the text:
  （确定）   fixed tables and arithmetic;
  （约定）   a declared counting or labelling convention;
  （分流派） results that depend on a named school;
  （传统对应）symbolic associations.
"""

from bazireason import deep_rules as D
from bazireason import demo_deep
from bazireason import rules as R

PILLARS = ("year", "month", "day", "hour")
POS = R.POSITION_ZH
ELEMENTS = "木火土金水"
SHEET_HEAD = "[规则代码·已核验的事实]"
ANSWER_HEAD = "[规则代码·针对这个问题的已核验结论]"
GROUNDING_RULE = (
    "以上由规则代码计算并核验。回答时以这些事实为准，不要改动其中的干支、十神、关系和结论；"
    "事实里没有的内容不要编造，规则不能决定的要说明。"
)


def pillar_line(pillars: dict) -> str:
    return " ".join(f"{R.PILLAR_ZH[name]}{pillars[name]}" for name in PILLARS)


def luck_line(chart: dict) -> str | None:
    luck = chart.get("luck")
    if not luck:
        return None
    after = luck["start_after"]
    cycles = "；".join(
        f"{c['pillar']}（{c['start_year']}–{c['end_year']}，虚岁 {c['start_age']}–{c['end_age']}）"
        for c in luck["cycles"]
    )
    way = "顺行" if luck["direction"] == "forward" else "逆行"
    return (
        f"大运（确定；起运时间按三天折一年，单一来源）：{way}，出生后约 {after['years']} 年 "
        f"{after['months']} 个月起运：{cycles}"
    )


def sheet_lines(pillars: dict, gender: str | None = None, chart: dict | None = None) -> list:
    """The full fact sheet of one chart."""
    f = R.chart_summary(pillars)
    day = pillars["day"][0]
    hidden = "；".join(
        f"{POS[n]}支{pillars[n][1]}藏"
        + "、".join(f"{s}({g})" for s, g in zip(f["hidden_stems"][n], f["hidden_ten_gods"][n], strict=True))
        for n in PILLARS
    )
    counts = f["element_counts"]
    missing = [e for e in ELEMENTS if counts[e] == 0]
    lines = [
        f"四柱（确定）：{pillar_line(pillars)}" + (f"；{'男' if gender == 'male' else '女'}命" if gender else ""),
        f"日主（确定）：{day}（{f['day_master_element']}，{f['day_master_polarity']}）",
        "天干十神（确定）："
        + "；".join(f"{POS[n]}干{pillars[n][0]}为{f['stem_ten_gods'][n]}" for n in ("year", "month", "hour")),
        f"地支藏干与十神（确定，本气在前）：{hidden}",
        "五行个数（约定：只数八个字，不算藏干）："
        + "、".join(f"{e}{counts[e]}" for e in ELEMENTS)
        + (f"；八字中没有{'、'.join(missing)}" if missing else "；五行俱全"),
        "十二长生（确定，日主在各支，阳顺阴逆）："
        + "、".join(f"{POS[n]}支{pillars[n][1]}为{f['life_stages'][n]}" for n in PILLARS),
        f"旬空（确定，按日柱）：日柱{pillars['day']}所在旬空{f['day_xun_kong'][0]}、{f['day_xun_kong'][1]}",
    ]
    if gender in ("male", "female"):
        way = R.luck_direction(pillars["year"][0], gender)
        lines.append(
            f"大运顺逆（确定）：年干{pillars['year'][0]}为{'阳' if R.is_yang(pillars['year'][0]) else '阴'}，"
            f"{'男' if gender == 'male' else '女'}命，大运{'顺' if way == 'forward' else '逆'}排"
        )
    if chart:
        extra = luck_line(chart)
        if extra:
            lines.append(extra)
    lines += [line for line in demo_deep.deep_lines(pillars) if not line.startswith("七、")]
    lines.append(
        "不能确定的部分：以上是规则推出的符号结构；现实中意味着什么、会不会发生某件事，规则无法确证。"
    )
    return lines


def fact_sheet(pillars: dict, gender: str | None = None, chart: dict | None = None) -> str:
    return SHEET_HEAD + "\n" + "\n".join(sheet_lines(pillars, gender, chart))


# ---- single facts with a short derivation (the "fact" intents) ---------------------------------
FACT_KINDS = (
    "ten_god_stem",
    "ten_god_branch",
    "hidden",
    "life_stage",
    "xun_kong",
    "luck_direction",
    "element_count",
    "day_master",
    "missing_element",
)


def fact_answer(kind: str, pillars: dict, position: str | None = None, gender: str | None = None):
    """(structured answer, derivation text) for one fact; None if the inputs do not decide it."""
    day = pillars["day"][0]
    if kind == "ten_god_stem":
        if position not in ("year", "month", "hour"):
            return None
        stem = pillars[position][0]
        god = R.ten_god(day, stem)
        return god, f"{POS[position]}干{stem}对日主{day}：{R.ten_god_reason(day, stem)}。"
    if kind == "ten_god_branch":
        if position not in PILLARS:
            return None
        branch = pillars[position][1]
        main = R.HIDDEN_STEMS[branch][0]
        god = R.branch_ten_gods(day, branch)[0]
        return god, f"{POS[position]}支{branch}本气为{main}；{main}对日主{day}为{god}。"
    if kind == "hidden":
        if position not in PILLARS:
            return None
        branch = pillars[position][1]
        stems = list(R.HIDDEN_STEMS[branch])
        return stems, f"{POS[position]}支{branch}藏{'、'.join(stems)}（本气在前）。"
    if kind == "life_stage":
        if position not in PILLARS:
            return None
        branch = pillars[position][1]
        stage = R.life_stage(day, branch)
        way = "顺" if R.is_yang(day) else "逆"
        return stage, f"日主{day}为{'阳' if way == '顺' else '阴'}干，十二长生{way}行；在{POS[position]}支{branch}为{stage}。"
    if kind == "xun_kong":
        pair = R.xun_kong(pillars["day"])
        return list(pair), f"日柱{pillars['day']}所在的旬，旬空为{pair[0]}、{pair[1]}。"
    if kind == "luck_direction":
        if gender not in ("male", "female"):
            return None
        way = "顺" if R.luck_direction(pillars["year"][0], gender) == "forward" else "逆"
        polarity = "阳" if R.is_yang(pillars["year"][0]) else "阴"
        sex = "男" if gender == "male" else "女"
        return way, f"年干{pillars['year'][0]}为{polarity}干，{sex}命：阳男阴女顺排，阴男阳女逆排，所以大运{way}排。"
    if kind == "element_count":
        counts = R.element_counts(pillars)
        return counts, "只数八个字（不算藏干）：" + "、".join(f"{e}{counts[e]}" for e in ELEMENTS) + "。"
    if kind == "missing_element":
        counts = R.element_counts(pillars)
        missing = [e for e in ELEMENTS if counts[e] == 0]
        text = "只数八个字（不算藏干）：" + "、".join(f"{e}{counts[e]}" for e in ELEMENTS)
        text += f"；没有{'、'.join(missing)}。" if missing else "；五行俱全。"
        return missing, text + "算上藏干结论可能不同，这是约定。"
    if kind == "day_master":
        element = R.STEM_ELEMENT[day]
        return day, f"日主是日柱天干{day}（{element}，{'阳' if R.is_yang(day) else '阴'}）。"
    raise ValueError(kind)


def star_answer(star: str, pillars: dict):
    """Branches of a star by both bases (the ambiguity of the basis is kept)."""
    return {basis: D.star_branches(star, pillars, basis) for basis in ("day", "year")}
