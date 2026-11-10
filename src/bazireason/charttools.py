"""The verified chart tool: engine pillars plus deterministic derived facts.

``verified_chart`` is what "with verified chart tools" means in Phase 2. The
four pillars come from the unchanged calendar engine under its declared
profile (fixed UTC+08:00 reference clock, 立春 and Jie instants, a stated
sect). Derived facts come from ``bazireason.rules``. Luck-cycle direction and
order are computed by rule and checked against lunar-python; the luck start
time is lunar-python's own convention and is labelled single-source.

Nothing here interprets fortune, and nothing is guessed: a request the engine
does not answer returns its status and reason codes without pillars.
"""

from datetime import datetime

from lunar_python import Solar

from bazireason import rules
from bazireason.calendar_engine import compute_chart
from bazireason.conventions import PILLARS

ENGINE_NOTE = "引擎 lunar-python 1.4.8；参考时钟固定 UTC+08:00；年以立春、月以节的时刻为界"


def luck_cycles(reference: datetime, gender: str, count: int = 8) -> dict:
    """Luck cycles for an aware UTC+08:00 instant (start time: lunar-python sect 1)."""
    eight = (
        Solar.fromYmdHms(
            reference.year,
            reference.month,
            reference.day,
            reference.hour,
            reference.minute,
            reference.second,
        )
        .getLunar()
        .getEightChar()
    )
    yun = eight.getYun(1 if gender == "male" else 0, 1)
    direction = rules.luck_direction(eight.getYear()[0], gender)
    sequence = rules.luck_sequence(eight.getMonth(), direction, count)
    cycles = []
    for expected, cycle in zip(sequence, yun.getDaYun(count + 1)[1:], strict=True):
        if cycle.getGanZhi() != expected or yun.isForward() != (direction == "forward"):
            raise AssertionError("Rule-derived luck cycle disagrees with lunar-python")
        cycles.append(
            {
                "pillar": expected,
                "start_year": cycle.getStartYear(),
                "end_year": cycle.getEndYear(),
                "start_age": cycle.getStartAge(),
                "end_age": cycle.getEndAge(),
            }
        )
    return {
        "direction": direction,
        "start_after": {
            "years": yun.getStartYear(),
            "months": yun.getStartMonth(),
            "days": yun.getStartDay(),
        },
        "start_date": yun.getStartSolar().toYmd(),
        "start_convention": "lunar-python sect 1 (3 days = 1 year); single source",
        "age_convention": "虚岁",
        "cycles": cycles,
    }


def verified_chart(inputs: dict, convention: dict, gender=None, until_year=None) -> dict:
    """Engine result plus derived facts; pillars are present only for status ok."""
    result = compute_chart(inputs, convention)
    chart = {
        "status": result["status"],
        "reason_codes": result["reason_codes"],
        "conventions": result["conventions"],
        "pillars": result["pillars"],
        "reference_utc8": None,
        "provenance": result["provenance"],
    }
    if result["status"] != "ok":
        return chart
    reference = datetime.fromisoformat(result["normalized_candidates"][0])
    chart["reference_utc8"] = result["normalized_candidates"]
    chart["facts"] = rules.chart_summary(result["pillars"])
    if gender in ("male", "female"):
        chart["gender"] = gender
        chart["luck"] = luck_cycles(reference, gender)
    if until_year is not None:
        chart["flow_years"] = {
            str(year): rules.year_pillar(year) for year in range(reference.year, until_year + 1)
        }
    return chart


def pillars_text(pillars: dict) -> str:
    return " ".join(f"{rules.PILLAR_ZH[name]}{pillars[name]}" for name in PILLARS)


def chart_tool_text(chart: dict, flow_years: bool = True) -> str:
    """The tool block shown to a model. Fixed wording, no interpretation of fortune."""
    lines = [f"[排盘工具·已核验] {ENGINE_NOTE}"]
    if chart["status"] != "ok":
        lines.append(
            f"状态：{chart['status']}；原因：{'、'.join(chart['reason_codes'])}。未给出四柱。"
        )
        return "\n".join(lines)
    sect = chart["conventions"]["sect"]
    moment = chart["reference_utc8"][0][:19].replace("T", " ")
    lines.append(
        f"参考时刻：{moment}（UTC+08:00）；日柱流派 sect={'未指定' if sect is None else sect}"
    )
    pillars, facts = chart["pillars"], chart["facts"]
    lines.append("四柱：" + pillars_text(pillars))
    lines.append(
        f"日主：{facts['day_master']}（{facts['day_master_element']}，"
        f"{facts['day_master_polarity']}）"
    )
    lines.append(
        "天干十神："
        + "；".join(
            f"{rules.POSITION_ZH[name]}干{pillars[name][0]} {facts['stem_ten_gods'][name]}"
            for name in ("year", "month", "hour")
        )
    )
    hidden = []
    for name in PILLARS:
        pairs = zip(facts["hidden_stems"][name], facts["hidden_ten_gods"][name], strict=True)
        hidden.append(
            f"{rules.POSITION_ZH[name]}支{pillars[name][1]} "
            + " ".join(f"{stem}({god})" for stem, god in pairs)
        )
    lines.append("地支藏干：" + "；".join(hidden))
    lines.append(
        "五行计数（八字明见）："
        + " ".join(f"{element}{count}" for element, count in facts["element_counts"].items())
    )
    relations = [rules.relation_text(key) for key in facts["relations"]]
    lines.append("干支关系：" + ("；".join(relations) if relations else "无"))
    strength = facts["strength"]
    lines.append(
        f"月令：日主{facts['day_master_element']}在{pillars['month'][1]}月为"
        f"“{facts['seasonal_state']}”；格局标签（简化约定）：{facts['month_pattern']}"
    )
    lines.append(
        "强弱计数（约定）："
        + "、".join(f"{key}{'是' if strength[key] else '否'}" for key in ("得令", "得地", "得势"))
        + f" → {strength['summary']}"
    )
    lines.append(
        "十二长生（日主在各支）："
        + " ".join(f"{rules.POSITION_ZH[name]}{facts['life_stages'][name]}" for name in PILLARS)
        + f"；日柱旬空：{facts['day_xun_kong']}"
    )
    if "luck" in chart:
        luck = chart["luck"]
        after = luck["start_after"]
        lines.append(
            f"大运（{'男' if chart['gender'] == 'male' else '女'}命，"
            f"{'顺行' if luck['direction'] == 'forward' else '逆行'}；出生后约 {after['years']} 年 "
            f"{after['months']} 个月 {after['days']} 天起运，按三天折一年，单一来源）："
            + "；".join(
                f"{c['pillar']}（{c['start_year']}–{c['end_year']}，虚岁 {c['start_age']}–"
                f"{c['end_age']}）"
                for c in luck["cycles"]
            )
        )
    if flow_years and chart.get("flow_years"):
        lines.append(
            "流年（立春起算）："
            + "，".join(f"{year}{pillar}" for year, pillar in chart["flow_years"].items())
        )
    return "\n".join(lines)
