"""Worked calendar answers in Chinese, with every pillar derived by an explicit rule.

The text produced here is the supervision target for the calendar part of the
Phase 2 training set. Each answer is recomputed from first principles by the
formulas written in the text (a day-number formula, 五虎遁, 五鼠遁) and is then
required to equal the calendar engine's result; a disagreement raises instead
of producing a training example.

Two information settings are supported:

* ``facts``: the request carries ``temporal_facts`` from the calendar tool
  (UTC+08:00 instants, the surrounding Jie instants and a day count). The
  engine's status is then derivable by rule.
* ``none``: no tool output. A model cannot know the exact instant of a solar
  term or every historical UTC offset, so the expected answer becomes
  ``needs_tool`` exactly when the result depends on such a fact. The rule that
  decides this is deterministic (``no_tool_expected``) and is tested against
  tzdata and the engine.
"""

from datetime import date, datetime, timedelta
from functools import lru_cache

from bazireason import rules, textstyle
from bazireason.calendar_engine import (
    FIRST_YEAR,
    LAST_YEAR,
    UTC8,
    _jie_of_table,
    compute_chart,
    load_timezone,
    parse_wall_time,
    temporal_facts,
)
from bazireason.conventions import BRANCHES, CYCLE, PILLARS, STEMS

CUMULATIVE_DAYS = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
# Calendar order of the month-start terms within a Gregorian year.
JIE_IN_YEAR = (
    "小寒",
    "立春",
    "惊蛰",
    "清明",
    "立夏",
    "芒种",
    "小暑",
    "立秋",
    "白露",
    "寒露",
    "立冬",
    "大雪",
)
NEEDS_TOOL = "needs_tool"
P2_STATUSES = ("ok", "ambiguous", "invalid", "unsupported", NEEDS_TOOL)
# Zones whose UTC offset a model is expected to know without a tool, with the first
# year from which the simple rule below holds (checked against tzdata in the tests).
FIXED_ZONES = {
    "UTC": (0, 1901),
    "Asia/Tokyo": (540, 1952),
    "Asia/Kolkata": (330, 1946),
    "Asia/Hong_Kong": (480, 1980),
    "Asia/Singapore": (480, 1982),
}
SEASONAL_ZONES = {  # zone: (standard minutes, daylight minutes); northern hemisphere
    "America/New_York": (-300, -240),
    "America/Los_Angeles": (-480, -420),
    "Europe/London": (0, 60),
}
SEASONAL_FROM_YEAR = 1976
DAYLIGHT_MONTHS = (5, 6, 7, 8, 9)
STANDARD_MONTHS = (12, 1, 2)
SHANGHAI_UNCERTAIN_YEARS = (1919, *range(1940, 1950))
SHANGHAI_DST_YEARS = range(1986, 1992)
SHANGHAI_DST_MONTHS = range(4, 10)
HOUR_RANGES = {
    branch: ("23:00–00:59" if i == 0 else f"{2 * i - 1:02d}:00–{2 * i:02d}:59")
    for i, branch in enumerate(BRANCHES)
}


def is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def day_number_formula(day: date) -> dict:
    """Sexagenary day index for 1901-2099 from small numbers only.

    1900-01-01 is 甲戌 (index 10) and a common year advances the cycle by
    365 mod 60 = 5 days, so the index is (9 + 5a + L + n) mod 60 with
    a = year - 1900, L = floor((a - 1) / 4) leap days before the year and n the
    day of the year.
    """
    a = day.year - 1900
    leaps = (a - 1) // 4
    of_year = CUMULATIVE_DAYS[day.month - 1] + day.day + int(is_leap(day.year) and day.month > 2)
    total = 9 + 5 * a + leaps + of_year
    return {
        "a": a,
        "leaps": leaps,
        "day_of_year": of_year,
        "before_month": CUMULATIVE_DAYS[day.month - 1],
        "leap_adjust": int(is_leap(day.year) and day.month > 2),
        "total": total,
        "index": total % 60,
    }


@lru_cache(maxsize=1)
def jie_windows() -> dict:
    """{term: ((month, day) earliest, (month, day) latest)} in UTC+08:00 over 1901-2099."""
    spans = {}
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        for moment, name in _jie_of_table(year):
            if not FIRST_YEAR <= moment.year <= LAST_YEAR:
                continue
            key = (moment.month, moment.day)
            low, high = spans.get(name, (key, key))
            spans[name] = (min(low, key), max(high, key))
    return spans


def window_text(name: str) -> str:
    (m1, d1), (m2, d2) = jie_windows()[name]
    return f"{m1} 月 {d1}–{d2} 日" if m1 == m2 else f"{m1} 月 {d1} 日–{m2} 月 {d2} 日"


def jie_window_of(day: date):
    """The term whose possible dates include this civil date, or None."""
    key = (day.month, day.day)
    for name, (low, high) in jie_windows().items():
        if low <= key <= high:
            return name
    return None


def previous_jie_by_date(day: date) -> str:
    """Latest month-start term before a date that is outside every Jie window."""
    key = (day.month, day.day)
    passed = [name for name in JIE_IN_YEAR if jie_windows()[name][1] < key]
    return passed[-1] if passed else "大雪"


def known_offset(zone: str, wall: datetime):
    """UTC offset in minutes that can be stated without a tool, else None."""
    if zone == "Asia/Shanghai":
        if wall.year in SHANGHAI_UNCERTAIN_YEARS:
            return None
        if wall.year in SHANGHAI_DST_YEARS and wall.month in SHANGHAI_DST_MONTHS:
            return None
        return 480
    if zone in FIXED_ZONES:
        minutes, since = FIXED_ZONES[zone]
        return minutes if wall.year >= since else None
    if zone in SEASONAL_ZONES and wall.year >= SEASONAL_FROM_YEAR:
        standard, daylight = SEASONAL_ZONES[zone]
        if wall.month in DAYLIGHT_MONTHS:
            return daylight
        if wall.month in STANDARD_MONTHS:
            return standard
    return None


def offset_text(minutes: int) -> str:
    sign = "+" if minutes >= 0 else "−"
    hours, rest = divmod(abs(minutes), 60)
    return f"UTC{sign}{hours}" + (f":{rest:02d}" if rest else "")


def mod60_text(value: int):
    """(text, remainder) showing value mod 60 with one division."""
    if value >= 0:
        quotient, remainder = divmod(value, 60)
        return f"{value} = 60×{quotient} + {remainder}，余 {remainder}", remainder
    quotient, rest = divmod(-value, 60)
    remainder = (60 - rest) % 60
    tail = f"60 − {rest} = {remainder}" if rest else "0"
    return f"{-value} = 60×{quotient} + {rest}，故 {value} mod 60 = {tail}", remainder


def index_text(index: int) -> str:
    return (
        f"{index} mod 10 = {index % 10} → {STEMS[index % 10]}，"
        f"{index} mod 12 = {index % 12} → {BRANCHES[index % 12]}"
    )


def clock(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")


# ---- derivation version 3: the checks before the decision (preregistration 12.14) ---------------
# Without tool facts, the zone's rule and the solar-term windows are written and compared before
# the answer is decided. Off by default; bazireason.steps6.set_steps(3) turns it on.
_CHECK_FIRST = [False]
KNOWN_ZONES_TEXT = "、".join(
    ["UTC", "Asia/Shanghai", *(z for z in FIXED_ZONES if z != "UTC"), *SEASONAL_ZONES]
)
# Reason codes decided before the offset (as in no_tool_expected).
_DECIDED_BEFORE_OFFSET = {
    "missing_datetime_or_timezone",
    "full_wall_time_required",
    "invalid_datetime_format",
    "invalid_date_or_timezone",
    "invalid_convention",
    "invalid_sect",
    "invalid_fold",
    "unsupported_convention",
}


def set_check_first(on: bool) -> None:
    _CHECK_FIRST[0] = bool(on)


def check_first() -> bool:
    return _CHECK_FIRST[0]


def zone_step(zone: str, wall: datetime):
    """(the 时区 step, UTC offset in minutes or None): the zone's rule, then the date against it.
    The conclusion always equals ``known_offset``. Compact: a calendar reply is capped at 700
    tokens (run6.MAX_TOKENS)."""
    y, m = wall.year, wall.month
    when = f"{y} 年 {m} 月"
    if zone == "Asia/Shanghai":
        rule = "Asia/Shanghai 除 1919 年、1940–1949 年和 1986–1991 年 4–9 月外为 UTC+8"
        if y in SHANGHAI_UNCERTAIN_YEARS:
            check, offset = f"{when}在 1919 年或 1940–1949 年 → 偏移需工具", None
        elif y in SHANGHAI_DST_YEARS and m in SHANGHAI_DST_MONTHS:
            check, offset = f"{when}在 1986–1991 年 4–9 月 → 偏移需工具", None
        elif y in SHANGHAI_DST_YEARS:
            check, offset = f"{when}在 1986–1991 年但不在 4–9 月 → UTC+8", 480
        else:
            check, offset = f"{when}不在例外中 → UTC+8", 480
    elif zone in FIXED_ZONES:
        minutes, since = FIXED_ZONES[zone]
        fixed = offset_text(minutes)
        rule = f"{zone} 自 {since} 年起为 {fixed}"
        if y >= since:
            check, offset = f"{y} 年不早于 {since} 年 → {fixed}", minutes
        else:
            check, offset = f"{y} 年早于 {since} 年 → 偏移需工具", None
    elif zone in SEASONAL_ZONES:
        standard, daylight = SEASONAL_ZONES[zone]
        rule = (
            f"{zone} 自 {SEASONAL_FROM_YEAR} 年起 5–9 月为 {offset_text(daylight)}、"
            f"12–2 月为 {offset_text(standard)}，其余月份要查工具"
        )
        if y < SEASONAL_FROM_YEAR:
            check, offset = f"{y} 年早于 {SEASONAL_FROM_YEAR} 年 → 偏移需工具", None
        elif m in DAYLIGHT_MONTHS:
            check, offset = f"{when}在 5–9 月 → {offset_text(daylight)}", daylight
        elif m in STANDARD_MONTHS:
            check, offset = f"{when}在 12–2 月 → {offset_text(standard)}", standard
        else:
            check, offset = f"{when}是切换月份 → 偏移需工具", None
    else:
        rule = f"凭记忆能确定偏移的时区只有 {KNOWN_ZONES_TEXT}"
        check, offset = f"{zone} 不在其中 → 偏移需工具", None
    if offset != known_offset(zone, wall):
        raise AssertionError(f"zone step {zone} {wall}: {offset} != {known_offset(zone, wall)}")
    return f"时区：{rule}；{check}。", offset


def window_step(reference: datetime):
    """(the 交节 step, the term whose possible dates hold the reference date, or None): the
    reference date against the nearest bounds of the solar-term windows around it."""
    day = reference.date()
    here = f"{day.month} 月 {day.day} 日"
    name = jie_window_of(day)
    if name:
        (m1, d1), (m2, d2) = jie_windows()[name]
        return (
            f"交节：{here}不早于{name}最早的 {m1} 月 {d1} 日、不晚于最晚的 {m2} 月 {d2} 日 "
            f"→ 在{name}窗口内，需工具。"
        ), name
    before = previous_jie_by_date(day)
    after = JIE_IN_YEAR[(JIE_IN_YEAR.index(before) + 1) % 12]
    pm, pd = jie_windows()[before][1]
    nm, nd = jie_windows()[after][0]
    last = "上年" if (pm, pd) > (day.month, day.day) else ""
    following = "次年" if (nm, nd) < (day.month, day.day) else ""
    return (
        f"交节：{here}晚于{last}{before}最晚的 {pm} 月 {pd} 日、早于{following}{after}最早的 "
        f"{nm} 月 {nd} 日 → 不在交节窗口内。"
    ), None


def reference_text(zone: str, offset: int, reference: datetime) -> str:
    """The 参考时刻 step of version 3 (the zone's offset is in the 时区 step just before)."""
    if offset == 480:
        return f"参考时刻：即 {clock(reference)}（UTC+08:00）。"
    shift = 480 - offset
    hours, minutes = divmod(abs(shift), 60)
    amount = f"{hours} 小时" + (f" {minutes} 分" if minutes else "")
    return (
        f"参考时刻：{offset_text(offset)} {'加' if shift > 0 else '减'} {amount}，"
        f"得 {clock(reference)}（UTC+08:00）。"
    )


def check_steps(zone: str, wall: datetime) -> tuple:
    """([时区, 参考时刻, 交节] steps, reference, term or None) for a zone whose offset is known."""
    text, offset = zone_step(zone, wall)
    if offset is None:
        raise ValueError(f"{zone} {wall}: the offset needs the tool")
    reference = wall + timedelta(minutes=480 - offset)
    window, name = window_step(reference)
    return [text, reference_text(zone, offset, reference), window], reference, name


def window_refusal(name: str) -> str:
    affected = "年柱和月柱" if name == "立春" else "月柱"
    return (
        f"没有该年{name}的精确时刻，就无法判断出生在节前还是节后，{affected}不能确定。"
        "请提供历法工具给出的交节时刻。"
    )


OFFSET_REFUSAL = (
    "没有时区工具，就无法可靠地把出生时间归一到 UTC+08:00。请提供历法工具的时间事实，"
    "或直接给出对应的 UTC+08:00 时刻。"
)


def _checked_answer(inputs, convention, result, expected, sect):
    """Version 3: the answer without tool facts with its checks first, or None when the answer
    does not depend on the offset (decided before it, as in no_tool_expected)."""
    raw, zone = inputs.get("datetime"), inputs.get("timezone")
    parsed = None
    if not any(r in _DECIDED_BEFORE_OFFSET for r in result["reason_codes"]):
        try:
            parsed = parse_wall_time(raw) if isinstance(raw, str) else None
        except ValueError:
            parsed = None
    if parsed is None or not isinstance(zone, str) or result["status"] == "unsupported":
        if expected["reason_codes"] in (["utc_offset_required"], ["jie_instant_required"]):
            raise AssertionError(f"decided before the offset, yet {expected}")
        return None
    wall = parsed[1]
    text, offset = zone_step(zone, wall)
    steps = [text]
    if offset is None:
        if expected["reason_codes"] != ["utc_offset_required"]:
            raise AssertionError(f"offset unknown, yet {expected}")
        return {"steps": [*steps, OFFSET_REFUSAL], "response": expected}
    reference = wall + timedelta(minutes=480 - offset)
    if result["status"] == "ambiguous" and result["reason_codes"] == ["missing_convention"]:
        steps.append(_nonanswer_text(inputs, convention, result, None, offset))
        return {"steps": steps, "response": expected}
    steps.append(reference_text(zone, offset, reference))
    if result["status"] in ("ok", "ambiguous"):
        window, name = window_step(reference)
        steps.append(window)
        if name:
            if expected["reason_codes"] != ["jie_instant_required"]:
                raise AssertionError(f"inside {name}, yet {expected}")
            return {"steps": [*steps, window_refusal(name)], "response": expected}
    if expected["reason_codes"] in (["utc_offset_required"], ["jie_instant_required"]):
        raise AssertionError(f"checks passed, yet {expected}")
    if expected["status"] != "ok":
        steps.append(_nonanswer_text(inputs, convention, result, None, offset))
        return {"steps": steps, "response": expected}
    pillars, derived = derive_pillars(reference, sect, None, brief=True)
    if sect is None:
        derived[2] = derived[2][:-1] + "（未指定流派，但时刻不在 23 点，两派结果相同）。"
    if pillars != expected["pillars"]:
        raise AssertionError(f"Worked answer {pillars} differs from engine {expected['pillars']}")
    return {"steps": [*steps, *derived], "response": expected}


def _year_step(reference, previous_jie, with_facts):
    civil = reference.year
    if with_facts:
        if previous_jie == "小寒" or (previous_jie == "大雪" and reference.month == 1):
            year, why = (
                civil - 1,
                f"上一个节是{previous_jie}，还没到 {civil} 年立春，年柱按 {civil - 1} 年",
            )
        else:
            year, why = civil, f"上一个节是{previous_jie}，已过 {civil} 年立春，取 {civil} 年"
    else:
        window = window_text("立春")
        if (reference.month, reference.day) < jie_windows()["立春"][0]:
            year, why = civil - 1, f"在 {civil} 年立春（约 {window}）之前，年柱按 {civil - 1} 年"
        else:
            year, why = civil, f"在 {civil} 年立春（约 {window}）之后，取 {civil} 年"
    text, index = mod60_text(year - 4)
    pillar = CYCLE[index]
    return (
        year,
        pillar,
        f"年柱：{why}；{year} − 4 = {year - 4}，{text}；{index_text(index)}，年柱{pillar}。",
    )


def _month_step(reference, previous_jie, next_jie, year_pillar, with_facts, brief=False):
    first = rules.first_month_stem(year_pillar[0])
    if brief:  # version 3: the 交节 step just before has compared the date with the windows
        name = previous_jie
        where = f"上一个节是{name}（见交节）"
    elif with_facts:
        name = previous_jie[0]
        where = (
            f"上一个节是{name}（{previous_jie[1]}），下一个节是{next_jie[0]}"
            f"（{next_jie[1]}），参考时刻在{name}之后"
        )
    else:
        name = previous_jie
        following = JIE_IN_YEAR[(JIE_IN_YEAR.index(name) + 1) % 12]
        where = (
            f"{reference.month} 月 {reference.day} 日在{name}（约 {window_text(name)}）之后、"
            f"{following}（约 {window_text(following)}）之前"
        )
    branch, number = rules.JIE_BRANCH[name], rules.JIE_NUMBER[name]
    pillar = rules.month_pillar_for(year_pillar[0], branch)
    return pillar, (
        f"月柱：{where}，月支为{branch}（第 {number} 月）。五虎遁：{year_pillar[0]}年寅月起{first}，"
        f"顺数到第 {number} 月为{pillar[0]}，月柱{pillar}。"
    )


def _day_step(reference, sect, days_since_2000):
    civil = reference.date()
    if days_since_2000 is not None:
        total = 54 + days_since_2000
        text, index = mod60_text(total)
        lead = (
            f"日柱：{civil.isoformat()} 距 2000-01-01（戊午，序数 54）为 {days_since_2000} 天；"
            f"54 + ({days_since_2000}) = {total}，{text}"
        )
    else:
        f = day_number_formula(civil)
        text, index = mod60_text(f["total"])
        if textstyle.version() >= 2:
            # One operation per step: the first training round showed that a model
            # cannot be relied on to add four numbers at once.
            product = 5 * f["a"]
            first, second = 9 + product, 9 + product + f["leaps"]
            total = (
                f"5×{f['a']} = {product}，9 + {product} = {first}，{first} + {f['leaps']} = "
                f"{second}，{second} + {f['day_of_year']} = {f['total']}"
            )
        else:
            total = f"9 + 5×{f['a']} + {f['leaps']} + {f['day_of_year']} = {f['total']}"
        lead = (
            f"日柱：a = {civil.year} − 1900 = {f['a']}；此前闰日 ⌊({f['a']} − 1)/4⌋ = {f['leaps']}；"
            f"年内日序 = {f['before_month']} + {civil.day}"
            + (" + 1（闰年且已过 2 月）" if f["leap_adjust"] else "")
            + f" = {f['day_of_year']}；{total}，{text}"
        )
    civil_index = index
    lead += f"；{index_text(index)}，当日为{CYCLE[index]}"
    if reference.hour == 23 and sect == 1:
        index = (index + 1) % 60
        lead += f"。sect=1 在 23:00 换日，日柱取次日{CYCLE[index]}"
    elif reference.hour == 23 and sect == 2:
        lead += "。sect=2 在 00:00 才换日，23 点仍用当日"
    return CYCLE[index], civil_index, lead + f"，日柱{CYCLE[index]}。"


def _hour_step(reference, civil_index):
    branch = rules.hour_branch_of(reference.hour)
    base = CYCLE[(civil_index + 1) % 60 if reference.hour == 23 else civil_index][0]
    first = rules.first_hour_stem(base)
    pillar = rules.hour_pillar_for(base, branch)
    late = f"23 点后时干按次日日干{base}起；" if reference.hour == 23 else ""
    return pillar, (
        f"时柱：{reference.strftime('%H:%M')} 属{branch}时（{HOUR_RANGES[branch]}）。{late}"
        f"五鼠遁：{base}日子时起{first}，顺数到{branch}（第 {BRANCHES.index(branch) + 1} 位）"
        f"为{pillar[0]}，时柱{pillar}。"
    )


def derive_pillars(reference: datetime, sect, entry=None, brief=False):
    """(pillars, [four step texts]) for a UTC+08:00 instant; ``entry`` is a tool-facts candidate.

    Without ``entry`` the date must lie outside every Jie window. ``sect`` may be
    None only when the hour is not 23.
    """
    if entry is not None:
        previous = (
            entry["previous_jie"]["name"],
            entry["previous_jie"]["utc8"][:19].replace("T", " "),
        )
        following = (entry["next_jie"]["name"], entry["next_jie"]["utc8"][:19].replace("T", " "))
        _, year_pillar, year_text = _year_step(reference, previous[0], True)
        month_pillar, month_text = _month_step(reference, previous, following, year_pillar, True)
        day_pillar, civil_index, day_text = _day_step(
            reference, sect, entry["days_since_2000_01_01"]
        )
    else:
        if jie_window_of(reference.date()):
            raise ValueError("A date inside a Jie window needs tool facts")
        previous = previous_jie_by_date(reference.date())
        _, year_pillar, year_text = _year_step(reference, previous, False)
        month_pillar, month_text = _month_step(reference, previous, None, year_pillar, False, brief)
        day_pillar, civil_index, day_text = _day_step(reference, sect, None)
    hour_pillar, hour_text = _hour_step(reference, civil_index)
    pillars = dict(zip(PILLARS, (year_pillar, month_pillar, day_pillar, hour_pillar), strict=True))
    return pillars, [year_text, month_text, day_text, hour_text]


def no_tool_expected(inputs: dict, convention: dict) -> dict:
    """Expected ``{status, pillars, reason_codes}`` when no tool output is available.

    The engine's answer is kept whenever it can be established from the request
    and common knowledge. It becomes ``needs_tool`` when it depends on a UTC
    offset outside ``known_offset`` or on the exact instant of a solar term.
    """
    result = compute_chart(inputs, convention)
    view = {
        "status": result["status"],
        "pillars": result["pillars"],
        "reason_codes": list(result["reason_codes"]),
    }
    raw, zone = inputs.get("datetime"), inputs.get("timezone")
    syntactic = {
        "missing_datetime_or_timezone",
        "full_wall_time_required",
        "invalid_datetime_format",
        "invalid_date_or_timezone",
        "invalid_convention",
        "invalid_sect",
        "invalid_fold",
        "unsupported_convention",
    }
    if any(reason in syntactic for reason in result["reason_codes"]):
        return view
    try:
        parsed = parse_wall_time(raw) if isinstance(raw, str) else None
    except ValueError:
        parsed = None
    if parsed is None or not isinstance(zone, str):
        return view
    _precision, first, _last = parsed
    if result["status"] == "unsupported":
        return view  # outside the study range: decided from the written year
    offset = known_offset(zone, first)
    if offset is None:
        return {"status": NEEDS_TOOL, "pillars": None, "reason_codes": ["utc_offset_required"]}
    reference = first + timedelta(minutes=480 - offset)
    if result["status"] == "ambiguous" and result["reason_codes"] == ["missing_convention"]:
        return view
    if result["status"] in ("ok", "ambiguous") and jie_window_of(reference.date()):
        return {"status": NEEDS_TOOL, "pillars": None, "reason_codes": ["jie_instant_required"]}
    return view


def _facts_reference(entry):
    return datetime.fromisoformat(entry["utc8_start"]).replace(tzinfo=None)


def _nonanswer_text(inputs, convention, result, facts, offset):
    """Explanation for a request that is not answered with pillars."""
    reasons = result["reason_codes"]
    raw, zone = inputs.get("datetime"), inputs.get("timezone")
    sect_note = "同时没有指定日柱流派（sect）。" if "missing_convention" in reasons else ""
    if "missing_datetime_or_timezone" in reasons:
        missing = "出生时间" if raw in (None, "") else "时区"
        return f"缺少{missing}，无法把出生时刻归一到 UTC+08:00 参考时钟，四柱不能确定。{sect_note}请补充后再算。"
    if "full_wall_time_required" in reasons:
        return (
            "只给了日期，没有钟点：一天有十二个时辰，时柱无法确定；若当天有节气交接或在 23 点前后，"
            f"其他柱也可能变化。{sect_note}请补充出生的具体时刻。"
        )
    if reasons == ["unsupported_convention"]:
        defaults = {
            "solar_time": ("none", "真太阳时或经度修正"),
            "clock_basis": ("fixed_utc8", "按出生地当地钟面而不归一到 UTC+08:00"),
            "year_boundary": ("lichun_instant", "不以立春时刻换年"),
            "month_boundary": ("jie_instant", "不以节的时刻换月"),
        }
        asked = [text for key, (value, text) in defaults.items() if convention.get(key) != value]
        return (
            f"请求的口径（{'、'.join(asked)}）不在本项目已验证的范围内：这里只验证了固定 UTC+08:00 "
            "参考时钟、立春换年、节换月的算法。为避免给出未经验证的结果，不提供四柱；"
            "可以改用已验证口径重新计算。"
        )
    if reasons == ["outside_study_range"]:
        return "归一到 UTC+08:00 后的参考年份不在 1901–2099 的支持范围内，本项目的历法核验不覆盖，不给出四柱。"
    if reasons == ["invalid_sect"]:
        return "日柱流派 sect 只能是 1（23:00 换日）或 2（00:00 换日），给出的值无效。"
    if reasons == ["invalid_fold"]:
        return "fold 只能是 0（重复时刻的第一次）或 1（第二次），给出的值无效。"
    if reasons == ["fold_not_applicable"]:
        return "该钟面时间在当地只出现一次，不是夏令时结束时的重复时刻，不应带 fold；请去掉 fold 后重算。"
    if reasons == ["invalid_datetime_format"]:
        return (
            "出生时间的写法无法按约定解析。请使用 YYYY-MM-DDTHH:MM:SS（也可只到分钟或只有日期），"
            "不要带 UTC 偏移、小数秒或其他写法。"
        )
    if reasons == ["invalid_date_or_timezone"]:
        try:
            parse_wall_time(raw)
        except ValueError:
            return f"“{raw}”不是真实存在的日期或时刻（例如平年没有 2 月 29 日、没有 24:00:00），请核对后重新给出。"
        return (
            f"“{zone}”不是精确的 IANA 时区键（大小写也必须一致，例如 Asia/Shanghai、"
            "America/New_York），无法据此换算，请给出准确的时区键。"
        )
    if reasons == ["nonexistent_civil_time"]:
        return (
            f"工具未能归一：{zone} 在这一天因夏令时开始而把钟拨快，{raw[11:16]} 这个钟面时间被跳过，"
            "当地并不存在这一时刻。请核对出生时间。"
        )
    entries = facts["candidates"] if facts else []
    if "dst_fold_required" in reasons:
        first, second = (_facts_reference(e) for e in entries[:2])
        return (
            f"该钟面时间因夏令时结束而出现两次：对应 UTC+08:00 的 {clock(first)} 与 {clock(second)}，"
            f"分别属{rules.hour_branch_of(first.hour)}时与{rules.hour_branch_of(second.hour)}时，"
            "四柱不同。请说明是第一次（fold=0）还是第二次（fold=1）。"
        )
    if "seconds_required" in reasons:
        entry = entries[0]
        name, moment = entry["next_jie"]["name"], entry["next_jie"]["utc8"][:19].replace("T", " ")
        return (
            f"出生时间只精确到分钟，而{name}的交节时刻 {moment} 正落在这一分钟内：节前与节后的月柱"
            "不同，需要精确到秒才能确定。"
        )
    if "missing_convention" in reasons:
        if entries:
            reference = _facts_reference(entries[0])
        else:
            reference = parse_wall_time(raw)[1] + timedelta(minutes=480 - offset)
        return (
            f"参考时刻 {clock(reference)}（UTC+08:00）处在 23:00–23:59:59：sect=1 此时已换日，"
            "sect=2 要到 00:00 才换日，两派的日柱不同。没有指定流派，不能替你默认，请先说明用哪一派。"
        )
    raise AssertionError(f"No explanation for {result['status']} {reasons}")


def explain_request(inputs: dict, convention: dict, mode: str) -> dict:
    """Worked answer for one chart request.

    Returns ``{"steps": [...], "response": {status, pillars, reason_codes}}``.
    ``mode`` is "facts" (tool facts were supplied) or "none".
    """
    if mode not in ("facts", "none"):
        raise ValueError("mode must be 'facts' or 'none'")
    result = compute_chart(inputs, convention)
    facts = temporal_facts(inputs) if mode == "facts" else None
    expected = (
        {
            "status": result["status"],
            "pillars": result["pillars"],
            "reason_codes": list(result["reason_codes"]),
        }
        if mode == "facts"
        else no_tool_expected(inputs, convention)
    )
    raw, zone = inputs.get("datetime"), inputs.get("timezone")
    sect = (convention or {}).get("sect")
    if _CHECK_FIRST[0] and mode == "none":  # derivation version 3 (preregistration 12.14)
        checked = _checked_answer(inputs, convention, result, expected, sect)
        if checked is not None:
            return checked
    offset = None
    if mode == "none" and isinstance(raw, str) and isinstance(zone, str):
        try:
            parsed = parse_wall_time(raw)
        except ValueError:
            parsed = None
        if parsed:
            offset = known_offset(zone, parsed[1])
    if expected["status"] == NEEDS_TOOL:
        if expected["reason_codes"] == ["utc_offset_required"]:
            text = (
                f"没有时区工具时，我不能凭记忆确定 {zone} 在这个日期的 UTC 偏移（夏令时交替期或历史上"
                "偏移有过调整），也就无法可靠地归一到 UTC+08:00。请提供历法工具的时间事实，或直接给出"
                "对应的 UTC+08:00 时刻。"
            )
        else:
            reference = parse_wall_time(raw)[1] + timedelta(minutes=480 - offset)
            name = jie_window_of(reference.date())
            affected = "年柱和月柱" if name == "立春" else "月柱"
            text = (
                f"参考日期 {reference.month} 月 {reference.day} 日落在{name}可能的交节日期"
                f"（约 {window_text(name)}）内。没有该年{name}的精确时刻，就无法判断出生在节前还是节后，"
                f"{affected}不能确定。请提供历法工具给出的交节时刻。"
            )
        return {"steps": [text], "response": expected}
    if expected["status"] != "ok":
        return {
            "steps": [_nonanswer_text(inputs, convention, result, facts, offset)],
            "response": expected,
        }
    steps = []
    if mode == "facts":
        entries = facts["candidates"]
        reference = _facts_reference(entries[0])
        if len(entries) == 2:
            other = _facts_reference(entries[1])
            steps.append(
                f"参考时刻：钟面时间因夏令时结束而重复，对应 UTC+08:00 的 {clock(reference)} 与 "
                f"{clock(other)}；两者同在一日、同属{rules.hour_branch_of(reference.hour)}时，"
                "四柱相同，以下按前者计算。"
            )
        else:
            end = datetime.fromisoformat(entries[0]["utc8_end"]).replace(tzinfo=None)
            note = (
                ""
                if end == reference
                else "（只精确到分钟，这一分钟内没有节气或换日分界，不影响结果）"
            )
            steps.append(f"参考时刻：工具给出 UTC+08:00 的 {clock(reference)}{note}。")
        pillars, derived = derive_pillars(reference, sect, entries[0])
    else:
        wall = parse_wall_time(raw)[1]
        reference = wall + timedelta(minutes=480 - offset)
        if offset == 480:
            steps.append(f"参考时刻：{zone} 在该日期为 UTC+8，参考时刻即 {clock(reference)}。")
        else:
            shift = 480 - offset
            hours, minutes = divmod(abs(shift), 60)
            amount = f"{hours} 小时" + (f" {minutes} 分" if minutes else "")
            steps.append(
                f"参考时刻：{zone} 在该日期为 {offset_text(offset)}，{'加' if shift > 0 else '减'} "
                f"{amount} 得 UTC+08:00 的 {clock(reference)}。"
            )
        pillars, derived = derive_pillars(reference, sect, None)
    if sect is None:
        derived[2] = derived[2][:-1] + "（未指定流派，但时刻不在 23 点，两派结果相同）。"
    steps.extend(derived)
    if pillars != expected["pillars"]:
        raise AssertionError(f"Worked answer {pillars} differs from engine {expected['pillars']}")
    return {"steps": steps, "response": expected}


def numbered(steps: list) -> str:
    if len(steps) == 1:
        return steps[0]
    return "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))


def reference_instant(inputs: dict):
    """The first UTC+08:00 instant of a request (naive), or None if not normalizable."""
    entries = temporal_facts(inputs)["candidates"]
    return _facts_reference(entries[0]) if entries else None


def shanghai_wall(reference: datetime) -> datetime:
    """Asia/Shanghai wall time (naive) showing a UTC+08:00 instant (naive)."""
    return (
        reference.replace(tzinfo=UTC8)
        .astimezone(load_timezone("Asia/Shanghai"))
        .replace(tzinfo=None)
    )
