"""Approach B of Phase 4: deterministic checks wrapped around the model.

W1 verify and repair  The rule code / engine recomputes what it can decide. If the reply's
                      final JSON differs, the model gets one more turn that names the fields
                      that differ (never their values).
W2 tool check         A supplied calendar answer is checked by rules that do not use the
                      tool: 五鼠遁, 五虎遁, clock to hour branch, an independent day-number
                      formula, the year away from 立春, the month branch away from a 节,
                      and the request's own status. Failed checks are named in the prompt.
W3 contest guard      Rule-code facts in the prompt, a loop guard and a citation check.
"""

import json
import re
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from bazireason import rules as R
from bazireason.baselines import parse_strict
from bazireason.calendar_engine import compute_chart
from bazireason.p2_prompts import extract_final_json
from bazireason.p3_eval import _equal

EXPLANATORY = {"reason_codes", "conventions", "reason", "unverified", "summary", "cause"}
FIELD_ZH = {"year": "年柱", "month": "月柱", "day": "日柱", "hour": "时柱", "status": "status"}
STRICT_TASKS = ("p3_strict",)
LOOP = re.compile(r"(.{20,}?)\1\1", re.S)
TITLE = re.compile(r"《([^》]{1,12})》")
# Approximate day-of-month ranges of the twelve 节 (UTC+08:00, 1901-2099), with a margin.
JIE_WINDOW = {1: (4, 8), 2: (2, 6), 3: (4, 8), 4: (3, 7), 5: (4, 8), 6: (4, 8),
              7: (5, 9), 8: (6, 10), 9: (6, 10), 10: (7, 10), 11: (6, 9), 12: (5, 9)}  # fmt: skip
UTC8 = timezone(timedelta(hours=8))  # the fixed reference clock, not Asia/Shanghai (DST 1986-91)
MONTH_BRANCH_AFTER_JIE = dict(zip(range(1, 13), "丑寅卯辰巳午未申酉戌亥子", strict=True))


# ---------------------------------------------------------------- W1: verify and repair
def is_strict(row) -> bool:
    return row["task"].startswith(STRICT_TASKS) or row["family"] == "strict_shift"


def reply_answer(row, text):
    if is_strict(row):
        try:
            return parse_strict(text)
        except (ValueError, TypeError):
            return None
    return extract_final_json(text or "")


def differing_fields(expected, answer):
    """Keys of the engine's object that the reply gets wrong; None if nothing parses."""
    if not isinstance(answer, dict):
        return None
    wrong = []
    for key, value in expected.items():
        if key in EXPLANATORY:
            continue
        if key == "pillars" and isinstance(value, dict):
            got = answer.get("pillars") if isinstance(answer.get("pillars"), dict) else {}
            wrong += [FIELD_ZH[p] for p in value if got.get(p) != value[p]]
        elif key not in answer or not _equal(key, value, answer[key]):
            wrong.append(key)
    return wrong


def indep_engine(row):
    """The project's own rule code on an independent row (not the lunar-python label)."""
    text = next(m["content"] for m in row["messages"] if m["role"] == "user")
    pair = "[甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥]"
    four = re.search(rf"({pair}) ({pair}) ({pair}) ({pair})", text).groups()
    pillars = dict(zip(("year", "month", "day", "hour"), four, strict=True))
    kind, name = row["meta"]["kind"], row["meta"]["pillar"].replace("time", "hour")
    day = pillars["day"][0]
    if kind == "ten_god_stem":
        return R.ten_god(day, pillars[name][0])
    if kind == "ten_god_branch":
        return R.ten_god(day, R.HIDDEN_STEMS[pillars[name][1]][0])
    if kind == "hidden":
        return list(R.HIDDEN_STEMS[pillars[name][1]])
    if kind == "life_stage":
        return R.life_stage(day, pillars[name][1])
    if kind == "xun_kong":
        return list(R.xun_kong(pillars["day"]))
    if kind == "luck_direction":
        gender = "male" if "男" in text else "female"
        return "顺" if R.luck_direction(pillars["year"][0], gender) == "forward" else "逆"
    if kind == "element_count":
        return R.element_counts(pillars)
    raise ValueError(kind)


def check_reply(row, text, score):
    """(agrees, fields) by the deterministic code. ``score`` is used for generated rows,
    whose stored expectation is the same code's output; independent rows are recomputed."""
    answer = reply_answer(row, text)
    if row["family"] == "indep":
        from bazireason.indep_rules import _norm

        engine = indep_engine(row)
        got = answer.get("answer") if isinstance(answer, dict) else None
        kind = row["meta"]["kind"]
        agrees = got is not None and _norm(kind, got) == _norm(kind, engine)
        return agrees, (None if not isinstance(answer, dict) else ([] if agrees else ["answer"]))
    if score.get("correct"):
        return True, []
    return False, differing_fields(row["meta"]["expected"], answer)


def prompt_messages(row) -> list:
    """The conversation sent to the model: the row's messages without a reference answer.

    Training-format rows end with the reference assistant reply; the independent rows
    carry their labels in ``meta`` and end with the question itself.
    """
    messages = row["messages"]
    return messages[:-1] if messages and messages[-1]["role"] == "assistant" else list(messages)


def repair_turn(row, text, fields) -> list:
    """The conversation for the second attempt: the first reply plus a value-free note."""
    if fields is None:
        note = "核验：上面的回答没有可解析的结论。请按题目要求重新逐步推导，最后给出完整结论。"
    elif fields:
        note = (
            f"核验：规则引擎独立核对了你的结论，其中 {'、'.join(fields)} 与引擎结果不一致"
            "（这里不提供正确值）。请逐项重新推导这些部分，再给出完整结论。"
        )
    else:
        note = "核验：你的结论与规则引擎的结果不一致（这里不提供正确值）。请重新逐步推导，再给出完整结论。"
    prompt = prompt_messages(row)
    if is_strict(row):
        note += "只输出一个 JSON 对象。"
    elif "结论JSON" in prompt[-1]["content"]:
        note += "最后一行仍以“结论JSON:”给出。"
    return prompt + [
        {"role": "assistant", "content": text or ""},
        {"role": "user", "content": note},
    ]


# ---------------------------------------------------------------- W2: independent tool check
def day_index(day: date) -> int:
    """Sexagenary day number (甲子 = 0) from the Julian day number; no calendar library."""
    jdn = day.toordinal() + 1721425
    return (jdn + 49) % 60


def tool_check(inputs: dict, convention: dict, supplied: dict) -> list:
    """Checks a supplied answer can fail without using the tool; returns the failed items."""
    status = compute_chart(inputs, convention)["status"]
    if supplied.get("status") != "ok":
        return [] if status != "ok" else ["status"]
    if status != "ok":
        return ["status"]
    pillars = supplied.get("pillars") or {}
    if not all(R.valid_pillar(pillars.get(p, "")) for p in ("year", "month", "day", "hour")):
        return ["format"]
    raw = inputs.get("datetime") or ""
    try:
        local = datetime.fromisoformat(raw)
    except ValueError:
        return []
    zone = inputs.get("timezone") or "Asia/Shanghai"
    try:
        moment = local.replace(tzinfo=ZoneInfo(zone)).astimezone(UTC8)
    except (KeyError, ValueError):
        return []
    moment = moment.replace(tzinfo=None)
    failed = []
    branch = R.hour_branch_of(moment.hour)
    if pillars["hour"][1] != branch:
        failed.append("时支")
    sect = convention.get("sect", 2)
    civil = moment.date() + timedelta(days=1 if (sect == 1 and moment.hour == 23) else 0)
    day = R.CYCLE[day_index(civil)]
    if pillars["day"] != day:
        failed.append("日柱")
    stem_day = R.CYCLE[day_index(moment.date() + timedelta(days=moment.hour == 23))][0]
    if pillars["hour"] != R.hour_pillar_for(stem_day, branch):
        failed.append("时干（五鼠遁）")
    if pillars["month"] != R.month_pillar_for(pillars["year"][0], pillars["month"][1]):
        failed.append("月干（五虎遁）")
    low, high = JIE_WINDOW[moment.month]
    if not low <= moment.day <= high:
        after = moment.day > high
        branch_m = MONTH_BRANCH_AFTER_JIE[moment.month if after else (moment.month - 2) % 12 + 1]
        if pillars["month"][1] != branch_m:
            failed.append("月支")
    if not (moment.month == 2 and 2 <= moment.day <= 6):
        year = moment.year if (moment.month, moment.day) > (2, 6) else moment.year - 1
        if pillars["year"] != R.year_pillar(year):
            failed.append("年柱")
    return failed


def tool_note(failed: list) -> str:
    if failed == ["status"]:
        return "【独立核对】请求本身不足以唯一确定四柱，工具却给出了确定的四柱。请自行判断后作答。"
    return (
        f"【独立核对】工具答案未通过不依赖该工具的检查：{'、'.join(failed)}。"
        "这里不提供正确值，请自行核对后作答。"
    )


# ---------------------------------------------------------------- W3: contest guard
def has_loop(text: str) -> bool:
    return bool(LOOP.search(text or ""))


def cut_at_loop(text: str) -> str:
    match = LOOP.search(text or "")
    return text[: match.start() + len(match.group(1))] if match else text


def unknown_titles(text: str, known) -> list:
    return [t for t in TITLE.findall(text or "") if t not in known]


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False)
