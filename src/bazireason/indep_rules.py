"""Independent rule questions: labels from lunar-python, wordings never used in training.

Phase 4 adds this set to test whether rule reasoning transfers beyond the project's own
question templates and label code. The labels come from lunar-python's ``EightChar``
(a third-party implementation), not from ``bazireason.rules``; the conventions are the
same (Phase 2 tests check the tables agree entry by entry), so the set measures transfer
to new wordings and a new label source, not a different school.
"""

import json
import random
from datetime import date, datetime, timedelta

from lunar_python import Solar

from bazireason import train_data as TD
from bazireason.p2_prompts import P2_SYSTEM, extract_final_json

PILLAR_NAMES = {"year": "年", "month": "月", "day": "日", "time": "时"}
ELEMENTS = "木火土金水"
STEM_ELEMENT = dict(zip("甲乙丙丁戊己庚辛壬癸", "木木火火土土金金水水", strict=True))
BRANCH_ELEMENT = dict(zip("子丑寅卯辰巳午未申酉戌亥", "水土木木土火火土金金土水", strict=True))
KINDS = (
    "ten_god_stem",
    "ten_god_branch",
    "hidden",
    "life_stage",
    "xun_kong",
    "luck_direction",
    "element_count",
)
# Two wordings per kind, written for Phase 4 and never used in any training set.
TEMPLATES = {
    "ten_god_stem": (
        "命盘四柱依次是 {chart}。站在日主{dm}的角度，{p}干{s}算哪一种十神？",
        "我在练习排十神：{chart}，{p}柱天干{s}相对日干{dm}是什么？",
    ),
    "ten_god_branch": (
        "{chart}。{p}支{b}的本气藏干，对日主{dm}来说是哪种十神？",
        "四柱：{chart}。只看{p}支{b}的本气，它是日元{dm}的什么十神？",
    ),
    "hidden": (
        "{chart}。{p}支{b}里藏着哪几个天干？",
        "帮我列一下这张盘（{chart}）{p}支{b}的全部藏干。",
    ),
    "life_stage": (
        "{chart}。日干{dm}在{p}支{b}处于十二长生的哪个阶段？",
        "按十二长生（阳顺阴逆）：{chart}，日主{dm}临{p}支{b}是什么状态？",
    ),
    "xun_kong": (
        "{chart}。按日柱{day}所在的旬，哪两个地支是空亡？",
        "日柱为{day}（全盘 {chart}），旬空落在哪两支？",
    ),
    "luck_direction": (
        "{chart}，{gender}命。按年干阴阳和性别，大运是顺排还是逆排？",
        "一位{gender}性，四柱 {chart}。其大运该往前数还是往后数？",
    ),
    "element_count": (
        "{chart}。不算藏干，只数八个字，金木水火土各有几个？",
        "统计一下 {chart} 这八个字的五行数量（不看地支藏干）。",
    ),
}
HINTS = {
    "ten_god_stem": "结论JSON 需含 answer（十神名称，如 正官）。",
    "ten_god_branch": "结论JSON 需含 answer（十神名称，如 正官）。",
    "hidden": '结论JSON 需含 answer（藏干列表，如 ["甲","丙"]）。',
    "life_stage": "结论JSON 需含 answer（十二长生阶段名，如 帝旺）。",
    "xun_kong": "结论JSON 需含 answer（两个地支的列表）。",
    "luck_direction": '结论JSON 需含 answer（"顺" 或 "逆"）。',
    "element_count": "结论JSON 需含 answer（对象，键为 木火土金水，值为个数）。",
}


def eight_char(moment: datetime, sect: int = 2):
    chart = (
        Solar.fromYmdHms(moment.year, moment.month, moment.day, moment.hour, moment.minute, 0)
        .getLunar()
        .getEightChar()
    )
    chart.setSect(sect)
    return chart


def chart_line(chart) -> str:
    return " ".join(
        getattr(chart, f"get{name.capitalize()}")() for name in ("year", "month", "day", "time")
    )


def label(chart, kind: str, pillar: str, gender: int):
    cap = pillar.capitalize()
    if kind == "ten_god_stem":
        return getattr(chart, f"get{cap}ShiShenGan")()
    if kind == "ten_god_branch":
        return getattr(chart, f"get{cap}ShiShenZhi")()[0]
    if kind == "hidden":
        return list(getattr(chart, f"get{cap}HideGan")())
    if kind == "life_stage":
        return getattr(chart, f"get{cap}DiShi")()
    if kind == "xun_kong":
        return list(chart.getDayXunKong())
    if kind == "luck_direction":
        return "顺" if chart.getYun(gender).isForward() else "逆"
    if kind == "element_count":
        text = "".join(getattr(chart, f"get{name}")() for name in ("Year", "Month", "Day", "Time"))
        counts = dict.fromkeys(ELEMENTS, 0)
        for index, char in enumerate(text):
            counts[(STEM_ELEMENT if index % 2 == 0 else BRANCH_ELEMENT)[char]] += 1
        return counts
    raise ValueError(kind)


def build(split: str, n: int, seed: int, blocked=None, exclude_prompts=frozenset()):
    """``n`` rows from the split's years, cycling over the kinds and both wordings."""
    if blocked is None:
        blocked = TD.benchmark_exclusions() | TD.contest_exclusions()
    rng = random.Random(f"indep:{seed}:{split}")
    years = sorted(TD.split_years()[split])
    rows = []
    index = 0
    while len(rows) < n:
        kind = KINDS[index % len(KINDS)]
        index += 1
        while True:
            day = date(rng.choice(years), 1, 1) + timedelta(days=rng.randrange(365))
            if day not in blocked and day.year in years:
                break
        moment = datetime(day.year, day.month, day.day, rng.randrange(24), rng.randrange(60))
        chart = eight_char(moment)
        pillar = rng.choice(("year", "month", "time") if "stem" in kind else tuple(PILLAR_NAMES))
        gender = rng.choice((1, 0))
        dm = chart.getDay()[0]
        branch = getattr(chart, f"get{pillar.capitalize()}")()[1]
        text = rng.choice(TEMPLATES[kind]).format(
            chart=chart_line(chart),
            dm=dm,
            p=PILLAR_NAMES[pillar],
            s=getattr(chart, f"get{pillar.capitalize()}")()[0],
            b=branch,
            day=chart.getDay(),
            gender="男" if gender == 1 else "女",
        )
        question = text + "\n" + HINTS[kind]
        if question in exclude_prompts:
            continue
        rows.append(
            {
                "id": f"IND-{split}-{len(rows) + 1:05d}",
                "family": "indep",
                "task": f"indep_{kind}",
                "split": split,
                "mode": "none",
                "group_id": f"indep-{len(rows)}",
                "messages": [
                    {"role": "system", "content": P2_SYSTEM},
                    {"role": "user", "content": question},
                ],
                "meta": {
                    "expected": {"answer": label(chart, kind, pillar, gender)},
                    "kind": kind,
                    "label_source": "lunar_python.EightChar",
                    "pillar": pillar,
                },
            }
        )
    return rows


def _norm(kind, value):
    if kind in ("hidden", "xun_kong"):
        return sorted(value) if isinstance(value, list) else None
    if kind == "luck_direction" and isinstance(value, str):
        return value.strip().removesuffix("行").removesuffix("排")[:1]
    if kind == "element_count" and isinstance(value, dict):
        try:
            return {k: int(value.get(k, -1)) for k in ELEMENTS}
        except (TypeError, ValueError):
            return None
    return value.strip() if isinstance(value, str) else value


def score_indep(row, text):
    kind = row["meta"]["kind"]
    answer = extract_final_json(text or "")
    got = answer.get("answer") if isinstance(answer, dict) else None
    want = row["meta"]["expected"]["answer"]
    return {
        "id": row["id"],
        "family": "indep",
        "task": row["task"],
        "kind": kind,
        "parsed": isinstance(answer, dict),
        "correct": got is not None and _norm(kind, got) == _norm(kind, want),
        "group": row["group_id"],
        "unseen_wording": True,
    }


def dumps(rows) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
