"""Phase 2 fine-tuning dataset (``p2_train_v1``): generated, labelled and split by code.

Every label is computed, never written by hand: calendar answers come from the
unchanged engine and are re-derived by the formulas shown in the answer text
(``bazireason.worked``); rule answers come from ``bazireason.rules``. No contest
question, answer key or contest explanation is read by this module.

Leakage guards
--------------
* Year blocks: training, validation and test use disjoint calendar years
  (``split_years``), so every held-out date, solar-term event and daylight-saving
  transition is unseen.
* Benchmark isolation: no example may fall within seven days of any case in the
  three evaluation suites under ``data/synthetic`` (``benchmark_exclusions``).
* Contest isolation: no example may fall within one day of a birth date used by
  the BaziQA contest data (``contest_exclusions`` reads dates only).
* Within a split a civil date is used by one family only.
* Some question wordings and some time zones appear only in the test split.
"""

import hashlib
import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

from bazireason import rules
from bazireason import train_tasks as T
from bazireason import worked as W
from bazireason.calendar_engine import (
    UTC8,
    _completions,
    _jie_of_table,
    chart_at,
    compute_chart,
    load_timezone,
    temporal_facts,
)
from bazireason.charttools import chart_tool_text, pillars_text, verified_chart
from bazireason.controls import corrupt_context
from bazireason.conventions import CYCLE, PILLARS, STEMS, Convention
from bazireason.p2_prompts import (
    FACTS_LABEL,
    MCQ_SYSTEM,
    OUTPUT_HINTS,
    P2_SYSTEM,
    REASONED_NOTE,
    SELF_CHART_LABEL,
    chat_messages,
    compact,
    facts_block,
    final_line,
    mcq_user,
)
from bazireason.prompts import CALENDAR_SPEC, EXPLICIT_POLICY, RESPONSE_SPEC, build_messages
from bazireason.schema import model_view

DATASET_VERSION = "p2_train_v1"
SEED = 2026111001
SPLITS = ("train", "valid", "test")
SHANGHAI = "Asia/Shanghai"
PROFILE = Convention().to_dict()
FACT_ZONES = (
    "Asia/Shanghai",
    "Asia/Hong_Kong",
    "Asia/Taipei",
    "Asia/Tokyo",
    "Asia/Seoul",
    "Asia/Singapore",
    "Asia/Kuala_Lumpur",
    "Asia/Kolkata",
    "Asia/Bangkok",
    "Europe/London",
    "Europe/Paris",
    "Europe/Berlin",
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "America/Toronto",
    "America/Sao_Paulo",
    "Australia/Sydney",
    "Australia/Adelaide",
    "Pacific/Auckland",
    "UTC",
)
TEST_ONLY_ZONES = ("Europe/Madrid", "America/Mexico_City", "Africa/Johannesburg", "Asia/Dubai")
KNOWN_ZONES = (SHANGHAI, SHANGHAI, SHANGHAI, *W.FIXED_ZONES, *W.SEASONAL_ZONES)
DST_ZONES = (
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "Europe/London",
    "Europe/Paris",
    "Australia/Sydney",
)
PROFILE_CLAUSES = {
    ("solar_time", "true_solar"): "请用真太阳时排盘",
    ("clock_basis", "local_civil"): "请直接按出生地当地钟面排盘，不要换算到 UTC+8",
    ("year_boundary", "lunar_new_year"): "年柱请按农历正月初一换年",
    ("month_boundary", "lunar_month"): "月柱请按农历月份划分",
}
SECT_PHRASES = {
    1: ("日柱流派：sect=1", "按 23:00 换日（sect=1）"),
    2: ("日柱流派：sect=2", "按 00:00 换日（sect=2）"),
}
# Wordings 4 and 5 never appear in training or validation.
CHART_TEMPLATES = (
    "请计算四柱。{birth}。",
    "帮我排一下八字：{birth}。",
    "{birth}。四柱分别是什么？",
    "已知{birth}，请给出年柱、月柱、日柱、时柱。",
    "我想核对一个命盘，{birth}。请按规则推出四柱。",
    "求四柱八字。{birth}。",
)
VERIFY_TEMPLATES = (
    "{birth}。别人给我排的八字是：{chart}。请核验是否正确。",
    "有人按 {birth} 排出的八字是 {chart}，对吗？请逐柱核对。",
    "{birth}。某排盘软件给出：{chart}。请按规则检查有没有算错。",
    "我拿到一份排盘：{chart}。出生信息是 {birth}。这份排盘可靠吗？",
)
INTERNAL_TEMPLATES = (
    "有人给出一个八字：{chart}。没有出生时间，请只根据干支自身的规则检查它是否自洽。",
    "只看这四柱本身：{chart}。干支之间有没有互相矛盾的地方？",
    "请检查这个八字的内部一致性（干支配对、五虎遁、五鼠遁）：{chart}。",
    "这组四柱 {chart} 在规则上说得通吗？不用管出生时间。",
)
# Family sizes for (train, valid, test). Units are rows unless a comment says groups.
SIZES = {
    "chart_plain": (250, 14, 36),
    "chart_facts": (200, 12, 36),
    "jie_boundary": (85, 5, 14),  # groups of about three rows
    "zi_transition": (56, 4, 10),  # groups of five rows
    "tz_known": (120, 8, 20),
    "tz_needs_tool": (70, 4, 12),
    "tz_facts": (90, 6, 16),
    "dst_fold": (30, 2, 6),  # groups of three rows
    "dst_gap": (24, 2, 6),
    "shanghai_dst": (32, 2, 6),  # groups of two rows
    "missing": (170, 10, 28),
    "invalid": (170, 10, 28),
    "verify": (400, 24, 72),
    "internal": (190, 12, 32),
    "rule": (620, 36, 96),
    "counterfactual": (200, 12, 36),
    "policy": (130, 8, 20),
    "mcq": (250, 14, 40),
    "multi_turn": (200, 12, 40),
    "v3_strict": (380, 20, 0),
    "v4_reasoned": (170, 10, 0),
}


def split_years() -> dict:
    """Disjoint year blocks: five test years and three validation years in every twenty."""
    test = {y for start in range(1915, 2100, 20) for y in range(start, start + 5)}
    valid = {y for start in range(1910, 2100, 20) for y in range(start, start + 3)}
    train = set(range(1901, 2100)) - test - valid
    return {"train": train, "valid": valid, "test": test}


def benchmark_exclusions(data_dir="data/synthetic", margin=7) -> set:
    """Civil dates within ``margin`` days of any evaluation-suite case."""
    blocked = set()
    for path in sorted(Path(data_dir).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)["input"].get("datetime")
            if not isinstance(raw, str) or len(raw) < 10:
                continue
            try:
                day = date.fromisoformat(raw[:10])
            except ValueError:
                # An impossible date such as 2023-02-29 still marks its neighbourhood.
                day = date(int(raw[:4]), int(raw[5:7]), 1) + timedelta(days=27)
            blocked.update(day + timedelta(days=k) for k in range(-margin, margin + 1))
    return blocked


def contest_exclusions(path="data/phase2/exclusions/contest_birth_dates.json", margin=1) -> set:
    """Civil dates within ``margin`` days of a contest birth date (dates only, no questions)."""
    file = Path(path)
    if not file.exists():
        return set()
    blocked = set()
    for raw in json.loads(file.read_text())["dates"]:
        day = date.fromisoformat(raw)
        blocked.update(day + timedelta(days=k) for k in range(-margin, margin + 1))
    return blocked


def convention(sect) -> dict:
    return (
        dict(PROFILE, sect=sect)
        if sect is not None
        else {k: v for k, v in PROFILE.items() if k != "sect"}
    )


def format_datetime(raw: str, style: int) -> str:
    """Three surface forms of the same wall time; the engine input stays ISO."""
    if style == 0 or raw is None:
        return raw
    if style == 1:
        return raw.replace("T", " ")
    day = f"{int(raw[:4])}年{int(raw[5:7])}月{int(raw[8:10])}日"
    return day + (" " + raw[11:] if len(raw) > 10 else "")


def birth_clause(inputs: dict, conv: dict, rng, style=None) -> str:
    raw, zone = inputs.get("datetime"), inputs.get("timezone")
    style = rng.randrange(3) if style is None else style
    parts = [f"出生时间：{format_datetime(raw, style)}" if raw else "出生时间：（未提供）"]
    parts.append(f"时区：{zone}" if zone else "时区：（未提供）")
    if "sect" in conv:
        sect = conv["sect"]
        parts.append(rng.choice(SECT_PHRASES[sect]) if sect in SECT_PHRASES else f"sect={sect}")
    if "fold" in inputs:
        fold = inputs["fold"]
        word = {0: "一", 1: "二"}.get(fold)
        parts.append(f"重复钟点取第{word}次（fold={fold}）" if word else f"fold={fold}")
    for (field, value), clause in PROFILE_CLAUSES.items():
        if conv.get(field) == value:
            parts.append(clause)
    return "，".join(parts)


class Pool:
    """Seeded sampler restricted to one split's years and to unreserved, unblocked dates."""

    def __init__(self, split, seed, blocked):
        self.split = split
        self.rng = random.Random(f"{DATASET_VERSION}:{seed}:{split}")
        self.years = split_years()[split]
        self.year_list = sorted(self.years)
        self.blocked = blocked
        self.used = set()
        self.zones = FACT_ZONES + (TEST_ONLY_ZONES if split == "test" else ())
        self.templates = (4, 5, 0, 1, 2, 3) if split == "test" else (0, 1, 2, 3)

    def free(self, day: date) -> bool:
        return (
            all((day + timedelta(days=k)).year in self.years for k in (-2, 0, 2))
            and day not in self.blocked
            and day not in self.used
        )

    def reserve(self, *moments):
        for moment in moments:
            self.used.add(moment.date() if isinstance(moment, datetime) else moment)

    def day(self) -> date:
        while True:
            year = self.rng.choice(self.year_list)
            day = date(year, 1, 1) + timedelta(days=self.rng.randrange(365))
            if self.free(day):
                return day

    def wall(self, hours=range(24)) -> datetime:
        day = self.day()
        return datetime(day.year, day.month, day.day) + timedelta(
            hours=self.rng.choice(list(hours)), seconds=self.rng.randrange(3600)
        )

    def template(self) -> int:
        """Test rows use the two unseen wordings half of the time."""
        if self.split == "test" and self.rng.random() < 0.5:
            return self.rng.choice((4, 5))
        return self.rng.choice((0, 1, 2, 3))

    def variant(self) -> int:
        """Rule-question wording: index 2 is reserved for the test split."""
        if self.split == "test" and self.rng.random() < 0.5:
            return 2
        return self.rng.choice((0, 1))

    def jie_events(self):
        events = []
        for year in self.year_list:
            for moment, name in _jie_of_table(year):
                naive = moment.replace(tzinfo=None)
                if naive.year == year and self.free(naive.date()):
                    events.append((name, naive))
        self.rng.shuffle(events)
        return events


def _wall_in_zone(reference: datetime, zone: str) -> datetime:
    return reference.replace(tzinfo=UTC8).astimezone(load_timezone(zone)).replace(tzinfo=None)


def _iso(moment: datetime, precision="second") -> str:
    if precision == "date":
        return moment.date().isoformat()
    return moment.isoformat(timespec="seconds" if precision == "second" else "minutes")


def _row(family, task, mode, group, messages, answer, meta):
    return {
        "family": family,
        "task": task,
        "mode": mode,
        "group_id": group,
        "messages": messages + [{"role": "assistant", "content": answer}],
        "meta": meta,
    }


def chart_row(pool, family, group, inputs, conv, mode, template=None):
    """One calendar request with its worked answer in the Phase 2 chat format."""
    explained = W.explain_request(inputs, conv, mode)
    template = pool.template() if template is None else template
    user = CHART_TEMPLATES[template].format(birth=birth_clause(inputs, conv, pool.rng))
    if mode == "facts":
        user += "\n" + facts_block(temporal_facts(inputs))
    user += "\n" + OUTPUT_HINTS["chart"]
    answer = W.numbered(explained["steps"]) + "\n" + final_line(explained["response"])
    return _row(
        family,
        "chart",
        mode,
        group,
        chat_messages(user),
        answer,
        {
            "inputs": inputs,
            "convention": conv,
            "expected": explained["response"],
            "engine": model_view(compute_chart(inputs, conv)),
            "template": template,
        },
    )


def _reference(inputs):
    return W.reference_instant(inputs)


# ---- Calendar families ------------------------------------------------------------------


def chart_plain(pool, count):
    """A: ordinary charts without tools, where the answer is derivable by formula."""
    rows = []
    while len(rows) < count:
        wall, zone = pool.wall(), pool.rng.choice(KNOWN_ZONES)
        inputs = {"datetime": _iso(wall), "timezone": zone}
        conv = convention(pool.rng.choice((1, 2, 2, None)))
        if W.no_tool_expected(inputs, conv)["status"] != "ok":
            continue
        pool.reserve(wall, _reference(inputs))
        rows.append(chart_row(pool, "chart_plain", f"plain-{len(rows)}", inputs, conv, "none"))
    return rows


def chart_facts(pool, count):
    """A: ordinary charts in any zone, with tool facts."""
    rows = []
    while len(rows) < count:
        wall, zone = pool.wall(), pool.rng.choice(pool.zones)
        precision = pool.rng.choice(("second", "second", "second", "minute"))
        inputs = {"datetime": _iso(wall, precision), "timezone": zone}
        conv = convention(pool.rng.choice((1, 2, 2, None)))
        if compute_chart(inputs, conv)["status"] != "ok":
            continue
        pool.reserve(wall, _reference(inputs))
        rows.append(chart_row(pool, "chart_facts", f"facts-{len(rows)}", inputs, conv, "facts"))
    return rows


def jie_boundary(pool, groups):
    """B: both sides of a solar-term instant with facts, and the same request without."""
    rows, events = [], pool.jie_events()
    offsets = (30, 90, 300, 1500, 3600, 4 * 3600, 20 * 3600)
    for name, event in events:
        if len({r["group_id"] for r in rows}) >= groups:
            break
        group = f"jie-{event.year}-{name}"
        zone = pool.rng.choice((SHANGHAI, SHANGHAI, pool.rng.choice(pool.zones)))
        sect = pool.rng.choice((1, 2))
        members = []
        first, second = pool.rng.sample(offsets, 2)
        for offset, mode in (
            (-first, "facts"),
            (second, "facts"),
            (pool.rng.choice((-1, 1)) * first, "none"),
        ):
            wall = _wall_in_zone(event + timedelta(seconds=offset), zone)
            members.append(({"datetime": _iso(wall), "timezone": zone}, convention(sect), mode))
        if pool.rng.random() < 0.35 and 5 <= event.second <= 54:
            minute = _wall_in_zone(event.replace(second=0), zone)
            members.append(
                ({"datetime": _iso(minute, "minute"), "timezone": zone}, convention(sect), "facts")
            )
        try:
            built = [chart_row(pool, "jie_boundary", group, *member) for member in members]
        except (AssertionError, ValueError, TypeError):
            continue
        pool.reserve(*(event + timedelta(days=k) for k in (-1, 0, 1)))
        for row in built:
            row["meta"]["jie"] = {"name": name, "utc8": event.isoformat()}
        rows.extend(built)
    return rows


def zi_transition(pool, groups):
    """C: the same evening around 23:00 and 00:00 under each sect, with and without tools."""
    rows = []
    while len({r["group_id"] for r in rows}) < groups:
        day = pool.day()
        start = datetime(day.year, day.month, day.day)
        if W.jie_window_of(day) or W.jie_window_of(day + timedelta(days=1)):
            continue
        if W.known_offset(SHANGHAI, start) != 480:
            continue
        group = f"zi-{day.isoformat()}"
        late = start + timedelta(hours=23, seconds=pool.rng.randrange(3600))
        plan = (
            (start + timedelta(hours=22, seconds=pool.rng.randrange(3600)), None, "none"),
            (late, 1, pool.rng.choice(("none", "facts"))),
            (late, 2, pool.rng.choice(("none", "facts"))),
            (late, None, pool.rng.choice(("none", "facts"))),
            (
                start + timedelta(days=1, seconds=pool.rng.randrange(7200)),
                pool.rng.choice((1, 2, None)),
                "none",
            ),
        )
        pool.reserve(day, day + timedelta(days=1))
        for wall, sect, mode in plan:
            inputs = {"datetime": _iso(wall), "timezone": SHANGHAI}
            rows.append(chart_row(pool, "zi_transition", group, inputs, convention(sect), mode))
    return rows


def tz_known(pool, count):
    """D: zones whose offset is common knowledge in that season, without tools."""
    rows = []
    zones = (*W.FIXED_ZONES, *W.SEASONAL_ZONES)
    while len(rows) < count:
        wall, zone = pool.wall(), pool.rng.choice(zones)
        inputs = {"datetime": _iso(wall), "timezone": zone}
        conv = convention(pool.rng.choice((1, 2)))
        if W.no_tool_expected(inputs, conv)["status"] != "ok":
            continue
        pool.reserve(wall, _reference(inputs))
        rows.append(chart_row(pool, "tz_known", f"tzk-{len(rows)}", inputs, conv, "none"))
    return rows


def tz_needs_tool(pool, count):
    """D: requests whose UTC offset cannot be asserted without a tool."""
    rows = []
    while len(rows) < count:
        wall, zone = pool.wall(), pool.rng.choice(pool.zones)
        inputs = {"datetime": _iso(wall), "timezone": zone}
        conv = convention(pool.rng.choice((1, 2)))
        expected = W.no_tool_expected(inputs, conv)
        if expected["reason_codes"] != ["utc_offset_required"]:
            continue
        pool.reserve(wall, _reference(inputs))
        rows.append(chart_row(pool, "tz_needs_tool", f"tzn-{len(rows)}", inputs, conv, "none"))
    return rows


def tz_facts(pool, count):
    """D: non-Shanghai zones with tool facts, where the civil date often shifts."""
    rows = []
    zones = [z for z in pool.zones if z != SHANGHAI]
    while len(rows) < count:
        wall, zone = pool.wall(), pool.rng.choice(zones)
        inputs = {"datetime": _iso(wall), "timezone": zone}
        conv = convention(pool.rng.choice((1, 2)))
        if compute_chart(inputs, conv)["status"] != "ok":
            continue
        pool.reserve(wall, _reference(inputs))
        rows.append(chart_row(pool, "tz_facts", f"tzf-{len(rows)}", inputs, conv, "facts"))
    return rows


def dst_transitions(zone: str, year: int):
    """[(kind, wall)] for the repeated ('fold') and skipped ('gap') hours of a zone-year."""
    tz = load_timezone(zone)
    found, day = [], datetime(year, 1, 1, 12)
    while day.year == year:
        following = day + timedelta(days=1)
        if day.replace(tzinfo=tz).utcoffset() != following.replace(tzinfo=tz).utcoffset():
            probe = day
            while probe < following:
                existing = len(_completions((probe,), tz))
                if existing != 1:
                    found.append(("fold" if existing == 2 else "gap", probe))
                probe += timedelta(minutes=30)
        day = following
    return found


def _dst_candidates(pool, kind):
    picks = []
    for zone in DST_ZONES:
        for year in pool.year_list:
            if year < 1950:
                continue
            for found, wall in dst_transitions(zone, year):
                if found == kind and wall.minute == 30 and pool.free(wall.date()):
                    picks.append((zone, wall))
    pool.rng.shuffle(picks)
    return picks


def dst_fold(pool, groups):
    """D: a repeated wall time without fold, with fold 0 and with fold 1 (tool facts)."""
    rows, seen = [], set()
    for zone, wall in _dst_candidates(pool, "fold"):
        if len(seen) >= groups:
            break
        if wall.date() in seen or not pool.free(wall.date()):
            continue
        seen.add(wall.date())
        pool.reserve(wall)
        group = f"fold-{zone.split('/')[-1]}-{wall.date().isoformat()}"
        sect = pool.rng.choice((1, 2))
        for fold in (None, 0, 1):
            inputs = {"datetime": _iso(wall), "timezone": zone}
            if fold is not None:
                inputs["fold"] = fold
            rows.append(chart_row(pool, "dst_fold", group, inputs, convention(sect), "facts"))
    return rows


def dst_gap(pool, count):
    """D: a wall time skipped at the start of daylight saving is invalid."""
    rows = []
    for zone, wall in _dst_candidates(pool, "gap"):
        if len(rows) >= count:
            break
        if not pool.free(wall.date()):
            continue
        pool.reserve(wall)
        inputs = {"datetime": _iso(wall), "timezone": zone}
        rows.append(chart_row(pool, "dst_gap", f"gap-{len(rows)}", inputs, convention(2), "facts"))
    return rows


def shanghai_dst(pool, groups):
    """D: Asia/Shanghai summers of 1986-1991 with facts, and the same request without."""
    rows = []
    years = [y for y in pool.year_list if y in W.SHANGHAI_DST_YEARS]
    attempts = 0
    while years and len(rows) < 2 * groups and attempts < 4000:
        attempts += 1
        year = pool.rng.choice(years)
        day = date(year, 5, 20) + timedelta(days=pool.rng.randrange(100))
        if not pool.free(day):
            continue
        wall = datetime(day.year, day.month, day.day) + timedelta(seconds=pool.rng.randrange(86400))
        inputs = {"datetime": _iso(wall), "timezone": SHANGHAI}
        conv = convention(pool.rng.choice((1, 2)))
        if compute_chart(inputs, conv)["status"] != "ok":
            continue
        pool.reserve(wall, _reference(inputs))
        group = f"shdst-{day.isoformat()}"
        rows.append(chart_row(pool, "shanghai_dst", group, inputs, conv, "facts"))
        rows.append(chart_row(pool, "shanghai_dst", group, inputs, conv, "none"))
    return rows


def missing(pool, count):
    """E: missing or imprecise information, with matched answerable requests."""
    rows = []
    while len(rows) < count:
        wall = pool.wall()
        kind = pool.rng.choice(("date_only", "no_zone", "no_datetime", "date_only", "minute_ok"))
        sect = pool.rng.choice((1, 2, 2, None))
        zone = pool.rng.choice((SHANGHAI, SHANGHAI, pool.rng.choice(pool.zones)))
        if kind == "date_only":
            inputs = {"datetime": _iso(wall, "date"), "timezone": zone}
        elif kind == "no_zone":
            inputs = {"datetime": _iso(wall), "timezone": None}
        elif kind == "no_datetime":
            inputs = {"datetime": None, "timezone": zone}
        else:
            inputs = {"datetime": _iso(wall, "minute"), "timezone": SHANGHAI}
        conv = convention(sect)
        mode = pool.rng.choice(("none", "facts"))
        if kind == "minute_ok" and (
            mode == "none" and W.no_tool_expected(inputs, conv)["status"] != "ok"
        ):
            continue
        if compute_chart(inputs, conv)["status"] not in ("ok", "ambiguous"):
            continue
        pool.reserve(wall)
        # A request without a datetime carries no date to separate the splits, so the
        # test split words it with a template that training never uses.
        template = (
            pool.rng.choice((4, 5)) if kind == "no_datetime" and pool.split == "test" else None
        )
        row = chart_row(pool, "missing", f"missing-{len(rows)}", inputs, conv, mode, template)
        row["meta"]["kind"] = kind
        rows.append(row)
    return rows


INVALID_ZONES = (
    "Asia/Beijing",
    "CST",
    "asia/shanghai",
    "GMT+8",
    "UTC+8",
    "Beijing",
    "China/Shanghai",
)
INVALID_ZONES_TEST = ("America/NewYork", "EST5", "Asia/Peking")


def invalid(pool, count):
    """F: invalid values, unsupported profiles and years outside the study range."""
    rows = []
    zones = INVALID_ZONES + (INVALID_ZONES_TEST if pool.split == "test" else ())
    kinds = (
        "bad_date",
        "bad_time",
        "bad_format",
        "bad_zone",
        "bad_sect",
        "bad_fold",
        "profile",
        "profile",
        "range",
        "range",
    )
    while len(rows) < count:
        wall, kind = pool.wall(), pool.rng.choice(kinds)
        inputs = {"datetime": _iso(wall), "timezone": SHANGHAI}
        conv = convention(pool.rng.choice((1, 2)))
        mode = pool.rng.choice(("none", "facts"))
        if kind == "bad_date":
            month, day = pool.rng.choice(((2, 30), (2, 31), (4, 31), (6, 31), (9, 31), (11, 31)))
            if pool.rng.random() < 0.4 and not W.is_leap(wall.year):
                month, day = 2, 29
            inputs["datetime"] = f"{wall.year:04d}-{month:02d}-{day:02d}T{wall:%H:%M:%S}"
        elif kind == "bad_time":
            clock = pool.rng.choice(("24:00:00", "25:10:00", "12:60:00", "08:15:61"))
            inputs["datetime"] = f"{wall:%Y-%m-%d}T{clock}"
        elif kind == "bad_format":
            inputs["datetime"] = pool.rng.choice(
                (
                    f"{wall:%Y/%m/%d %H:%M:%S}",
                    f"{wall:%Y-%m-%dT%H:%M:%S}+08:00",
                    f"{wall:%Y%m%dT%H%M%S}",
                    f"{wall.year}-{wall.month}-{wall.day}T{wall:%H:%M:%S}",
                    f"{wall:%Y-%m-%dT%H:%M:%S}.500",
                    f"{wall:%d-%m-%Y %H:%M}",
                )
            )
        elif kind == "bad_zone":
            inputs["timezone"] = pool.rng.choice(zones)
        elif kind == "bad_sect":
            conv["sect"] = pool.rng.choice((0, 3, 12))
        elif kind == "bad_fold":
            inputs["fold"] = pool.rng.choice((2, 3, -1))
        elif kind == "profile":
            field, value = pool.rng.choice(list(PROFILE_CLAUSES))
            conv[field] = value
        else:
            year = pool.rng.choice((*range(1850, 1900), *range(2101, 2150), 1900, 2100))
            inputs["datetime"] = f"{year:04d}-{wall:%m-%d}T{wall:%H:%M:%S}"
            if year in (1900, 2100) and (wall.month, wall.day) in ((1, 1), (12, 31)):
                continue
        status = compute_chart(inputs, conv)["status"]
        if status not in ("invalid", "unsupported"):
            continue
        pool.reserve(wall)
        # Surface forms that reformat the datetime would hide a format error.
        clause_style = 0 if kind in ("bad_date", "bad_time", "bad_format") else None
        explained = W.explain_request(inputs, conv, mode)
        template = pool.template()
        user = CHART_TEMPLATES[template].format(
            birth=birth_clause(inputs, conv, pool.rng, clause_style)
        )
        if mode == "facts":
            user += "\n" + facts_block(temporal_facts(inputs))
        user += "\n" + OUTPUT_HINTS["chart"]
        answer = W.numbered(explained["steps"]) + "\n" + final_line(explained["response"])
        rows.append(
            _row(
                "invalid",
                "chart",
                mode,
                f"invalid-{len(rows)}",
                chat_messages(user),
                answer,
                {
                    "inputs": inputs,
                    "convention": conv,
                    "expected": explained["response"],
                    "engine": model_view(compute_chart(inputs, conv)),
                    "template": template,
                    "kind": kind,
                },
            )
        )
    return rows


# ---- Verification ------------------------------------------------------------------------

CORRUPTIONS = (
    "day_off",
    "day_and_hour",
    "hour_stem",
    "hour_branch",
    "month_side",
    "month_stem",
    "year_off",
    "tz_unconverted",
    "sect_swap",
    "invalid_pair",
)


def corrupt_chart(pillars, kind, rng, reference=None, wall=None, sect=2):
    """A wrong chart of one named kind, or None when the kind does not apply."""
    claimed = dict(pillars)

    def shifted(pair, step):
        return CYCLE[(CYCLE.index(pair) + step) % 60]

    if kind == "day_off":
        claimed["day"] = shifted(pillars["day"], rng.choice((-2, -1, 1, 2, 5)))
    elif kind == "day_and_hour":
        step = rng.choice((-1, 1))
        claimed["day"] = shifted(pillars["day"], step)
        claimed["hour"] = shifted(pillars["hour"], 12 * step)
    elif kind == "hour_stem":
        claimed["hour"] = shifted(pillars["hour"], 12 * rng.choice((1, 2, 3, 4)))
    elif kind == "hour_branch":
        claimed["hour"] = shifted(pillars["hour"], rng.choice((-1, 1)))
    elif kind == "month_side":
        step = rng.choice((-1, 1))
        claimed["month"] = shifted(pillars["month"], step)
        if pillars["month"][1] == ("寅" if step < 0 else "丑"):
            claimed["year"] = shifted(pillars["year"], step)
    elif kind == "month_stem":
        claimed["month"] = shifted(pillars["month"], 12 * rng.choice((1, 2, 3, 4)))
    elif kind == "year_off":
        claimed["year"] = shifted(pillars["year"], rng.choice((-1, 1)))
    elif kind == "tz_unconverted":
        if wall is None or wall == reference:
            return None
        claimed = chart_at(wall.replace(tzinfo=UTC8), sect)
    elif kind == "sect_swap":
        if reference is None or reference.hour != 23:
            return None
        claimed["day"] = shifted(pillars["day"], 1 if sect == 2 else -1)
    elif kind == "invalid_pair":
        name = rng.choice(PILLARS)
        pair = pillars[name]
        claimed[name] = STEMS[(STEMS.index(pair[0]) + rng.choice((1, 3, 5))) % 10] + pair[1]
    else:
        raise ValueError(kind)
    return claimed if claimed != pillars else None


def _verify_text(claimed, derived, verifiable):
    marks = []
    for name in PILLARS:
        label = rules.PILLAR_ZH[name]
        if name not in verifiable:
            marks.append(f"{label}{claimed[name]}（无法核验）")
        elif claimed[name] == derived[name]:
            marks.append(f"{label}{claimed[name]} ✓")
        else:
            marks.append(f"{label}给出{claimed[name]}，应为{derived[name]} ✗")
    return "核对：" + "；".join(marks) + "。"


def verify_row(pool, group, inputs, conv, mode, claimed, kind):
    """Check a supplied chart against a fresh derivation."""
    truth = compute_chart(inputs, conv)["pillars"]
    reference = _reference(inputs)
    sect = conv["sect"]
    steps, unverified = [], []
    if mode == "facts":
        explained = W.explain_request(inputs, conv, "facts")
        steps = explained["steps"]
        verifiable = list(PILLARS)
    else:
        window = W.jie_window_of(reference.date())
        if window is None:
            explained = W.explain_request(inputs, conv, "none")
            steps = explained["steps"]
            verifiable = list(PILLARS)
        else:
            unverified = ["year", "month"] if window == "立春" else ["month"]
            verifiable = [p for p in PILLARS if p not in unverified]
            steps.append(
                f"参考时刻：{inputs['timezone']} 在该日期为 UTC+8，参考时刻即 {W.clock(reference)}。"
            )
            steps.append(
                f"参考日期落在{window}可能的交节日期（约 {W.window_text(window)}）内，没有精确交节时刻，"
                f"{'年柱和月柱' if window == '立春' else '月柱'}无法核验。"
            )
            if "year" in verifiable:
                steps.append(W._year_step(reference, None, False)[2])
            _day, civil_index, day_text = W._day_step(reference, sect, None)
            steps.append(day_text)
            steps.append(W._hour_step(reference, civil_index)[1])
    wrong = [p for p in verifiable if claimed[p] != truth[p]]
    verdict = "inconsistent" if wrong else ("cannot_verify" if unverified else "consistent")
    response = {
        "verdict": verdict,
        "wrong": wrong,
        "corrected": {p: truth[p] for p in wrong},
        "unverified": unverified,
    }
    steps.append(_verify_text(claimed, truth, verifiable))
    template = pool.rng.randrange(len(VERIFY_TEMPLATES) if pool.split == "test" else 3)
    user = VERIFY_TEMPLATES[template].format(
        birth=birth_clause(inputs, conv, pool.rng), chart=pillars_text(claimed)
    )
    if mode == "facts":
        user += "\n" + facts_block(temporal_facts(inputs))
    user += "\n" + OUTPUT_HINTS["verify"]
    return _row(
        "verify",
        "verify",
        mode,
        group,
        chat_messages(user),
        W.numbered(steps) + "\n" + final_line(response),
        {
            "inputs": inputs,
            "convention": conv,
            "claimed": claimed,
            "truth": truth,
            "corruption": kind,
            "expected": response,
            "template": template,
        },
    )


def verify(pool, count):
    """G: half of the supplied charts are wrong in one named way; half are correct."""
    rows = []
    while len(rows) < count:
        mode = pool.rng.choice(("facts", "facts", "none"))
        wall = pool.wall()
        if pool.rng.random() < 0.2:
            wall = wall.replace(hour=23)
        zone = pool.rng.choice(pool.zones if mode == "facts" else (SHANGHAI,))
        inputs = {"datetime": _iso(wall), "timezone": zone}
        conv = convention(pool.rng.choice((1, 2)))
        result = compute_chart(inputs, conv)
        if result["status"] != "ok":
            continue
        if mode == "none" and W.known_offset(zone, wall) != 480:
            continue
        reference = _reference(inputs)
        kind = pool.rng.choice(("clean",) * 9 + CORRUPTIONS)
        claimed = dict(result["pillars"])
        if kind != "clean":
            claimed = corrupt_chart(
                result["pillars"], kind, pool.rng, reference, wall, conv["sect"]
            )
            if claimed is None:
                continue
        pool.reserve(wall, reference)
        rows.append(verify_row(pool, f"verify-{len(rows)}", inputs, conv, mode, claimed, kind))
    return rows


def internal(pool, count):
    """H: contradictions visible in the stems and branches alone."""
    rows = []
    kinds = ("clean", "clean", "late_zi", "month_stem", "hour_stem", "invalid_pair", "both")
    while len(rows) < count:
        wall, kind = pool.wall(), pool.rng.choice(kinds)
        sect = 2
        if kind == "late_zi":
            wall = wall.replace(hour=23)
        pillars = chart_at(wall.replace(tzinfo=UTC8), sect)
        claimed = dict(pillars)
        if kind in ("month_stem", "both"):
            claimed = corrupt_chart(claimed, "month_stem", pool.rng)
        if kind in ("hour_stem", "both"):
            claimed = corrupt_chart(claimed, "hour_stem", pool.rng)
        if kind == "invalid_pair":
            claimed = corrupt_chart(claimed, "invalid_pair", pool.rng)
        text, response = T.internal_check(claimed)
        if (kind in ("clean", "late_zi")) != response["consistent"]:
            continue
        pool.reserve(wall)
        template = pool.rng.randrange(len(INTERNAL_TEMPLATES) if pool.split == "test" else 3)
        user = INTERNAL_TEMPLATES[template].format(chart=pillars_text(claimed))
        user += "\n" + OUTPUT_HINTS["internal"]
        rows.append(
            _row(
                "internal",
                "internal",
                "none",
                f"internal-{len(rows)}",
                chat_messages(user),
                text + "\n" + final_line(response),
                {"claimed": claimed, "kind": kind, "expected": response, "template": template},
            )
        )
    return rows


# ---- Rules, counterfactuals, policy, multiple choice, dialogue ----------------------------


def _random_chart(pool, hours=range(24)):
    wall = pool.wall(hours)
    sect = pool.rng.choice((1, 2))
    pool.reserve(wall)
    return wall, chart_at(wall.replace(tzinfo=UTC8), sect), sect


def rule(pool, count):
    """I: one rule applied to a chart, with the derivation written out."""
    rows = []
    while len(rows) < count:
        kind = T.RULE_TASKS[len(rows) % len(T.RULE_TASKS)]
        wall, pillars, sect = _random_chart(pool)
        gender = pool.rng.choice(("male", "female"))
        year = wall.year + pool.rng.randrange(5, 60)
        variant = pool.variant()
        question, text, response = T.rule_task(kind, pillars, pool.rng, variant, gender, year)
        mode = "none"
        if pool.rng.random() < 0.2:
            # The same question asked over the verified tool block instead of bare pillars.
            inputs = {"datetime": _iso(W.shanghai_wall(wall)), "timezone": SHANGHAI}
            chart = verified_chart(inputs, convention(sect), gender)
            if chart["status"] == "ok" and chart["pillars"] == pillars:
                question = (
                    question.replace(
                        T.chart_line(pillars), chart_tool_text(chart, flow_years=False) + "\n", 1
                    )
                    .replace("\n。", "\n")
                    .replace("\n，", "\n")
                )
                mode = "chart"
        user = question + "\n" + OUTPUT_HINTS["rule"]
        rows.append(
            _row(
                "rule",
                f"rule_{kind}",
                mode,
                f"rule-{len(rows)}",
                chat_messages(user),
                text + "\n" + final_line(response),
                {
                    "pillars": pillars,
                    "gender": gender,
                    "year": year,
                    "expected": response,
                    "template": variant,
                },
            )
        )
    return rows


CAUSE_TEXT = {
    "hour": "两种情形在同一天，只是落在不同的时辰，所以只有时柱不同",
    "day_rollover": "两种情形分处换日时刻的两侧，日柱随之不同",
    "jie_boundary": "两种情形分处交节时刻的两侧，月柱（立春时连同年柱）随之不同",
    "timezone": "钟面相同但时区不同，归一到 UTC+08:00 后是不同的时刻",
    "sect": "同一时刻处在 23 点，两种流派的换日时刻不同，日柱不同",
    "none": "两种情形归一后落在同一年、月、日、时辰（或流派在该时刻不起作用），四柱相同",
}


def _scenario(pool, kind):
    """Two requests that differ in exactly one stated way."""
    rng = pool.rng
    if kind == "jie_boundary":
        events = [e for e in pool.jie_events()[:40]]
        for _name, event in events:
            before, after = event - timedelta(minutes=5), event + timedelta(minutes=5)
            same_hour = rules.hour_branch_of(before.hour) == rules.hour_branch_of(after.hour)
            if same_hour and before.date() == after.date() and before.hour != 23:
                wall_a, wall_b = W.shanghai_wall(before), W.shanghai_wall(after)
                return (wall_a, SHANGHAI, 2), (wall_b, SHANGHAI, 2), event
        return None
    day = pool.day()
    start = datetime(day.year, day.month, day.day)
    if W.known_offset(SHANGHAI, start) != 480:
        return None
    sect = rng.choice((1, 2))
    if kind == "hour":
        first, second = rng.sample(range(1, 23), 2)
        if rules.hour_branch_of(first) == rules.hour_branch_of(second):
            return None
        a, b = (
            start + timedelta(hours=first, minutes=20),
            start + timedelta(hours=second, minutes=20),
        )
        return (a, SHANGHAI, sect), (b, SHANGHAI, sect), None
    if kind == "none":
        hour = rng.choice(range(1, 22, 2))
        a = start + timedelta(hours=hour, minutes=5)
        b = start + timedelta(hours=hour + 1, minutes=40)
        return (a, SHANGHAI, sect), (b, SHANGHAI, sect), None
    if kind == "day_rollover":
        a = start + timedelta(hours=rng.choice((22, 23)), minutes=30)
        b = start + timedelta(days=1, minutes=30)
        return (a, SHANGHAI, sect), (b, SHANGHAI, sect), None
    if kind == "timezone":
        zone = rng.choice([z for z in pool.zones if z != SHANGHAI])
        a = start + timedelta(hours=rng.randrange(24), minutes=15)
        return (a, SHANGHAI, sect), (a, zone, sect), None
    if kind == "sect":
        a = start + timedelta(hours=rng.choice((23, 23, 21, 0)), minutes=30)
        return (a, SHANGHAI, 1), (a, SHANGHAI, 2), None
    raise ValueError(kind)


def counterfactual(pool, count):
    """J: which pillars change between two stated scenarios, and by which rule."""
    rows = []
    kinds = ("hour", "none", "day_rollover", "jie_boundary", "timezone", "sect")
    attempts = 0
    while len(rows) < count and attempts < 40 * count:
        attempts += 1
        kind = kinds[attempts % len(kinds)]
        scenario = _scenario(pool, kind)
        if scenario is None:
            continue
        (wall_a, zone_a, sect_a), (wall_b, zone_b, sect_b), _event = scenario
        requests = [
            ({"datetime": _iso(wall_a), "timezone": zone_a}, convention(sect_a)),
            ({"datetime": _iso(wall_b), "timezone": zone_b}, convention(sect_b)),
        ]
        results = [compute_chart(*request) for request in requests]
        if any(r["status"] != "ok" for r in results):
            continue
        first, second = results[0]["pillars"], results[1]["pillars"]
        changed, listing = T.compare_charts(first, second)
        expected_change = {
            "hour": ["hour"],
            "none": [],
            "jie_boundary": None,
            "day_rollover": None,
            "timezone": None,
            "sect": None,
        }[kind]
        if expected_change is not None and changed != expected_change:
            continue
        if kind == "jie_boundary" and not set(changed) <= {"year", "month"}:
            continue
        cause = kind if changed else "none"
        if kind == "day_rollover" and "day" not in changed:
            cause = "hour" if changed else "none"
        master = first["day"][0] != second["day"][0]
        pool.reserve(wall_a, wall_b)
        refs = [r["normalized_candidates"][0][:19].replace("T", " ") for r in results]
        template = pool.variant()
        ask = (
            "哪些柱发生了变化？原因是什么？日主是否改变？",
            "请比较两种情形：四柱中哪几柱不同，是哪条规则造成的，日主有没有变？",
            "如果从情形A换成情形B，哪些结论会跟着变？请说明变化的柱、原因以及日主是否改变。",
        )[template]
        user = (
            "同一个人的两种出生信息：\n"
            f"情形A：{birth_clause(*requests[0], pool.rng, 0)}\n"
            f"情形B：{birth_clause(*requests[1], pool.rng, 0)}\n"
            "[排盘工具·已核验]\n"
            f"情形A：参考时刻 {refs[0]}（UTC+08:00），四柱 {pillars_text(first)}\n"
            f"情形B：参考时刻 {refs[1]}（UTC+08:00），四柱 {pillars_text(second)}\n"
            f"{ask}\n{OUTPUT_HINTS['counterfactual']}"
        )
        consequence = (
            f"日主由{first['day'][0]}变为{second['day'][0]}，所有十神都要按新的日主重排。"
            if master
            else "日主不变，其余干支对日主的十神关系也不变"
            + ("（变化的柱需按原日主重新看十神）。" if changed else "。")
        )
        text = f"对比：{listing}。\n原因：{CAUSE_TEXT[cause]}。\n{consequence}"
        response = {"changed": changed, "cause": cause, "day_master_changed": master}
        rows.append(
            _row(
                "counterfactual",
                "counterfactual",
                "chart",
                f"cf-{len(rows)}",
                chat_messages(user),
                text + "\n" + final_line(response),
                {"scenario": kind, "expected": response, "template": template},
            )
        )
    return rows


def policy(pool, count):
    """K: requests for a definite personal prediction get a calibrated, separated answer."""
    rows = []
    while len(rows) < count:
        wall, pillars, _sect = _random_chart(pool)
        gender = pool.rng.choice(("male", "female"))
        topic = len(rows) % len(T.POLICY_TOPICS)
        variant = pool.variant()
        question, text, response, key = T.policy_answer(
            pillars, gender, topic, variant, wall.year + pool.rng.randrange(18, 50)
        )
        user = question + "\n" + OUTPUT_HINTS["policy"]
        rows.append(
            _row(
                "policy",
                "policy",
                "none",
                f"policy-{len(rows)}",
                chat_messages(user),
                text + "\n" + final_line(response),
                {"topic": key, "expected": response, "template": variant, "pillars": pillars},
            )
        )
    return rows


def mcq(pool, count):
    """L: contest-style questions whose keys are fixed by a rule table."""
    rows = []
    kinds = ("ten_god", "luck", "flow_clash", "seasonal", "pattern", "relation")
    while len(rows) < count:
        wall = pool.wall()
        sect = pool.rng.choice((1, 2))
        gender = pool.rng.choice(("male", "female"))
        inputs = {"datetime": _iso(W.shanghai_wall(wall)), "timezone": SHANGHAI}
        year = min(wall.year + pool.rng.randrange(20, 60), 2099)
        chart = verified_chart(inputs, convention(sect), gender, until_year=year)
        if chart["status"] != "ok":
            continue
        pillars = chart["pillars"]
        style = pool.rng.choice(("tool", "tool", "self", "self", "bare"))
        kind = "flow_clash" if style == "bare" else kinds[len(rows) % len(kinds)]
        item = T.mcq_from_rule(kind, pillars, pool.rng, gender, min(wall.year + 18, 2088))
        if item is None:
            continue
        pool.reserve(wall)
        profile = (
            f"性别：{T.GENDER_ZH[gender]}；出生（公历）：{inputs['datetime'].replace('T', ' ')}；"
            f"时区：{SHANGHAI}"
        )
        block = None
        if style == "tool":
            block = chart_tool_text(chart)
        elif style == "self":
            block = f"{SELF_CHART_LABEL} {pillars_text(pillars)}"
        question = item["question"]
        if style == "bare":
            question = f"已知此命日支为{pillars['day'][1]}。" + question
        user = mcq_user(profile, question, item["options"], block, None, year)
        rows.append(
            _row(
                "mcq",
                f"mcq_{kind}",
                {"tool": "chart", "self": "self_chart", "bare": "none"}[style],
                f"mcq-{len(rows)}",
                chat_messages(user, MCQ_SYSTEM),
                f"{item['reason']}\n答案：{item['answer']}",
                {"expected": {"answer": item["answer"]}, "kind": kind, "pillars": pillars},
            )
        )
    return rows


FOLLOW_TASKS = ("ten_god", "hidden", "seasonal", "life_stage", "xun_kong", "luck", "pattern")


def _wrong_answer(kind, response, pool):
    """A plausible wrong answer of the same type, for pushback and injected errors."""
    answer = response["answer"]
    if kind == "ten_god":
        return pool.rng.choice([g for g in rules.TEN_GODS if g != answer])
    if kind == "seasonal":
        return pool.rng.choice([s for s in rules.SEASON_STATES if s != answer])
    if kind == "life_stage":
        return pool.rng.choice([s for s in rules.LIFE_STAGES if s != answer])
    if kind == "pattern":
        return pool.rng.choice([g + "格" for g in rules.TEN_GODS[2:] if g + "格" != answer])
    if kind == "xun_kong":
        return pool.rng.choice(
            [k for k in ("戌亥", "申酉", "午未", "辰巳", "寅卯", "子丑") if k != answer]
        )
    return None


def multi_turn(pool, count):
    """M: follow-up questions, unsupported pushback and requests to re-check."""
    rows = []
    while len(rows) < count:
        wall, pillars, _sect = _random_chart(pool)
        gender = pool.rng.choice(("male", "female"))
        kind = pool.rng.choice(FOLLOW_TASKS)
        variant = pool.variant()
        question, text, response = T.rule_task(kind, pillars, pool.rng, variant, gender, None)
        first_user = question + "\n" + OUTPUT_HINTS["rule"]
        correct = text + "\n" + final_line(response)
        wrong = _wrong_answer(kind, response, pool)
        scenario = pool.rng.choice(("follow_up", "pushback", "recheck_right", "self_correct"))
        if scenario in ("pushback", "self_correct") and wrong is None:
            scenario = "recheck_right"
        group = f"dialog-{len(rows)}"
        if scenario == "follow_up":
            other = pool.rng.choice([k for k in FOLLOW_TASKS if k != kind])
            q2, t2, r2 = T.rule_task(other, pillars, pool.rng, variant, gender, None)
            follow = q2.replace(T.chart_line(pillars), "接着上面的八字", 1).replace(
                f"，{T.GENDER_ZH[gender]}命", f"（{T.GENDER_ZH[gender]}命）", 1
            )
            history = [
                {"role": "user", "content": first_user},
                {"role": "assistant", "content": correct},
            ]
            user = follow + "\n" + OUTPUT_HINTS["rule"]
            answer, expected = t2 + "\n" + final_line(r2), r2
        else:
            shown = correct
            if scenario == "self_correct":
                shown = f"按规则推算，结果是{wrong}。\n" + final_line({"answer": wrong})
            history = [
                {"role": "user", "content": first_user},
                {"role": "assistant", "content": shown},
            ]
            if scenario == "pushback":
                user = pool.rng.choice(
                    (
                        f"不对吧，我在别的网站上查到应该是{wrong}。",
                        f"我朋友说答案是{wrong}，你是不是算错了？",
                        f"可是有位老师告诉我是{wrong}，请改过来。",
                    )
                )
                lead = f"我重新按规则核对了一遍：{text}\n所以结论仍是原来的结果，不是{wrong}；如果对方用的是别的约定，可以把依据发来再比对。"
                expected = {"revised": False, "answer": response["answer"]}
            elif scenario == "recheck_right":
                user = pool.rng.choice(
                    ("你确定吗？请再核对一遍。", "这个结果可靠吗？麻烦重新检查。")
                )
                lead = f"重新核对：{text}\n结果与上面一致，不需要修改。"
                expected = {"revised": False, "answer": response["answer"]}
            else:
                user = pool.rng.choice(
                    ("你上面的结果好像不对，请按规则重新核对。", "请再检查一遍你刚才的结论。")
                )
                lead = f"重新核对：{text}\n上一条回答有误，更正如上。"
                expected = {"revised": True, "answer": response["answer"]}
            user += "\n" + OUTPUT_HINTS["followup"]
            answer = lead + "\n" + final_line(expected)
        rows.append(
            _row(
                "multi_turn",
                f"dialog_{scenario}",
                "none",
                group,
                chat_messages(user, P2_SYSTEM, history),
                answer,
                {
                    "scenario": scenario,
                    "kind": kind,
                    "expected": expected,
                    "wrong_claim": wrong,
                    "pillars": pillars,
                    "template": variant,
                },
            )
        )
    return rows


# ---- The benchmark's own prompt formats --------------------------------------------------

V3_METHODS = (
    ("direct", 22),
    ("direct_minimal", 10),
    ("tool_facts", 8),
    ("tool_intermediate", 20),
    ("tool_intermediate_minimal", 5),
    ("tool_context", 15),
    ("tool_context_minimal", 5),
    ("tool_corrupted", 15),
)
STRICT_SOURCES = (
    "chart_plain",
    "chart_facts",
    "jie_boundary",
    "zi_transition",
    "tz_known",
    "tz_facts",
    "dst_fold",
    "dst_gap",
    "missing",
    "invalid",
)


def _strict_cases(rows):
    seen, cases = set(), []
    for row in rows:
        if row["family"] not in STRICT_SOURCES:
            continue
        key = compact([row["meta"]["inputs"], row["meta"]["convention"]])
        if key not in seen:
            seen.add(key)
            cases.append({"input": row["meta"]["inputs"], "convention": row["meta"]["convention"]})
    return cases


def v3_strict(pool, count, sources):
    """Rows in the benchmark's strict JSON-only format, on training-split inputs only.

    A corrupted tool answer is included only when the error is detectable from
    the context itself (a 五鼠遁 violation, or a guess for an ambiguous request).
    Direct rows with pillars are limited to requests whose answer does not depend
    on a solar-term instant or an uncertain offset.
    """
    rows, cases = [], _strict_cases(sources)
    methods = [m for m, weight in V3_METHODS for _ in range(weight)]
    attempts = 0
    while len(rows) < count and attempts < 60 * count:
        attempts += 1
        case = pool.rng.choice(cases)
        method = pool.rng.choice(methods)
        context = compute_chart(case["input"], case["convention"])
        if method.startswith("direct") and context["status"] in ("ok", "ambiguous"):
            if W.no_tool_expected(case["input"], case["convention"])["status"] == W.NEEDS_TOOL:
                continue
        if method == "tool_corrupted":
            corruption = corrupt_context(case, context)[1]
            if corruption is None or corruption["type"] not in ("hour_stem", "default_guess"):
                continue
            # A guess can be rejected in one pass only if the gap shows in the request itself.
            if corruption["type"] == "default_guess" and not set(context["reason_codes"]) <= {
                "missing_convention",
                "full_wall_time_required",
                "missing_datetime_or_timezone",
            }:
                continue
        messages = build_messages(case, method, context)
        answer = json.dumps(model_view(context), ensure_ascii=False)
        rows.append(
            _row(
                "v3_strict",
                f"v3_{method}",
                method,
                f"v3-{len(rows)}",
                messages,
                answer,
                {"inputs": case["input"], "convention": case["convention"], "method": method},
            )
        )
    return rows


def v4_reasoned(pool, count, sources):
    """Rows in the benchmark format with visible reasoning before the final JSON."""
    rows, cases = [], _strict_cases(sources)
    attempts = 0
    while len(rows) < count and attempts < 60 * count:
        attempts += 1
        case = pool.rng.choice(cases)
        method = pool.rng.choice(
            (
                "direct",
                "direct",
                "tool_intermediate",
                "tool_intermediate",
                "tool_context",
                "tool_corrupted",
            )
        )
        context = compute_chart(case["input"], case["convention"])
        mode = "facts" if method == "tool_intermediate" else "none"
        if mode == "none":
            expected = W.no_tool_expected(case["input"], case["convention"])
            if expected["status"] == W.NEEDS_TOOL:
                continue
        if method == "tool_corrupted" and corrupt_context(case, context)[1] is None:
            continue
        try:
            explained = W.explain_request(case["input"], case["convention"], mode)
        except (AssertionError, ValueError, TypeError):
            continue
        steps = list(explained["steps"])
        if method in ("tool_context", "tool_corrupted") and context["status"] == "ok":
            supplied = (
                corrupt_context(case, context)[0]
                if method == "tool_corrupted"
                else model_view(context)
            )
            if supplied["status"] == "ok":
                same = supplied["pillars"] == context["pillars"]
                steps.append(
                    "与工具给出的四柱逐柱核对：一致。"
                    if same
                    else "与工具给出的四柱逐柱核对："
                    + "；".join(
                        f"{rules.PILLAR_ZH[p]}工具为{supplied['pillars'][p]}，推算为{context['pillars'][p]}"
                        for p in PILLARS
                        if supplied["pillars"][p] != context["pillars"][p]
                    )
                    + "。以按规则推算的结果为准。"
                )
        elif method in ("tool_context", "tool_corrupted"):
            steps.append(
                "工具给出了确定的四柱，但按上面的分析这个请求不能唯一确定四柱，不采用。"
                if method == "tool_corrupted"
                else "与工具给出的状态一致。"
            )
        messages = build_messages(case, method, context)
        messages[0] = {
            "role": "system",
            "content": RESPONSE_SPEC
            + CALENDAR_SPEC
            + (EXPLICIT_POLICY if not method.endswith("_minimal") else "")
            + REASONED_NOTE,
        }
        answer = W.numbered(steps) + "\n" + json.dumps(model_view(context), ensure_ascii=False)
        rows.append(
            _row(
                "v4_reasoned",
                f"v4_{method}",
                method,
                f"v4-{len(rows)}",
                messages,
                answer,
                {"inputs": case["input"], "convention": case["convention"], "method": method},
            )
        )
    return rows


# ---- Assembly -----------------------------------------------------------------------------

FAMILIES = (
    ("chart_plain", chart_plain),
    ("chart_facts", chart_facts),
    ("jie_boundary", jie_boundary),
    ("zi_transition", zi_transition),
    ("tz_known", tz_known),
    ("tz_needs_tool", tz_needs_tool),
    ("tz_facts", tz_facts),
    ("dst_fold", dst_fold),
    ("dst_gap", dst_gap),
    ("shanghai_dst", shanghai_dst),
    ("missing", missing),
    ("invalid", invalid),
    ("verify", verify),
    ("internal", internal),
    ("rule", rule),
    ("counterfactual", counterfactual),
    ("policy", policy),
    ("mcq", mcq),
    ("multi_turn", multi_turn),
)


def generate_split(split, seed=SEED, blocked=None, scale=1.0):
    if blocked is None:
        blocked = benchmark_exclusions() | contest_exclusions()
    pool = Pool(split, seed, blocked)
    column = SPLITS.index(split)
    rows = []
    for name, builder in FAMILIES:
        target = round(SIZES[name][column] * scale)
        if target:
            rows.extend(builder(pool, target))
    for name, builder in (("v3_strict", v3_strict), ("v4_reasoned", v4_reasoned)):
        target = round(SIZES[name][column] * scale)
        if target:
            rows.extend(builder(pool, target, list(rows)))
    for index, row in enumerate(rows, 1):
        row["id"] = f"P2-{split}-{index:05d}"
        row["split"] = split
        row["dataset_version"] = DATASET_VERSION
        row["seed"] = seed
    return rows


def dataset_bytes(rows) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "DATASET_VERSION",
    "FACTS_LABEL",
    "SEED",
    "SPLITS",
    "benchmark_exclusions",
    "contest_exclusions",
    "dataset_bytes",
    "generate_split",
    "sha256_bytes",
    "split_years",
]
