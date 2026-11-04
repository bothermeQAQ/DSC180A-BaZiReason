"""Version 3 benchmark: matched H1 distance ladders and knowledge-dependent H3 families.

Every family keeps its matched variants under one group ID, and no two families
share a civil date. The design metadata in `design` is for analysis only; it is
never shown to a model. Labels remain lunar-python references; each family
asserts its intended status so a design error fails loudly instead of silently
changing a label. The v1 and v2 files are unchanged historical artifacts.
"""

import random
from datetime import date, datetime, timedelta

from bazireason.calendar_engine import UTC8, compute_chart, jie_around, load_timezone
from bazireason.conventions import Convention
from bazireason.schema import model_view

SEED = 1802026
VERSION = "benchmark_v3"
SHANGHAI = "Asia/Shanghai"
PROFILE = Convention().to_dict()
NO_SECT = {k: v for k, v in PROFILE.items() if k != "sect"}
CHINA_DST_YEARS = range(1986, 1992)

# H1 Jie ladders: twelve month-start terms once each, plus two more 立春 (year boundary).
JIE_EVENTS = (
    ("立春", 1978),
    ("惊蛰", 1982),
    ("清明", 1994),
    ("立夏", 1997),
    ("芒种", 2003),
    ("小暑", 2009),
    ("立秋", 2012),
    ("白露", 2016),
    ("寒露", 2021),
    ("立冬", 2029),
    ("大雪", 2034),
    ("小寒", 2037),
    ("立春", 2041),
    ("立春", 2047),
)
JIE_RUNGS = (  # offset seconds, distance class
    (-5 * 86400, "far"),
    (-3600, "mid"),
    (-120, "near"),
    (120, "near"),
    (3600, "mid"),
    (5 * 86400, "far"),
)
ZI_DATES = ("1981-03-17", "1997-08-26", "2013-11-29", "2035-05-24")
ZI_RUNGS = (  # clock, day offset, distance class, seconds from nearest 23:00/00:00 boundary
    ("21:30:00", 0, "far", -5400),
    ("22:59:59", 0, "near", -1),
    ("23:00:00", 0, "near", 0),
    ("23:59:59", 0, "near", -1),
    ("00:00:00", 1, "near", 0),
    ("01:30:00", 1, "far", 5400),
)
TIMEZONE_PAIRS = (  # base UTC+08:00 instant, other zone
    ("1994-05-27T10:20:00", "UTC"),
    ("2003-09-02T09:40:00", "America/Los_Angeles"),
    ("2010-01-19T15:10:00", "Europe/London"),
    ("2016-03-29T17:50:00", "Asia/Kolkata"),
    ("2021-12-02T07:25:00", "Australia/Adelaide"),
    ("2038-08-11T05:35:00", "America/Sao_Paulo"),
)
FOLD_FAMILIES = (  # zone, repeated wall time, does the fold change the chart?
    ("America/Los_Angeles", "2023-11-05T01:30:00", True),
    ("Europe/London", "2022-10-30T01:30:00", True),
    ("America/Chicago", "2021-11-07T01:30:00", True),
    ("America/New_York", "2019-11-03T01:30:00", False),
    ("America/Denver", "2018-11-04T01:30:00", False),
)
GAP_FAMILIES = (
    ("America/Los_Angeles", "2023-03-12T02:30:00"),
    ("Europe/London", "2022-03-27T01:30:00"),
)
SECT_DATES = ("1984-04-12", "1993-02-23", "2002-06-27", "2014-09-19", "2028-04-13", "2043-11-23")
MINUTE_EVENTS = (("立夏", 1980), ("白露", 2000), ("大雪", 2017), ("清明", 2043))
MINUTE_ZI_DATE = "2006-01-23"
TIME_MISSING_DATES = ("1979-07-19", "2007-05-30", "2031-10-02")
ZONE_MISSING_DATES = ("1985-12-11", "2010-08-03", "2027-03-15")
# Distinct zones keep the two missing-datetime requests from being identical.
DATETIME_MISSING = (("2001-11-14", SHANGHAI), ("2019-04-21", "Europe/London"))
UNSUPPORTED_PROFILES = (
    ("solar_time", "true_solar"),
    ("clock_basis", "local_civil"),
    ("year_boundary", "lunar_new_year"),
    ("month_boundary", "lunar_month"),
)


def convention_dependence(inputs, convention, result=None):
    """Corrected evaluation class (audit defect E6): compare the two sects directly.

    ok answers are convention_dependent only if the other sect changes a pillar.
    An ok answer with an unspecified sect is independent by construction.
    """
    result = result or compute_chart(inputs, convention)
    if result["status"] != "ok":
        return result["status"]
    sect = (convention or {}).get("sect")
    if sect is None:
        return "convention_independent"
    other = compute_chart(inputs, {**convention, "sect": 3 - sect})
    if other["pillars"] != result["pillars"]:
        return "convention_dependent"
    return "convention_independent"


def jie_instant(term, year):
    from lunar_python import Solar

    table = Solar.fromYmdHms(year, 6, 15, 12, 0, 0).getLunar().getJieQiTable()
    return datetime.fromisoformat(table[term].toYmdHms())


def _branch(moment):
    return ((moment.hour + 1) // 2) % 12


def _days_to_jie(moment):
    ref = moment.replace(tzinfo=UTC8)
    (prev_t, _), (next_t, _) = jie_around(ref)
    return min(ref - prev_t, next_t - ref) / timedelta(days=1)


def _wall(moment):
    return moment.isoformat(timespec="seconds")


class _Builder:
    def __init__(self, seed):
        self.seed = seed
        self.cases = []
        self.dates = {}

    def add(self, category, group, inputs, convention, intended, design):
        result = compute_chart(inputs, convention)
        if result["status"] != intended:
            raise AssertionError((group, inputs, convention, intended, result["status"]))
        raw = inputs.get("datetime")
        if isinstance(raw, str) and len(raw) >= 10:
            owner = self.dates.setdefault(raw[:10], group)
            if owner != group:
                raise AssertionError(f"{raw[:10]} is shared by {owner} and {group}")
        self.cases.append(
            {
                "input": inputs,
                "convention": convention,
                "category": category,
                "group_id": group,
                "expected": model_view(result),
                "evaluation_class": convention_dependence(inputs, convention, result),
                "design": design,
                "reference_implementation": result["provenance"],
                "validation_status": "single_engine_reference",
                "dataset_version": VERSION,
                "seed": self.seed,
            }
        )


def _h3(kind, knowledge_dependent, named, surface_form):
    return {
        "type": kind,
        "knowledge_dependent": knowledge_dependent,
        "named_in_explicit_policy": named,
        "surface_form": surface_form,
    }


def _jie_ladders(b):
    for term, year in JIE_EVENTS:
        event = jie_instant(term, year)
        clock = event - event.replace(hour=0, minute=0, second=0)
        if not timedelta(hours=4) <= clock <= timedelta(hours=20):
            raise AssertionError(f"{term} {year} at {event} leaves the 04:00-20:00 window")
        if _branch(event - timedelta(minutes=2)) != _branch(event + timedelta(minutes=2)):
            raise AssertionError(f"{term} {year} near rungs cross an hour-branch boundary")
        if year in CHINA_DST_YEARS:
            raise AssertionError("Shanghai wall time differs from UTC+08:00 in DST years")
        for offset, distance in JIE_RUNGS:
            moment = event + timedelta(seconds=offset)
            b.add(
                "jie_ladder",
                f"jie-{year}-{term}",
                {"datetime": _wall(moment), "timezone": SHANGHAI},
                dict(PROFILE),
                "ok",
                {
                    "hypotheses": ["H1", "H2"],
                    "h1": {
                        "boundary": "jie",
                        "term": term,
                        "event_utc8": event.replace(tzinfo=UTC8).isoformat(),
                        "offset_seconds": offset,
                        "distance": distance,
                        "side": "before" if offset < 0 else "after",
                        "relevant_pillars": ["year", "month"] if term == "立春" else ["month"],
                    },
                },
            )


def _zi_ladders(b):
    for day in ZI_DATES:
        start = date.fromisoformat(day)
        if _days_to_jie(datetime.combine(start, datetime.min.time())) < 2.5:
            raise AssertionError(f"Zi ladder {day} is within 2.5 days of a Jie")
        for sect in (1, 2):
            for clock, shift, distance, offset in ZI_RUNGS:
                wall = (start + timedelta(days=shift)).isoformat() + "T" + clock
                b.add(
                    "zi_ladder",
                    f"zi-{day}",
                    {"datetime": wall, "timezone": SHANGHAI},
                    Convention(sect).to_dict(),
                    "ok",
                    {
                        "hypotheses": ["H1", "H2"],
                        "h1": {
                            "boundary": "zi",
                            "clock": clock,
                            "seconds_from_boundary": offset,
                            "distance": distance,
                            "relevant_pillars": ["day", "hour"],
                        },
                    },
                )


def _timezone_pairs(b):
    for i, (base, zone) in enumerate(TIMEZONE_PAIRS):
        moment = datetime.fromisoformat(base)
        if _days_to_jie(moment) < 1:
            raise AssertionError(f"{base} is within a day of a Jie")
        local = moment.replace(tzinfo=UTC8).astimezone(load_timezone(zone)).replace(tzinfo=None)
        for z, wall in ((SHANGHAI, moment), (zone, local)):
            b.add(
                "timezone_pair",
                f"tz-{i + 1}",
                {"datetime": _wall(wall), "timezone": z},
                dict(PROFILE),
                "ok",
                {"hypotheses": ["H2"], "equivalent_instant_utc8": base + "+08:00"},
            )


def _dst_families(b):
    for zone, wall, fold_matters in FOLD_FAMILIES:
        group = f"fold-{zone.split('/')[-1]}-{wall[:10]}"
        control = _wall(datetime.fromisoformat(wall) - timedelta(days=7))
        members = (
            (
                {"datetime": wall, "timezone": zone},
                "ambiguous" if fold_matters else "ok",
                "no_fold",
            ),
            ({"datetime": wall, "timezone": zone, "fold": 0}, "ok", "fold_given"),
            ({"datetime": wall, "timezone": zone, "fold": 1}, "ok", "fold_given"),
            ({"datetime": control, "timezone": zone}, "ok", "no_fold"),
        )
        for inputs, intended, form in members:
            b.add(
                "dst_fold",
                group,
                inputs,
                dict(PROFILE),
                intended,
                {"hypotheses": ["H3", "H2"], "h3": _h3("fold", True, True, form)},
            )
    for zone, wall in GAP_FAMILIES:
        group = f"gap-{zone.split('/')[-1]}-{wall[:10]}"
        later = _wall(datetime.fromisoformat(wall) + timedelta(days=7))
        for inputs, intended in (
            ({"datetime": wall, "timezone": zone}, "invalid"),
            ({"datetime": later, "timezone": zone}, "ok"),
        ):
            b.add(
                "dst_gap",
                group,
                inputs,
                dict(PROFILE),
                intended,
                {"hypotheses": ["H3"], "h3": _h3("nonexistent_time", True, True, "wall_time")},
            )


def _sect_families(b):
    for day in SECT_DATES:
        start = date.fromisoformat(day)
        if _days_to_jie(datetime.combine(start, datetime.min.time())) < 2.5:
            raise AssertionError(f"Sect family {day} is within 2.5 days of a Jie")
        nxt = (start + timedelta(days=1)).isoformat()
        members = (
            (f"{day}T23:30:00", dict(NO_SECT), "ambiguous", "no_sect"),
            (f"{day}T23:30:00", Convention(1).to_dict(), "ok", "sect_given"),
            (f"{day}T23:30:00", Convention(2).to_dict(), "ok", "sect_given"),
            (f"{day}T22:30:00", dict(NO_SECT), "ok", "no_sect"),
            (f"{nxt}T00:30:00", dict(NO_SECT), "ok", "no_sect"),
        )
        for wall, convention, intended, form in members:
            b.add(
                "h3_sect",
                f"sect-{day}",
                {"datetime": wall, "timezone": SHANGHAI},
                convention,
                intended,
                {"hypotheses": ["H3"], "h3": _h3("sect", True, True, form)},
            )


def _minute_families(b):
    for term, year in MINUTE_EVENTS:
        event = jie_instant(term, year)
        if not 5 <= event.second <= 54 or year in CHINA_DST_YEARS:
            raise AssertionError(
                f"{term} {year} second {event.second} is too close to a minute edge"
            )
        minute = event.replace(second=0)
        for shift, intended in ((-1, "ok"), (0, "ambiguous"), (1, "ok")):
            wall = (minute + timedelta(minutes=shift)).isoformat(timespec="minutes")
            b.add(
                "h3_precision",
                f"minute-{year}-{term}",
                {"datetime": wall, "timezone": SHANGHAI},
                dict(PROFILE),
                intended,
                {
                    "hypotheses": ["H3", "H2"],
                    "h3": _h3("seconds", True, False, "minute_precision"),
                    "jie_utc8": event.replace(tzinfo=UTC8).isoformat(),
                },
            )
    for clock in ("22:59", "23:00"):
        b.add(
            "h3_precision",
            f"minute-zi-{MINUTE_ZI_DATE}",
            {"datetime": f"{MINUTE_ZI_DATE}T{clock}", "timezone": SHANGHAI},
            Convention(1).to_dict(),
            "ok",
            {"hypotheses": ["H3"], "h3": _h3("seconds", True, False, "minute_precision")},
        )


def _surface_families(b):
    for day in TIME_MISSING_DATES:
        for wall, intended, form in (
            (day, "ambiguous", "date_only"),
            (day + "T14:20:00", "ok", "complete"),
        ):
            b.add(
                "h3_missing",
                f"time-{day}",
                {"datetime": wall, "timezone": SHANGHAI},
                dict(PROFILE),
                intended,
                {"hypotheses": ["H3"], "h3": _h3("time_of_day", False, True, form)},
            )
    for day in ZONE_MISSING_DATES:
        for zone, intended, form in (
            (None, "ambiguous", "zone_missing"),
            (SHANGHAI, "ok", "complete"),
        ):
            b.add(
                "h3_missing",
                f"zone-{day}",
                {"datetime": day + "T09:20:00", "timezone": zone},
                dict(PROFILE),
                intended,
                {"hypotheses": ["H3"], "h3": _h3("timezone", False, False, form)},
            )
    for day, zone in DATETIME_MISSING:
        for wall, intended, form in (
            (None, "ambiguous", "datetime_missing"),
            (day + "T18:10:00", "ok", "complete"),
        ):
            b.add(
                "h3_missing",
                f"datetime-{day}",
                {"datetime": wall, "timezone": zone},
                dict(PROFILE),
                intended,
                {"hypotheses": ["H3"], "h3": _h3("datetime", False, False, form)},
            )


def _invalid_and_unsupported(b):
    families = (
        (
            "invalid_input",
            "feb29",
            "invalid_date",
            False,
            (
                ({"datetime": "2023-02-29T10:10:00", "timezone": SHANGHAI}, PROFILE, "invalid"),
                ({"datetime": "2024-02-29T10:10:00", "timezone": SHANGHAI}, PROFILE, "ok"),
            ),
        ),
        (
            "invalid_input",
            "apr31",
            "invalid_date",
            False,
            (
                ({"datetime": "2022-04-31T15:15:00", "timezone": SHANGHAI}, PROFILE, "invalid"),
                ({"datetime": "2022-04-30T15:15:00", "timezone": SHANGHAI}, PROFILE, "ok"),
            ),
        ),
        (
            "invalid_input",
            "hour24",
            "invalid_time",
            False,
            (
                ({"datetime": "2026-08-19T24:00:00", "timezone": SHANGHAI}, PROFILE, "invalid"),
                ({"datetime": "2026-08-20T00:00:00", "timezone": SHANGHAI}, PROFILE, "ok"),
            ),
        ),
        (
            "invalid_input",
            "zone-names",
            "invalid_zone",
            True,
            tuple(
                ({"datetime": "2005-03-18T11:45:00", "timezone": z}, PROFILE, s)
                for z, s in (
                    (SHANGHAI, "ok"),
                    ("Asia/Beijing", "invalid"),
                    ("CST", "invalid"),
                    ("asia/shanghai", "invalid"),
                )
            ),
        ),
        (
            "unsupported_input",
            "profiles",
            "unsupported_profile",
            False,
            (({"datetime": "2005-07-21T16:40:00", "timezone": SHANGHAI}, PROFILE, "ok"),)
            + tuple(
                (
                    {"datetime": "2005-07-21T16:40:00", "timezone": SHANGHAI},
                    {**PROFILE, field: value},
                    "unsupported",
                )
                for field, value in UNSUPPORTED_PROFILES
            ),
        ),
        (
            "unsupported_input",
            "range-low",
            "unsupported_year",
            False,
            (
                ({"datetime": "1900-07-07T12:30:00", "timezone": SHANGHAI}, PROFILE, "unsupported"),
                ({"datetime": "1901-07-07T12:30:00", "timezone": SHANGHAI}, PROFILE, "ok"),
            ),
        ),
        (
            "unsupported_input",
            "range-high",
            "unsupported_year",
            False,
            (
                ({"datetime": "2100-03-03T08:30:00", "timezone": SHANGHAI}, PROFILE, "unsupported"),
                ({"datetime": "2099-03-03T08:30:00", "timezone": SHANGHAI}, PROFILE, "ok"),
            ),
        ),
        # Repeated LMT->CST wall time whose two instants both precede 1901 (defect E4).
        (
            "unsupported_input",
            "lmt-1901",
            "unsupported_year",
            False,
            (
                ({"datetime": "1900-12-31T23:59:59", "timezone": SHANGHAI}, PROFILE, "unsupported"),
                ({"datetime": "1901-01-01T00:10:00", "timezone": SHANGHAI}, PROFILE, "ok"),
            ),
        ),
    )
    for category, group, kind, knowledge, members in families:
        for inputs, convention, intended in members:
            b.add(
                category,
                group,
                dict(inputs),
                dict(convention),
                intended,
                {"hypotheses": ["H3"], "h3": _h3(kind, knowledge, True, "complete")},
            )


def _ordinary(b, rng, count=12):
    made = 0
    while made < count:
        year = rng.randint(1950, 2049)
        moment = datetime(year, rng.randint(1, 12), rng.randint(1, 28)) + timedelta(
            seconds=rng.randint(2 * 3600, 21 * 3600 - 1)
        )
        clock = moment.hour * 3600 + moment.minute * 60 + moment.second
        # Hour branches change at odd hours; keep ordinary cases 10 minutes away.
        near_hour = min(abs(clock - h * 3600) for h in range(1, 24, 2))
        if (
            year in CHINA_DST_YEARS
            or moment.date().isoformat() in b.dates
            or _days_to_jie(moment) < 3
            or near_hour < 600
        ):
            continue
        made += 1
        b.add(
            "ordinary",
            f"ordinary-{made}",
            {"datetime": _wall(moment), "timezone": SHANGHAI},
            dict(PROFILE),
            "ok",
            {"hypotheses": ["H2"]},
        )


def generate_benchmark_v3(seed=SEED):
    b = _Builder(seed)
    _jie_ladders(b)
    _zi_ladders(b)
    _timezone_pairs(b)
    _dst_families(b)
    _sect_families(b)
    _minute_families(b)
    _surface_families(b)
    _invalid_and_unsupported(b)
    _ordinary(b, random.Random(seed))
    for i, case in enumerate(b.cases, 1):
        case["case_id"] = f"B3-{i:03d}"
    return b.cases


def pilot_case_ids_v3(cases):
    """Operational pilot: the first complete family of every category (not a random sample)."""
    first_group = {}
    for case in cases:
        first_group.setdefault(case["category"], case["group_id"])
    chosen = set(first_group.values())
    return [c["case_id"] for c in cases if c["group_id"] in chosen]
