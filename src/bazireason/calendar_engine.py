"""A narrow, explicit wrapper around lunar-python with strict input handling.

A request is answered only when every admissible completion of the supplied
information gives the same chart under the declared profile. The completions
considered are an unspecified sect (1 or 2), the fold of a repeated wall time,
and the unknown seconds of a minute-precision time. A date without a time of
day is never answerable, because the hour pillar is unknown.
"""

import bisect
import re
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from importlib.metadata import version
from importlib.resources import files
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from lunar_python import Solar

from bazireason.conventions import JIE, PILLARS, Convention

UTC8 = timezone(timedelta(hours=8))
FIRST_YEAR, LAST_YEAR = 1901, 2099
DAY_ANCHOR = date(2000, 1, 1)  # 戊午 in the continuous sexagenary day cycle
# [0-9] rather than \d: Python's \d also matches non-ASCII decimal digits.
_FORMATS = {
    "second": re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})"),
    "minute": re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2})"),
    "date": re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})"),
}


@lru_cache(maxsize=1)
def tzdata_keys() -> frozenset:
    """Exact zone keys listed by the pinned tzdata package (case-sensitive)."""
    text = files("tzdata").joinpath("zones").read_text(encoding="utf-8")
    return frozenset(line.strip() for line in text.splitlines() if line.strip())


@lru_cache(maxsize=128)
def load_timezone(key: str) -> ZoneInfo:
    """Read the declared tzdata dependency, independent of host TZPATH/cache.

    reset_tzpath alone does not invalidate ZoneInfo's global constructor cache.
    Loading a package resource also avoids changing another application's
    timezone configuration. Checking tzdata's own key list first makes
    validation case-sensitive on every filesystem: on case-insensitive macOS
    volumes, "asia/shanghai" would otherwise open the Asia/Shanghai file.
    """
    if not isinstance(key, str) or key not in tzdata_keys():
        raise ZoneInfoNotFoundError(f"Not an exact tzdata zone key: {key!r}")
    try:
        resource = files("tzdata.zoneinfo").joinpath(*key.split("/"))
        with resource.open("rb") as stream:
            return ZoneInfo.from_file(stream, key=key)
    except (OSError, ValueError) as error:
        raise ZoneInfoNotFoundError(key) from error


def parse_wall_time(raw: str):
    """Return (precision, first, last) for the three accepted wall-time formats.

    Accepted: "YYYY-MM-DDTHH:MM:SS", "YYYY-MM-DDTHH:MM" and "YYYY-MM-DD".
    Returns None for every other string (offsets, fractions, spaces, week
    dates). Raises ValueError when a well-formed string names an impossible
    date or time such as 24:00:00. The result never depends on the Python
    version's ISO parser.
    """
    for precision, pattern in _FORMATS.items():
        match = pattern.fullmatch(raw)
        if match:
            first = datetime(*(int(part) for part in match.groups()))
            last = {
                "second": first,
                "minute": first.replace(second=59),
                "date": first.replace(hour=23, minute=59, second=59),
            }[precision]
            return precision, first, last
    return None


def _response(status, reasons, convention=None, pillars=None, normalized=None, candidates=None):
    return {
        "status": status,
        "pillars": pillars,
        "reason_codes": reasons,
        "conventions": convention,
        "normalized_datetime": normalized,
        "normalized_candidates": candidates,
        "provenance": {
            "engine": "lunar-python",
            "version": version("lunar-python"),
            "tzdata_version": version("tzdata"),
            "label_status": "single_engine_reference_not_independent_truth",
        },
    }


def chart_at(ref: datetime, sect: int) -> dict:
    """Four pillars for an aware UTC+08:00 instant under one sect."""
    eight = (
        Solar.fromYmdHms(ref.year, ref.month, ref.day, ref.hour, ref.minute, ref.second)
        .getLunar()
        .getEightChar()
    )
    eight.setSect(sect)
    values = (eight.getYear(), eight.getMonth(), eight.getDay(), eight.getTime())
    return dict(zip(PILLARS, values, strict=True))


def _varies(outcomes, axis):
    """True if changing only key component `axis` changes some outcome."""
    groups = {}
    for key, value in outcomes.items():
        rest = key[:axis] + key[axis + 1 :]
        groups.setdefault(rest, []).append(value)
    return any(any(v != values[0] for v in values) for values in groups.values())


def _completions(wall_times, zone):
    """Map fold -> local instants, keeping a fold only if every wall time exists in it."""
    by_fold = {}
    for fold in (0, 1):
        found = []
        for wall in wall_times:
            local = wall.replace(tzinfo=zone, fold=fold)
            back = local.astimezone(timezone.utc).astimezone(zone)
            if back.replace(tzinfo=None) == wall and back.fold == fold:
                found.append(local)
        if len(found) == len(wall_times):
            by_fold[fold] = found
    return by_fold


def _supported(local):
    try:
        ref = local.astimezone(UTC8)
    except OverflowError:
        return None
    return ref if FIRST_YEAR <= ref.year <= LAST_YEAR else None


def compute_chart(inputs: dict, convention: dict | None = None) -> dict:
    """Return ok, ambiguous, unsupported or invalid without guessing missing data.

    The zone must be an exact tzdata key. Optional fold (0 or 1) selects one
    instant of a repeated wall time. A convention without "sect" leaves the sect
    unspecified; it is not silently defaulted.
    """
    if not isinstance(inputs, dict):
        return _response("invalid", ["input_must_be_object"])
    if convention is not None and not isinstance(convention, dict):
        return _response("invalid", ["convention_must_be_object"])
    convention = {} if convention is None else convention
    sect_given = "sect" in convention
    try:
        profile = Convention(**convention)
    except (TypeError, ValueError):
        return _response("invalid", ["invalid_convention"])
    selected = profile.to_dict()
    if not sect_given:
        selected["sect"] = None
    elif type(profile.sect) is not int or profile.sect not in (1, 2):
        return _response("invalid", ["invalid_sect"])
    if any(
        not isinstance(value, str) or not value.strip()
        for name, value in selected.items()
        if name != "sect"
    ):
        return _response("invalid", ["invalid_convention"])
    # A nonanswer repeats the profile only when the input fully specified one.
    shown = selected if sect_given else None
    if selected != {**Convention().to_dict(), "sect": selected["sect"]}:
        return _response("unsupported", ["unsupported_convention"], shown)
    raw, key = inputs.get("datetime"), inputs.get("timezone")
    if raw is None or raw == "" or key is None or key == "":
        reasons = ["missing_datetime_or_timezone"] + ([] if sect_given else ["missing_convention"])
        return _response("ambiguous", reasons, shown)
    if not isinstance(raw, str):
        return _response("invalid", ["datetime_must_be_string"], shown)
    if not isinstance(key, str):
        return _response("invalid", ["timezone_must_be_string"], shown)
    try:
        parsed = parse_wall_time(raw)
    except ValueError:
        return _response("invalid", ["invalid_date_or_timezone"], shown)
    if parsed is None:
        return _response("invalid", ["invalid_datetime_format"], shown)
    try:
        zone = load_timezone(key)
    except ZoneInfoNotFoundError:
        return _response("invalid", ["invalid_date_or_timezone"], shown)
    precision, first, last = parsed
    fold = inputs.get("fold")
    if precision == "date":
        if fold is not None and (type(fold) is not int or fold not in (0, 1)):
            return _response("invalid", ["invalid_fold"], shown)
        existing = []
        try:
            for wall in (first, first.replace(hour=12), last):
                for found in _completions((wall,), zone).values():
                    existing.extend(found)
        except OverflowError:
            return _response("unsupported", ["outside_study_range"], shown)
        if not existing:
            return _response("invalid", ["nonexistent_civil_time"], shown)
        if not any(_supported(local) for local in existing):
            return _response("unsupported", ["outside_study_range"], shown)
        reasons = ([] if sect_given else ["missing_convention"]) + ["full_wall_time_required"]
        return _response("ambiguous", reasons, shown)
    wall_times = (first,) if first == last else (first, last)
    try:
        by_fold = _completions(wall_times, zone)
    except OverflowError:
        return _response("unsupported", ["outside_study_range"], shown)
    if not by_fold:
        return _response("invalid", ["nonexistent_civil_time"], shown)
    if fold is not None and (type(fold) is not int or fold not in (0, 1)):
        return _response("invalid", ["invalid_fold"], shown)
    if fold is not None and fold not in by_fold:
        return _response("invalid", ["fold_not_applicable"], shown)
    folds = [fold] if fold is not None else sorted(by_fold)
    sects = [profile.sect] if sect_given else [1, 2]
    # Range applies to every completion first (an LMT fold can lie wholly before 1901).
    outcomes, instants = {}, set()
    for f in folds:
        for wall, local in zip(wall_times, by_fold[f], strict=True):
            ref = _supported(local)
            if ref is not None:
                instants.add(ref)
            for s in sects:
                outcomes[(f, wall, s)] = chart_at(ref, s) if ref is not None else None
    if not instants:
        return _response("unsupported", ["outside_study_range"], shown)
    candidates = sorted(ref.isoformat() for ref in instants)
    values = list(outcomes.values())
    if all(value == values[0] for value in values):
        normalized = candidates[0] if len(candidates) == 1 else None
        return _response("ok", [], selected, values[0], normalized, candidates)
    reasons = []
    if not sect_given and _varies(outcomes, 2):
        reasons.append("missing_convention")
    if len(folds) == 2 and _varies(outcomes, 0):
        reasons.append("dst_fold_required")
    if len(wall_times) == 2 and _varies(outcomes, 1):
        reasons.append("seconds_required")
    return _response("ambiguous", reasons, shown, candidates=candidates)


@lru_cache(maxsize=512)
def _jie_of_table(year):
    table = Solar.fromYmdHms(year, 6, 15, 12, 0, 0).getLunar().getJieQiTable()
    return tuple(
        (datetime.fromisoformat(solar.toYmdHms()).replace(tzinfo=UTC8), name)
        for name, solar in table.items()
        if name in JIE
    )


def jie_around(ref: datetime):
    """(previous, next) month-start Jie around an aware UTC+08:00 instant.

    A Jie at exactly `ref` counts as previous: the new month starts at its instant.
    """
    events = sorted(
        set().union(*(_jie_of_table(y) for y in (ref.year - 1, ref.year, ref.year + 1)))
    )
    index = bisect.bisect_right([moment for moment, _ in events], ref)
    return events[index - 1], events[index]


def temporal_facts(inputs: dict) -> dict:
    """Intermediate facts for the tool_intermediate condition; never pillars.

    Computed from the input alone, never from the convention or a stored label:
    admissible UTC+08:00 instants, the surrounding month-start Jie and a day
    count from 2000-01-01. The model must still apply every pillar rule.
    """
    facts = {
        "reference_clock": "UTC+08:00",
        "input_precision": None,
        "repeated_wall_time": False,
        "candidates": [],
        "normalization_error": None,
    }
    raw = inputs.get("datetime") if isinstance(inputs, dict) else None
    key = inputs.get("timezone") if isinstance(inputs, dict) else None
    if raw is None or raw == "" or key is None or key == "":
        facts["normalization_error"] = "missing_datetime_or_timezone"
        return facts
    try:
        parsed = parse_wall_time(raw) if isinstance(raw, str) else None
    except ValueError:
        facts["normalization_error"] = "impossible_date_or_time"
        return facts
    if parsed is None:
        facts["normalization_error"] = "unparseable_datetime"
        return facts
    try:
        zone = load_timezone(key) if isinstance(key, str) else None
    except ZoneInfoNotFoundError:
        zone = None
    if zone is None:
        facts["normalization_error"] = "unknown_timezone"
        return facts
    precision, first, last = parsed
    facts["input_precision"] = precision
    wall_times = (first,) if first == last else (first, last)
    fold = inputs.get("fold")
    try:
        if precision == "date":
            # Report the span of the civil day; any existing instants of it count.
            by_fold = {0: [first.replace(tzinfo=zone), last.replace(tzinfo=zone)]}
        else:
            by_fold = _completions(wall_times, zone)
            facts["repeated_wall_time"] = len(by_fold) == 2
            if fold in (0, 1) and type(fold) is int and fold in by_fold:
                by_fold = {fold: by_fold[fold]}
        for f, locals_ in sorted(by_fold.items()):
            start, end = locals_[0].astimezone(UTC8), locals_[-1].astimezone(UTC8)
            (prev_t, prev_name), (next_t, next_name) = jie_around(start)
            entry = {
                "utc8_start": start.isoformat(),
                "utc8_end": end.isoformat(),
                "civil_date_utc8": start.date().isoformat(),
                "days_since_2000_01_01": (start.date() - DAY_ANCHOR).days,
                "previous_jie": {"name": prev_name, "utc8": prev_t.isoformat()},
                "next_jie": {"name": next_name, "utc8": next_t.isoformat()},
            }
            if facts["repeated_wall_time"]:
                entry["fold"] = f
            facts["candidates"].append(entry)
    except (OverflowError, ValueError):
        facts["candidates"] = []
        facts["normalization_error"] = "conversion_overflow"
        return facts
    if not facts["candidates"]:
        facts["normalization_error"] = "nonexistent_wall_time"
    return facts
