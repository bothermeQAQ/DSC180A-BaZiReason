"""Deterministic experimental controls: corrupted tool answers and zero-call baselines.

Nothing here reads a stored label. Corruptions are computed from the case input
and a fresh engine result; heuristic baselines use surface features only.
"""

import copy
import re
from datetime import datetime, timedelta

from bazireason.calendar_engine import chart_at, compute_chart, jie_around, parse_wall_time
from bazireason.conventions import CYCLE, Convention
from bazireason.schema import model_view

CORRUPTION_SCHEMES = ("v2", "v3")
_NEAR_JIE = timedelta(days=1)
_ZI_WINDOW_HOURS = (21, 22, 23, 0, 1)


def _first_instant(context):
    value = (
        context.get("normalized_datetime") or (context.get("normalized_candidates") or [None])[0]
    )
    return datetime.fromisoformat(value)


def corrupt_context(case, context, scheme="v3"):
    """Return (supplied context, corruption or None) for the tool_corrupted condition.

    v3 errors, chosen from the input instant only:
      * jie_side: within one day of a Jie, month (and year at 立春) from the
        other side of that Jie, a plausible boundary bug;
      * day_rollover: from 21:00 to 01:59, the day pillar off by one toward the
        other side of the 23:00/00:00 boundary;
      * hour_stem: otherwise the hour stem shifted by two (cycle index + 12),
        which the five-rats rule exposes as internally inconsistent;
      * default_guess: for an ambiguous request, a confident answer obtained by
        silently defaulting the missing information (sect 2, fold 0, :00 seconds,
        12:00:00, or Asia/Shanghai).
    Invalid and unsupported requests are not corrupted. Scheme v2 reproduces the
    historical audit_v2 control: hour stem only, scorable cases only.
    """
    if scheme not in CORRUPTION_SCHEMES:
        raise ValueError("Unknown corruption scheme")
    supplied = copy.deepcopy(model_view(context))
    if context["status"] == "ok":
        pillars = supplied["pillars"]
        ref = _first_instant(context)
        kind, other = "hour_stem", {"hour": CYCLE[(CYCLE.index(pillars["hour"]) + 12) % 60]}
        if scheme == "v3":
            sect = context["conventions"]["sect"] or 2
            (prev_t, prev_name), (next_t, next_name) = jie_around(ref)
            if min(ref - prev_t, next_t - ref) <= _NEAR_JIE:
                after = ref - prev_t <= next_t - ref
                moment, name = (prev_t, prev_name) if after else (next_t, next_name)
                side = chart_at(moment - timedelta(seconds=1) if after else moment, sect)
                fields = ("year", "month") if name == "立春" else ("month",)
                kind, other = "jie_side", {f: side[f] for f in fields}
            elif ref.hour in _ZI_WINDOW_HOURS:
                advanced = ref.hour <= 1 or (ref.hour == 23 and sect == 1)
                shift = -1 if advanced else 1
                kind = "day_rollover"
                other = {"day": CYCLE[(CYCLE.index(pillars["day"]) + shift) % 60]}
        if any(pillars[f] == value for f, value in other.items()):
            raise AssertionError("Corruption must change every designated field")
        pillars.update(other)
        return supplied, {"type": kind, "fields": dict(other)}
    if scheme == "v3" and context["status"] == "ambiguous":
        guess = _default_guess(case, context["reason_codes"])
        if guess is not None:
            return model_view(guess), {"type": "default_guess", "fields": dict(guess["pillars"])}
    return supplied, None


def _default_guess(case, reasons):
    inputs = dict(case["input"])
    convention = dict(case["convention"] or {})
    if "missing_convention" in reasons:
        convention["sect"] = 2
    if "dst_fold_required" in reasons:
        inputs["fold"] = 0
    if "seconds_required" in reasons:
        inputs["datetime"] += ":00"
    if "full_wall_time_required" in reasons:
        inputs["datetime"] += "T12:00:00"
    if "missing_datetime_or_timezone" in reasons:
        if inputs.get("datetime") in (None, ""):
            return None
        if inputs.get("timezone") in (None, ""):
            inputs["timezone"] = "Asia/Shanghai"
    guess = compute_chart(inputs, {**Convention().to_dict(), **convention})
    return guess if guess["status"] == "ok" else None


# ---- Zero-call baselines -------------------------------------------------
_CONSTANT = dict.fromkeys(("year", "month", "day", "hour"), "甲子")
_SECOND = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}")
_PARTIAL = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}(T[0-9]{2}:[0-9]{2})?")
_REGIONS = (
    "Africa/",
    "America/",
    "Antarctica/",
    "Asia/",
    "Atlantic/",
    "Australia/",
    "Europe/",
    "Indian/",
    "Pacific/",
    "Etc/",
)


def _response(status, reasons=(), conventions=None, pillars=None):
    return {
        "status": status,
        "pillars": pillars,
        "reason_codes": list(reasons),
        "conventions": conventions,
    }


def heuristic_surface(case):
    """Zero-knowledge floor: surface rules, no sexagenary, Jie or timezone data.

    It validates Gregorian dates, reads field presence and string shapes, and
    abstains whenever a sect, fold-free precision or time of day looks missing.
    It never computes pillars: every answer is the constant chart 甲子 x 4.
    """
    inputs, convention = case["input"], case["convention"]
    defaults = Convention().to_dict()
    if not isinstance(convention, dict) or set(convention) - set(defaults):
        return _response("invalid", ["invalid_convention"])
    if any(convention.get(k, v) != v for k, v in defaults.items() if k != "sect"):
        return _response("unsupported", ["unsupported_convention"])
    if "sect" not in convention:
        return _response("ambiguous", ["missing_convention"])
    if convention["sect"] not in (1, 2) or type(convention["sect"]) is not int:
        return _response("invalid", ["invalid_sect"])
    profile = {**defaults, **convention}
    raw, zone = inputs.get("datetime"), inputs.get("timezone")
    if raw in (None, "") or zone in (None, ""):
        return _response("ambiguous", ["missing_field"], profile)
    if not isinstance(raw, str) or not isinstance(zone, str):
        return _response("invalid", ["invalid_type"], profile)
    if not _SECOND.fullmatch(raw):
        if _PARTIAL.fullmatch(raw):
            return _response("ambiguous", ["full_time_required"], profile)
        return _response("invalid", ["invalid_datetime_format"], profile)
    try:
        parse_wall_time(raw)  # Gregorian validity only, identical on every Python version
    except ValueError:
        return _response("invalid", ["invalid_date"], profile)
    if not 1901 <= int(raw[:4]) <= 2099:
        return _response("unsupported", ["outside_study_range"], profile)
    if zone != "UTC" and not zone.startswith(_REGIONS):
        return _response("invalid", ["invalid_timezone"], profile)
    fold = inputs.get("fold")
    if fold is not None and (type(fold) is not int or fold not in (0, 1)):
        return _response("invalid", ["invalid_fold"], profile)
    return _response("ok", [], profile, dict(_CONSTANT))


def always_abstain(case):
    return _response("ambiguous", ["clarification_requested"])


def always_answer(case):
    convention = case["convention"] if isinstance(case["convention"], dict) else {}
    profile = {**Convention().to_dict(), "sect": None, **convention}
    return _response("ok", [], profile, dict(_CONSTANT))


BASELINES = {
    "heuristic_surface": heuristic_surface,
    "always_abstain": always_abstain,
    "always_answer": always_answer,
}
