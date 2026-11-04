"""A narrow, explicit wrapper around lunar-python, with timezone validation."""

import os
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, reset_tzpath

from lunar_python import Solar

from bazireason.conventions import PILLARS, Convention

# Force the pinned tzdata wheel instead of a machine-specific system tz database.
# This module is intended for this research application, not embedded applications.
os.environ["PYTHONTZPATH"] = ""
reset_tzpath(())
UTC8 = timezone(timedelta(hours=8))


def _response(status, reasons, convention=None, pillars=None, normalized=None):
    return {
        "status": status,
        "pillars": pillars,
        "reason_codes": reasons,
        "conventions": convention,
        "normalized_datetime": normalized,
        "provenance": {
            "engine": "lunar-python",
            "version": version("lunar-python"),
            "tzdata_version": version("tzdata"),
            "label_status": "single_engine_reference_not_independent_truth",
        },
    }


def compute_chart(inputs: dict, convention: dict | None = None) -> dict:
    """Return ok, ambiguous, unsupported or invalid without guessing missing data.

    Input datetime is a naive ISO local wall time with explicit seconds. The zone
    is an IANA identifier. Optional fold disambiguates a repeated civil time.
    """
    if not isinstance(inputs, dict):
        return _response("invalid", ["input_must_be_object"])
    if convention is not None and not isinstance(convention, dict):
        return _response("invalid", ["convention_must_be_object"])
    if convention is None or "sect" not in convention:
        return _response("ambiguous", ["missing_convention"])
    try:
        profile = Convention(**convention)
    except (TypeError, ValueError):
        return _response("invalid", ["invalid_convention"])
    selected = profile.to_dict()
    if type(profile.sect) is not int or profile.sect not in (1, 2):
        return _response("invalid", ["invalid_sect"], selected)
    if selected != Convention(sect=profile.sect).to_dict():
        return _response("unsupported", ["unsupported_convention"], selected)
    if not inputs.get("datetime") or not inputs.get("timezone"):
        return _response("ambiguous", ["missing_datetime_or_timezone"], selected)
    try:
        raw = inputs["datetime"]
        if not isinstance(raw, str) or "T" not in raw or len(raw) != 19:
            return _response("ambiguous", ["full_wall_time_required"], selected)
        wall = datetime.fromisoformat(raw)
        if wall.tzinfo is not None or wall.microsecond:
            return _response("invalid", ["naive_seconds_required"], selected)
        zone = ZoneInfo(inputs["timezone"])
    except (ValueError, TypeError, ZoneInfoNotFoundError):
        return _response("invalid", ["invalid_date_or_timezone"], selected)
    candidates = {}
    for fold in (0, 1):
        local = wall.replace(tzinfo=zone, fold=fold)
        utc = local.astimezone(timezone.utc)
        back = utc.astimezone(zone)
        if back.replace(tzinfo=None) == wall and back.fold == fold:
            candidates[fold] = local
    if not candidates:
        return _response("invalid", ["nonexistent_civil_time"], selected)
    fold = inputs.get("fold")
    if fold is not None and (type(fold) is not int or fold not in (0, 1)):
        return _response("invalid", ["invalid_fold"], selected)
    if len(candidates) == 2 and fold is None:
        return _response("ambiguous", ["dst_fold_required"], selected)
    if fold is not None and fold not in candidates:
        return _response("invalid", ["fold_not_applicable"], selected)
    local = candidates[fold if fold is not None else next(iter(candidates))]
    ref = local.astimezone(UTC8)
    if not 1901 <= ref.year <= 2099:
        return _response("unsupported", ["outside_study_range"], selected)
    eight = (
        Solar.fromYmdHms(ref.year, ref.month, ref.day, ref.hour, ref.minute, ref.second)
        .getLunar()
        .getEightChar()
    )
    eight.setSect(profile.sect)
    values = (eight.getYear(), eight.getMonth(), eight.getDay(), eight.getTime())
    return _response("ok", [], selected, dict(zip(PILLARS, values, strict=True)), ref.isoformat())
