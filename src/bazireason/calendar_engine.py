"""A narrow, explicit wrapper around lunar-python, with timezone validation."""

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from importlib.metadata import version
from importlib.resources import files
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from lunar_python import Solar

from bazireason.conventions import PILLARS, Convention

UTC8 = timezone(timedelta(hours=8))


@lru_cache(maxsize=128)
def load_timezone(key: str) -> ZoneInfo:
    """Read the declared tzdata dependency, independent of host TZPATH/cache.

    reset_tzpath alone does not invalidate ZoneInfo's global constructor cache.
    Loading a package resource also avoids changing another application's
    timezone configuration when this module is imported.
    """
    if not isinstance(key, str) or any(part in ("", ".", "..") for part in key.split("/")):
        raise ZoneInfoNotFoundError("Timezone must be a normalized relative IANA key")
    try:
        resource = files("tzdata.zoneinfo").joinpath(*key.split("/"))
        with resource.open("rb") as stream:
            return ZoneInfo.from_file(stream, key=key)
    except (OSError, ValueError) as error:
        raise ZoneInfoNotFoundError(key) from error


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
        return _response("invalid", ["invalid_sect"])
    if any(
        not isinstance(value, str) or not value.strip()
        for name, value in selected.items()
        if name != "sect"
    ):
        return _response("invalid", ["invalid_convention"])
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
        zone = load_timezone(inputs["timezone"])
    except (ValueError, TypeError, ZoneInfoNotFoundError):
        return _response("invalid", ["invalid_date_or_timezone"], selected)
    candidates = {}
    for fold in (0, 1):
        local = wall.replace(tzinfo=zone, fold=fold)
        try:
            utc = local.astimezone(timezone.utc)
            back = utc.astimezone(zone)
        except OverflowError:
            return _response("unsupported", ["outside_study_range"], selected)
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
    try:
        ref = local.astimezone(UTC8)
    except OverflowError:
        return _response("unsupported", ["outside_study_range"], selected)
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
