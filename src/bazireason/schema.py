"""Strict structured response checks shared by mocks and live inference."""

from bazireason.conventions import CYCLE, PILLARS, Convention

STATUSES = {"ok", "ambiguous", "unsupported", "invalid"}
NONANSWERS = STATUSES - {"ok"}
CONVENTION_FIELDS = set(Convention().to_dict())


def valid_convention(value):
    """Check representation, leaving scientific profile support to scoring.

    sect may be null when the input omitted it (a determinable answer at a time
    the sect cannot affect).
    """
    return (
        isinstance(value, dict)
        and set(value) == CONVENTION_FIELDS
        and (value["sect"] is None or type(value["sect"]) is int)
        and all(isinstance(value[k], str) and value[k].strip() for k in value if k != "sect")
    )


def valid_reasons(reasons):
    return isinstance(reasons, list) and all(isinstance(x, str) and x.strip() for x in reasons)


def valid_response(value):
    if not isinstance(value, dict) or set(value) != {
        "status",
        "pillars",
        "reason_codes",
        "conventions",
    }:
        return False
    if not isinstance(value["status"], str) or value["status"] not in STATUSES:
        return False
    reasons = value["reason_codes"]
    if not valid_reasons(reasons):
        return False
    if value["conventions"] is not None and not valid_convention(value["conventions"]):
        return False
    p = value["pillars"]
    if value["status"] == "ok":
        return (
            isinstance(p, dict)
            and set(p) == set(PILLARS)
            and all(isinstance(x, str) and x in CYCLE for x in p.values())
            and not reasons
            and isinstance(value["conventions"], dict)
        )
    return p is None and bool(reasons)


def is_nonanswer(value, status=None):
    """A structurally clear nonanswer: non-ok status, null pillars, nonblank reasons.

    The conventions echo is deliberately not inspected, so abstention credit
    does not depend on reproducing a profile object exactly.
    """
    return (
        isinstance(value, dict)
        and isinstance(value.get("status"), str)
        and value.get("status") in NONANSWERS
        and (status is None or value.get("status") == status)
        and "pillars" in value
        and value["pillars"] is None
        and valid_reasons(value.get("reason_codes"))
        and bool(value.get("reason_codes"))
    )


def model_view(result):
    return {k: result[k] for k in ("status", "pillars", "reason_codes", "conventions")}
