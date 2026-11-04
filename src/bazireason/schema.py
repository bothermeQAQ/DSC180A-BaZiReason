"""Strict structured response checks shared by mocks and live inference."""

from bazireason.conventions import CYCLE, PILLARS, Convention

STATUSES = {"ok", "ambiguous", "unsupported", "invalid"}
CONVENTION_FIELDS = set(Convention().to_dict())


def valid_convention(value):
    """Check representation, leaving scientific profile support to scoring."""
    return (
        isinstance(value, dict)
        and set(value) == CONVENTION_FIELDS
        and type(value["sect"]) is int
        and all(isinstance(value[k], str) and value[k].strip() for k in value if k != "sect")
    )


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
    if not isinstance(reasons, list) or any(
        not isinstance(x, str) or not x.strip() for x in reasons
    ):
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


def model_view(result):
    return {k: result[k] for k in ("status", "pillars", "reason_codes", "conventions")}
