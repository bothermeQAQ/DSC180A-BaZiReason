"""Strict structured response checks shared by mocks and live inference."""

from bazireason.conventions import CYCLE, PILLARS

STATUSES = {"ok", "ambiguous", "unsupported", "invalid"}


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
    if not isinstance(reasons, list) or any(not isinstance(x, str) for x in reasons):
        return False
    if value["conventions"] is not None and not isinstance(value["conventions"], dict):
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
