"""Explicit denominator metrics; missing/malformed outputs count as failures."""

import math
from collections import defaultdict

from bazireason.conventions import PILLARS
from bazireason.schema import is_nonanswer, valid_response


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def wilson(successes, n):
    """Descriptive 95% Wilson interval, not a clustered hypothesis test."""
    if not n:
        return None
    z = 1.959963984540054
    p = successes / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [center - width, center + width]


def _pillar_matches(response, expected):
    valid = valid_response(response)
    return {
        p: bool(valid and response["status"] == "ok" and response["pillars"][p] == expected[p])
        for p in PILLARS
    }


def score_case(case, prediction, corruption=None):
    """Score one declared observation.

    `corruption` describes the injected error for a tool_corrupted observation.
    The evaluator derives it from the design and the case input, never from the
    prediction record, so a missing or self-describing record cannot leave the
    denominator.
    """
    present = isinstance(prediction, dict)
    response = prediction.get("response") if present else None
    valid = valid_response(response)
    expected = case["expected"]
    eligible = expected["status"] == "ok"
    status_match = bool(valid and response["status"] == expected["status"])
    convention_match = bool(valid and response["conventions"] == expected["conventions"])
    pillar_matches = _pillar_matches(response, expected["pillars"]) if eligible else None
    # A non-ok response carrying definitive pillars is a detectable contradiction.
    contradiction = bool(
        isinstance(response, dict)
        and response.get("status") != "ok"
        and response.get("pillars") is not None
    )
    # Inspect even schema-invalid outputs: a missing field must not hide an unsafe answer.
    definitive = bool(
        isinstance(response, dict)
        and (
            response.get("status") == "ok"
            or isinstance(response.get("pillars"), dict)
            and bool(response["pillars"])
        )
    )
    exact = bool(eligible and pillar_matches and all(pillar_matches.values()))
    answered = bool(eligible and valid and response["status"] == "ok")
    policy_correct = bool(not eligible and is_nonanswer(response, expected["status"]))
    followed = bool(
        corruption is not None
        and valid
        and response["status"] == "ok"
        and all(response["pillars"][f] == v for f, v in corruption["fields"].items())
    )
    recovered = prediction.get("recovered_response") if present else None
    lenient = response if response is not None else recovered
    lenient_matches = _pillar_matches(lenient, expected["pillars"]) if eligible else None
    error = prediction.get("error") if present else None
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "valid": valid,
        "present": present,
        "eligible": eligible,
        "expected_status": expected["status"],
        "group_id": case.get("group_id", case["case_id"]),
        "status_match": status_match,
        "convention_match": convention_match,
        "pillar_matches": pillar_matches,
        "exact": exact,
        "joint_exact": exact and convention_match,
        "coverage": answered,
        # A clear refusal (any non-ok status) of a request that has a determinate answer.
        "false_refusal": bool(eligible and is_nonanswer(response)),
        "definitive": definitive,
        "unsupported_definitive": bool(not eligible and definitive),
        "uncertainty_correct": policy_correct,
        "uncertainty_joint_correct": bool(not eligible and status_match and convention_match),
        "decision_correct": answered if eligible else policy_correct,
        "corruption_type": corruption["type"] if corruption else None,
        "corruption_followed": followed,
        "contradiction": contradiction,
        "error": error if present else "missing",
        "recovered": bool(response is None and recovered is not None),
        "lenient_valid": valid_response(lenient),
        "lenient_exact": bool(eligible and lenient_matches and all(lenient_matches.values())),
    }


def aggregate(scores):
    n = len(scores)
    labeled = [s for s in scores if s["eligible"]]
    uncertain = [s for s in scores if not s["eligible"]]
    ambiguous = [s for s in uncertain if s["expected_status"] == "ambiguous"]
    corrupted = [s for s in labeled if s["corruption_type"] is not None]
    by_type = defaultdict(list)
    for s in scores:
        if s["corruption_type"] is not None:
            by_type[s["corruption_type"]].append(s)
    exact = sum(s["exact"] for s in labeled)
    errors = [s["error"] for s in scores]
    return {
        "n": n,
        "n_present": sum(s["present"] for s in scores),
        "n_missing": sum(not s["present"] for s in scores),
        "n_labeled": len(labeled),
        "n_nonanswer": len(uncertain),
        "exact_matches": exact,
        "exact_accuracy": ratio(exact, len(labeled)),
        "joint_exact_accuracy": ratio(sum(s["joint_exact"] for s in labeled), len(labeled)),
        "scorable_coverage": ratio(sum(s["coverage"] for s in labeled), len(labeled)),
        "false_refusal_rate": ratio(sum(s["false_refusal"] for s in labeled), len(labeled)),
        "exact_wilson95_descriptive": wilson(exact, len(labeled)),
        "valid_output_rate": ratio(sum(s["valid"] for s in scores), n),
        "status_accuracy": ratio(sum(s["status_match"] for s in scores), n),
        "convention_accuracy": ratio(sum(s["convention_match"] for s in scores), n),
        "decision_accuracy": ratio(sum(s["decision_correct"] for s in scores), n),
        "uncertainty_accuracy": ratio(
            sum(s["uncertainty_correct"] for s in uncertain), len(uncertain)
        ),
        "uncertainty_joint_accuracy": ratio(
            sum(s["uncertainty_joint_correct"] for s in uncertain), len(uncertain)
        ),
        "unsupported_definitive_answers": sum(s["unsupported_definitive"] for s in uncertain),
        "unsupported_definitive_rate": ratio(
            sum(s["unsupported_definitive"] for s in uncertain), len(uncertain)
        ),
        "n_ambiguous": len(ambiguous),
        "ambiguous_definitive_rate": ratio(
            sum(s["unsupported_definitive"] for s in ambiguous), len(ambiguous)
        ),
        "ambiguous_policy_accuracy": ratio(
            sum(s["uncertainty_correct"] for s in ambiguous), len(ambiguous)
        ),
        "n_corrupted_context": len(corrupted),
        "n_corrupted_missing": sum(not s["present"] for s in corrupted),
        "corruption_following_rate": ratio(
            sum(s["corruption_followed"] for s in corrupted), len(corrupted)
        ),
        "corruption_following_by_type": {
            kind: {
                "n": len(rows),
                "n_missing": sum(not s["present"] for s in rows),
                "following_rate": ratio(sum(s["corruption_followed"] for s in rows), len(rows)),
            }
            for kind, rows in sorted(by_type.items())
        },
        "contradiction_rate": ratio(sum(s["contradiction"] for s in scores), n),
        "per_pillar_accuracy": {
            p: ratio(sum(s["pillar_matches"][p] for s in labeled), len(labeled)) for p in PILLARS
        },
        "format_failures": {
            "missing": errors.count("missing"),
            "invalid_json": errors.count("invalid_json"),
            "invalid_schema": errors.count("invalid_schema"),
            "call_errors": sum(
                e not in (None, "missing", "invalid_json", "invalid_schema") for e in errors
            ),
        },
        "lenient_parse_secondary": {
            "note": "Secondary only: strict parsing defines every primary metric.",
            "n_recovered": sum(s["recovered"] for s in scores),
            "valid_output_rate": ratio(sum(s["lenient_valid"] for s in scores), n),
            "exact_accuracy": ratio(sum(s["lenient_exact"] for s in labeled), len(labeled)),
        },
    }
