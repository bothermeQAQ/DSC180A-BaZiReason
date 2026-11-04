"""Explicit denominator metrics; missing/malformed outputs count as failures."""

import math

from bazireason.conventions import PILLARS
from bazireason.schema import valid_response


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


def score_case(case, prediction):
    response = prediction.get("response") if isinstance(prediction, dict) else None
    valid = valid_response(response)
    expected = case["expected"]
    eligible = expected["status"] == "ok"
    status_match = bool(valid and response["status"] == expected["status"])
    convention_match = bool(valid and response["conventions"] == expected["conventions"])
    pillar_matches = (
        {
            p: bool(
                valid
                and response["status"] == "ok"
                and response["pillars"][p] == expected["pillars"][p]
            )
            for p in PILLARS
        }
        if eligible
        else None
    )
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
    supplied = prediction.get("supplied_context") if isinstance(prediction, dict) else None
    corrupted = bool(
        eligible and isinstance(prediction, dict) and prediction.get("context_corrupted")
    )
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "valid": valid,
        "present": isinstance(prediction, dict),
        "eligible": eligible,
        "expected_status": expected["status"],
        "group_id": case.get("group_id", case["case_id"]),
        "status_match": status_match,
        "convention_match": convention_match,
        "pillar_matches": pillar_matches,
        "exact": exact,
        "joint_exact": exact and convention_match,
        "coverage": bool(eligible and valid and response["status"] == "ok"),
        "definitive": definitive,
        "unsupported_definitive": bool(not eligible and definitive),
        "context_corrupted": corrupted,
        "corrupted_hour_followed": bool(
            corrupted
            and valid
            and response["status"] == "ok"
            and isinstance(supplied, dict)
            and response["pillars"]["hour"] == supplied.get("pillars", {}).get("hour")
        ),
        "uncertainty_correct": bool(not eligible and status_match and convention_match),
        "contradiction": contradiction,
    }


def aggregate(scores):
    n = len(scores)
    labeled = [s for s in scores if s["eligible"]]
    uncertain = [s for s in scores if not s["eligible"]]
    ambiguous = [s for s in uncertain if s["expected_status"] == "ambiguous"]
    corrupted = [s for s in scores if s["context_corrupted"]]
    exact = sum(s["exact"] for s in labeled)
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
        "exact_wilson95_descriptive": wilson(exact, len(labeled)),
        "valid_output_rate": ratio(sum(s["valid"] for s in scores), n),
        "status_accuracy": ratio(sum(s["status_match"] for s in scores), n),
        "convention_accuracy": ratio(sum(s["convention_match"] for s in scores), n),
        "uncertainty_accuracy": ratio(
            sum(s["uncertainty_correct"] for s in uncertain), len(uncertain)
        ),
        "unsupported_definitive_answers": sum(s["unsupported_definitive"] for s in uncertain),
        "unsupported_definitive_rate": ratio(
            sum(s["unsupported_definitive"] for s in uncertain), len(uncertain)
        ),
        "n_ambiguous": len(ambiguous),
        "ambiguous_definitive_rate": ratio(
            sum(s["unsupported_definitive"] for s in ambiguous), len(ambiguous)
        ),
        "n_corrupted_context": len(corrupted),
        "corrupted_hour_following_rate": ratio(
            sum(s["corrupted_hour_followed"] for s in corrupted), len(corrupted)
        ),
        "contradiction_rate": ratio(sum(s["contradiction"] for s in scores), n),
        "per_pillar_accuracy": {
            p: ratio(sum(s["pillar_matches"][p] for s in labeled), len(labeled)) for p in PILLARS
        },
    }
