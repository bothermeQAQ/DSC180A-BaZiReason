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
    response = prediction.get("response") if prediction else None
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
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "valid": valid,
        "eligible": eligible,
        "status_match": status_match,
        "convention_match": convention_match,
        "pillar_matches": pillar_matches,
        "exact": bool(eligible and pillar_matches and all(pillar_matches.values())),
        "uncertainty_correct": bool(not eligible and status_match and convention_match),
        "contradiction": contradiction,
    }


def aggregate(scores):
    n = len(scores)
    labeled = [s for s in scores if s["eligible"]]
    uncertain = [s for s in scores if not s["eligible"]]
    exact = sum(s["exact"] for s in labeled)
    return {
        "n": n,
        "n_labeled": len(labeled),
        "n_nonanswer": len(uncertain),
        "exact_matches": exact,
        "exact_accuracy": ratio(exact, len(labeled)),
        "exact_wilson95_descriptive": wilson(exact, len(labeled)),
        "valid_output_rate": ratio(sum(s["valid"] for s in scores), n),
        "status_accuracy": ratio(sum(s["status_match"] for s in scores), n),
        "convention_accuracy": ratio(sum(s["convention_match"] for s in scores), n),
        "uncertainty_accuracy": ratio(
            sum(s["uncertainty_correct"] for s in uncertain), len(uncertain)
        ),
        "contradiction_rate": ratio(sum(s["contradiction"] for s in scores), n),
        "per_pillar_accuracy": {
            p: ratio(sum(s["pillar_matches"][p] for s in labeled), len(labeled)) for p in PILLARS
        },
    }
