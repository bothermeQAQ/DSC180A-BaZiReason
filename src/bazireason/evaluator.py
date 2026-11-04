"""Evaluate a complete declared experiment, retaining failures in denominators."""

import json
from collections import defaultdict

from bazireason.metrics import aggregate, ratio, score_case
from bazireason.schema import valid_response


def evaluate(cases, predictions, methods=("direct", "tool_context"), trials=1):
    if type(trials) is not int or trials < 1:
        raise ValueError("Trials must be a positive integer")
    case_ids = {c["case_id"] for c in cases}
    if len(case_ids) != len(cases):
        raise ValueError("Duplicate dataset case ID")
    indexed = {}
    for pred in predictions:
        key = (pred["case_id"], pred["method"], pred["trial"])
        if key in indexed:
            raise ValueError("Duplicate prediction")
        if key[0] not in case_ids or key[1] not in methods or not 0 <= key[2] < trials:
            raise ValueError("Unexpected prediction identity")
        indexed[key] = pred
    output = {
        "synthetic": any(p.get("synthetic") for p in predictions),
        "missing_predictions": len(cases) * len(methods) * trials - len(indexed),
        "methods": {},
    }
    per_case = {}
    for method in methods:
        scores = []
        categories = defaultdict(list)
        for trial in range(trials):
            for case in cases:
                score = score_case(case, indexed.get((case["case_id"], method, trial)))
                scores.append(score)
                categories[case["category"]].append(score)
                per_case[(case["case_id"], method, trial)] = score
        consistency_pairs = 0
        consistent = 0
        for case in cases:
            for a in range(trials):
                for b in range(a + 1, trials):
                    consistency_pairs += 1
                    pa = indexed.get((case["case_id"], method, a), {}).get("response")
                    pb = indexed.get((case["case_id"], method, b), {}).get("response")
                    consistent += bool(
                        valid_response(pa)
                        and valid_response(pb)
                        and json.dumps(pa, sort_keys=True) == json.dumps(pb, sort_keys=True)
                    )
        output["methods"][method] = {
            "overall": aggregate(scores),
            "by_category": {k: aggregate(v) for k, v in categories.items()},
            "consistency_pairs": consistency_pairs,
            "pairwise_response_consistency": ratio(consistent, consistency_pairs),
        }
    if set(methods) == {"direct", "tool_context"}:
        pairs = [
            (
                per_case[(c["case_id"], "direct", t)]["exact"],
                per_case[(c["case_id"], "tool_context", t)]["exact"],
            )
            for c in cases
            if c["expected"]["status"] == "ok"
            for t in range(trials)
        ]
        output["paired"] = {
            "n": len(pairs),
            "tool_only_correct": sum(not a and b for a, b in pairs),
            "direct_only_correct": sum(a and not b for a, b in pairs),
        }
    return output
