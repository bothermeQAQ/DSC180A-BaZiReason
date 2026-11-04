"""Evaluate a complete declared experiment, retaining failures in denominators."""

import json
import random
from collections import defaultdict

from bazireason.metrics import aggregate, ratio, score_case
from bazireason.schema import valid_response


def paired_estimate(rows, resamples=1000, seed=1802026):
    """Paired observation mean; resample whole group IDs, preserving within-group trials.

    Rows are (group_id, treatment_outcome - comparator_outcome). The percentile
    interval is descriptive and conditions on this small, synthetic case suite.
    """
    groups = defaultdict(list)
    for group, delta in rows:
        groups[group].append(delta)
    result = {
        "n": len(rows),
        "n_groups": len(groups),
        "mean_difference": ratio(sum(delta for _, delta in rows), len(rows)),
        "group_bootstrap95_descriptive": None,
        "bootstrap_samples": resamples,
        "bootstrap_seed": seed,
    }
    if len(groups) < 2 or not resamples:
        return result
    values = list(groups.values())
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sampled = [rng.choice(values) for _ in values]
        draws.append(sum(sum(v) for v in sampled) / sum(len(v) for v in sampled))
    draws.sort()
    result["group_bootstrap95_descriptive"] = [
        draws[int(0.025 * (resamples - 1))],
        draws[int(0.975 * (resamples - 1))],
    ]
    return result


def evaluate(
    cases,
    predictions,
    methods=("direct", "tool_context"),
    trials=1,
    bootstrap_samples=1000,
    bootstrap_seed=1802026,
):
    if type(trials) is not int or trials < 1:
        raise ValueError("Trials must be a positive integer")
    if (
        not isinstance(methods, (list, tuple))
        or not methods
        or any(not isinstance(m, str) or not m for m in methods)
        or len(set(methods)) != len(methods)
    ):
        raise ValueError("Methods must be nonempty unique strings")
    if type(bootstrap_samples) is not int or not 0 <= bootstrap_samples <= 10000:
        raise ValueError("Bootstrap samples must be an integer in 0-10000")
    if any(
        not isinstance(c.get("case_id"), str) or not valid_response(c.get("expected"))
        for c in cases
    ):
        raise ValueError("Dataset IDs and reference response schemas must be valid")
    case_ids = {c["case_id"] for c in cases}
    if len(case_ids) != len(cases):
        raise ValueError("Duplicate dataset case ID")
    indexed = {}
    for pred in predictions:
        if (
            not isinstance(pred, dict)
            or not isinstance(pred.get("case_id"), str)
            or not isinstance(pred.get("method"), str)
            or type(pred.get("trial")) is not int
        ):
            raise ValueError("Invalid prediction identity")
        key = (pred["case_id"], pred["method"], pred["trial"])
        if key in indexed:
            raise ValueError("Duplicate prediction")
        if key[0] not in case_ids or key[1] not in methods or not 0 <= key[2] < trials:
            raise ValueError("Unexpected prediction identity")
        indexed[key] = pred
    output = {
        "synthetic": any(p.get("synthetic") for p in predictions),
        "n_predictions_present": len(indexed),
        "evidence_status": "observations_present" if indexed else "no_observations",
        "record_kinds": dict(
            sorted(
                (kind, sum(p.get("record_kind", "unspecified") == kind for p in predictions))
                for kind in {p.get("record_kind", "unspecified") for p in predictions}
            )
        ),
        "evidence_note": (
            "Missing outputs remain failures; deterministic/mock records are not LLM evidence."
        ),
        "missing_predictions": len(cases) * len(methods) * trials - len(indexed),
        "methods": {},
    }
    per_case = {}
    for method in methods:
        scores = []
        categories = defaultdict(list)
        statuses = defaultdict(list)
        for trial in range(trials):
            for case in cases:
                score = score_case(case, indexed.get((case["case_id"], method, trial)))
                scores.append(score)
                categories[case["category"]].append(score)
                statuses[case["expected"]["status"]].append(score)
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
            "by_expected_status": {k: aggregate(v) for k, v in statuses.items()},
            "consistency_pairs": consistency_pairs,
            "pairwise_response_consistency": ratio(consistent, consistency_pairs),
        }
    if {"direct", "tool_context"} <= set(methods):
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
    comparisons = {}
    pairs_to_score = [
        (m, "direct")
        for m in ("tool_context", "tool_facts", "tool_corrupted")
        if m in methods and "direct" in methods
    ]
    pairs_to_score += [
        (m, m + "_minimal")
        for m in ("direct", "tool_context", "tool_facts", "tool_corrupted")
        if m in methods and m + "_minimal" in methods
    ]
    for treatment, comparator in pairs_to_score:
        outcomes = {}
        for metric, predicate in (
            ("exact", lambda c: c["expected"]["status"] == "ok"),
            ("joint_exact", lambda c: c["expected"]["status"] == "ok"),
            ("coverage", lambda c: c["expected"]["status"] == "ok"),
            ("unsupported_definitive", lambda c: c["expected"]["status"] != "ok"),
            ("ambiguous_definitive", lambda c: c["expected"]["status"] == "ambiguous"),
            ("nonanswer_policy_correct", lambda c: c["expected"]["status"] != "ok"),
            ("ambiguous_policy_correct", lambda c: c["expected"]["status"] == "ambiguous"),
        ):
            score_key = (
                "unsupported_definitive"
                if metric == "ambiguous_definitive"
                else "uncertainty_correct"
                if metric.endswith("policy_correct")
                else metric
            )
            rows = [
                (
                    c.get("group_id", c["case_id"]),
                    int(per_case[(c["case_id"], treatment, t)][score_key])
                    - int(per_case[(c["case_id"], comparator, t)][score_key]),
                )
                for c in cases
                if predicate(c)
                for t in range(trials)
            ]
            outcomes[metric] = paired_estimate(rows, bootstrap_samples, bootstrap_seed)
        comparisons[f"{treatment}_minus_{comparator}"] = outcomes
    output["comparisons"] = comparisons
    return output
