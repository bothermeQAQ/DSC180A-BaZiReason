"""Evaluate a complete declared experiment, retaining failures in denominators."""

import json
import math
import random
from collections import defaultdict

from bazireason.calendar_engine import compute_chart
from bazireason.controls import corrupt_context
from bazireason.metrics import aggregate, ratio, score_case
from bazireason.prompts import condition
from bazireason.schema import valid_response

# Pre-specified contrasts (treatment, comparator). The first entry under each
# hypothesis is its primary contrast; the remainder are secondary.
PRIMARY_CONTRASTS = {
    "H1": "direct: near-boundary minus far-boundary accuracy of the boundary-relevant "
    "pillars, within Jie and Zi ladder families (h1_boundary)",
    "H2": "tool_intermediate minus direct exact accuracy on scorable cases",
    "H3": "direct minus direct_minimal decision accuracy on knowledge-dependent H3 "
    "families (h3_matched)",
}
H2_CONTRASTS = (
    ("tool_intermediate", "direct"),
    ("tool_facts", "direct"),
    ("tool_context", "direct"),
    ("tool_intermediate", "tool_facts"),
    ("tool_context", "tool_intermediate"),
    ("tool_corrupted", "tool_context"),
)
COMPARED_METRICS = (
    ("exact", True),
    ("joint_exact", True),
    ("coverage", True),
    ("false_refusal", True),
    ("unsupported_definitive", False),
    ("ambiguous_definitive", "ambiguous"),
    ("nonanswer_policy_correct", False),
    ("ambiguous_policy_correct", "ambiguous"),
    ("decision_correct", None),
)
_SCORE_KEY = {
    "ambiguous_definitive": "unsupported_definitive",
    "nonanswer_policy_correct": "uncertainty_correct",
    "ambiguous_policy_correct": "uncertainty_correct",
}


def mcnemar_exact(treatment_only, comparator_only):
    """Two-sided exact McNemar p-value from discordant counts (descriptive)."""
    n = treatment_only + comparator_only
    if not n:
        return None
    tail = sum(math.comb(n, k) for k in range(min(treatment_only, comparator_only) + 1))
    return min(1.0, 2 * tail / 2**n)


def paired_estimate(rows, resamples=1000, seed=1802026):
    """Paired observation mean; resample whole clusters, preserving within-cluster rows.

    Rows are (cluster_id, treatment_outcome - comparator_outcome). The percentile
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


def resampling_clusters(cases):
    """Merge group IDs that share a civil date, so related variants resample together.

    The historical audit_v2 suite placed matched variants of one input (for
    example 2024-06-15 noon with a missing zone, an invalid zone or other
    profiles) in separate groups. Linking every case whose datetime begins with
    the same date is conservative: it can only merge clusters.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first_by_date = {}
    for case in cases:
        group = "group:" + str(case.get("group_id", case["case_id"]))
        find(group)
        raw = case["input"].get("datetime") if isinstance(case.get("input"), dict) else None
        if isinstance(raw, str) and len(raw) >= 10:
            day = raw[:10]
            if day in first_by_date:
                parent[find(group)] = find(first_by_date[day])
            else:
                first_by_date[day] = group
    return {c["case_id"]: find("group:" + str(c.get("group_id", c["case_id"])))[6:] for c in cases}


def _first_case(cases, group):
    return next(c["case_id"] for c in cases if c.get("group_id", c["case_id"]) == group)


def _design(case, key):
    design = case.get("design")
    return design.get(key) if isinstance(design, dict) else None


def _boundary_analysis(cases, per_case, method, trials, clusters, resamples, seed):
    output = {}
    for boundary in ("jie", "zi"):
        members = [c for c in cases if (_design(c, "h1") or {}).get("boundary") == boundary]
        if not members:
            continue
        by_distance = defaultdict(list)
        family_rows = defaultdict(lambda: {"near": [], "far": []})
        for c in members:
            h1 = c["design"]["h1"]
            for t in range(trials):
                s = per_case[(c["case_id"], method, t)]
                relevant = bool(s["pillar_matches"]) and all(
                    s["pillar_matches"][p] for p in h1["relevant_pillars"]
                )
                by_distance[h1["distance"]].append((relevant, s["exact"]))
                if h1["distance"] in ("near", "far"):
                    family_rows[(c["group_id"], t)][h1["distance"]].append(relevant)
        rows = [
            (
                clusters[next(c["case_id"] for c in members if c["group_id"] == group)],
                sum(v["near"]) / len(v["near"]) - sum(v["far"]) / len(v["far"]),
            )
            for (group, _t), v in sorted(family_rows.items())
            if v["near"] and v["far"]
        ]
        output[boundary] = {
            "by_distance": {
                d: {
                    "n": len(v),
                    "relevant_pillar_accuracy": ratio(sum(r for r, _ in v), len(v)),
                    "exact_accuracy": ratio(sum(e for _, e in v), len(v)),
                }
                for d, v in sorted(by_distance.items())
            },
            "near_minus_far_relevant_accuracy": paired_estimate(rows, resamples, seed),
        }
    return output


def _summary(rows):
    answerable = [s for s in rows if s["eligible"]]
    ambiguous = [s for s in rows if s["expected_status"] == "ambiguous"]
    return {
        "n": len(rows),
        "n_ambiguous": len(ambiguous),
        "decision_accuracy": ratio(sum(s["decision_correct"] for s in rows), len(rows)),
        "ambiguous_policy_accuracy": ratio(
            sum(s["uncertainty_correct"] for s in ambiguous), len(ambiguous)
        ),
        "ambiguous_definitive_rate": ratio(
            sum(s["unsupported_definitive"] for s in ambiguous), len(ambiguous)
        ),
        "false_refusal_rate": ratio(sum(s["false_refusal"] for s in answerable), len(answerable)),
        "scorable_coverage": ratio(sum(s["coverage"] for s in answerable), len(answerable)),
    }


def _h3_families(cases, per_case, method, trials):
    """{(stratum, group, trial): family-level decisions} for H3 families that mix
    answerable and nonanswerable members (the matched complete/incomplete design)."""
    families = defaultdict(list)
    for c in cases:
        h3 = _design(c, "h3")
        if not h3:
            continue
        key = "knowledge_dependent" if h3["knowledge_dependent"] else "surface"
        for t in range(trials):
            families[(key, c["group_id"], t)].append(per_case[(c["case_id"], method, t)])
    return {
        k: all(s["decision_correct"] for s in rows)
        for k, rows in families.items()
        if any(s["eligible"] for s in rows) and any(not s["eligible"] for s in rows)
    }


def _h3_analysis(cases, per_case, method, trials):
    members = [c for c in cases if _design(c, "h3")]
    if not members:
        return {}
    by_type, by_dependence = defaultdict(list), defaultdict(list)
    for c in members:
        h3 = c["design"]["h3"]
        for t in range(trials):
            s = per_case[(c["case_id"], method, t)]
            by_type[h3["type"]].append(s)
            key = "knowledge_dependent" if h3["knowledge_dependent"] else "surface"
            by_dependence[key].append(s)
    families = _h3_families(cases, per_case, method, trials)
    family_result = {}
    for key in ("knowledge_dependent", "surface"):
        outcomes = [v for (k, _g, _t), v in families.items() if k == key]
        family_result[key] = {
            "n_family_trials": len(outcomes),
            "all_decisions_correct_rate": ratio(sum(outcomes), len(outcomes)),
        }
    return {
        "by_type": {k: _summary(v) for k, v in sorted(by_type.items())},
        "by_dependence": {k: _summary(v) for k, v in sorted(by_dependence.items())},
        "matched_families": family_result,
    }


def evaluate(
    cases,
    predictions,
    methods=("direct", "tool_context"),
    trials=1,
    bootstrap_samples=1000,
    bootstrap_seed=1802026,
    corruption_scheme="v3",
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
    clusters = resampling_clusters(cases)
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
        "n_groups": len({c.get("group_id", c["case_id"]) for c in cases}),
        "n_resampling_clusters": len(set(clusters.values())),
        "primary_contrasts": PRIMARY_CONTRASTS,
        "corruption_scheme": corruption_scheme,
        "methods": {},
    }
    # The corrupted set is fixed by the design: case input plus a fresh engine result.
    corruptions = {}
    if any(m in methods for m in ("tool_corrupted", "tool_corrupted_minimal")):
        for c in cases:
            context = compute_chart(c["input"], c["convention"])
            corruptions[c["case_id"]] = corrupt_context(c, context, corruption_scheme)[1]
    per_case = {}
    for method in methods:
        corrupted_method = method in ("tool_corrupted", "tool_corrupted_minimal")
        scores = []
        categories = defaultdict(list)
        statuses = defaultdict(list)
        for trial in range(trials):
            for case in cases:
                corruption = corruptions.get(case["case_id"]) if corrupted_method else None
                score = score_case(case, indexed.get((case["case_id"], method, trial)), corruption)
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
            "h1_boundary": _boundary_analysis(
                cases, per_case, method, trials, clusters, bootstrap_samples, bootstrap_seed
            ),
            "h3_matched": _h3_analysis(cases, per_case, method, trials),
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
    model_methods = [m for m in methods if condition(m)[1] is not None]
    pairs_to_score = []
    for policy_suffix in ("", "_minimal"):
        pairs_to_score += [
            (t + policy_suffix, c + policy_suffix)
            for t, c in H2_CONTRASTS
            if t + policy_suffix in methods and c + policy_suffix in methods
        ]
    pairs_to_score += [(m, m + "_minimal") for m in model_methods if m + "_minimal" in methods]
    comparisons = {}
    for treatment, comparator in pairs_to_score:
        outcomes = {}
        for metric, scope in COMPARED_METRICS:
            score_key = _SCORE_KEY.get(metric, metric)

            def selected(c, scope=scope):
                if scope is None:
                    return True
                if scope == "ambiguous":
                    return c["expected"]["status"] == "ambiguous"
                return (c["expected"]["status"] == "ok") is scope

            paired = [
                (
                    clusters[c["case_id"]],
                    int(per_case[(c["case_id"], treatment, t)][score_key]),
                    int(per_case[(c["case_id"], comparator, t)][score_key]),
                )
                for c in cases
                if selected(c)
                for t in range(trials)
            ]
            estimate = paired_estimate(
                [(g, a - b) for g, a, b in paired], bootstrap_samples, bootstrap_seed
            )
            treatment_only = sum(a and not b for _, a, b in paired)
            comparator_only = sum(b and not a for _, a, b in paired)
            estimate["discordant_treatment_only"] = treatment_only
            estimate["discordant_comparator_only"] = comparator_only
            estimate["mcnemar_exact_p_descriptive"] = mcnemar_exact(treatment_only, comparator_only)
            outcomes[metric] = estimate
        if comparator == treatment + "_minimal":
            # Pre-specified H3 outcome: every decision in a matched family is right.
            a = _h3_families(cases, per_case, treatment, trials)
            b = _h3_families(cases, per_case, comparator, trials)
            family = {}
            for stratum in ("knowledge_dependent", "surface"):
                keys = sorted(k for k in a if k[0] == stratum)
                estimate = paired_estimate(
                    [(clusters[_first_case(cases, k[1])], int(a[k]) - int(b[k])) for k in keys],
                    bootstrap_samples,
                    bootstrap_seed,
                )
                t_only = sum(a[k] and not b[k] for k in keys)
                c_only = sum(b[k] and not a[k] for k in keys)
                estimate["discordant_treatment_only"] = t_only
                estimate["discordant_comparator_only"] = c_only
                estimate["mcnemar_exact_p_descriptive"] = mcnemar_exact(t_only, c_only)
                family[stratum] = estimate
            outcomes["h3_family_all_decisions_correct"] = family
        tool_pair = {condition(treatment)[0], condition(comparator)[0]} != {"direct"}
        if tool_pair and condition(treatment)[0] != condition(comparator)[0]:
            outcomes["note"] = (
                "Tool conditions can reveal answerability (for example a failed "
                "normalization); nonanswer contrasts here are secondary."
            )
        comparisons[f"{treatment}_minus_{comparator}"] = outcomes
    output["comparisons"] = comparisons
    return output
