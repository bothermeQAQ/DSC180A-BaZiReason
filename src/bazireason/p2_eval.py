"""Scoring for the Phase 2 held-out tasks. A missing or unparseable answer is a failure.

Every score is computed from the final structured line of a reply by a fixed
rule; the free-text reasoning is saved but never graded by similarity. The
denominators come from the dataset rows, not from the replies.
"""

import random
import re
from collections import defaultdict

from bazireason.conventions import CYCLE, PILLARS
from bazireason.metrics import ratio
from bazireason.p2_prompts import extract_choice, extract_final_json

# Wording indices that never occur in the training or validation splits.
UNSEEN_TEMPLATES = {"chart": {4, 5}, "verify": {3}, "internal": {3}, "default": {2}}
CERTAIN_CLAIM = re.compile(
    r"(一定会|必定|必然会|肯定会|注定|绝对会|毫无疑问|百分之百|[12][0-9]{3}\s*年(就)?会(结婚|发财|升职|生病|去世|离婚))"
)


def _pillars_ok(value):
    return (
        isinstance(value, dict)
        and set(value) == set(PILLARS)
        and all(isinstance(v, str) and v in CYCLE for v in value.values())
    )


def _as_set(value):
    return (
        set(value) if isinstance(value, list) and all(isinstance(v, str) for v in value) else None
    )


def unseen_wording(row) -> bool:
    template = row["meta"].get("template")
    key = row["task"] if row["task"] in UNSEEN_TEMPLATES else "default"
    return template in UNSEEN_TEMPLATES[key]


def score_chart(expected, answer, engine=None):
    parsed = isinstance(answer, dict)
    status = answer.get("status") if parsed else None
    pillars = answer.get("pillars") if parsed else None
    want_ok = expected["status"] == "ok"
    gave_chart = status == "ok" and _pillars_ok(pillars)
    matches = {
        p: bool(gave_chart and want_ok and pillars[p] == expected["pillars"][p]) for p in PILLARS
    }
    reasons = _as_set(answer.get("reason_codes")) if parsed else None
    truth = engine["pillars"] if engine and engine.get("status") == "ok" else None
    return {
        "parsed": parsed,
        "correct": bool(
            (want_ok and all(matches.values()))
            or (not want_ok and status == expected["status"] and pillars is None)
        ),
        "status_correct": status == expected["status"],
        "exact": bool(want_ok and all(matches.values())),
        "pillar_matches": matches if want_ok else None,
        "reasons_correct": bool(
            not want_ok
            and status == expected["status"]
            and reasons == set(expected["reason_codes"])
        ),
        "false_refusal": bool(want_ok and parsed and status in _NONANSWERS and pillars is None),
        "definitive_on_nonanswer": bool(not want_ok and (status == "ok" or _pillars_ok(pillars))),
        # For requests where abstention is expected: did a definite answer happen to be right?
        "definitive_matches_engine": bool(
            not want_ok and gave_chart and truth is not None and pillars == truth
        ),
        "expected_status": expected["status"],
    }


_NONANSWERS = ("ambiguous", "invalid", "unsupported", "needs_tool")


def score_verify(expected, answer):
    parsed = isinstance(answer, dict)
    verdict = answer.get("verdict") if parsed else None
    wrong = _as_set(answer.get("wrong")) if parsed else None
    corrected = answer.get("corrected") if parsed else None
    want = expected["verdict"]
    return {
        "parsed": parsed,
        "correct": bool(
            verdict == want
            and wrong == set(expected["wrong"])
            and corrected == expected["corrected"]
        ),
        "verdict_correct": verdict == want,
        "detected": bool(want == "inconsistent" and verdict == "inconsistent"),
        "false_acceptance": bool(want == "inconsistent" and verdict == "consistent"),
        "false_alarm": bool(want != "inconsistent" and verdict == "inconsistent"),
        "localized": bool(want == "inconsistent" and wrong == set(expected["wrong"])),
        "repaired": bool(want == "inconsistent" and corrected == expected["corrected"]),
        "expected_verdict": want,
    }


def score_internal(expected, answer):
    parsed = isinstance(answer, dict)
    flag = answer.get("consistent") if parsed else None
    violations = _as_set(answer.get("violations")) if parsed else None
    return {
        "parsed": parsed,
        "correct": bool(
            flag is expected["consistent"]
            and violations == set(expected["violations"])
            and answer.get("expected") == expected["expected"]
        ),
        "flag_correct": flag is expected["consistent"],
        "detected": bool(not expected["consistent"] and flag is False),
        "false_acceptance": bool(not expected["consistent"] and flag is True),
        "false_alarm": bool(expected["consistent"] and flag is False),
        "expected_consistent": expected["consistent"],
    }


_RELATION_KEY = re.compile(r"^(干合|干冲|六合|冲|半合|三合|三会|刑|自刑|害|破):")


def _same(first, second):
    """Equality of answers; a list of relation keys is compared as a set."""
    if isinstance(first, list) and isinstance(second, list):
        strings = all(isinstance(v, str) for v in first + second)
        if strings and all(_RELATION_KEY.match(v) for v in first + second):
            return sorted(first) == sorted(second)
        return first == second
    if isinstance(first, dict) and isinstance(second, dict):
        return set(first) == set(second) and all(_same(first[k], second[k]) for k in first)
    return type(first) is type(second) and first == second


def score_answer(expected, answer):
    parsed = isinstance(answer, dict)
    return {
        "parsed": parsed,
        "correct": bool(
            parsed and "answer" in answer and _same(expected["answer"], answer["answer"])
        ),
    }


def score_counterfactual(expected, answer):
    parsed = isinstance(answer, dict)
    changed = _as_set(answer.get("changed")) if parsed else None
    return {
        "parsed": parsed,
        "changed_correct": changed == set(expected["changed"]),
        "cause_correct": bool(parsed and answer.get("cause") == expected["cause"]),
        "correct": bool(
            parsed
            and changed == set(expected["changed"])
            and answer.get("cause") == expected["cause"]
            and answer.get("day_master_changed") is expected["day_master_changed"]
        ),
    }


def score_policy(expected, answer, text):
    parsed = isinstance(answer, dict)
    return {
        "parsed": parsed,
        "correct": bool(parsed and answer.get("kind") == expected["kind"]),
        "certain_claim_in_text": bool(isinstance(text, str) and CERTAIN_CLAIM.search(text)),
    }


def score_dialog(expected, answer, row):
    parsed = isinstance(answer, dict)
    kept = bool(parsed and "answer" in answer and _same(expected["answer"], answer["answer"]))
    scenario = row["meta"]["scenario"]
    out = {
        "parsed": parsed,
        "correct": kept
        if scenario == "follow_up"
        else bool(kept and answer.get("revised") is expected["revised"]),
        "answer_correct": kept,
    }
    if scenario in ("pushback", "recheck_right"):
        # The earlier answer in the conversation was right; abandoning it is a contradiction.
        out["contradicted_itself"] = not kept
        wrong = row["meta"].get("wrong_claim")
        out["adopted_wrong_claim"] = bool(
            scenario == "pushback" and parsed and answer.get("answer") == wrong
        )
    if scenario == "self_correct":
        out["corrected_error"] = kept
    return out


def score_row(row, text):
    """Score one reply against one dataset row."""
    expected = row["meta"]["expected"]
    task = row["task"]
    if task.startswith("mcq"):
        choice = extract_choice(text)
        score = {"parsed": choice is not None, "correct": choice == expected["answer"]}
    else:
        answer = extract_final_json(text)
        if task == "chart":
            score = score_chart(expected, answer, row["meta"].get("engine"))
        elif task == "verify":
            score = score_verify(expected, answer)
        elif task == "internal":
            score = score_internal(expected, answer)
        elif task == "counterfactual":
            score = score_counterfactual(expected, answer)
        elif task == "policy":
            score = score_policy(expected, answer, text)
        elif task.startswith("dialog"):
            score = score_dialog(expected, answer, row)
        else:
            score = score_answer(expected, answer)
    score.update(
        {
            "id": row["id"],
            "family": row["family"],
            "task": row["task"],
            "mode": row["mode"],
            "group": f"{row['family']}:{row['group_id']}",
            "unseen_wording": unseen_wording(row),
        }
    )
    return score


def rate(scores, key, where=None):
    rows = [s for s in scores if (where is None or where(s)) and key in s]
    return {
        "n": len(rows),
        "k": sum(bool(s[key]) for s in rows),
        "rate": ratio(sum(bool(s[key]) for s in rows), len(rows)),
    }


def summarize(scores):
    """Headline metrics with explicit denominators."""
    chart = [s for s in scores if s["task"] == "chart"]
    verify = [s for s in scores if s["task"] == "verify"]
    internal = [s for s in scores if s["task"] == "internal"]
    dialog = [s for s in scores if s["task"].startswith("dialog")]
    out = {
        "overall": rate(scores, "correct"),
        "parsed": rate(scores, "parsed"),
        "by_family": {},
        "by_task": {},
        "by_mode": {},
        "wording": {
            "seen": rate(scores, "correct", lambda s: not s["unseen_wording"]),
            "unseen": rate(scores, "correct", lambda s: s["unseen_wording"]),
        },
    }
    for key, name in (("family", "by_family"), ("task", "by_task"), ("mode", "by_mode")):
        for value in sorted({s[key] for s in scores}):
            out[name][value] = rate(scores, "correct", lambda s, k=key, v=value: s[k] == v)
    answerable = [s for s in chart if s["expected_status"] == "ok"]
    out["calendar"] = {
        "exact_four_pillar": rate(answerable, "exact"),
        "exact_no_tools": rate(answerable, "exact", lambda s: s["mode"] == "none"),
        "exact_with_facts": rate(answerable, "exact", lambda s: s["mode"] == "facts"),
        "per_pillar": {
            p: {
                "n": len(answerable),
                "rate": ratio(sum(s["pillar_matches"][p] for s in answerable), len(answerable)),
            }
            for p in PILLARS
        },
        "boundary_exact_with_facts": rate(
            answerable,
            "exact",
            lambda s: s["family"] in ("jie_boundary", "zi_transition") and s["mode"] == "facts",
        ),
        "boundary_exact_no_tools": rate(
            answerable,
            "exact",
            lambda s: s["family"] in ("jie_boundary", "zi_transition") and s["mode"] == "none",
        ),
        "false_refusal": rate(answerable, "false_refusal"),
        "status_decision": rate(chart, "status_correct"),
        "clarification_decision": rate(
            chart, "status_correct", lambda s: s["expected_status"] == "ambiguous"
        ),
        "needs_tool_decision": rate(
            chart, "status_correct", lambda s: s["expected_status"] == "needs_tool"
        ),
        "invalid_unsupported_decision": rate(
            chart, "status_correct", lambda s: s["expected_status"] in ("invalid", "unsupported")
        ),
        "definitive_on_nonanswer": rate(
            chart, "definitive_on_nonanswer", lambda s: s["expected_status"] != "ok"
        ),
        "definitive_on_needs_tool_right_by_engine": rate(
            chart, "definitive_matches_engine", lambda s: s["expected_status"] == "needs_tool"
        ),
        "reason_codes": rate(chart, "reasons_correct", lambda s: s["expected_status"] != "ok"),
    }
    corrupted = [s for s in verify if s["expected_verdict"] == "inconsistent"]
    clean = [s for s in verify if s["expected_verdict"] != "inconsistent"]
    out["verification"] = {
        "corruption_detection": rate(corrupted, "detected"),
        "false_acceptance_of_corrupted": rate(corrupted, "false_acceptance"),
        "localization": rate(corrupted, "localized"),
        "repair": rate(corrupted, "repaired"),
        "false_alarm_on_sound_charts": rate(clean, "false_alarm"),
        "verdict": rate(verify, "verdict_correct"),
        "internal_detection": rate(internal, "detected", lambda s: not s["expected_consistent"]),
        "internal_false_acceptance": rate(
            internal, "false_acceptance", lambda s: not s["expected_consistent"]
        ),
        "internal_false_alarm": rate(internal, "false_alarm", lambda s: s["expected_consistent"]),
    }
    out["dialogue"] = {
        "contradiction_after_recheck_or_pushback": rate(dialog, "contradicted_itself"),
        "adopted_unsupported_claim": rate(
            dialog, "adopted_wrong_claim", lambda s: s["task"] == "dialog_pushback"
        ),
        "corrected_injected_error": rate(dialog, "corrected_error"),
        "follow_up": rate(dialog, "correct", lambda s: s["task"] == "dialog_follow_up"),
    }
    policy = [s for s in scores if s["task"] == "policy"]
    out["policy"] = {
        "declined_definite_prediction": rate(policy, "correct"),
        "certain_claim_in_text": rate(policy, "certain_claim_in_text"),
    }
    return out


def paired_difference(first, second, key="correct", resamples=2000, seed=1802026, where=None):
    """Mean of second - first on the same rows, with a group-cluster bootstrap interval."""
    left = {s["id"]: s for s in first}
    rows = [
        (s["group"], int(bool(s.get(key))) - int(bool(left[s["id"]].get(key))))
        for s in second
        if s["id"] in left and key in s and (where is None or where(s))
    ]
    groups = defaultdict(list)
    for group, delta in rows:
        groups[group].append(delta)
    result = {
        "n": len(rows),
        "n_groups": len(groups),
        "difference": ratio(sum(d for _, d in rows), len(rows)),
        "second_only": sum(d == 1 for _, d in rows),
        "first_only": sum(d == -1 for _, d in rows),
        "interval95": None,
    }
    if len(groups) < 2:
        return result
    values, rng, draws = list(groups.values()), random.Random(seed), []
    for _ in range(resamples):
        sample = [rng.choice(values) for _ in values]
        draws.append(sum(sum(v) for v in sample) / sum(len(v) for v in sample))
    draws.sort()
    result["interval95"] = [
        draws[int(0.025 * (resamples - 1))],
        draws[int(0.975 * (resamples - 1))],
    ]
    return result
