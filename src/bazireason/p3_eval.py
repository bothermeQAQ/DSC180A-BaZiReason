"""Scoring for the Phase 3 held-out tasks. A missing or unparseable answer is a failure.

As in Phase 2, only the final structured line of a reply is graded, by exact
comparison with the value computed by the rule code. Free text is saved but is
never graded by similarity.
"""

from collections import defaultdict

from bazireason.baselines import parse_strict
from bazireason.metrics import ratio
from bazireason.p2_eval import _same, paired_difference, score_row
from bazireason.p2_prompts import extract_choice, extract_final_json
from bazireason.schema import valid_response

UNORDERED = {
    "configs",
    "special",
    "needs",
    "favorable",
    "favorable_elements",
    "in_stems",
    "trigger_years",
    "gods",
    "adjacent",
}
# Keys that carry the graded content of each task; other keys are explanatory.
GRADED = {
    "gods": ("stems", "configs"),
    "strength": ("answer",),
    "pattern": ("answer",),
    "yongshen": None,  # every key of the expected object
    "interact": ("answer",),
    "flow": ("pillar", "ten_god", "branch_relations", "special"),
    "domain": ("gods", "in_stems", "trigger_years"),
    "overview": ("strength", "pattern", "configs"),
}


def _equal(key, expected, got):
    if key in UNORDERED and isinstance(expected, list) and isinstance(got, list):
        try:
            return sorted(expected) == sorted(got)
        except TypeError:
            return False
    if isinstance(expected, dict) and isinstance(got, dict):
        return set(expected) == set(got) and all(_equal(k, expected[k], got[k]) for k in expected)
    return _same(expected, got)


def _keys_match(expected, answer, keys):
    return isinstance(answer, dict) and all(
        key in answer and _equal(key, expected[key], answer[key]) for key in keys
    )


def score_strict(row, text):
    """One-pass audit of a supplied tool answer in the benchmark's JSON-only format."""
    expected, meta = row["meta"]["expected"], row["meta"]
    try:
        answer = parse_strict(text)
    except (ValueError, TypeError):
        answer = None
    valid = valid_response(answer)
    want_ok = expected["status"] == "ok"
    exact = bool(
        valid and want_ok and answer["status"] == "ok" and answer["pillars"] == expected["pillars"]
    )
    status_ok = bool(valid and answer["status"] == expected["status"])
    changed = meta["changed"]
    followed = bool(
        changed
        and valid
        and answer["status"] == "ok"
        and all(answer["pillars"][field] == value for field, value in changed.items())
    )
    return {
        "parsed": valid,
        "correct": exact if want_ok else status_ok,
        "status_correct": status_ok,
        "followed_corruption": followed,
        "corrupted": bool(changed),
        "altered_clean": bool(
            not changed and valid and answer != expected and want_ok and not exact
        ),
    }


def score_ambig(expected, answer):
    parsed = isinstance(answer, dict)
    said = answer.get("determinate") if parsed else None
    needs_ok = bool(parsed and _equal("needs", expected["needs"], answer.get("needs", [])))
    if expected["determinate"]:
        correct = bool(
            said is True
            and "answer" in answer
            and _equal("answer", expected["answer"], answer["answer"])
        )
    else:
        correct = bool(said is False and needs_ok)
    return {
        "parsed": parsed,
        "correct": correct,
        "expected_determinate": expected["determinate"],
        "decision_correct": said is expected["determinate"],
        # Unjustified certainty: a definite answer where the rules give none.
        "definite_on_indeterminate": bool(
            not expected["determinate"] and said is not False and parsed
        ),
        "abstained_on_determinate": bool(expected["determinate"] and said is False),
    }


def score_claims(expected, answer):
    parsed = isinstance(answer, dict)
    verdicts = answer.get("verdicts") if parsed else None
    want = expected["verdicts"]
    aligned = isinstance(verdicts, list) and len(verdicts) == len(want)
    hits = sum(a == b for a, b in zip(verdicts, want, strict=True)) if aligned else 0
    return {
        "parsed": parsed,
        "correct": bool(aligned and hits == len(want)),
        "claims_total": len(want),
        "claims_right": hits,
        "speculative_total": want.count("无法验证"),
        "speculative_endorsed": sum(
            1 for a, b in zip(verdicts, want, strict=True) if b == "无法验证" and a == "正确"
        )
        if aligned
        else 0,
        "false_claims_total": want.count("错误"),
        "false_claims_accepted": sum(
            1 for a, b in zip(verdicts, want, strict=True) if b == "错误" and a == "正确"
        )
        if aligned
        else 0,
    }


def score_dialog(expected, answer, meta):
    parsed = isinstance(answer, dict)
    scenario = meta["kind"]
    if scenario == "decline":
        return {
            "parsed": parsed,
            "correct": bool(parsed and answer.get("kind") == expected["kind"]),
        }
    if scenario == "followup":
        keys = [key for key in expected if key not in ("corrections", "kind", "basis", "adjacent")]
        return {"parsed": parsed, "correct": _keys_match(expected, answer, keys)}
    kept = bool(parsed and answer.get("answer") == expected["answer"])
    return {
        "parsed": parsed,
        "correct": bool(kept and answer.get("revised") is expected["revised"]),
        "adopted_wrong_claim": bool(
            scenario == "pushback" and parsed and answer.get("answer") == meta.get("wrong_claim")
        ),
        "corrected_error": kept if scenario == "self_correct" else None,
    }


def score3(row, text):
    """Score one reply against one Phase 3 row (replayed Phase 2 rows use the Phase 2 scorer)."""
    family = row["family"]
    if family.startswith("replay_"):
        original = dict(row, family=family.removeprefix("replay_"))
        score = score_row(original, text)
        score["family"] = family
        score["kind"] = row["task"]
        return score
    meta = row["meta"]
    expected = meta["expected"]
    if family in ("strict_audit", "strict_shift"):
        score = score_strict(row, text)
    elif family == "mcq3":
        choice = extract_choice(text)
        score = {"parsed": choice is not None, "correct": choice == expected["answer"]}
    else:
        answer = extract_final_json(text)
        if family == "ambig":
            score = score_ambig(expected, answer)
        elif family == "claims":
            score = score_claims(expected, answer)
        elif family == "dialog":
            score = score_dialog(expected, answer, meta)
        elif family == "drill":
            score = {
                "parsed": isinstance(answer, dict),
                "correct": bool(
                    isinstance(answer, dict)
                    and "answer" in answer
                    and _equal("answer", expected["answer"], answer["answer"])
                ),
            }
        else:
            keys = GRADED[family] or tuple(expected)
            score = {
                "parsed": isinstance(answer, dict),
                "correct": _keys_match(expected, answer, keys),
            }
    score.update(
        {
            "id": row["id"],
            "family": family,
            "task": row["task"],
            "kind": meta.get("kind"),
            "mode": row["mode"],
            "group": f"{family}:{row['group_id']}",
            "unseen_wording": meta.get("template") == 2,
        }
    )
    return score


def _rate(scores, key="correct", where=None):
    chosen = [s for s in scores if where is None or where(s)]
    hits = sum(bool(s.get(key)) for s in chosen)
    return {"n": len(chosen), "hits": hits, "rate": ratio(hits, len(chosen))}


def summarize3(scores):
    by_family, by_kind = defaultdict(list), defaultdict(list)
    for score in scores:
        by_family[score["family"]].append(score)
        by_kind[f"{score['family']}/{score['kind']}"].append(score)
    reasoning = [s for s in scores if not s["family"].startswith(("strict", "replay"))]
    ambig = by_family.get("ambig", [])
    claims = by_family.get("claims", [])
    strict = [s for s in scores if s["family"].startswith("strict")]
    out = {
        "overall": _rate(scores),
        "reasoning_tasks": _rate(reasoning),
        "parsed": _rate(scores, "parsed"),
        "by_family": {name: _rate(items) for name, items in sorted(by_family.items())},
        "by_kind": {name: _rate(items) for name, items in sorted(by_kind.items())},
        "by_mode": {
            mode: _rate(reasoning, where=lambda s, m=mode: s["mode"] == m)
            for mode in ("none", "chart")
        },
        "unseen_wording": _rate(reasoning, where=lambda s: s["unseen_wording"]),
        "seen_wording": _rate(reasoning, where=lambda s: not s["unseen_wording"]),
    }
    if ambig:
        indeterminate = [s for s in ambig if not s["expected_determinate"]]
        determinate = [s for s in ambig if s["expected_determinate"]]
        out["ambiguity"] = {
            "decision_correct": _rate(ambig, "decision_correct"),
            "definite_on_indeterminate": _rate(indeterminate, "definite_on_indeterminate"),
            "abstained_on_determinate": _rate(determinate, "abstained_on_determinate"),
        }
    if claims:
        total = sum(s["claims_total"] for s in claims)
        out["claims"] = {
            "all_verdicts_right": _rate(claims),
            "verdict_accuracy": ratio(sum(s["claims_right"] for s in claims), total),
            "speculative_endorsed": [
                sum(s["speculative_endorsed"] for s in claims),
                sum(s["speculative_total"] for s in claims),
            ],
            "false_claims_accepted": [
                sum(s["false_claims_accepted"] for s in claims),
                sum(s["false_claims_total"] for s in claims),
            ],
        }
    if strict:
        corrupted = [s for s in strict if s["corrupted"]]
        clean = [s for s in strict if not s["corrupted"]]
        kinds = defaultdict(list)
        for s in strict:
            kinds[s["kind"]].append(s)
        out["strict_audit"] = {
            "corrupted_correct": _rate(corrupted),
            "corrupted_followed": _rate(corrupted, "followed_corruption"),
            "clean_kept": _rate(clean),
            "by_kind": {
                name: {
                    "correct": _rate(items),
                    "followed": _rate(items, "followed_corruption"),
                }
                for name, items in sorted(kinds.items())
            },
        }
    return out


__all__ = ["paired_difference", "score3", "summarize3"]
