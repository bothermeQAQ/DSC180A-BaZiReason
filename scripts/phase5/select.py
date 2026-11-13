"""Apply the frozen Phase 5 selection rules (docs/phase5_preregistration.md, section 4).

    .venv/bin/python scripts/phase5/select.py alpha    # M*: best alpha on development data
    .venv/bin/python scripts/phase5/select.py final    # M* eligibility and G* on validation

Reads saved outputs only. "final" writes results/phase5/selection.json, which opens the
sealed split.
"""

import json
import sys
from pathlib import Path

from bazireason.p2_eval import paired_difference

ROOT = Path("results/phase5")
ALPHAS = (0.15, 0.25, 0.35)
SAFE = str.maketrans({":": "-", "/": "-"})


def records(split, label, part, condition):
    path = ROOT / split / label.translate(SAFE) / f"{part}_{condition}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def correct(recs):
    return sum(bool(r["score"]["correct"]) for r in recs)


def balanced_decision(recs):
    ok = [r["score"] for r in recs if r["score"].get("expected_status") == "ok"]
    other = [r["score"] for r in recs if r["score"].get("expected_status") not in (None, "ok")]
    rate = lambda xs: sum(bool(s.get("status_correct")) for s in xs) / len(xs) if xs else 0.0  # noqa: E731
    return round((rate(ok) + rate(other)) / 2, 4), {"answerable": len(ok), "without_answer": len(other)}


def alpha():
    table = {}
    for a in ALPHAS:
        label = f"mix:p3r3:p4a1:{a}"
        parts = {p: correct(records("dev", label, p, "w0")) for p in ("calendar", "rules", "indep", "blind")}
        table[a] = {"parts": parts, "total": sum(parts.values())}
    best = max(ALPHAS, key=lambda a: (table[a]["total"], -a))
    out = {"rule": "most correct development answers (w0: calendar, rules, indep, blind); ties to the smaller alpha",
           "table": {str(k): v for k, v in table.items()}, "alpha": best, "label": f"mix:p3r3:p4a1:{best}"}  # fmt: skip
    (ROOT / "dev_alpha.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))


def final():
    m = json.loads((ROOT / "dev_alpha.json").read_text())["label"]
    cal = {lab: records("valid", lab, "calendar", "w0") for lab in ("p2", "p3r3", "p4a1", m)}
    g1_cal = records("valid", "p5g1", "calendar", "w0")
    diff = paired_difference([r["score"] for r in cal["p2"]], [r["score"] for r in cal[m]])
    bda = {lab: balanced_decision(recs)[0] for lab, recs in cal.items()}
    bda_g1 = balanced_decision(g1_cal)[0]
    rule_parts = ("rules", "indep", "blind")
    w0 = {lab: sum(correct(records("valid", lab, p, "w0")) for p in rule_parts) for lab in ("p4a1", m)}
    n_items = sum(len(records("valid", "p4a1", p, "w0")) for p in rule_parts)
    gates = {
        "a_calendar_lower_bound_ge_-7": diff["interval95"] is not None and diff["interval95"][0] >= -0.07,
        "b_rules_within_3_points_of_A1": (w0[m] - w0["p4a1"]) / n_items >= -0.03,
        "c_balanced_decision_above_P2_R3_A1": bda[m] > max(bda["p2"], bda["p3r3"], bda["p4a1"]),
    }
    candidate = all(gates.values())
    pool = ["base", "p3r3", "p4a1", "p5g1"] + ([m] if candidate else [])
    convo = {}
    for lab in pool:
        path = ROOT / "valid" / lab.translate(SAFE) / "convo_grv.jsonl"
        if path.exists():
            convo[lab] = sum(json.loads(x)["dialogue_ok"] for x in path.read_text(encoding="utf-8").splitlines())
    excluded = [lab for lab in pool if lab != "p4a1" and lab in convo and "p4a1" in convo and convo[lab] < convo["p4a1"] - 3]
    pool = [lab for lab in pool if lab not in excluded]
    gr = {}
    for lab in pool:
        recs = [r for p in rule_parts for r in records("valid", lab, p, "gr")]
        gr[lab] = {"first_pass_correct": correct(recs), "with_contradiction": sum(bool(r["check"]["contradictions"]) for r in recs), "n": len(recs)}
    order = {"p4a1": 0}
    g_star = max(pool, key=lambda lab: (gr[lab]["first_pass_correct"], -gr[lab]["with_contradiction"], -order.get(lab, 1)))
    out = {
        "rules": "docs/phase5_preregistration.md section 4 (frozen before validation scoring)",
        "m_star": m,
        "m_star_calendar_minus_p2": diff,
        "balanced_decision_accuracy": dict(bda, p5g1=bda_g1),
        "w0_rules_indep_blind_correct": w0,
        "n_items": n_items,
        "m_star_gates": gates,
        "m_star_is_candidate": candidate,
        "convo_grv_dialogues_ok": convo,
        "excluded_by_conversation_guard": excluded,
        "gr_first_pass": gr,
        "g_star": g_star,
    }
    (ROOT / "selection.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    {"alpha": alpha, "final": final}[sys.argv[1]]()
