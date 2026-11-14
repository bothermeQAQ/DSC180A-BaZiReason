"""Apply the frozen Phase 6 decision rules (docs/phase6_preregistration.md, sections 5 and 6).

    .venv/bin/python scripts/phase6/decide6.py screen    # dev6: U1 recipe -> results/phase6/screening.json
    .venv/bin/python scripts/phase6/decide6.py select    # valid6: selection -> results/phase6/selection.json

Reads saved records only. "select" writes results/phase6/selection.json, which opens sealed6.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/phase5")
sys.path.insert(0, "scripts/phase6")
import make_report6 as M  # noqa: E402

from bazireason.p2_eval import paired_difference  # noqa: E402

ROOT = Path("results/phase6")
STEPS = (
    ("p6s1", "p6u0", "small steps"),
    ("p6s2", "p6s1", "wording"),
    ("p6s3", "p6s2", "evidence and self-check"),
)
DROP_NAMES = {
    "wording": ["wording", "contrast"],
    "evidence and self-check": ["evidence", "verify", "selfcheck", "twins"],
}


def paired(split, a, b, level="L0", parts=M.RULE_PARTS):
    ra = [r for p in parts for r in M.load(split, a, p, level)]
    rb = {r["id"]: r for p in parts for r in M.load(split, b, p, level)}
    d = paired_difference(M.scores(ra), M.scores([rb[r["id"]] for r in ra]))
    return {
        "difference": d["difference"],
        "interval95": d["interval95"],
        "a": M.correct(ra),
        "b": sum(bool(rb[r["id"]]["score"]["correct"]) for r in ra),
        "n": len(ra),
    }


def screen():
    out = {"rule": "docs/phase6_preregistration.md section 5", "steps": [], "drop": []}
    for run, before, component in STEPS:
        d = paired("dev6", before, run)
        hi = d["interval95"][1] if d["interval95"] else None
        dropped = hi is not None and hi < 0
        out["steps"].append(
            {
                "run": run,
                "before": before,
                "component": component,
                "l0_all_rule_items": d,
                "dropped": dropped,
            }
        )
        if dropped and component in DROP_NAMES:
            out["drop"] += DROP_NAMES[component]
    targets = {}
    stm = {}
    summary = {}
    M.statements_section([], summary, "dev6")
    for lab, v in summary.get("statements", {}).get("dev6", {}).items():
        stm[lab] = [
            sum(v["wrong"].get(k, 0) for k in M.COMPUTED_KINDS),
            sum(v["checked"].get(k, 0) for k in M.COMPUTED_KINDS),
        ]
    targets["computed_statements_wrong"] = stm
    targets["blind"] = {lab: paired("dev6", "p6s1", lab, parts=("blind",)) for lab in ("p6s2",)}
    for lab in ("p6s2", "p6s3"):
        if M.load("dev6", lab, "rules", "L1"):
            a = [r for p in M.RULE_PARTS for r in M.load("dev6", lab, p, "L0")]
            b = {r["id"]: r for p in M.RULE_PARTS for r in M.load("dev6", lab, p, "L1")}
            d = paired_difference(M.scores(a), M.scores([b[r["id"]] for r in a]))
            targets.setdefault("l1_minus_l0", {})[lab] = {
                "difference": d["difference"],
                "interval95": d["interval95"],
            }
    out["targets"] = targets
    cal = {lab: M.load("dev6", lab, "calendar") for lab in ("p6u0", "p6s1", "p6s2", "p6s3")}
    cal = {k: v for k, v in cal.items() if v}
    if "p6u0" in cal:
        u0 = M.correct(cal["p6u0"]) / len(cal["p6u0"])
        gaps = {k: M.correct(v) / len(v) - u0 for k, v in cal.items() if k != "p6u0"}
        out["calendar_gap_to_u0"] = gaps
        out["calendar_share"] = 0.35 if gaps and all(g < -0.10 for g in gaps.values()) else None
    runs = [r for r, _, _ in STEPS if M.load("dev6", r, "rules")]
    best = max(runs, key=lambda r: (M.correct(M.rule_records("dev6", r)), -runs.index(r)))
    out["best_s_run_on_dev6"] = best
    (ROOT / "screening.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(
        json.dumps(
            {k: out[k] for k in ("drop", "calendar_share", "best_s_run_on_dev6") if k in out},
            ensure_ascii=False,
        )
    )


def select():
    screening = json.loads((ROOT / "screening.json").read_text())
    candidates = [
        lab
        for lab in ("p6u1", "p6u1hi", screening["best_s_run_on_dev6"])
        if M.rule_records("valid6", lab)
    ]
    p2 = M.load("valid6", "p2", "calendar")
    table = {}
    for lab in candidates:
        cal = M.load("valid6", lab, "calendar")
        d = paired_difference(
            M.scores(p2), M.scores([{r["id"]: r for r in cal}[x["id"]] for x in p2])
        )
        recs = M.rule_records("valid6", lab)
        table[lab] = {"l0_all_rule_items": [M.correct(recs), len(recs)], "calendar_minus_p2": d,
                      "calendar_gate": d["interval95"] is not None and d["interval95"][0] >= -0.07,
                      "layers": 36 if lab == "p6u1hi" else 16}  # fmt: skip
    passing = [lab for lab in candidates if table[lab]["calendar_gate"]]
    pool = passing or candidates
    order = {lab: i for i, lab in enumerate(("p6s1", "p6s2", "p6s3", "p6u1", "p6u1hi"))}
    chosen = max(
        pool,
        key=lambda lab: (table[lab]["l0_all_rule_items"][0], -table[lab]["layers"], -order[lab]),
    )
    out = {"rule": "docs/phase6_preregistration.md section 6 (frozen before validation scoring)", "candidates": table,
           "passing_calendar_gate": passing, "calendar_claim_failed": not passing, "selected": chosen}  # fmt: skip
    (ROOT / "selection.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(out, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    {"screen": screen, "select": select}[sys.argv[1]]()
