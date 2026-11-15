"""Apply the frozen Phase 6 decision rules (docs/phase6_preregistration.md, sections 5 and 6).

    .venv/bin/python scripts/phase6/decide6.py screen    # dev6: U1 recipe -> results/phase6/screening.json
    .venv/bin/python scripts/phase6/decide6.py select    # valid6: selection -> results/phase6/selection.json
    .venv/bin/python scripts/phase6/decide6.py flags     # U1 build options implied by screening.json

Round 2 (section 10; every decision is appended to results/phase6/round2.json):

    .venv/bin/python scripts/phase6/decide6.py structure # adapter options of round 2
    .venv/bin/python scripts/phase6/decide6.py next      # "<recipe> <parent>" of the next continuation, or "stop"

Reads saved records only. "select" writes results/phase6/selection.json, which opens sealed6.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/phase5")
sys.path.insert(0, "scripts/phase6")
import build_train6  # noqa: E402
import make_report6 as M  # noqa: E402
import run6  # noqa: E402

from bazireason import prompts6 as P6  # noqa: E402
from bazireason.p2_eval import paired_difference  # noqa: E402

ROOT = Path("results/phase6")
STEPS = (
    ("p6s1", "p6u0", "small steps"),
    ("p6s2", "p6s1", "wording"),
    ("p6s3", "p6s2", "evidence and self-check"),
)
CONTINUATIONS = ("p6u3", "p6u4", "p6u5")
ORDER = ("p6s1", "p6s2", "p6s3", "p6u1", "p6u1hi", "p6u2", *CONTINUATIONS)  # earlier runs first
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


def flags():
    """The U1 build options that the screening decision implies (read by queue_u1.sh)."""
    d = json.loads((ROOT / "screening.json").read_text())
    out = ["--drop", *d["drop"]] if d["drop"] else []
    if d.get("calendar_share"):
        out += ["--calendar-share", str(d["calendar_share"])]
    print(" ".join(out))


def best_s():
    print(json.loads((ROOT / "screening.json").read_text())["best_s_run_on_dev6"])


def small_steps_dropped():
    d = json.loads((ROOT / "screening.json").read_text())
    sys.exit(0 if d["steps"][0]["dropped"] else 1)


def parameters(lab):
    """Trainable parameters of a run, from its training plan ("smaller adapter" in a tie)."""
    plan = ROOT / "training" / run6.FOLDERS[lab] / "training_plan.json"
    return json.loads(plan.read_text())["trainable_parameters"]


def best(labels, split):
    """Highest L0 accuracy on all rule items; ties: the smaller adapter, then the earlier run."""
    return max(
        labels,
        key=lambda lab: (
            M.correct(M.rule_records(split, lab)),
            -parameters(lab),
            -ORDER.index(lab),
        ),
    )


def record(entry):
    path = ROOT / "round2.json"
    log = json.loads(path.read_text()) if path.exists() else []
    log.append(entry)
    path.write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n")


def structure():
    """Round 2 (section 10.2): rank 16 on the last 16 layers, or U1-hi's structure if U1-hi
    was trained and beats U1 on dev6 L0 with a paired lower bound above 0. Prints the build
    options."""
    entry = {"decision": "structure", "options": "", "u1hi_minus_u1": None}
    if M.rule_records("dev6", "p6u1hi"):
        d = paired("dev6", "p6u1", "p6u1hi")
        entry["u1hi_minus_u1"] = d
        if d["interval95"] and d["interval95"][0] > 0:
            shape = build_train6.structure_of("u1hi")
            entry["options"] = f"--layers {shape['layers']} --rank {shape['rank']}" + (
                " --grad-checkpoint" if shape["grad_checkpoint"] else ""
            )
    record(entry)
    print(entry["options"])


def nxt():
    """Round-2 loop (section 10.2). The first continuation starts from the best of U1, U1-hi
    and U2 on dev6; a continuation that beats its parent on dev6 becomes the next parent;
    at most three continuations. Prints "<recipe> <parent recipe>" or "stop"."""
    done = [lab for lab in CONTINUATIONS if M.rule_records("dev6", lab)]
    score = lambda lab: M.correct(M.rule_records("dev6", lab))  # noqa: E731
    if not done:
        if not M.rule_records("dev6", "p6u2"):
            sys.exit("U2 has no dev6 result yet")
        pool = [lab for lab in ("p6u1", "p6u1hi", "p6u2") if M.rule_records("dev6", lab)]
        parent = best(pool, "dev6")
        entry = {"decision": "next", "next": "u3", "parent": parent[2:],
                 "dev6_correct": {lab: score(lab) for lab in pool}}  # fmt: skip
    else:
        last = done[-1]
        parent = "p6" + P6.manifest(last[2:])["parent"]
        better = score(last) > score(parent)
        following = CONTINUATIONS[len(done)] if len(done) < len(CONTINUATIONS) else None
        entry = {"decision": "next", "next": following[2:] if better and following else None,
                 "parent": last[2:] if better and following else None,
                 "dev6_correct": {last: score(last), parent: score(parent)},
                 "child_minus_parent": paired("dev6", parent, last)}  # fmt: skip
    record(entry)
    print(f"{entry['next']} {entry['parent']}" if entry["next"] else "stop")


def sealed_extra():
    """Section 10.4: the sealed L0 runs added by round 2 (U1, U2 and every ancestor of M)."""
    m = json.loads((ROOT / "selection.json").read_text())["selected"]
    labels = ["p6u1", "p6u2", *("p6" + r for r in P6.lineage(m[2:])[1:])]
    keep = [lab for lab in dict.fromkeys(labels) if lab != m and M.rule_records("dev6", lab)]
    print(" ".join(keep))


def select():
    screening = json.loads((ROOT / "screening.json").read_text())
    pool = dict.fromkeys(
        (screening["best_s_run_on_dev6"], "p6u1", "p6u1hi", "p6u2", *CONTINUATIONS)
    )  # section 10.4
    candidates = [lab for lab in pool if M.rule_records("valid6", lab)]
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
                      "trainable_parameters": parameters(lab)}  # fmt: skip
    passing = [lab for lab in candidates if table[lab]["calendar_gate"]]
    chosen = best(passing or candidates, "valid6")
    out = {"rule": "docs/phase6_preregistration.md sections 6 and 10.4 (frozen before validation scoring)", "candidates": table,
           "passing_calendar_gate": passing, "calendar_claim_failed": not passing, "selected": chosen}  # fmt: skip
    (ROOT / "selection.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(out, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    {
        "screen": screen,
        "select": select,
        "flags": flags,
        "best": best_s,
        "small-steps-dropped": small_steps_dropped,
        "structure": structure,
        "next": nxt,
        "sealed-extra": sealed_extra,
    }[sys.argv[1]]()
