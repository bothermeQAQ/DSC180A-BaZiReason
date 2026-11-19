"""Apply the frozen Phase 6 decision rules (docs/phase6_preregistration.md, sections 5 and 6).

    .venv/bin/python scripts/phase6/decide6.py screen    # dev6: U1 recipe -> results/phase6/screening.json
    .venv/bin/python scripts/phase6/decide6.py select    # valid6: selection -> results/phase6/selection.json
    .venv/bin/python scripts/phase6/decide6.py flags     # U1 build options implied by screening.json

Round 2 (section 10; every decision is appended to results/phase6/round2.json):

    .venv/bin/python scripts/phase6/decide6.py structure # adapter options of round 2
    .venv/bin/python scripts/phase6/decide6.py next      # "<recipe> <parent>" of the next continuation, or "stop"
    .venv/bin/python scripts/phase6/decide6.py ud        # 12.14: UD's parent and twin, after the chain
    .venv/bin/python scripts/phase6/decide6.py user 10 start|stop "<the user's words>"  # 12.15

Reads saved records only. "select" writes results/phase6/selection.json, which opens sealed6.
"""

import datetime
import json
import sys
import time
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
# Round-2 chain (sections 10.2 and 10.5): the first level is a fork with two arms.
# Level 3 forks on the derivation version (10.5, 12.6), level 4 on composition rows (12.7).
LEVELS = (("p6u3", "p6u3v"), ("p6u4", "p6u4c"), ("p6u5",), ("p6u6",), ("p6u7",), ("p6u8",), ("p6u9",), ("p6u10",))  # fmt: skip
# Level 4 and the levels after it ask about years of the chart's life (preregistration 12.11)
# and weight the calendar rows by family (12.13); later levels inherit both from their
# parent's manifest.
ARM_OPTIONS = {
    "p6u3v": "--steps 2",
    "p6u4": "--years life --calendar-weight-by family",
    "p6u4c": "--compose-share 0.10 --years life --calendar-weight-by family",
}
CONTINUATIONS = tuple(lab for level in LEVELS for lab in level)
# Preregistration 12.15 (the user's message of 18 November, 16:05): level 10 starts only after the
# user's choice is recorded ("decide6.py user 10 start|stop <the user's words>"). Until then "next"
# waits, and the GPU is idle.
HELD = (10,)
# Preregistration 12.14: UD, built after the chain from its best run with derivation version 3.
# A candidate like every continuation, never a level of the chain.
EXTRA = ("p6ud",)
ORDER = ("p6s1", "p6s2", "p6s3", "p6s2s", "p6u1", "p6u1hi", "p6u2", *CONTINUATIONS, *EXTRA)  # earlier first
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


def recorded():
    path = ROOT / "round2.json"
    return json.loads(path.read_text()) if path.exists() else []


def record(entry):
    log = recorded()
    log.append(entry)
    (ROOT / "round2.json").write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n")


def users_choice(level, wait=True, every=60):
    """Preregistration 12.15: the user's recorded choice for a held level, "start" or "stop".
    With wait, reads round2.json again every minute until the choice is there; the first time,
    says so on stderr (the queue's log)."""
    told = False
    while True:
        try:
            found = [e["choice"] for e in recorded() if e.get("decision") == "user" and e.get("level") == level]  # fmt: skip
        except json.JSONDecodeError:  # being written by "user": read it again
            found = []
        if found or not wait:
            return found[0] if found else None
        if not told:
            print(f"HOLD level {level} {datetime.datetime.now():%m-%d %H:%M:%S}: waits for the user's choice "
                  f"(decide6.py user {level} start|stop)", file=sys.stderr, flush=True)  # fmt: skip
            told = True
        time.sleep(every)


def user():
    """Preregistration 12.15: records the user's choice for a held level, once:
    decide6.py user <level> start|stop "<the user's words>"."""
    args = sys.argv[2:]
    if len(args) < 3 or not args[0].isdigit() or int(args[0]) not in HELD or args[1] not in ("start", "stop"):  # fmt: skip
        sys.exit("usage: decide6.py user <held level> start|stop <the user's words>")
    level, choice, said = int(args[0]), args[1], " ".join(args[2:])
    earlier = users_choice(level, wait=False)
    if earlier:
        sys.exit(f"level {level}: the user's choice is already recorded ({earlier})")
    record({"decision": "user", "level": level, "choice": choice, "said": said,
            "time": f"{datetime.datetime.now():%Y-%m-%d %H:%M}"})  # fmt: skip
    print(f"level {level}: {choice}")


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


def prompt():
    """Section 12.4: U2's system prompt. Short, unless S2s minus S2 on dev6 has a 95% upper
    bound below 0 on the rule items or on the calendar rows. Prints the build option."""
    rules = paired("dev6", "p6s2", "p6s2s")
    cal = paired("dev6", "p6s2", "p6s2s", parts=("calendar",))
    worse = any(d["interval95"] and d["interval95"][1] < 0 for d in (rules, cal))
    entry = {"decision": "prompt", "options": "--system long" if worse else "--system short",
             "s2s_minus_s2_rule_items": rules, "s2s_minus_s2_calendar": cal}  # fmt: skip
    record(entry)
    print(entry["options"])


def nxt():
    """Round-2 loop (sections 10.2 and 10.5).

    The first level starts from the best of U1, U1-hi and U2 on dev6. Each later level starts
    from the best arm of the level before it. The loop ends when two levels in a row fail to
    exceed the best dev6 accuracy reached so far in the chain (the first parent included), or
    after the last level. Prints one line per arm, "<recipe> <parent recipe> [options]", or
    "stop".

    A level counts as reached when one of its arms has a dev6 result. If another arm of it has
    none (the queue was restarted between the arms, or the arm failed), that arm is asked for
    once more before the chain goes on, so that a restart cannot drop an arm of a pair."""
    score = lambda lab: M.correct(M.rule_records("dev6", lab))  # noqa: E731
    done = [[lab for lab in level if M.rule_records("dev6", lab)] for level in LEVELS]
    done = done[: next((i for i, arms in enumerate(done) if not arms), len(done))]
    arm_line = lambda run, parent: f"{run} {parent} {ARM_OPTIONS.get('p6' + run, '')}".strip()  # noqa: E731
    # A level that was interrupted (the queue restarted, or an arm failed): its arms without a
    # result are asked for once more, with the same parent, before the chain goes on.
    if done:
        missing = [lab[2:] for lab in LEVELS[len(done) - 1] if lab not in done[-1]]
        asked = any(
            e.get("decision") == "complete" and e.get("runs") == missing for e in recorded()
        )
        if missing and not asked:
            parent = P6.manifest(done[-1][0][2:])["parent"]
            record({"decision": "complete", "level": len(done) + 2, "runs": missing, "parent": parent,
                    "dev6_correct": {lab: score(lab) for lab in done[-1]}})  # fmt: skip
            print("\n".join(arm_line(run, parent) for run in missing))
            return
    entry = {"decision": "next", "level": len(done) + 3, "runs": [], "parent": None}
    if not done:
        if not M.rule_records("dev6", "p6u2"):
            sys.exit("U2 has no dev6 result yet")
        pool = [lab for lab in ("p6u1", "p6u1hi", "p6u2") if M.rule_records("dev6", lab)]
        parent = best(pool, "dev6")
        entry["dev6_correct"] = {lab: score(lab) for lab in pool}
    else:
        first = "p6" + P6.manifest(done[0][0][2:])["parent"]
        top, strikes, tops = score(first), 0, []
        for arms in done:
            parent = best(arms, "dev6")
            tops.append(parent)
            if score(parent) > top:
                top, strikes = score(parent), 0
            else:
                strikes += 1
        entry["dev6_correct"] = {
            first: score(first),
            **{a: score(a) for arms in done for a in arms},
        }
        entry["levels_without_a_new_best"] = strikes
        if strikes >= 2 or len(done) == len(LEVELS):
            parent = None
    # 12.15: a held level waits for the user's choice; "stop" ends the chain here.
    if parent and entry["level"] in HELD and users_choice(entry["level"]) == "stop":
        parent, entry["ended_by"] = None, "user"
    if parent:
        entry["parent"] = parent[2:]
        entry["runs"] = [lab[2:] for lab in LEVELS[len(done)]]
    record(entry)
    lines = [arm_line(run, entry["parent"]) for run in entry["runs"]]
    print("\n".join(lines) if lines else "stop")


def ud():
    """Preregistration 12.14, after the chain: UD's parent is the chain's best run on dev6 (its
    runs and its first parent; ties as in best()), and its twin the chain level built from that
    parent (the first arm of a two-arm level). Prints "ud <parent> --steps 3 --seed-from <twin>"
    (without a twin: no --seed-from) and records the decision once; a restart repeats it."""
    earlier = [e for e in recorded() if e.get("decision") == "ud"]
    if not earlier:
        nexts = [e for e in recorded() if e.get("decision") == "next"]
        stopped = bool(nexts) and not nexts[-1]["runs"]
        reached = [lab for level in LEVELS for lab in level if M.rule_records("dev6", lab)]
        after_cutoff = datetime.date.today() > datetime.date(2026, 11, 27)
        if not reached or not (stopped or len(reached) == len(CONTINUATIONS) or after_cutoff):
            sys.exit("the chain has not ended")
        first = "p6" + P6.manifest(reached[0][2:])["parent"]
        parent = best([first, *reached], "dev6")
        twins = [lab for level in LEVELS for lab in level
                 if lab in reached and P6.manifest(lab[2:]).get("parent") == parent[2:]]  # fmt: skip
        twin = twins[0] if twins else None
        record({"decision": "ud", "parent": parent[2:], "twin": twin[2:] if twin else None,
                "dev6_correct": {lab: M.correct(M.rule_records("dev6", lab)) for lab in [first, *reached]}})  # fmt: skip
        earlier = [e for e in recorded() if e.get("decision") == "ud"]
    entry = earlier[0]
    print(f"ud {entry['parent']} --steps 3" + (f" --seed-from {entry['twin']}" if entry["twin"] else ""))


def sealed_extra():
    """Sections 10.4 and 10.5: the sealed L0 runs added by round 2 (U1, U2, the two arms of
    the fork and every ancestor of M)."""
    m = json.loads((ROOT / "selection.json").read_text())["selected"]
    # Sections 12.4 and 12.7: S2 and S2s (H7), U3 and U3v (H9), U4 and U4c (H11), S1 and S1-8B
    # (H10), U1, U2 and M's ancestors
    labels = ["p6s2", "p6s2s", "p6u3", "p6u3v", "p6u4", "p6u4c", "p6s1", "p8s1", "p6u1", "p6u2"]
    labels += ["p6" + r for r in P6.lineage(m[2:])[1:]]
    # Section 12.14: UD and its twin (H12)
    for e in recorded():
        if e.get("decision") == "ud":
            labels += ["p6ud"] + (["p6" + e["twin"]] if e["twin"] else [])
    keep = [lab for lab in dict.fromkeys(labels) if lab != m and M.rule_records("dev6", lab)]
    print(" ".join(keep))


def select():
    screening = json.loads((ROOT / "screening.json").read_text())
    pool = dict.fromkeys(
        (screening["best_s_run_on_dev6"], "p6s2s", "p6u1", "p6u1hi", "p6u2", *CONTINUATIONS, *EXTRA)
    )  # sections 12.4 and 12.14
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
        "prompt": prompt,
        "next": nxt,
        "ud": ud,
        "user": user,
        "sealed-extra": sealed_extra,
    }[sys.argv[1]]()
