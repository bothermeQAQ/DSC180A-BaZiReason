"""Descriptive sealed6 numbers for docs/phase6_round2_report.md, section 0 (prints only; writes nothing).

    PYTHONPATH=src:scripts/phase6:scripts/phase5 .venv/bin/python scripts/phase6/sealed_extras6.py

Reads the saved sealed6 records. The registered comparisons are in results/phase6/RESULTS.md
(make_report6.py); this script adds the tables and breakdowns the report quotes besides them:

  1. every run at L0 (all rule items, without the combinations, the combinations, calendar);
  2. paired differences of the selected model against the others (decide6.paired);
  3. level minus L0 for each model with levels of help;
  4. L5 (the Phase 5 hybrid system) against the selected model alone, paired by item;
  5. the selected model's right answers by question type;
  6. its calendar errors by kind;
  7. its answers right at L0 but wrong at L2 or L3: unreadable, or which conclusion field changed.

Items 5 to 7 are descriptive counts, not registered measures.
"""

import collections
import json
import re
import sys

sys.path.insert(0, "scripts/phase5")
sys.path.insert(0, "scripts/phase6")
import decide6 as D  # noqa: E402
import make_report6 as R  # noqa: E402

S = "sealed6"
NO_COMBO = tuple(p for p in R.RULE_PARTS if p != "combo")
L5_PARTS = ("rules", "indep", "blind")


def fmt(d):
    lo, hi = d["interval95"]
    return f"{100 * d['difference']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"


def every_run(selected):
    rows = []
    for lab in R.labels_in(S):
        recs = R.rule_records(S, lab)
        if not recs:
            continue
        combo = R.load(S, lab, "combo")
        cal = R.load(S, lab, "calendar")
        rows.append((R.correct(recs), lab, len(recs), R.correct(combo), len(combo), cal))
    print("1. Every run on sealed6, model alone (L0)\n")
    print("| Run | All rule items | Without combinations | Combinations | Calendar |")
    print("|---|---:|---:|---:|---:|")
    for k, lab, n, kc, nc, cal in sorted(rows, key=lambda x: (-x[0], x[1])):
        star = "**" if lab == selected else ""
        cal_cell = R.pct(R.correct(cal), len(cal)) if cal else "–"
        print(
            f"| {star}{R.name(lab)}{star} | {R.pct(k, n)} | {R.pct(k - kc, n - nc)} | {kc}/{nc} | {cal_cell} |"
        )
    print()


def against(selected):
    print(f"2. {R.name(selected)} minus each run, paired, points [95%]\n")
    print("| Minus | All rule items | Without combinations | Calendar |")
    print("|---|---|---|---|")
    for lab in ("p4a1", "p5g1", "p2", "base", "p6u2", "p6u5", "p6u6", "p6u7"):
        cells = [fmt(D.paired(S, lab, selected)), fmt(D.paired(S, lab, selected, parts=NO_COMBO))]
        cells.append(fmt(D.paired(S, lab, selected, parts=("calendar",))))
        print(f"| {R.name(lab)} | " + " | ".join(cells) + " |")
    print()


def levels():
    print("3. Level minus L0, all rule items, paired, points [95%]\n")
    for lab in R.labels_in(S):
        base = R.rule_records(S, lab, "L0")
        cells = []
        for lv in ("L1", "L2", "L3"):
            recs = R.rule_records(S, lab, lv)
            if base and recs:
                cells.append(f"{lv} {R.diff(base, recs)[0]}")
        if cells:
            print(f"- {R.name(lab)}: " + "; ".join(cells))
    print()


def l5(selected):
    print(f"4. L5 (system accuracy) against {R.name(selected)} alone, paired by item\n")
    alone = {p: R.load(S, selected, p) for p in L5_PARTS}
    for gen in R.labels_in(S):
        if not R.load(S, gen, "rules", "L5"):
            continue
        system = {
            p: [dict(r, score=r["final_score"]) for r in R.load(S, gen, p, "L5")] for p in L5_PARTS
        }
        reply = sum(R.correct(R.load(S, gen, p, "L5")) for p in L5_PARTS)
        for p in L5_PARTS:
            print(f"- system with {R.name(gen)}, {p}: {R.correct(system[p])}/{len(system[p])}; "
                  f"alone {R.correct(alone[p])}/{len(alone[p])}; alone minus system {R.diff(system[p], alone[p])[0]}")  # fmt: skip
        a = [r for p in L5_PARTS for r in system[p]]
        b = [r for p in L5_PARTS for r in alone[p]]
        print(f"- system with {R.name(gen)}, all three: {R.correct(a)}/{len(a)}; alone {R.correct(b)}/{len(b)}; "
              f"alone minus system {R.diff(a, b)[0]}; the generator's own replies inside the system {reply}/{len(a)}")  # fmt: skip
    print()


def by_type(selected):
    print(f"5. {R.name(selected)} at L0 by question type (combinations and facts by kind)\n")
    n, k = collections.Counter(), collections.Counter()
    for r in R.rule_records(S, selected):
        intent = r.get("intent") or ""
        head = intent.split(":")[0]
        t = ":".join(intent.split(":")[:2]) if head in ("combo", "fact") else head
        n[t] += 1
        k[t] += bool(r["score"]["correct"])
    for t in sorted(n, key=lambda x: (k[x] / n[x], x)):
        print(f"- {t}: {R.pct(k[t], n[t])}")
    print()


def calendar(selected):
    print(f"6. {R.name(selected)}'s calendar errors by kind\n")
    wrong = [r for r in R.load(S, selected, "calendar") if not r["score"]["correct"]]
    kinds, pillars = collections.Counter(), collections.Counter()
    for r in wrong:
        s = r["score"]
        if not s.get("parsed"):
            kinds["no readable answer"] += 1
        elif s.get("false_refusal"):
            kinds["answerable request refused"] += 1
        elif s.get("definitive_on_nonanswer"):
            kinds["definite chart where none exists"] += 1
        elif s.get("status_correct") and s.get("expected_status") == "ok":
            kinds["right decision, wrong pillars"] += 1
            pillars.update(p for p, ok in (s.get("pillar_matches") or {}).items() if not ok)
        elif s.get("status_correct"):
            kinds["right decision, wrong reasons"] += 1
        else:
            kinds["other wrong decision"] += 1
    print(f"- {len(wrong)} wrong: {dict(kinds)}")
    print(f"- wrong pillars among the right decisions: {dict(pillars)}")
    print()


def last_json(text):
    for m in reversed(list(re.finditer(r"\{", text or ""))):
        try:
            return json.loads(text[m.start() :].strip().split("\n")[0])
        except ValueError:
            continue
    return None


def lost(selected):
    print(
        f"7. {R.name(selected)}: right at L0, wrong at L2 or L3 (template rules and blind wording)\n"
    )
    for lv in ("L2", "L3"):
        unreadable, fields, total = 0, collections.Counter(), 0
        for part in ("rules", "blind"):
            first = {r["id"]: r for r in R.load(S, selected, part)}
            for r in R.load(S, selected, part, lv):
                if not first[r["id"]]["score"]["correct"] or r["score"]["correct"]:
                    continue
                total += 1
                if not r["score"]["parsed"]:
                    unreadable += 1
                    continue
                j0, j1 = last_json(first[r["id"]]["raw_output"]), last_json(r["raw_output"])
                if isinstance(j0, dict) and isinstance(j1, dict):
                    head = (r.get("intent") or "").split(":")[0]
                    fields.update(
                        f"{head}.{f}" for f in set(j0) | set(j1) if j0.get(f) != j1.get(f)
                    )
        top = dict(sorted(fields.items(), key=lambda kv: (-kv[1], kv[0]))[:10])  # ties by name
        print(
            f"- {lv}: {total} lost, {unreadable} without a readable conclusion; changed fields: {top}"
        )
    print()


def main():
    selected = json.load(open(R.ROOT / "selection.json"))["selected"]
    every_run(selected)
    against(selected)
    levels()
    l5(selected)
    by_type(selected)
    calendar(selected)
    lost(selected)


if __name__ == "__main__":
    main()
