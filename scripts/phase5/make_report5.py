"""Build results/phase5/RESULTS.md and summary.json from the saved Phase 5 outputs.

    .venv/bin/python scripts/phase5/make_report5.py

Reads saved records only; runs no model. Sections appear only for the runs that exist.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from bazireason.p2_eval import paired_difference

ROOT = Path("results/phase5")
SAFE = str.maketrans({":": "-", "/": "-"})
NAMES = {"base": "Untuned base", "p2": "Phase 2", "p3r2": "P3 round 2", "p3r3": "P3 round 3", "p4a1": "P4 A1"}
COND = {"w0": "model alone", "gf": "full fact sheet", "go": "oracle-routed", "gr": "routed", "grv": "routed + verified"}
RULE_PARTS = ("rules", "indep", "blind")


def name(label):
    if label.startswith("mix"):
        _, a, b, alpha = label.split(":")
        return f"Merged {alpha} ({NAMES.get(a, a)}→{NAMES.get(b, b)})"
    return NAMES.get(label, label)


def load(split, label, part, cond):
    path = ROOT / split / label.translate(SAFE) / f"{part}_{cond}.jsonl"
    if not path.exists():
        return None
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def labels_in(split):
    folder = ROOT / split
    if not folder.exists():
        return []
    found = [p.name for p in folder.iterdir() if p.is_dir()]
    back = {lab.translate(SAFE): lab for lab in ("base", "p2", "p3r2", "p3r3", "p4a1")}
    out = []
    for f in sorted(found):
        if f in back:
            out.append(back[f])
        elif f.startswith("mix-"):
            _, a, b, alpha = f.split("-", 3)
            out.append(f"mix:{a}:{b}:{alpha}")
    return out


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "–"


def diff(first, second, key="correct"):
    d = paired_difference(first, second, key=key)
    if d["interval95"] is None:
        return f"{100 * d['difference']:+.1f}", d
    lo, hi = d["interval95"]
    return f"{100 * d['difference']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]", d


def scores(recs, final=False):
    out = []
    for r in recs:
        s = dict(r["final_score"] if final else r["score"])
        s.setdefault("group", r["id"])
        s["id"] = r["id"]
        out.append(s)
    return out


def contradiction_scores(recs, final=False):
    out = []
    for r in recs:
        bad = r["final_contradictions"] if final else r.get("check", {}).get("contradictions", [])
        out.append({"id": r["id"], "group": r["id"], "contradiction": bool(bad)})
    return out


def condition_table(lines, summary, split):
    lines += [f"### Conditions on `{split}` (rules + independent + blind)\n",
              "| System | " + " | ".join(COND[c] for c in COND) + " |", "|---|" + "---:|" * len(COND)]  # fmt: skip
    block = {}
    for label in labels_in(split):
        cells, rec = [], {}
        for cond in COND:
            parts = [load(split, label, p, cond) for p in RULE_PARTS]
            if not all(parts):
                cells.append("–")
                continue
            recs = [r for p in parts for r in p]
            k = sum(bool(r["score"]["correct"]) for r in recs)
            text = pct(k, len(recs))
            if cond == "grv":
                kf = sum(bool(r["final_score"]["correct"]) for r in recs)
                text = f"{pct(k, len(recs))} → final {pct(kf, len(recs))}"
                rec["grv_final"] = kf
            cells.append(text)
            rec[cond] = {"k": k, "n": len(recs)}
        if rec:
            lines.append(f"| {name(label)} | " + " | ".join(cells) + " |")
            block[label] = rec
    lines.append("")
    summary.setdefault("conditions", {})[split] = block


def by_part_table(lines, summary, split, conds=("w0", "gr", "grv")):
    lines += [f"### By part on `{split}` (first pass; grv shows the final answer)\n",
              "| System | Condition | Template rules | Independent | Blind |", "|---|---|---:|---:|---:|"]  # fmt: skip
    for label in labels_in(split):
        for cond in conds:
            row = []
            for p in RULE_PARTS:
                recs = load(split, label, p, cond)
                if not recs:
                    row.append("–")
                    continue
                key = "final_score" if cond == "grv" else "score"
                row.append(pct(sum(bool(r[key]["correct"]) for r in recs), len(recs)))
            if any(c != "–" for c in row):
                lines.append(f"| {name(label)} | {COND[cond]} | " + " | ".join(row) + " |")
    lines.append("")


def faithfulness(lines, summary, split):
    lines += [f"### Faithfulness on `{split}` (rules + independent + blind)\n",
              "A reply is unfaithful when a fixed-form statement in it contradicts the rule code "
              "(bazireason.claim_check) or its conclusion contradicts the plan it was given.\n",
              "| System | Condition | Replies with a contradicted statement | Conclusion contradicts the plan | Repaired | Engine fallback | Final replies with a contradiction |",
              "|---|---|---:|---:|---:|---:|---:|"]  # fmt: skip
    block = {}
    for label in labels_in(split):
        for cond in COND:
            parts = [load(split, label, p, cond) for p in RULE_PARTS]
            if not all(parts):
                continue
            recs = [r for p in parts for r in p]
            n = len(recs)
            bad = sum(bool(r["check"]["contradictions"]) for r in recs)
            disagree = sum(r["check"]["conclusion_agrees"] is False for r in recs)
            rep = fb = fin = "–"
            if cond == "grv":
                rep = sum(bool(r["repaired"]) for r in recs)
                fb = sum(bool(r["fallback"]) for r in recs)
                fin = pct(sum(bool(r["final_contradictions"]) for r in recs), n)
            lines.append(f"| {name(label)} | {COND[cond]} | {pct(bad, n)} | {pct(disagree, n)} | {rep} | {fb} | {fin} |")
            block[f"{label}/{cond}"] = {"n": n, "contradiction": bad, "conclusion_disagrees": disagree}
    lines.append("")
    summary.setdefault("faithfulness", {})[split] = block


def router(lines, summary, split):
    rows = []
    for label in labels_in(split):
        recs = load(split, label, "blind", "gr")
        if recs:
            rows = recs
            break
    if not rows:
        return
    ok = sum(bool(r["routed_ok"]) for r in rows)
    wrong = Counter((r["intent_oracle"].split(":")[0], r["intent_used"].split(":")[0]) for r in rows if not r["routed_ok"])
    computable_refused = sum(r["intent_used"] == "prediction" and r["intent_oracle"] != "prediction" for r in rows)
    lines += [f"### Router on `{split}` blind questions\n",
              f"Intent exactly right: {pct(ok, len(rows))}. Computable questions sent to the prediction refusal: {computable_refused}.\n"]  # fmt: skip
    if wrong:
        lines.append("Most common errors (true → routed): " + "; ".join(f"{a} → {b} ×{n}" for (a, b), n in wrong.most_common(8)) + "\n")
    summary.setdefault("router", {})[split] = {"ok": ok, "n": len(rows), "computable_refused": computable_refused}


def conversations(lines, summary, split):
    found = []
    for label in labels_in(split):
        for cond in ("w0", "grv"):
            path = ROOT / split / label.translate(SAFE) / f"convo_{cond}.jsonl"
            if path.exists():
                found.append((label, cond, [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]))
    if not found:
        return
    lines += [f"### Conversations on `{split}` (five turns per dialogue)\n",
              "| System | Condition | Dialogues fully right | … at first pass | Turn 1 | Turn 2 (why) | Turn 3 (pushback) | Turn 4 (prediction) | Turn 5 (switch) |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    block = {}
    for label, cond, rows in found:
        n = len(rows)
        per = [sum(bool(r["turns"][k]["final_score"]["correct"]) for r in rows) for k in range(5)]
        ok = sum(bool(r["dialogue_ok"]) for r in rows)
        first = sum(bool(r["dialogue_ok_first_pass"]) for r in rows)
        lines.append(f"| {name(label)} | {COND[cond]} | {pct(ok, n)} | {pct(first, n)} | " + " | ".join(f"{k}/{n}" for k in per) + " |")
        block[f"{label}/{cond}"] = {"n": n, "ok": ok, "first_pass_ok": first, "per_turn": per}
    lines.append("")
    summary.setdefault("conversations", {})[split] = block


def calendar(lines, summary, split):
    labs = [lab for lab in labels_in(split) if load(split, lab, "calendar", "w0")]
    if not labs:
        return
    lines += [f"### Model alone on calendar requests (`{split}`)\n",
              "| Model | Calendar rows right | Minus Phase 2 [95%] | Balanced decision accuracy | Answerable refused | Definite where none exists |",
              "|---|---:|---|---:|---:|---:|"]  # fmt: skip
    ref = load(split, "p2", "calendar", "w0")
    block = {}
    for lab in labs:
        recs = load(split, lab, "calendar", "w0")
        sc = [r["score"] for r in recs]
        k = sum(bool(s["correct"]) for s in sc)
        d = diff([r["score"] for r in ref], sc)[0] if ref and lab != "p2" else "–"
        ok = [s for s in sc if s.get("expected_status") == "ok"]
        other = [s for s in sc if s.get("expected_status") not in (None, "ok")]
        bda = (sum(bool(s.get("status_correct")) for s in ok) / max(len(ok), 1) + sum(bool(s.get("status_correct")) for s in other) / max(len(other), 1)) / 2
        refused = sum(bool(s.get("false_refusal")) for s in sc)
        definite = sum(bool(s.get("definitive_on_nonanswer")) for s in sc)
        lines.append(f"| {name(lab)} | {pct(k, len(sc))} | {d} | {100 * bda:.1f}% | {refused}/{len(ok)} | {definite}/{len(other)} |")
        block[lab] = {"k": k, "n": len(sc), "balanced_decision": round(bda, 4), "false_refusal": refused, "definite_on_nonanswer": definite}
    lines.append("")
    summary.setdefault("calendar", {})[split] = block


def model_alone_rules(lines, summary, split):
    labs = [lab for lab in labels_in(split) if all(load(split, lab, p, "w0") for p in RULE_PARTS)]
    if not labs:
        return
    lines += [f"### Model alone on rule questions (`{split}`)\n", "| Model | Template rules | Independent | Blind | All |", "|---|---:|---:|---:|---:|"]
    for lab in labs:
        parts = {p: load(split, lab, p, "w0") for p in RULE_PARTS}
        cells = [pct(sum(bool(r["score"]["correct"]) for r in parts[p]), len(parts[p])) for p in RULE_PARTS]
        allr = [r for p in RULE_PARTS for r in parts[p]]
        lines.append(f"| {name(lab)} | " + " | ".join(cells) + f" | {pct(sum(bool(r['score']['correct']) for r in allr), len(allr))} |")
    lines.append("")


def primaries(lines, summary):
    sel_path = ROOT / "selection.json"
    if not sel_path.exists():
        return
    sel = json.loads(sel_path.read_text())
    g = sel["g_star"]
    lines += ["## Primary comparisons on the sealed split (pre-registered)\n"]
    prim = {}
    sys_recs = [r for p in RULE_PARTS for r in (load("sealed", g, p, "grv") or [])]
    a1 = [r for p in RULE_PARTS for r in (load("sealed", "p4a1", p, "w0") or [])]
    if sys_recs and a1:
        text, d = diff(scores(a1), scores(sys_recs, final=True))
        prim["P-1 system (grv, G*) minus A1 alone"] = (text, d)
    other = "p4a1" if g == "base" else g
    gt = [r for p in RULE_PARTS for r in (load("sealed", other, p, "gr") or [])]
    gb = [r for p in RULE_PARTS for r in (load("sealed", "base", p, "gr") or [])]
    if gt and gb:
        text, d = diff(scores(gb), scores(gt))
        prim[f"P-2 {name(other)} minus untuned base, both routed (gr)"] = (text, d)
    if sel.get("m_star_is_candidate"):
        m = sel["m_star"]
        cm, cp = load("sealed", m, "calendar", "w0"), load("sealed", "p2", "calendar", "w0")
        if cm and cp:
            text, d = diff(scores(cp), scores(cm))
            prim[f"P-3 {name(m)} minus Phase 2, calendar rows (margin −5)"] = (text, d)
    else:
        prim["P-3 merged adapter, calendar non-inferiority"] = ("not tested: M* was not a candidate on validation", None)
    lines += ["| Comparison | Difference, points [95%] |", "|---|---|"] + [f"| {k} | {v[0]} |" for k, v in prim.items()] + [""]
    summary["primary"] = {k: v[1] for k, v in prim.items()}
    summary["selection"] = sel


def main():
    lines = ["# Phase 5 results\n",
             "Generated by `scripts/phase5/make_report5.py` from the files in this folder. Real local inference "
             "(MLX, Apple M5, 32 GB), greedy decoding, one base model with adapters swapped in place. Design: "
             "`docs/phase5_preregistration.md`. Development data (`dev`) were read freely; `valid` was used for "
             "selection only; `sealed` was scored once after `selection.json` existed.\n"]  # fmt: skip
    summary = {}
    primaries(lines, summary)
    for split in ("sealed", "valid", "dev"):
        if not (ROOT / split).exists():
            continue
        lines.append(f"## Split `{split}`\n")
        condition_table(lines, summary, split)
        by_part_table(lines, summary, split)
        faithfulness(lines, summary, split)
        router(lines, summary, split)
        conversations(lines, summary, split)
        model_alone_rules(lines, summary, split)
        calendar(lines, summary, split)
    for extra in ("dev_alpha.json", "bank_check.json", "latency_valid.json", "latency_sealed.json"):
        path = ROOT / extra
        if path.exists():
            summary[extra.removesuffix(".json")] = json.loads(path.read_text())
    (ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str) + "\n")
    print(f"Wrote {ROOT / 'RESULTS.md'} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
