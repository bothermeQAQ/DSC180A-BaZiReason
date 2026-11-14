"""Build results/phase6/RESULTS.md and summary.json from saved Phase 6 records (no model).

    PYTHONPATH=src .venv/bin/python scripts/phase6/make_report6.py

Every table names its level. L0 is the model alone; L1-L4 add help but never an answer
substitution; L5 (the hybrid system) is reported last and apart. Sections appear only for
runs that exist.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "scripts/phase5")
sys.path.insert(0, "scripts/phase6")
import run6  # noqa: E402

from bazireason import workflow as W  # noqa: E402
from bazireason.claim_check import check_claims  # noqa: E402
from bazireason.p2_eval import paired_difference  # noqa: E402

ROOT = Path("results/phase6")
RULE_PARTS = ("rules", "indep", "blind", "combo", "cf")
NAMES = {
    "base": "Untuned base", "p2": "Phase 2", "p4a1": "P4 A1", "p5g1": "P5 G1",
    "p6u0": "U0 joint, earlier data", "p6s1": "S1 small steps", "p6s2": "S2 + wording", "p6s3": "S3 + evidence, self-check",
    "p6u1": "U1", "p6u1hi": "U1-hi",
}  # fmt: skip
ORDER = list(NAMES)
LOOKUP_KINDS = ("day_master", "hidden", "pillar", "spouse_star")
COMPUTED_KINDS = (
    "xun_kong",
    "element_count",
    "pattern",
    "relation",
    "stem_relation",
    "flow_year",
    "ten_god",
    "strength",
    "star",
    "life_stage",
)
LEVEL_NAMES = {
    "L0": "alone",
    "L1": "raw facts",
    "L2": "rule text",
    "L3": "right conclusion",
    "L3w": "wrong conclusion",
}


def name(label):
    return NAMES.get(label, label)


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "–"


def load(split, label, part, level="L0"):
    path = ROOT / split / label / f"{part}_{level}.jsonl"
    return (
        [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        if path.exists()
        else None
    )


def labels_in(split):
    folder = ROOT / split
    found = [p.name for p in folder.iterdir() if p.is_dir()] if folder.exists() else []
    return sorted(found, key=lambda x: ORDER.index(x) if x in ORDER else 99)


def correct(recs):
    return sum(bool(r["score"]["correct"]) for r in recs)


def scores(recs):
    out = []
    for r in recs:
        s = dict(r["score"])
        s.setdefault("group", r["id"])
        s["id"] = r["id"]
        out.append(s)
    return out


def diff(a, b):
    """Paired difference b − a in points with its 95% interval (records aligned by id)."""
    ids = [r["id"] for r in a]
    bmap = {r["id"]: r for r in b}
    if set(ids) != set(bmap):
        return "–", None
    d = paired_difference(scores(a), scores([bmap[i] for i in ids]))
    if d["interval95"] is None:
        return f"{100 * d['difference']:+.1f}", d
    lo, hi = d["interval95"]
    return f"{100 * d['difference']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]", d


def rule_records(split, label, level="L0"):
    parts = [load(split, label, p, level) for p in RULE_PARTS]
    return [r for p in parts for r in p] if all(parts) else None


# ---- sections ----------------------------------------------------------------------------------------


def fit_section(lines, summary):
    path = ROOT / "diag_fit.json"
    fits = sorted(ROOT.glob("fit_*.json"))
    if not path.exists() and not fits:
        return
    lines += ["## Fit to the training distribution (model alone, L0)\n",
              "Near-perfect on the training distribution with a low test score means a coverage problem; "
              "a low score on fresh training-distribution questions means under-fitting.\n",
              "| Model | Sample | Correct |", "|---|---|---:|"]  # fmt: skip
    out = {}
    if path.exists():
        d = json.loads(path.read_text())
        labels = {"literal": "A1's own training rows (literal)", "fresh": "fresh rows, A1's generator, training years",
                  "dev6": "dev6 rule rows (validation years)"}  # fmt: skip
        for k, v in d.items():
            lines.append(f"| P4 A1 | {labels[k]} | {pct(v['correct'], v['n'])} |")
        out["a1"] = d
    for f in fits:
        d = json.loads(f.read_text())
        lines.append(
            f"| {name(d['label'])} | fresh Phase 6 procedure questions, training years | {pct(d['correct'], d['n'])} |"
        )
        out[d["label"]] = d
    lines.append("")
    summary["fit"] = out


def l0_section(lines, summary, split):
    labs = [lab for lab in labels_in(split) if rule_records(split, lab)]
    if not labs:
        return
    lines += [f"### Model alone (L0) on `{split}`\n",
              "| Model | Template rules | Independent | Blind wording | Combinations | Counterfactual | All rule items | Calendar |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    block = {}
    for lab in labs:
        cells, entry = [], {}
        for part in RULE_PARTS:
            recs = load(split, lab, part)
            cells.append(pct(correct(recs), len(recs)))
            entry[part] = [correct(recs), len(recs)]
        allr = rule_records(split, lab)
        cal = load(split, lab, "calendar")
        entry["all"] = [correct(allr), len(allr)]
        if cal:
            entry["calendar"] = [correct(cal), len(cal)]
        lines.append(
            f"| {name(lab)} | "
            + " | ".join(cells)
            + f" | **{pct(correct(allr), len(allr))}** | {pct(correct(cal), len(cal)) if cal else '–'} |"
        )
        block[lab] = entry
    lines.append("")
    refs = [r for r in ("p4a1", "p6u0", "base") if r in labs]
    if refs:
        lines += [f"Paired differences on all rule items (`{split}`, L0), points [95%]:\n",
                  "| Model | " + " | ".join(f"minus {name(r)}" for r in refs) + " |", "|---|" + "---|" * len(refs)]  # fmt: skip
        for lab in labs:
            row = [
                diff(rule_records(split, r), rule_records(split, lab))[0] if lab != r else "–"
                for r in refs
            ]
            lines.append(f"| {name(lab)} | " + " | ".join(row) + " |")
        lines.append("")
    summary.setdefault("l0", {})[split] = block


def statements_section(lines, summary, split):
    labs = [lab for lab in labels_in(split) if rule_records(split, lab)]
    if not labs:
        return
    items = {it["id"]: it for p in RULE_PARTS for it in run6.load_items(split, p)}
    lines += [f"### Statements in L0 replies (`{split}`, claim checker)\n",
              "Look-ups: day master, hidden stems, pillars, spouse star. Computed: 旬空, element counts, pattern, relations, flow years, Ten Gods, strength, stars, life stages.\n",
              "| Model | Look-up statements wrong | Computed statements wrong | 旬空 | Element counts | Pattern | Relations | Replies with a wrong statement | Loops |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    block = {}
    for lab in labs:
        checked, wrong, replies_bad, loops = Counter(), Counter(), 0, 0
        recs = rule_records(split, lab)
        for r in recs:
            meta = items[r["id"]]["meta"]
            claims = check_claims(r["raw_output"], meta["pillars"], meta.get("gender"))
            for c in claims:
                checked[c["kind"]] += 1
                wrong[c["kind"]] += not c["ok"]
            replies_bad += any(not c["ok"] for c in claims if c["kind"] != "repetition")
            loops += W.has_loop(r["raw_output"])

        def frac(kinds, wrong=wrong, checked=checked):
            return pct(sum(wrong[k] for k in kinds), sum(checked[k] for k in kinds))

        lines.append(f"| {name(lab)} | {frac(LOOKUP_KINDS)} | {frac(COMPUTED_KINDS)} | {frac(('xun_kong',))} | {frac(('element_count',))} | "
                     f"{frac(('pattern',))} | {frac(('relation', 'stem_relation'))} | {pct(replies_bad, len(recs))} | {loops} |")  # fmt: skip
        block[lab] = {
            "checked": dict(checked),
            "wrong": dict(wrong),
            "replies_with_wrong": replies_bad,
            "n": len(recs),
            "loops": loops,
        }
    lines.append("")
    summary.setdefault("statements", {})[split] = block


def wording_section(lines, summary, split):
    labs = [lab for lab in labels_in(split) if load(split, lab, "rules")]
    if not labs:
        return
    items = {it["id"]: it for it in run6.load_items(split, "rules")}
    lines += [f"### Wording (`{split}`, L0)\n",
              "| Model | Template rules, training wordings | Template rules, held-out wording | Blind wording |", "|---|---:|---:|---:|"]  # fmt: skip
    block = {}
    for lab in labs:
        recs = load(split, lab, "rules")
        seen = [r for r in recs if items[r["id"]]["meta"].get("template") != 2]
        held = [r for r in recs if items[r["id"]]["meta"].get("template") == 2]
        blind = load(split, lab, "blind") or []
        lines.append(
            f"| {name(lab)} | {pct(correct(seen), len(seen))} | {pct(correct(held), len(held))} | {pct(correct(blind), len(blind))} |"
        )
        block[lab] = {
            "seen": [correct(seen), len(seen)],
            "held_out": [correct(held), len(held)],
            "blind": [correct(blind), len(blind)],
        }
    lines.append("")
    summary.setdefault("wording", {})[split] = block


def intent_section(lines, summary, split):
    labs = [lab for lab in labels_in(split) if rule_records(split, lab)]
    if not labs:
        return
    table = defaultdict(dict)
    for lab in labs:
        c = defaultdict(lambda: [0, 0])
        for r in rule_records(split, lab):
            intent = r.get("intent") or "?"
            fam = intent.split(":")[0] + (
                ":" + intent.split(":")[1] if intent.startswith(("fact", "combo")) else ""
            )
            c[fam][0] += bool(r["score"]["correct"])
            c[fam][1] += 1
        for fam, v in c.items():
            table[fam][lab] = v
    fams = sorted(table, key=lambda f: -max(v[1] for v in table[f].values()))
    lines += [f"### By question type (`{split}`, L0, all rule items)\n", "| Type | n | " + " | ".join(name(lab) for lab in labs) + " |",
              "|---|---:|" + "---:|" * len(labs)]  # fmt: skip
    for fam in fams:
        n = max(v[1] for v in table[fam].values())
        lines.append(
            f"| {fam} | {n} | "
            + " | ".join(
                f"{100 * table[fam][lab][0] / table[fam][lab][1]:.0f}%"
                if lab in table[fam]
                else "–"
                for lab in labs
            )
            + " |"
        )
    lines.append("")
    summary.setdefault("by_type", {})[split] = {f: dict(v) for f, v in table.items()}


def cf_section(lines, summary, split):
    labs = [lab for lab in labels_in(split) if load(split, lab, "cf")]
    if not labs:
        return
    items = {it["id"]: it for it in run6.load_items(split, "cf")}
    lines += [f"### Counterfactual pairs (`{split}`, L0): one character changed, the answer changes\n",
              "| Model | Items right | Pairs right on both charts | Pairs with the same answer on both charts |", "|---|---:|---:|---:|"]  # fmt: skip
    block = {}
    for lab in labs:
        recs = load(split, lab, "cf")
        pairs = defaultdict(list)
        for r in recs:
            pairs[items[r["id"]]["meta"]["pair"]].append(r)
        both = sum(all(r["score"]["correct"] for r in p) for p in pairs.values())
        from bazireason.p2_prompts import extract_final_json

        same = sum(
            len(p) == 2
            and extract_final_json(p[0]["raw_output"]) == extract_final_json(p[1]["raw_output"])
            for p in pairs.values()
        )
        lines.append(
            f"| {name(lab)} | {pct(correct(recs), len(recs))} | {pct(both, len(pairs))} | {pct(same, len(pairs))} |"
        )
        block[lab] = {"both": both, "pairs": len(pairs), "same_answer": same}
    lines.append("")
    summary.setdefault("cf", {})[split] = block


def calendar_section(lines, summary, split):
    labs = [lab for lab in labels_in(split) if load(split, lab, "calendar")]
    if not labs:
        return
    ref = load(split, "p2", "calendar")
    lines += [f"### Calendar requests (`{split}`, L0)\n",
              "| Model | Right | Minus Phase 2 [95%] | Balanced decision accuracy | Answerable refused | Definite where none exists |",
              "|---|---:|---|---:|---:|---:|"]  # fmt: skip
    block = {}
    for lab in labs:
        recs = load(split, lab, "calendar")
        sc = [r["score"] for r in recs]
        ok = [s for s in sc if s.get("expected_status") == "ok"]
        other = [s for s in sc if s.get("expected_status") not in (None, "ok")]
        bda = (
            sum(bool(s.get("status_correct")) for s in ok) / max(len(ok), 1)
            + sum(bool(s.get("status_correct")) for s in other) / max(len(other), 1)
        ) / 2
        d = diff(ref, recs)[0] if ref and lab != "p2" else "–"
        refused = sum(bool(s.get("false_refusal")) for s in sc)
        definite = sum(bool(s.get("definitive_on_nonanswer")) for s in sc)
        lines.append(
            f"| {name(lab)} | {pct(correct(recs), len(recs))} | {d} | {100 * bda:.1f}% | {refused}/{len(ok)} | {definite}/{len(other)} |"
        )
        block[lab] = {
            "k": correct(recs),
            "n": len(recs),
            "balanced_decision": round(bda, 4),
            "refused": refused,
            "definite": definite,
        }
    lines.append("")
    summary.setdefault("calendar", {})[split] = block


def levels_section(lines, summary, split):
    labs = [
        lab
        for lab in labels_in(split)
        if any(rule_records(split, lab, lv) for lv in ("L1", "L2", "L3", "L3w"))
    ]
    if not labs:
        return
    lines += [f"### Levels of help (`{split}`, all rule items; each level its own row of prompts)\n",
              "| Model | L0 alone | L1 raw facts | L2 rule text | L3 right conclusion | L3w wrong conclusion: right / follows the wrong one |",
              "|---|---:|---:|---:|---:|---:|"]  # fmt: skip
    block = {}
    for lab in labs:
        cells, entry = [], {}
        for lv in ("L0", "L1", "L2", "L3"):
            recs = rule_records(split, lab, lv)
            cells.append(pct(correct(recs), len(recs)) if recs else "–")
            if recs:
                entry[lv] = [correct(recs), len(recs)]
        w = rule_records(split, lab, "L3w")
        if w:
            follows = sum(r["follows_wrong"] for r in w)
            cells.append(f"{pct(correct(w), len(w))} / {pct(follows, len(w))}")
            entry["L3w"] = [correct(w), follows, len(w)]
        else:
            cells.append("–")
        lines.append(f"| {name(lab)} | " + " | ".join(cells) + " |")
        block[lab] = entry
    lines.append("")
    rows = []
    for lab in labs:
        a, b = rule_records(split, lab, "L0"), rule_records(split, lab, "L1")
        if a and b:
            rows.append(f"| {name(lab)} | {diff(a, b)[0]} |")
    if rows:
        lines += [
            "L1 minus L0 (does the model use raw facts?), points [95%]:\n",
            "| Model | L1 − L0 |",
            "|---|---|",
            *rows,
            "",
        ]
    summary.setdefault("levels", {})[split] = block


def selfcheck_section(lines, summary, split):
    labs = [lab for lab in labels_in(split) if rule_records(split, lab, "L4")]
    if not labs:
        return
    lines += [f"### Self-correction (`{split}`, L4: the model's own L0 reply, no value given)\n",
              "Fix rate: wrong first answers made right. Break rate: right first answers made wrong (60 sampled per part).\n",
              "| Model | Check | Fixed | Broke |", "|---|---|---:|---:|"]  # fmt: skip
    block = {}
    words = {
        "L4a": "a: re-check",
        "L4b": "b: told there is an error",
        "L4c": "c: told which field is wrong",
    }
    for lab in labs:
        recs = rule_records(split, lab, "L4")
        for v in ("L4a", "L4b", "L4c"):
            rs = [r for r in recs if r["level"] == v]
            wrong = [r for r in rs if not r["first_correct"]]
            right = [r for r in rs if r["first_correct"]]
            fixed, broke = sum(r["fixed"] for r in wrong), sum(r["broke"] for r in right)
            lines.append(
                f"| {name(lab)} | {words[v]} | {pct(fixed, len(wrong))} | {pct(broke, len(right))} |"
            )
            block.setdefault(lab, {})[v] = {
                "fixed": fixed,
                "wrong": len(wrong),
                "broke": broke,
                "right": len(right),
            }
    lines.append("")
    summary.setdefault("selfcheck", {})[split] = block


def main():
    lines = ["# Phase 6 results\n",
             "Generated by `scripts/phase6/make_report6.py` from the files in this folder. Real local inference (MLX, Apple M5, 32 GB), "
             "greedy decoding. Design: `docs/phase6_preregistration.md`. **Every table names its level; L0 is the model alone.** "
             "`dev6` is read freely, `valid6` is for selection only, `sealed6` is scored once after `selection.json`.\n"]  # fmt: skip
    summary = {}
    fit_section(lines, summary)
    for split in ("sealed6", "valid6", "dev6"):
        if not (ROOT / split).exists():
            continue
        lines.append(f"## Split `{split}`\n")
        for section in (
            l0_section,
            statements_section,
            wording_section,
            cf_section,
            intent_section,
            calendar_section,
            levels_section,
            selfcheck_section,
        ):
            section(lines, summary, split)
    (ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=str) + "\n"
    )
    print(f"Wrote {ROOT / 'RESULTS.md'} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
