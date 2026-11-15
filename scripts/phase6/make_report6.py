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
    "p6u2": "U2 short prompt", "p6u3": "U3 continuation", "p6u4": "U4 continuation", "p6u5": "U5 continuation",
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
    samples = {"": "sample A: Phase 6 procedure questions, training years (for S2, S3 and the U runs it "
                   "includes questions they train on)",
               "_clean": "sample A, only the charts the model never trained on",
               "_b": "sample B: 100 per question type, template wordings, charts never trained on"}  # fmt: skip
    for f in fits:
        d = json.loads(f.read_text())
        kind = next(k for k in ("_clean", "_b", "") if f.stem.endswith(k))
        lines.append(f"| {name(d['label'])} | {samples[kind]} | {pct(d['correct'], d['n'])} |")
        out[f.stem[4:]] = d
    lines.append("")
    by_b = {k[:-2]: v for k, v in out.items() if k.endswith("_b")}
    if by_b:
        labs = sorted(by_b, key=lambda x: ORDER.index(x) if x in ORDER else 99)
        lines += ["Sample B by question type (right answers of the scored questions):\n",
                  "| Type | " + " | ".join(name(x) for x in labs) + " |", "|---|" + "---:|" * len(labs)]  # fmt: skip
        for t in sorted({t for v in by_b.values() for t in v["by_type"]}):
            cells = [pct(*by_b[x]["by_type"][t]) if t in by_b[x]["by_type"] else "–" for x in labs]
            lines.append(f"| {t} | " + " | ".join(cells) + " |")
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
    lines += [f"### Statements in L0 replies (`{split}`, claim checker)\n",
              "Look-ups: day master, hidden stems, pillars, spouse star. Computed: 旬空, element counts, pattern, relations, flow years, Ten Gods, strength, stars, life stages.\n",
              "| Model | Look-up statements wrong | Computed statements wrong | 旬空 | Element counts | Pattern | Relations | Replies with a wrong statement | Loops |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    block = {}
    for lab in labs:
        cc = claim_counts(split, lab)
        checked, wrong, replies_bad, loops = (
            Counter(cc["checked"]),
            Counter(cc["wrong"]),
            cc["replies_with_wrong"],
            cc["loops"],
        )
        recs = rule_records(split, lab)

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


def truncation_section(lines, summary, split):
    path = ROOT / "truncation.json"
    if not path.exists():
        return
    data = json.loads(path.read_text()).get(split, {})
    labs = [lab for lab in labels_in(split) if lab in data]
    if not labs:
        return
    lines += [f"### Replies cut off at the token cap (`{split}`; scripts/phase6/truncation6.py)\n",
              "Caps: 1,100 tokens for rule items, 700 for calendar rows. A cut-off reply usually has no final JSON and scores wrong.\n",
              "| Model | Level | Rule items cut off | of which wrong | Mean reply tokens | Calendar cut off |",
              "|---|---|---:|---:|---:|---:|"]  # fmt: skip
    block = {}
    for lab in labs:
        d = data[lab]
        levels = sorted({k.rsplit("_", 1)[1] for k in d})
        for lv in levels:
            rule = [d[f"{p}_{lv}"] for p in RULE_PARTS if f"{p}_{lv}" in d]
            if not rule:
                continue
            n, cut = sum(v["n"] for v in rule), sum(v["cut_off"] for v in rule)
            wrong = sum(v["cut_off_and_wrong"] for v in rule)
            mean = sum(v["mean_tokens"] * v["n"] for v in rule) / n
            cal = d.get(f"calendar_{lv}")
            lines.append(
                f"| {name(lab)} | {lv} | {pct(cut, n)} | {wrong} | {mean:.0f} | {pct(cal['cut_off'], cal['n']) if cal else '–'} |"
            )
            block.setdefault(lab, {})[lv] = {
                "cut": cut,
                "n": n,
                "wrong": wrong,
                "mean_tokens": round(mean, 1),
            }
    lines.append("")
    summary.setdefault("truncation", {})[split] = block


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


# ---- training, decisions, sealed primaries, system ---------------------------------------------------

_CLAIMS = {}


def claim_counts(split, lab):
    """Claim-checker counts over a model's L0 rule replies (cached: the checker is slow)."""
    if (split, lab) in _CLAIMS:
        return _CLAIMS[(split, lab)]
    items = {it["id"]: it for p in RULE_PARTS for it in run6.load_items(split, p)}
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
    out = {
        "checked": dict(checked),
        "wrong": dict(wrong),
        "replies_with_wrong": replies_bad,
        "n": len(recs),
        "loops": loops,
    }
    _CLAIMS[(split, lab)] = out
    return out


def balanced(recs):
    sc = [r["score"] for r in recs]
    ok = [s for s in sc if s.get("expected_status") == "ok"]
    other = [s for s in sc if s.get("expected_status") not in (None, "ok")]
    return (sum(bool(s.get("status_correct")) for s in ok) / max(len(ok), 1)
            + sum(bool(s.get("status_correct")) for s in other) / max(len(other), 1)) / 2  # fmt: skip


def training_section(lines, summary):
    plans = sorted((ROOT / "training").glob("*/training_plan.json"))
    if not plans:
        return
    labels = {v: k for k, v in run6.FOLDERS.items()}
    lines += ["## Training runs\n",
              "| Run | Starts from | Rows | Sequence tokens | Layers / rank | Iterations done | Train loss, last 200 iterations | Validation loss, last | Peak memory | Hours |",
              "|---|---|---:|---:|---|---:|---:|---:|---:|---:|"]  # fmt: skip
    block = {}
    for path in plans:
        plan = json.loads(path.read_text())
        if not (path.parent / "metrics.jsonl").exists():
            continue
        mets = [
            json.loads(x)
            for x in (path.parent / "metrics.jsonl").read_text().splitlines()
            if x.strip()
        ]
        train = [m for m in mets if m["kind"] == "train"]
        valid = [m for m in mets if m["kind"] == "valid"]
        if not train or not valid:
            continue
        last = train[-1]["iteration"]
        tail = [m["train_loss"] for m in train if m["iteration"] > last - 200]
        cfg = Path(plan["config_file"])
        layers = rank = "?"
        if cfg.exists():
            text = cfg.read_text()
            for ln in text.splitlines():
                if ln.strip().startswith("num_layers:"):
                    layers = ln.split(":")[1].strip()
                if ln.strip().startswith("rank:"):
                    rank = ln.split(":")[1].strip()
        lab = labels.get(plan["run_name"], plan["run_name"])
        done = f"{last}/{plan['iterations']}" + ("" if last >= plan["iterations"] else " (running)")
        origin = "untuned base"
        if plan.get("init_adapter"):
            parent = Path(plan["init_adapter"]).parent.name
            origin = name(labels.get(parent, parent))
        lines.append(f"| {name(lab)} | {origin} | {plan['train_rows']:,} | {plan['train_sequence_tokens_per_epoch']:,} | {layers} / {rank} | {done} | "
                     f"{sum(tail) / len(tail):.4f} | {valid[-1]['val_loss']:.4f} | {max(m['peak_memory_gb'] for m in train):.1f} GB | "
                     f"{train[-1]['elapsed_seconds'] / 3600:.2f} |")  # fmt: skip
        block[lab] = {"rows": plan["train_rows"], "tokens": plan["train_sequence_tokens_per_epoch"], "iterations": [last, plan["iterations"]],
                      "train_loss_tail": round(sum(tail) / len(tail), 5), "val_loss": valid[-1]["val_loss"],
                      "hours": round(train[-1]["elapsed_seconds"] / 3600, 2)}  # fmt: skip
    lines += [
        "",
        "Average per-token loss is dominated by template text; the fit table above is the measure of what was learned.\n",
    ]
    summary["training"] = block


def decisions_section(lines, summary):
    scr, sel = ROOT / "screening.json", ROOT / "selection.json"
    if scr.exists():
        d = json.loads(scr.read_text())
        lines += ["## Screening decisions (dev6, frozen rule: preregistration section 5)\n",
                  "| Component | Run minus the run before it, all rule items, L0, points [95%] | Correct | Dropped |", "|---|---|---:|---|"]  # fmt: skip
        for s in d["steps"]:
            v = s["l0_all_rule_items"]
            iv = v["interval95"]
            cell = f"{100 * v['difference']:+.1f}" + (
                f" [{100 * iv[0]:+.1f}, {100 * iv[1]:+.1f}]" if iv else ""
            )
            lines.append(f"| {s['component']} ({name(s['run'])} − {name(s['before'])}) | {cell} | {v['a']} vs {v['b']} of {v['n']} | "
                         f"{'yes' if s['dropped'] else 'no'} |")  # fmt: skip
        lines += ["", f"Dropped from U1: {', '.join(d['drop']) or 'nothing'}. Calendar share for U1: {d.get('calendar_share') or 'unchanged (28%)'}. "
                  f"Best S run on dev6: {name(d['best_s_run_on_dev6'])}.\n"]  # fmt: skip
        summary["screening"] = d
    if sel.exists():
        d = json.loads(sel.read_text())
        lines += ["## Selection (valid6, frozen rule: preregistration sections 6 and 10.4)\n",
                  "| Candidate | All rule items, L0 | Calendar minus Phase 2 [95%] | Calendar gate (lower bound ≥ −7) |", "|---|---:|---|---|"]  # fmt: skip
        for lab, v in d["candidates"].items():
            c = v["calendar_minus_p2"]
            iv = c["interval95"]
            cell = f"{100 * c['difference']:+.1f}" + (
                f" [{100 * iv[0]:+.1f}, {100 * iv[1]:+.1f}]" if iv else ""
            )
            lines.append(
                f"| {name(lab)} | {pct(*v['l0_all_rule_items'])} | {cell} | {'passes' if v['calendar_gate'] else 'fails'} |"
            )
        lines += ["", f"**Selected: {name(d['selected'])}.**" + (" No candidate passed the calendar gate; the calendar claim fails." if d["calendar_claim_failed"] else ""), ""]  # fmt: skip
        summary["selection"] = d


def round2_section(lines, summary):
    """Round 2 (preregistration section 10): the measurement behind it and its decisions."""
    share, log = ROOT / "token_share.json", ROOT / "round2.json"
    block = {}
    if share.exists():
        d = json.loads(share.read_text())
        lines += ["## Where the training tokens go (`token_share.json`)\n",
                  f"The loss is computed on the final reply only. The main system prompt has {d['main_system_prompt_tokens']} tokens; "
                  f"the short one of round 2 has {d['short_system_prompt_tokens']}.\n",
                  "| Training file | Rows | System prompts | of which the main prompt | User turns | Earlier replies | Final reply |",
                  "|---|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
        for k, v in d["files"].items():
            sh = v["share"]
            lines.append(f"| {k} | {v['rows']:,} | {100 * sh['system']:.1f}% | {100 * v['main_system_prompt_share']:.1f}% | "
                         f"{100 * sh['user']:.1f}% | {100 * sh['earlier_replies']:.1f}% | {100 * sh['final_reply']:.1f}% |")  # fmt: skip
        t = d["tokenization"]
        lines += ["", f"Tokenization: {len(t['pairs_merged_into_one_token'])} of {t['pairs_checked']} stem-branch pairs are merged into "
                  f"one token, and {len(t['single_characters_not_one_token'])} stems or branches need more than one token.\n"]  # fmt: skip
        block["token_share"] = d
    if log.exists():
        entries = json.loads(log.read_text())
        lines += ["## Round-2 decisions (dev6, frozen rules: preregistration section 10)\n"]
        for e in entries:
            if e["decision"] == "structure":
                d = e["u1hi_minus_u1"]
                why = "U1-hi was not trained" if d is None else (
                    f"U1-hi minus U1 {100 * d['difference']:+.1f} [{100 * d['interval95'][0]:+.1f}, {100 * d['interval95'][1]:+.1f}]")  # fmt: skip
                lines.append(
                    f"- **Adapter structure:** {e['options'] or 'rank 16 on the last 16 layers'} ({why})."
                )
            else:
                counts = ", ".join(f"{name(k)} {v}" for k, v in e["dev6_correct"].items())
                what = (
                    f"{name('p6' + e['next'])} continues {name('p6' + e['parent'])}"
                    if e["next"]
                    else "the loop ends"
                )
                lines.append(
                    f"- **Loop:** {what} (dev6 right answers on all rule items: {counts})."
                )
        lines.append("")
        block["decisions"] = entries
    weights = {}
    for lab in ("p6u3", "p6u4", "p6u5"):
        manifest = Path("data/phase6/train") / lab[2:] / "manifest.json"
        if manifest.exists():
            weights[lab] = json.loads(manifest.read_text())["weighting"]
    if weights:
        lines += ["Question-type weights of the continuations (parent's error on fit sample B; weight = max(error, 0.05)):\n",
                  "| Type | " + " | ".join(f"{name(x)}: wrong of scored | weight" for x in weights) + " |",
                  "|---|" + "---:|---:|" * len(weights)]  # fmt: skip
        for t in next(iter(weights.values()))["weights"]:
            cells = [
                f"{w['scored_wrong'][t][1]}/{w['scored_wrong'][t][0]} | {w['weights'][t]:.2f}"
                for w in weights.values()
            ]
            lines.append(f"| {t} | " + " | ".join(cells) + " |")
        lines.append("")
        block["weights"] = weights
    if block:
        summary["round2"] = block


def round2_compare_section(lines, summary, split):
    """H7 (U2 against U1) and H8 (each continuation against its parent) at L0."""
    pairs = [("H7: short system prompt at equal compute", "p6u1", "p6u2")]
    for lab in ("p6u3", "p6u4", "p6u5"):
        manifest = Path("data/phase6/train") / lab[2:] / "manifest.json"
        if manifest.exists():
            pairs.append(
                (
                    "H8: weighted continuation",
                    "p6" + json.loads(manifest.read_text())["parent"],
                    lab,
                )
            )
    rows = []
    for title, a, b in pairs:
        ra, rb = rule_records(split, a), rule_records(split, b)
        if not ra or not rb:
            continue
        cell, d = diff(ra, rb)
        ca, cb = load(split, a, "calendar"), load(split, b, "calendar")
        cal = diff(ca, cb)[0] if ca and cb else "–"
        rows.append((title, a, b, cell, cal, d))
    if not rows:
        return
    lines += [f"### Round 2 (`{split}`, L0): paired differences, points [95%]\n",
              "| Hypothesis | Comparison | All rule items | Calendar |", "|---|---|---|---|"]  # fmt: skip
    for title, a, b, cell, cal, _ in rows:
        lines.append(f"| {title} | {name(b)} minus {name(a)} | {cell} | {cal} |")
    lines.append("")
    summary.setdefault("round2_compare", {})[split] = [
        {"hypothesis": t, "a": a, "b": b, "difference": d} for t, a, b, _, _, d in rows
    ]


def primary_section(lines, summary, split):
    sel = ROOT / "selection.json"
    if split != "sealed6" or not sel.exists():
        return
    m = json.loads(sel.read_text())["selected"]
    if not rule_records(split, m):
        return
    lines += [f"### Pre-registered comparisons on `{split}` (M = {name(m)}; preregistration section 7)\n",
              "| | Comparison | Result, points [95%] | Target | Met |", "|---|---|---|---|---|"]  # fmt: skip
    block = {"M": m}

    def add(code, what, a, b, target, test):
        if not a or not b:
            return
        text, d = diff(a, b)
        met = test(d) if d and d.get("interval95") else None
        lines.append(
            f"| {code} | {what} | {text} | {target} | {'yes' if met else 'no' if met is not None else '–'} |"
        )
        block[code] = {
            "difference": d["difference"] if d else None,
            "interval95": d["interval95"] if d else None,
            "met": met,
        }

    above0 = lambda d: d["interval95"][0] > 0  # noqa: E731
    add(
        "P-1",
        f"{name(m)} − A1, L0, all rule items",
        rule_records(split, "p4a1"),
        rule_records(split, m),
        "lower bound > 0 (aim +10)",
        above0,
    )
    add(
        "P-2",
        f"{name(m)} − A1, L0, blind questions",
        load(split, "p4a1", "blind"),
        load(split, m, "blind"),
        "secondary",
        above0,
    )
    add(
        "P-3",
        f"{name(m)} − untuned base, L0, all rule items",
        rule_records(split, "base"),
        rule_records(split, m),
        "lower bound > 0",
        above0,
    )
    add("P-4", f"{name(m)} − Phase 2, calendar", load(split, "p2", "calendar"), load(split, m, "calendar"), "lower bound ≥ −5",
        lambda d: d["interval95"][0] >= -0.05)  # fmt: skip
    for lab in (m, "p4a1", "p5g1"):
        add("P-5" if lab == m else "P-5 (ref.)", f"{name(lab)}: L1 − L0, all rule items", rule_records(split, lab), rule_records(split, lab, "L1"),
            "lower bound > 0" if lab == m else "reference", above0)  # fmt: skip
    add(
        "H1",
        "U0 − A1, L0, all rule items",
        rule_records(split, "p4a1"),
        rule_records(split, "p6u0"),
        "lower bound > 0",
        above0,
    )
    add("H1", "U0 − Phase 2, calendar", load(split, "p2", "calendar"), load(split, "p6u0", "calendar"), "lower bound ≥ −5",
        lambda d: d["interval95"][0] >= -0.05)  # fmt: skip
    lines.append("")
    cal = {lab: load(split, lab, "calendar") for lab in (m, "p2")}
    if all(cal.values()):
        lines.append(
            f"P-4, balanced decision accuracy on twins: {name(m)} {100 * balanced(cal[m]):.1f}%, Phase 2 {100 * balanced(cal['p2']):.1f}%.\n"
        )
        block["P-4 balanced"] = {m: balanced(cal[m]), "p2": balanced(cal["p2"])}
    l4 = {lab: rule_records(split, lab, "L4") for lab in (m, "p4a1")}
    if all(l4.values()):
        cells = []
        for lab in (m, "p4a1"):
            rs = [r for r in l4[lab] if r["level"] == "L4a"]
            wrong, right = (
                [r for r in rs if not r["first_correct"]],
                [r for r in rs if r["first_correct"]],
            )
            fixed, broke = sum(r["fixed"] for r in wrong), sum(r["broke"] for r in right)
            cells.append(
                f"{name(lab)} fixes {pct(fixed, len(wrong))} and breaks {pct(broke, len(right))}"
            )
            block.setdefault("P-6", {})[lab] = {
                "fixed": fixed,
                "wrong": len(wrong),
                "broke": broke,
                "right": len(right),
            }
        v = block["P-6"][m]
        met = (
            v["wrong"]
            and v["fixed"] / v["wrong"] >= 0.25
            and v["broke"] / max(v["right"], 1) <= 0.02
        )
        lines.append(
            f"P-6, self-correction (L4a): {'; '.join(cells)}. Target: fix at least 25%, break at most 2%: {'met' if met else 'not met'}.\n"
        )
    if rule_records(split, "p4a1"):
        a, b = claim_counts(split, "p4a1"), claim_counts(split, m)
        ra, rb = a["replies_with_wrong"] / a["n"], b["replies_with_wrong"] / b["n"]
        lines.append(f"P-7, L0 replies with a contradicted checkable statement: {name(m)} {pct(b['replies_with_wrong'], b['n'])}, "
                     f"A1 {pct(a['replies_with_wrong'], a['n'])}. Target: at most half of A1's rate: {'met' if rb <= ra / 2 else 'not met'}.\n")  # fmt: skip
        block["P-7"] = {
            m: [b["replies_with_wrong"], b["n"]],
            "p4a1": [a["replies_with_wrong"], a["n"]],
            "met": rb <= ra / 2,
        }
    summary.setdefault("primary", {})[split] = block


def l5_section(lines, summary, split):
    labs = [lab for lab in labels_in(split) if load(split, lab, "rules", "L5")]
    if not labs:
        return
    lines += [f"### L5: the Phase 5 hybrid system (`{split}`). System accuracy, not model accuracy\n",
              "Router, conclusion supplied by the rule code, value repair and engine fallback (Phase 5 code, condition grv). "
              "The generator's own L0 score is shown for contrast.\n",
              "| Generator | Part | Generator alone (L0) | Generator's reply inside the system | Final system answer | Repaired | Engine fallback | Routed right |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    block = {}
    for lab in labs:
        for part in ("rules", "indep", "blind"):
            recs = load(split, lab, part, "L5")
            if not recs:
                continue
            l0 = load(split, lab, part)
            n = len(recs)
            final = sum(bool(r["final_score"]["correct"]) for r in recs)
            routed = sum(bool(r.get("routed_ok")) for r in recs)
            lines.append(f"| {name(lab)} | {part} | {pct(correct(l0), len(l0)) if l0 else '–'} | {pct(correct(recs), n)} | **{pct(final, n)}** | "
                         f"{sum(bool(r['repaired']) for r in recs)} | {sum(bool(r['fallback']) for r in recs)} | {pct(routed, n)} |")  # fmt: skip
            block.setdefault(lab, {})[part] = {"n": n, "reply": correct(recs), "final": final, "routed": routed,
                                               "l0": correct(l0) if l0 else None}  # fmt: skip
    lines.append("")
    summary.setdefault("l5", {})[split] = block


def main():
    lines = ["# Phase 6 results\n",
             "Generated by `scripts/phase6/make_report6.py` from the files in this folder. Real local inference (MLX, Apple M5, 32 GB), "
             "greedy decoding. Design: `docs/phase6_preregistration.md`. **Every table names its level; L0 is the model alone.** "
             "`dev6` is read freely, `valid6` is for selection only, `sealed6` is scored once after `selection.json`.\n"]  # fmt: skip
    summary = {}
    fit_section(lines, summary)
    training_section(lines, summary)
    round2_section(lines, summary)
    decisions_section(lines, summary)
    for split in ("sealed6", "valid6", "dev6"):
        if not (ROOT / split).exists():
            continue
        lines.append(f"## Split `{split}`\n")
        for section in (
            primary_section,
            l0_section,
            round2_compare_section,
            statements_section,
            truncation_section,
            wording_section,
            cf_section,
            intent_section,
            calendar_section,
            levels_section,
            selfcheck_section,
            l5_section,
        ):
            section(lines, summary, split)
    (ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=str) + "\n"
    )
    print(f"Wrote {ROOT / 'RESULTS.md'} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
