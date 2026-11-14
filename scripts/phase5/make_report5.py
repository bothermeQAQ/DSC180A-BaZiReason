"""Build results/phase5/RESULTS.md and summary.json from the saved Phase 5 outputs.

    .venv/bin/python scripts/phase5/make_report5.py

Reads saved records only; runs no model. Sections appear only for the runs that exist.
"""

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from bazireason.p2_eval import paired_difference

ROOT = Path("results/phase5")
SAFE = str.maketrans({":": "-", "/": "-"})
NAMES = {
    "base": "Untuned base",
    "p2": "Phase 2",
    "p3r2": "P3 round 2",
    "p3r3": "P3 round 3",
    "p4a1": "P4 A1",
    "p5g1": "P5 G1",
}
COND = {
    "w0": "model alone",
    "gf": "full fact sheet",
    "go": "oracle-routed",
    "gr": "routed",
    "grv": "routed + verified",
}
RULE_PARTS = ("rules", "indep", "blind")


def name(label):
    if label.startswith("mix"):
        _, a, b, alpha = label.split(":")
        return f"Merged {alpha} ({NAMES.get(a, a)}→{NAMES.get(b, b)})"
    return NAMES.get(label, label)


def load(split, label, part, cond, tag=""):
    path = ROOT / split / label.translate(SAFE) / f"{part}_{cond}{tag}.jsonl"
    if not path.exists() and cond == "w0" and not tag:  # w0 does not depend on the prompt version
        path = path.with_name(f"{part}_w0_p2.jsonl")
    if not path.exists():
        return None
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def labels_in(split):
    folder = ROOT / split
    if not folder.exists():
        return []
    found = [p.name for p in folder.iterdir() if p.is_dir()]
    back = {lab.translate(SAFE): lab for lab in ("base", "p2", "p3r2", "p3r3", "p4a1", "p5g1")}
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
            lines.append(
                f"| {name(label)} | {COND[cond]} | {pct(bad, n)} | {pct(disagree, n)} | {rep} | {fb} | {fin} |"
            )
            block[f"{label}/{cond}"] = {
                "n": n,
                "contradiction": bad,
                "conclusion_disagrees": disagree,
            }
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
    short = lambda i: i if i.startswith("fact") else i.split(":")[0]  # noqa: E731
    wrong = Counter(
        (short(r["intent_oracle"]), short(r["intent_used"])) for r in rows if not r["routed_ok"]
    )
    computable_refused = sum(
        r["intent_used"] == "prediction" and computable(r["intent_oracle"]) for r in rows
    )
    lines += [f"### Router on `{split}` blind questions\n",
              f"Intent exactly right: {pct(ok, len(rows))}. Computable questions sent to the prediction refusal: {computable_refused}.\n"]  # fmt: skip
    if wrong:
        lines.append(
            "Most common errors (true → routed): "
            + "; ".join(f"{a} → {b} ×{n}" for (a, b), n in wrong.most_common(8))
            + "\n"
        )
    summary.setdefault("router", {})[split] = {
        "ok": ok,
        "n": len(rows),
        "computable_refused": computable_refused,
    }


def conversations(lines, summary, split):
    found = []
    for label in labels_in(split):
        for cond in ("w0", "grv"):
            path = ROOT / split / label.translate(SAFE) / f"convo_{cond}.jsonl"
            if path.exists():
                found.append(
                    (
                        label,
                        cond,
                        [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()],
                    )
                )
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
        lines.append(
            f"| {name(label)} | {COND[cond]} | {pct(ok, n)} | {pct(first, n)} | "
            + " | ".join(f"{k}/{n}" for k in per)
            + " |"
        )
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
        bda = (
            sum(bool(s.get("status_correct")) for s in ok) / max(len(ok), 1)
            + sum(bool(s.get("status_correct")) for s in other) / max(len(other), 1)
        ) / 2
        refused = sum(bool(s.get("false_refusal")) for s in sc)
        definite = sum(bool(s.get("definitive_on_nonanswer")) for s in sc)
        lines.append(
            f"| {name(lab)} | {pct(k, len(sc))} | {d} | {100 * bda:.1f}% | {refused}/{len(ok)} | {definite}/{len(other)} |"
        )
        block[lab] = {
            "k": k,
            "n": len(sc),
            "balanced_decision": round(bda, 4),
            "false_refusal": refused,
            "definite_on_nonanswer": definite,
        }
    lines.append("")
    summary.setdefault("calendar", {})[split] = block


def model_alone_rules(lines, summary, split):
    labs = [lab for lab in labels_in(split) if all(load(split, lab, p, "w0") for p in RULE_PARTS)]
    if not labs:
        return
    lines += [
        f"### Model alone on rule questions (`{split}`)\n",
        "| Model | Template rules | Independent | Blind | All |",
        "|---|---:|---:|---:|---:|",
    ]
    for lab in labs:
        parts = {p: load(split, lab, p, "w0") for p in RULE_PARTS}
        cells = [
            pct(sum(bool(r["score"]["correct"]) for r in parts[p]), len(parts[p]))
            for p in RULE_PARTS
        ]
        allr = [r for p in RULE_PARTS for r in parts[p]]
        lines.append(
            f"| {name(lab)} | "
            + " | ".join(cells)
            + f" | {pct(sum(bool(r['score']['correct']) for r in allr), len(allr))} |"
        )
    lines.append("")


def dev_prompt_v2(lines, summary):
    """Development runs with prompt version 2 (files tagged _p2): A1 before and after, and G1."""
    found = [lab for lab in ("p4a1", "p5g1") if load("dev", lab, "rules", "go", "_p2")]
    if not found:
        return
    lines += ["### Prompt version 2 on `dev` (the version used for validation and sealed runs)\n",
              "| System | Part | Model alone | Oracle-routed | Routed | Routed + verified, final | Repairs | Fallbacks | Contradicted first-pass replies (go) |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    block = {}
    for lab in found:
        for part in RULE_PARTS:
            w0, go, gr, grv = (
                load("dev", lab, part, c, "" if c == "w0" else "_p2")
                for c in ("w0", "go", "gr", "grv")
            )
            if not (go and gr and grv):
                continue
            n = len(go)

            def k(recs, key="score"):
                return sum(bool(r[key]["correct"]) for r in recs) if recs else None

            cells = [pct(k(w0), n) if w0 else "–", pct(k(go), n), pct(k(gr), n), pct(k(grv, "final_score"), n),
                     str(sum(bool(r["repaired"]) for r in grv)), str(sum(bool(r["fallback"]) for r in grv)),
                     str(sum(bool(r["check"]["contradictions"]) for r in go))]  # fmt: skip
            lines.append(f"| {name(lab)} | {part} | " + " | ".join(cells) + " |")
            block[f"{lab}/{part}"] = {
                "n": n,
                "w0": k(w0),
                "go": k(go),
                "gr": k(gr),
                "grv_final": k(grv, "final_score"),
            }
    lines.append("")
    summary["dev_prompt_v2"] = block


def two_by_three(lines, summary, split="valid"):
    """{base, R3, A1} x {w0, gr} on rules + indep + blind (section 6)."""
    cells = {}
    for lab in ("base", "p3r3", "p4a1", "p5g1"):
        for cond in ("w0", "gr"):
            parts = [load(split, lab, p, cond) for p in RULE_PARTS]
            if all(parts):
                recs = [r for p in parts for r in p]
                cells[(lab, cond)] = (sum(bool(r["score"]["correct"]) for r in recs), len(recs))
    if not any(lab != "p5g1" and (lab, "w0") in cells for lab in ("base", "p3r3")):
        return
    lines += [f"### Training versus grounding on `{split}` (first pass, rules + independent + blind)\n",
              "| Generator | Model alone (w0) | Routed conclusion supplied (gr) | Gain from grounding |", "|---|---:|---:|---:|"]  # fmt: skip
    block = {}
    for lab in ("base", "p3r3", "p4a1", "p5g1"):
        w, g = cells.get((lab, "w0")), cells.get((lab, "gr"))
        if not (w or g):
            continue
        gain = f"{100 * (g[0] - w[0]) / g[1]:+.1f} points" if w and g else "–"
        lines.append(
            f"| {name(lab)} | {pct(*w) if w else '–'} | {pct(*g) if g else '–'} | {gain} |"
        )
        block[lab] = {"w0": w, "gr": g}
    lines.append("")
    summary["training_vs_grounding"] = block


def by_intent(lines, summary, split="sealed"):
    """Correct answers by oracle intent family: A1 alone, the system's first pass and its final answer."""
    sel_path = ROOT / "selection.json"
    if not sel_path.exists():
        return
    g = json.loads(sel_path.read_text())["g_star"]
    a1 = [r for p in RULE_PARTS for r in (load(split, "p4a1", p, "w0") or [])]
    sys_recs = [r for p in RULE_PARTS for r in (load(split, g, p, "grv") or [])]
    if not (a1 and sys_recs):
        return
    fam = lambda r: (r.get("intent_oracle") or "?").split(":")[0]  # noqa: E731
    table = defaultdict(lambda: [0, 0, 0, 0])
    for r in a1:
        table[fam(r)][0] += bool(r["score"]["correct"])
        table[fam(r)][3] += 1
    for r in sys_recs:
        table[fam(r)][1] += bool(r["score"]["correct"])
        table[fam(r)][2] += bool(r["final_score"]["correct"])
    lines += [f"### By intent on `{split}` (rules + independent + blind)\n",
              f"| Intent | Items | A1 alone | {name(g)} routed, first pass | System final |", "|---|---:|---:|---:|---:|"]  # fmt: skip
    for k in sorted(table, key=lambda k: -table[k][3]):
        a, f, fin, n = table[k]
        lines.append(f"| {k} | {n} | {pct(a, n)} | {pct(f, n)} | {pct(fin, n)} |")
    lines.append("")
    summary.setdefault("by_intent", {})[split] = {
        k: dict(zip(("a1_w0", "first_pass", "final", "n"), v, strict=True))
        for k, v in table.items()
    }


def computable(intent):
    """A question with a single rule-code answer: not ambiguous by design, not a prediction, not unknown."""
    return not intent.startswith("ambig") and intent not in ("prediction", "unknown")


def gates(lines, summary):
    """Section 6 gates on the sealed split."""
    sel_path = ROOT / "selection.json"
    if not sel_path.exists():
        return
    g = json.loads(sel_path.read_text())["g_star"]
    blind = load("sealed", g, "blind", "gr")
    sys_recs = [r for p in RULE_PARTS for r in (load("sealed", g, p, "grv") or [])]
    if not blind or not sys_recs:
        return
    out = {}
    ok = sum(bool(r["routed_ok"]) for r in blind)
    out["router intent accuracy on sealed blind questions ≥ 90%"] = (
        pct(ok, len(blind)),
        ok / len(blind) >= 0.90,
    )
    to_refusal = [
        r
        for p in RULE_PARTS
        for r in load("sealed", g, p, "gr")
        if r["intent_used"] == "prediction" and r["intent_oracle"] != "prediction"
    ]
    refused = [r for r in to_refusal if computable(r["intent_oracle"])]
    others = Counter(r["intent_oracle"] for r in to_refusal if not computable(r["intent_oracle"]))
    note = "".join(
        f"; also {n} × {i} (no rule answer, so not computable; scored wrong)"
        for i, n in others.items()
    )
    out["no computable question routed to the prediction refusal"] = (
        f"{len(refused)}{note}",
        not refused,
    )
    final_bad = sum(bool(r["final_contradictions"]) for r in sys_recs)
    first_bad = sum(bool(r["check"]["contradictions"]) for r in sys_recs)
    out["final displayed replies with a contradicted statement < 3%"] = (
        f"{pct(final_bad, len(sys_recs))} (first pass {pct(first_bad, len(sys_recs))})", final_bad / len(sys_recs) < 0.03)  # fmt: skip
    lat = ROOT / "latency_valid.json"
    if lat.exists():
        L = json.loads(lat.read_text())["systems"]
        if f"{g}/grv" in L and "p4a1/w0" in L:
            a, b = L[f"{g}/grv"]["median_seconds"], L["p4a1/w0"]["median_seconds"]
            out["median reply time of the system ≤ 2 × the model alone (A1)"] = (
                f"{a:.1f} s vs {b:.1f} s ({a / b:.2f}×)",
                a <= 2 * b,
            )
    lines += [
        "## Gates (pre-registered, section 6)\n",
        "| Gate | Measured | Pass |",
        "|---|---|---|",
    ]
    lines += [f"| {k} | {v[0]} | {'yes' if v[1] else 'no'} |" for k, v in out.items()] + [""]
    summary["gates"] = {k: {"measured": v[0], "pass": v[1]} for k, v in out.items()}


def adapter_checks(lines, summary, label="p5g1"):
    """The execution plan's three checks for any new adapter, applied to G1 (section 4 froze them for M* only)."""
    rows = []
    for split in ("valid", "sealed"):
        cal = {lab: load(split, lab, "calendar", "w0") for lab in ("p2", "p3r3", "p4a1", label)}
        if not all(cal.values()):
            continue
        sc = {lab: [r["score"] for r in recs] for lab, recs in cal.items()}
        text, d = diff(sc["p2"], sc[label])
        bda = {}
        for lab, xs in sc.items():
            ok = [s for s in xs if s.get("expected_status") == "ok"]
            other = [s for s in xs if s.get("expected_status") not in (None, "ok")]
            bda[lab] = (
                sum(bool(s.get("status_correct")) for s in ok) / len(ok)
                + sum(bool(s.get("status_correct")) for s in other) / len(other)
            ) / 2
        lo = d["interval95"][0] if d["interval95"] else None
        rows.append(
            (
                split,
                "calendar minus P2: 95% lower bound ≥ −7 points (section 4a)",
                text,
                lo is not None and lo >= -0.07,
            )
        )
        rows.append(
            (
                split,
                "calendar minus P2: 95% lower bound ≥ −5 points (the P-3 margin)",
                text,
                lo is not None and lo >= -0.05,
            )
        )
        others = max(bda[lab] for lab in ("p2", "p3r3", "p4a1"))
        rows.append((split, "balanced decision accuracy above P2, R3 and A1 (section 4c)",
                     f"{100 * bda[label]:.2f}% vs best other {100 * others:.2f}%", bda[label] > others))  # fmt: skip
        w0 = {
            lab: [r for p in RULE_PARTS for r in (load(split, lab, p, "w0") or [])]
            for lab in ("p4a1", label)
        }
        if all(w0.values()) and len(w0["p4a1"]) == len(w0[label]):
            a, b = (sum(bool(r["score"]["correct"]) for r in w0[lab]) for lab in ("p4a1", label))
            gap = (b - a) / len(w0[label])
            rows.append(
                (
                    split,
                    "model alone, rules + indep + blind: within 3 points of A1 (section 4b)",
                    f"{b} vs {a} ({100 * gap:+.1f} points)",
                    gap >= -0.03,
                )
            )
    if not rows:
        return
    lines += [f"## Adapter checks for {name(label)} (exploratory)\n",
              "The execution plan asks every new adapter to pass three checks. The pre-registration froze them for the merged adapter only, "
              f"so here they are applied to {name(label)} after the fact. They did not affect selection.\n",
              "| Split | Check | Measured | Pass |", "|---|---|---|---|"]  # fmt: skip
    lines += [
        f"| {sp} | {check} | {m} | {'yes' if ok else 'no'} |" for sp, check, m, ok in rows
    ] + [""]
    summary["adapter_checks"] = {
        label: [{"split": sp, "check": c, "measured": m, "pass": ok} for sp, c, m, ok in rows]
    }


def costs(lines, summary):
    lat, bank = ROOT / "latency_valid.json", ROOT / "bank_check.json"
    if lat.exists():
        L = json.loads(lat.read_text())
        lines += [f"## Interactive cost (one reply at a time, {L['n']} validation blind questions)\n",
                  "| System | Condition | Median s | 90th pct s | Prompt tokens | Completion tokens | Repair turns | Peak memory GB |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
        for key, v in L["systems"].items():
            lab, cond = key.split("/")
            lines.append(f"| {name(lab)} | {COND[cond]} | {v['median_seconds']} | {v['p90_seconds']} | {v['mean_prompt_tokens']} | "
                         f"{v['mean_completion_tokens']} | {v['repair_turns']} | {v['peak_memory_gb']} |")  # fmt: skip
        lines.append("")
    if bank.exists():
        B = json.loads(bank.read_text())
        same = all(B["same_as_fresh_load"].values())
        swap = sorted(B["swap_ms"])
        lines += ["## Adapter bank\n",
                  f"Greedy replies with each adapter swapped into the one loaded base equal those of a freshly loaded model: "
                  f"{'yes, for every adapter' if same else B['same_as_fresh_load']}. Swap time median {statistics.median(swap):.1f} ms "
                  f"(range {swap[0]}–{swap[-1]} ms, {len(swap)} swaps); a fresh load takes about {max(B['fresh_load_seconds'].values())} s; "
                  f"bank active memory {B['bank_active_memory_gb']} GB against {B['single_model_active_memory_gb']} GB for one model.\n"]  # fmt: skip
    extras = ROOT / "addenda" / "extras.json"
    if extras.exists():
        X = json.loads(extras.read_text())
        lines += ["## Format, loops and titles (rules + independent + blind)\n",
                  "Format = a readable 结论JSON. Accuracy among well-formed replies separates format from content.\n",
                  "| Split / system / condition | Replies | Well-formed | Correct | Correct among well-formed | Loops | Titles outside the rule cards |",
                  "|---|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
        for key, v in X["format_and_loops"].items():
            split, lab, cond = key.split("/")
            lines.append(f"| {split} · {name(lab)} · {COND.get(cond.removesuffix('_p2'), cond)}{' (v2)' if cond.endswith('_p2') else ''} | {v['n']} | "
                         f"{pct(v['format_ok'], v['n'])} | {pct(v['correct'], v['n'])} | {pct(v['correct_when_format_ok'], v['format_ok'])} | {v['loops']} | {v['titles_not_in_cards']} |")  # fmt: skip
        lines.append("")
        lines += ["## Adapter routing (exploratory, simulated from stored replies)\n",
                  "A per-intent adapter table is chosen on development data (ties to A1) and applied to validation and sealed items.\n",
                  "| Condition | Table from dev | Split | Routed | Best single adapter | Routed minus best | Any adapter right (ceiling) |",
                  "|---|---|---|---:|---:|---:|---:|"]  # fmt: skip
        for cond, R in X["adapter_routing"].items():
            if not R:
                continue
            table = (
                ", ".join(f"{k}→{name(v)}" for k, v in R["table_from_dev"].items() if v != "p4a1")
                or "all A1"
            )
            for split in ("valid", "sealed"):
                if split in R:
                    v = R[split]
                    lines.append(f"| {COND[cond]} | others A1; {table} | {split} | {pct(v['routed'], v['n'])} | {name(v['best_single'])} "
                                 f"{pct(v['single'][v['best_single']], v['n'])} | {v['routed_minus_best_single']:+d} | {pct(v['oracle_any_adapter'], v['n'])} |")  # fmt: skip
        lines.append("")
    flags = ROOT / "addenda" / "checker_flags_valid.json"
    if flags.exists():
        F = json.loads(flags.read_text())
        lines += ["## Claim checker: read sample of its flags on validation replies\n",
                  f"{F['sample']} of {F['flagged_replies']} flagged replies, drawn at random and read by hand: {F['real']} real errors, "
                  f"{F['borderline']} borderline (a wrong value corrected in the same sentence), {F['false_alarm']} false alarms "
                  f"(upper 95% bound {F['false_alarm_rate_upper95']}). Details: `addenda/checker_flags_valid.json`.\n"]  # fmt: skip
    if extras.exists() and "engine_fallbacks" in json.loads(extras.read_text()):
        E = json.loads(extras.read_text())["engine_fallbacks"]
        lines += ["## Engine fallbacks\n",
                  "The runners do not claim-check a reply that is the engine's own derivation. Checked here instead: "
                  f"{E['with_a_contradicted_statement']} of {E['fallback_replies']} fallback replies (validation and sealed, every system) "
                  "contain a contradicted statement (`addenda/extras.json`).\n"]  # fmt: skip


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
    if g == "p5g1":
        ga = [r for p in RULE_PARTS for r in (load("sealed", "p4a1", p, "gr") or [])]
        if gt and ga:
            text, d = diff(scores(ga), scores(gt))
            prim["P-2b P5 G1 minus P4 A1, both routed (gr): what the Phase 5 training added"] = (
                text,
                d,
            )
    if sys_recs and a1:
        text, d = diff(scores(a1), scores(sys_recs))
        prim["(first pass, before repair) system minus A1 alone"] = (text, d)
    if sel.get("m_star_is_candidate"):
        m = sel["m_star"]
        cm, cp = load("sealed", m, "calendar", "w0"), load("sealed", "p2", "calendar", "w0")
        if cm and cp:
            text, d = diff(scores(cp), scores(cm))
            prim[f"P-3 {name(m)} minus Phase 2, calendar rows (margin −5)"] = (text, d)
    else:
        prim["P-3 merged adapter, calendar non-inferiority"] = (
            "not tested: M* was not a candidate on validation",
            None,
        )
    lines += (
        ["| Comparison | Difference, points [95%] |", "|---|---|"]
        + [f"| {k} | {v[0]} |" for k, v in prim.items()]
        + [""]
    )
    summary["primary"] = {k: v[1] for k, v in prim.items()}
    summary["selection"] = sel


def ladder(lines, summary):
    """Each step of the system on the same items: what grounding, training, routing and verification add."""
    steps = [
        (
            "grounding: A1 routed (gr) minus A1 alone (w0)",
            ("p4a1", "w0", False),
            ("p4a1", "gr", False),
        ),
        (
            "training before Phase 5: A1 minus base, both routed",
            ("base", "gr", False),
            ("p4a1", "gr", False),
        ),
        (
            "Phase 5 training: G1 minus A1, both routed",
            ("p4a1", "gr", False),
            ("p5g1", "gr", False),
        ),
        (
            "router errors: G1 oracle-routed (go) minus G1 routed (gr)",
            ("p5g1", "gr", False),
            ("p5g1", "go", False),
        ),
        (
            "verification with G1: grv final minus gr first pass",
            ("p5g1", "gr", False),
            ("p5g1", "grv", True),
        ),
        (
            "verification with A1: grv final minus gr first pass",
            ("p4a1", "gr", False),
            ("p4a1", "grv", True),
        ),
        (
            "verification with base: grv final minus gr first pass",
            ("base", "gr", False),
            ("base", "grv", True),
        ),
        (
            "merging (model alone): M* minus A1",
            ("p4a1", "w0", False),
            ("mix:p3r3:p4a1:0.15", "w0", False),
        ),
        (
            "merging (routed): M* minus A1",
            ("p4a1", "gr", False),
            ("mix:p3r3:p4a1:0.15", "gr", False),
        ),
        ("untuned base: routed minus alone", ("base", "w0", False), ("base", "gr", False)),
        ("R3: routed minus alone", ("p3r3", "w0", False), ("p3r3", "gr", False)),
        ("G1: routed minus alone", ("p5g1", "w0", False), ("p5g1", "gr", False)),
        ("G1: full fact sheet (gf) minus alone", ("p5g1", "w0", False), ("p5g1", "gf", False)),
        ("G1: routed minus full fact sheet", ("p5g1", "gf", False), ("p5g1", "gr", False)),
    ]
    out = {}
    for split in ("sealed", "valid"):
        rows = []
        for title, (la, ca, fa), (lb, cb, fb) in steps:
            a = [r for p in RULE_PARTS for r in (load(split, la, p, ca) or [])]
            b = [r for p in RULE_PARTS for r in (load(split, lb, p, cb) or [])]
            if not a or len(a) != len(b):
                continue
            ka = sum(bool((r["final_score"] if fa else r["score"])["correct"]) for r in a)
            kb = sum(bool((r["final_score"] if fb else r["score"])["correct"]) for r in b)
            text, d = diff(scores(a, final=fa), scores(b, final=fb))
            rows.append(f"| {title} | {ka} → {kb} of {len(a)} | {text} |")
            out.setdefault(split, {})[title] = {"from": ka, "to": kb, "n": len(a), "difference": d}
        if rows:
            lines += [f"## What each step adds on `{split}` (rules + independent + blind, paired)\n",
                      "| Step | Correct answers | Difference, points [95%] |", "|---|---|---|", *rows, ""]  # fmt: skip
    summary["ladder"] = out


def main():
    lines = ["# Phase 5 results\n",
             "Generated by `scripts/phase5/make_report5.py` from the files in this folder. Real local inference "
             "(MLX, Apple M5, 32 GB), greedy decoding, one base model with adapters swapped in place. Design: "
             "`docs/phase5_preregistration.md`. Development data (`dev`) were read freely; `valid` was used for "
             "selection only; `sealed` was scored once after `selection.json` existed.\n"]  # fmt: skip
    summary = {}
    primaries(lines, summary)
    ladder(lines, summary)
    gates(lines, summary)
    adapter_checks(lines, summary)
    costs(lines, summary)
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
        if split == "valid":
            two_by_three(lines, summary)
        if split == "sealed":
            by_intent(lines, summary)
        if split == "dev":
            dev_prompt_v2(lines, summary)
    for extra in ("dev_alpha.json", "bank_check.json", "latency_valid.json", "latency_sealed.json"):
        path = ROOT / extra
        if path.exists():
            summary[extra.removesuffix(".json")] = json.loads(path.read_text())
    (ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=str) + "\n"
    )
    print(f"Wrote {ROOT / 'RESULTS.md'} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
