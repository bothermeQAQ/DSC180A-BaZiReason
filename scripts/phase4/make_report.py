"""Build results/phase4/RESULTS.md and summary.json from the Phase 4 result files.

    python scripts/phase4/make_report.py

Reads only saved outputs; runs no model. Missing inputs are listed at the end.
"""

import json
from pathlib import Path

from bazireason.p2_eval import paired_difference

ROOT = Path("results/phase4")
LABELS = {"p2": "Phase 2", "p3r2": "P3 round 2", "p3r3": "P3 round 3", "p4a1": "P4 A1"}
MISSING = []


def load(path):
    path = Path(path)
    if not path.exists():
        MISSING.append(str(path))
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def scores(path):
    path = Path(path)
    if not path.exists():
        MISSING.append(str(path))
        return None
    return [json.loads(line)["score"] for line in path.read_text(encoding="utf-8").splitlines()]


def frac(k, n):
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "–"


def count(rows, where=lambda s: True, key="correct"):
    chosen = [s for s in rows if where(s)]
    return sum(bool(s.get(key)) for s in chosen), len(chosen)


def diff(first, second, where=None):
    if not first or not second:
        return "–", None
    d = paired_difference(first, second, where=where)
    if d["interval95"] is None:
        return f"{100 * d['difference']:+.1f} pp", d
    low, high = d["interval95"]
    return f"{100 * d['difference']:+.1f} pp [{100 * low:+.1f}, {100 * high:+.1f}]", d


REASONING = lambda s: s["family"] != "strict_shift"  # noqa: E731
CALENDAR = lambda s: s["task"] == "chart"  # noqa: E731


def training(lines, summary):
    done = load(ROOT / "training/bazireason-qwen3-4b-lora-p4-a1/training_summary.json")
    if not done:
        return
    summary["training_a1"] = {
        k: done[k]
        for k in ("train_rows", "iterations", "optimizer_updates", "duration_seconds",
                  "initial_validation_loss", "selected_validation_loss", "peak_memory_gb",
                  "selected_adapter_sha256", "init_adapter")
    }  # fmt: skip
    lines += [
        "## 1. Approach A training run\n",
        "| Run | Starts from | Rows | Iterations (updates) | Minutes | Validation loss | Peak memory |",
        "|---|---|---:|---:|---:|---|---:|",
        f"| `bazireason-qwen3-4b-lora-p4-a1` | Phase 3 round 3 | {done['train_rows']:,} | "
        f"{done['iterations']:,} ({done['optimizer_updates']:,}) | {done['duration_seconds'] / 60:.0f} | "
        f"{done['initial_validation_loss']:.3f} → {done['selected_validation_loss']:.4f} | "
        f"{done['peak_memory_gb']:.1f} GB |",
        "",
    ]


def selection(lines, summary):
    sel = load(ROOT / "selection.json")
    if not sel:
        return
    summary["selection"] = sel
    lines += [
        "## 2. Adapter selection on the Phase 4 validation split (rule frozen in advance)\n",
        f"Rule: {sel['rule']}\n",
        "| Adapter | Phase 3 part | Phase 2 part | Independent part | Total | Calendar rows minus P2 [95%] | Eligible |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for r in sel["candidates"]:
        cal = r["calendar_vs_p2"]
        ci = cal["interval95"] or [0, 0]
        mark = " **(selected)**" if r["label"] == sel["selected"] else ""
        lines.append(
            f"| {LABELS.get(r['label'], r['label'])}{mark} | {r['correct']['p3']}/{r['rows']['p3']} | "
            f"{r['correct']['p2']}/{r['rows']['p2']} | {r['correct']['indep']}/{r['rows']['indep']} | "
            f"{r['total_correct']} | {100 * cal['difference']:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] | "
            f"{'yes' if r['eligible'] else 'no'} |"
        )
    lines.append("")


def final_w0(lines, summary):
    F = ROOT / "final_test"
    runs = {
        lab: {p: scores(F / f"{lab}_{p}.jsonl") for p in ("p3", "p2", "indep")} for lab in LABELS
    }
    runs = {lab: v for lab, v in runs.items() if all(v.values())}
    if not runs:
        return runs
    rep = {
        lab: {p: load(F / f"{lab}_{p}.jsonl.scores.json")["summary"] for p in ("p3", "p2")}
        for lab in runs
    }
    cols = list(runs)
    lines += [
        "## 3. Untouched final test split (scored once, after the selection)\n",
        "Model alone (W0), greedy decoding, the prompts of Phases 2 and 3.\n",
        "| Measure | " + " | ".join(LABELS[c] for c in cols) + " |",
        "|---|" + "---:|" * len(cols),
    ]

    def row(name, fn):
        lines.append(f"| {name} | " + " | ".join(fn(c) for c in cols) + " |")

    row("**Phase 3 reasoning items**", lambda c: frac(*count(runs[c]["p3"], REASONING)))
    row(
        "… wording never seen in training",
        lambda c: frac(*count(runs[c]["p3"], lambda s: REASONING(s) and s.get("unseen_wording"))),
    )
    row(
        "… wording seen in training",
        lambda c: frac(
            *count(runs[c]["p3"], lambda s: REASONING(s) and not s.get("unseen_wording"))
        ),
    )
    for fam in (
        "drill",
        "gods",
        "strength",
        "pattern",
        "yongshen",
        "interact",
        "flow",
        "domain",
        "claims",
        "ambig",
        "mcq3",
        "dialog",
        "overview",
    ):
        row(f"… `{fam}`", lambda c, f=fam: frac(*count(runs[c]["p3"], lambda s: s["family"] == f)))
    amb = lambda c, k: rep[c]["p3"]["ambiguity"][k]  # noqa: E731
    row(
        "Ambiguity: right decision",
        lambda c: frac(amb(c, "decision_correct")["hits"], amb(c, "decision_correct")["n"]),
    )
    row(
        "Ambiguity: definite where none exists (lower is better)",
        lambda c: frac(
            amb(c, "definite_on_indeterminate")["hits"], amb(c, "definite_on_indeterminate")["n"]
        ),
    )
    row(
        "Ambiguity: declined although determinate (lower is better)",
        lambda c: frac(
            amb(c, "abstained_on_determinate")["hits"], amb(c, "abstained_on_determinate")["n"]
        ),
    )
    sa = lambda c, k: rep[c]["p3"]["strict_audit"][k]  # noqa: E731
    row(
        "One-pass audit: wrong tool answers corrected",
        lambda c: frac(sa(c, "corrupted_correct")["hits"], sa(c, "corrupted_correct")["n"]),
    )
    row(
        "One-pass audit: wrong tool answers followed (lower is better)",
        lambda c: frac(sa(c, "corrupted_followed")["hits"], sa(c, "corrupted_followed")["n"]),
    )
    cal = lambda c, k: rep[c]["p2"]["calendar"][k]  # noqa: E731
    row(
        "**Calendar rows (Phase 2 part, task = chart)**",
        lambda c: frac(*count(runs[c]["p2"], CALENDAR)),
    )
    row(
        "Exact four pillars",
        lambda c: frac(cal(c, "exact_four_pillar")["k"], cal(c, "exact_four_pillar")["n"]),
    )
    row(
        "… without tool facts",
        lambda c: frac(cal(c, "exact_no_tools")["k"], cal(c, "exact_no_tools")["n"]),
    )
    row(
        "… with tool facts",
        lambda c: frac(cal(c, "exact_with_facts")["k"], cal(c, "exact_with_facts")["n"]),
    )
    row(
        "Asks for the tool when the zone offset is unknown",
        lambda c: frac(cal(c, "needs_tool_decision")["k"], cal(c, "needs_tool_decision")["n"]),
    )
    row(
        "Answerable chart refused (lower is better)",
        lambda c: frac(cal(c, "false_refusal")["k"], cal(c, "false_refusal")["n"]),
    )
    row(
        "Definite chart for a request without one (lower is better)",
        lambda c: frac(
            cal(c, "definitive_on_nonanswer")["k"], cal(c, "definitive_on_nonanswer")["n"]
        ),
    )
    for fam in (
        "jie_boundary",
        "zi_transition",
        "tz_needs_tool",
        "verify",
        "counterfactual",
        "rule",
        "mcq",
        "multi_turn",
        "policy",
    ):
        row(f"… `{fam}`", lambda c, f=fam: frac(*count(runs[c]["p2"], lambda s: s["family"] == f)))
    row("Phase 2 part, all rows", lambda c: frac(*count(runs[c]["p2"])))
    row(
        "**Independent rule questions (lunar-python labels, unseen wordings)**",
        lambda c: frac(*count(runs[c]["indep"])),
    )
    for kind in (
        "ten_god_stem",
        "ten_god_branch",
        "hidden",
        "life_stage",
        "xun_kong",
        "luck_direction",
        "element_count",
    ):
        row(
            f"… `{kind}`",
            lambda c, k=kind: frac(*count(runs[c]["indep"], lambda s: s["kind"] == k)),
        )
    lines.append("")
    sel = (load(ROOT / "selection.json") or {}).get("selected", "p3r3")
    lines += ["**Pre-registered primary comparisons** (paired, row-group bootstrap 95% intervals)\n",
              "| Comparison | Difference |", "|---|---|"]  # fmt: skip
    prim = {}
    if "p2" in runs and sel in runs:
        prim["P-1 calendar rows, selected minus P2 (margin -5)"] = diff(
            runs["p2"]["p2"], runs[sel]["p2"], CALENDAR
        )
    if "p3r3" in runs and sel in runs:
        prim["P-2 Phase 3 reasoning, selected minus R3"] = diff(
            runs["p3r3"]["p3"], runs[sel]["p3"], REASONING
        )
        prim["P-3 independent questions, selected minus R3"] = diff(
            runs["p3r3"]["indep"], runs[sel]["indep"]
        )
    if "p2" in runs and sel in runs:
        prim["P-3 independent questions, selected minus P2"] = diff(
            runs["p2"]["indep"], runs[sel]["indep"]
        )
    for k, (text, _) in prim.items():
        lines.append(f"| {k} | {text} |")
    if sel == "p2":
        lines += [
            "",
            "The frozen rule selected Phase 2 itself, so P-1 and P-3 against P2 compare P2 with "
            "itself (difference 0 by construction). The same non-inferiority test for every other "
            "adapter is in the exploratory table below.",
        ]
    lines += [
        "",
        "**Other paired comparisons (exploratory)**\n",
        "| Comparison | Calendar rows | Phase 3 reasoning | Independent | Phase 2 rule rows |",
        "|---|---|---|---|---|",
    ]
    for a, b in (
        ("p2", "p4a1"),
        ("p3r3", "p4a1"),
        ("p2", "p3r3"),
        ("p2", "p3r2"),
        ("p3r2", "p3r3"),
    ):
        if a in runs and b in runs:
            lines.append(
                f"| {LABELS[b]} minus {LABELS[a]} | {diff(runs[a]['p2'], runs[b]['p2'], CALENDAR)[0]} | "
                f"{diff(runs[a]['p3'], runs[b]['p3'], REASONING)[0]} | {diff(runs[a]['indep'], runs[b]['indep'])[0]} | "
                f"{diff(runs[a]['p2'], runs[b]['p2'], lambda s: s['family'] == 'rule')[0]} |"
            )
    lines.append("")
    summary["final_test_w0"] = {
        c: {
            p: dict(
                zip(
                    ("k", "n"),
                    count(runs[c][p], REASONING if p == "p3" else (lambda s: True)),
                    strict=True,
                )
            )
            for p in runs[c]
        }
        for c in cols
    }
    summary["primary"] = {k: v[1] for k, v in prim.items()}
    return runs


def approach_b(lines, summary, runs):
    F = ROOT / "final_test"
    lines += [
        "## 4. Approach B on the final test\n",
        "**W1, verify and repair:** the rule code / engine checks the final JSON. A reply that disagrees gets one more turn, which names the differing fields without their values. The table compares W0 with W1.\n",
        "| Adapter | Part | W0 | W1 | Re-asked | Fixed by the repair turn | Broken by it |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    out = {}
    for lab in ("p2", "p3r3", "p4a1"):
        for part, where in (
            ("p3", REASONING),
            ("p2", CALENDAR),
            ("p2", lambda s: s["family"] == "rule"),
            ("indep", lambda s: True),
        ):
            path = F / f"{lab}_{part}_w1.jsonl"
            if not path.exists() or lab not in runs:
                MISSING.append(str(path))
                continue
            recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
            recs = [r for r in recs if where(r["score"])]
            w1 = sum(bool(r["score"]["correct"]) for r in recs)
            w0 = sum(r["first_correct"] for r in recs)
            asked = sum(r["regenerated"] for r in recs)
            fixed = sum(
                r["regenerated"] and not r["first_correct"] and r["score"]["correct"] for r in recs
            )
            broke = sum(r["first_correct"] and not r["score"]["correct"] for r in recs)
            name = {"p3": "Phase 3 reasoning", "indep": "independent"}.get(
                part, "calendar rows" if where is CALENDAR else "Phase 2 rule rows"
            )
            lines.append(
                f"| {LABELS[lab]} | {name} | {frac(w0, len(recs))} | {frac(w1, len(recs))} | {asked} | {fixed} | {broke} |"
            )
            out[f"{lab}/{name}"] = {
                "w0": w0,
                "w1": w1,
                "n": len(recs),
                "asked": asked,
                "fixed": fixed,
                "broken": broke,
            }
    lines += ["", "**W2, independent tool check** on the `strict_shift` rows (one-pass JSON with a supplied tool answer).\n",
              "| Adapter | Check flagged (corrupted / clean) | Corrected, W0 → W2 | Followed the error, W0 → W2 (lower is better) | Clean answers kept, W0 → W2 |",
              "|---|---|---|---|---|"]  # fmt: skip
    for lab in ("p2", "p3r3", "p4a1"):
        path = F / f"{lab}_p3_w2.jsonl"
        if not path.exists() or lab not in runs:
            MISSING.append(str(path))
            continue
        recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        base = {s["id"]: s for s in runs[lab]["p3"]}
        bad = [r for r in recs if r["score"].get("corrupted")]
        clean = [r for r in recs if not r["score"].get("corrupted")]
        flagged = (sum(bool(r["check"]) for r in bad), sum(bool(r["check"]) for r in clean))
        cor0 = sum(bool(base[r["id"]]["correct"]) for r in bad)
        cor2 = sum(bool(r["score"]["correct"]) for r in bad)
        fol0 = sum(bool(base[r["id"]].get("followed_corruption")) for r in bad)
        fol2 = sum(bool(r["score"].get("followed_corruption")) for r in bad)
        kept0 = sum(bool(base[r["id"]]["correct"]) for r in clean)
        kept2 = sum(bool(r["score"]["correct"]) for r in clean)
        lines.append(
            f"| {LABELS[lab]} | {flagged[0]}/{len(bad)} / {flagged[1]}/{len(clean)} | {cor0} → {cor2} of {len(bad)} | "
            f"{fol0} → {fol2} | {kept0} → {kept2} of {len(clean)} |"
        )
        out[f"{lab}/w2"] = {
            "corrected": [cor0, cor2],
            "followed": [fol0, fol2],
            "n_bad": len(bad),
            "flagged": flagged,
        }
    lines.append("")
    summary["approach_b"] = out


def benchmark(lines, summary):
    files = {
        "Phase 2": "results/phase2/v3/tuned_v2_strict.jsonl.scores.json",
        "P3 round 2": "results/phase3/v3/p3r2_strict.jsonl.scores.json",
        "P3 round 3": "results/phase3/v3/p3r3_strict.jsonl.scores.json",
        "P4 A1": str(ROOT / "v3/p4a1_strict.jsonl.scores.json"),
    }
    rep = {k: load(v) for k, v in files.items()}
    rep = {k: v for k, v in rep.items() if v}
    get = lambda r, m, k: r["methods"].get(m, {}).get("overall", {}).get(k)  # noqa: E731
    pct = lambda v: "–" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    lines += ["## 5. benchmark_v3 (inspected; the checkpoint's prompts and evaluator)\n",
              "| Condition | Measure | " + " | ".join(rep) + " |", "|---|---|" + "---:|" * len(rep)]  # fmt: skip
    for m, k, text in (
        ("direct", "decision_accuracy", "right status (261)"),
        ("direct", "false_refusal_rate", "answerable request refused (lower is better)"),
        (
            "direct",
            "ambiguous_definitive_rate",
            "definite chart for an ambiguous request (lower is better)",
        ),
        ("tool_context", "exact_accuracy", "exact four pillars, answer supplied"),
        (
            "tool_corrupted",
            "corruption_following_rate",
            "followed the wrong tool answer, W0 (lower is better)",
        ),
    ):
        lines.append(
            f"| `{m}` | {text} | " + " | ".join(pct(get(r, m, k)) for r in rep.values()) + " |"
        )
    w2 = {}
    for lab in ("p2", "p3r3", "p4a1"):
        r = load(ROOT / f"v3/{lab}_w2_tool_corrupted.jsonl.scores.json")
        if r:
            w2[LABELS[lab]] = r
    if w2:
        lines += ["", "With the independent tool check (W2), `tool_corrupted`:\n",
                  "| Measure | " + " | ".join(w2) + " |", "|---|" + "---:|" * len(w2)]  # fmt: skip
        lines.append(
            "| followed the wrong tool answer (lower is better) | "
            + " | ".join(
                pct(get(r, "tool_corrupted", "corruption_following_rate")) for r in w2.values()
            )
            + " |"
        )
        for kind in ("hour_stem", "day_rollover", "default_guess", "jie_side"):
            vals = [
                (get(r, "tool_corrupted", "corruption_following_by_type") or {})
                .get(kind, {})
                .get("following_rate")
                for r in w2.values()
            ]
            lines.append(f"| … `{kind}` | " + " | ".join(pct(v) for v in vals) + " |")
        lines.append(
            "| exact four pillars despite the wrong answer | "
            + " | ".join(pct(get(r, "tool_corrupted", "exact_accuracy")) for r in w2.values())
            + " |"
        )
    reasoned = {
        "Phase 2": "results/phase2/v3/tuned_v2_reasoned.jsonl.scores.json",
        "P3 round 3": "results/phase3/v3/p3r3_reasoned.jsonl.scores.json",
        "P4 A1": str(ROOT / "v3/p4a1_reasoned.jsonl.scores.json"),
    }
    rr = {k: load(v) for k, v in reasoned.items()}
    rr = {k: v for k, v in rr.items() if v}
    if rr:
        lines += [
            "",
            "Reasoned protocol, exact four pillars:\n",
            "| Condition | " + " | ".join(rr) + " |",
            "|---|" + "---:|" * len(rr),
        ]
        for m in ("direct", "tool_intermediate"):
            lines.append(
                f"| `{m}` | "
                + " | ".join(pct(get(r, m, "exact_accuracy")) for r in rr.values())
                + " |"
            )
    lines.append("")
    summary["benchmark_v3"] = {
        k: {m: get(r, m, "corruption_following_rate") for m in ("tool_corrupted",)}
        for k, r in {**rep, **{f"{k} W2": v for k, v in w2.items()}}.items()
    }


def baziqa(lines, summary):
    s = load(ROOT / "baziqa/summary.json")
    if not s:
        return
    summary["baziqa"] = s
    lines += ["## 6. BaziQA (inspected; 555 questions, chance 24.7%, three sampled runs; not tuned on its keys)\n",
              "| Condition | All 555 [95%] | BaZi-only | Official keys | Unanswered | Loops | Loop guard used | Book named / not in cards |",
              "|---|---|---:|---:|---:|---:|---:|---|"]  # fmt: skip
    for name, b in s["conditions"].items():
        lo, hi = b["full"]["interval95"]
        lines.append(
            f"| `{name}` | {100 * b['full']['accuracy_mean']:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] | "
            f"{100 * b['bazi_only']['accuracy_mean']:.1f}% | {100 * b['official_key']['accuracy_mean']:.1f}% | "
            f"{b['unanswered']}/{b['replies']} | {b['loops']} | {b['loop_guard_used']} | {b['names_a_book']} / {b['names_a_book_not_in_cards']} |"
        )
    lines += [
        "",
        "| Paired comparison (person-cluster bootstrap) | All 555 | BaZi-only | Official keys |",
        "|---|---|---|---|",
    ]
    for v in s["comparisons"].values():
        cells = []
        for t in ("full", "bazi_only", "official_key"):
            d = v[t]
            cells.append(
                f"{100 * d['difference']:+.1f} pp [{100 * d['interval95'][0]:+.1f}, {100 * d['interval95'][1]:+.1f}]"
            )
        lines.append(f"| {v['effect_of']} | " + " | ".join(cells) + " |")
    lines.append("")


def main():
    lines = [
        "# Phase 4 results\n",
        "Generated by `scripts/phase4/make_report.py` from the files in this folder. Real local inference on an Apple M5 with 32 GB. "
        "Design and frozen rules: `docs/phase4_preregistration.md`. The Phase 3 test splits, benchmark_v3 and BaziQA were inspected "
        "before Phase 4 and are development or previously inspected material here; only `data/phase4/final_test` is untouched, "
        "and it was scored once, after the selection (`final_test_access.jsonl`).\n",
    ]
    summary = {}
    training(lines, summary)
    selection(lines, summary)
    runs = final_w0(lines, summary) or {}
    approach_b(lines, summary, runs)
    benchmark(lines, summary)
    baziqa(lines, summary)
    if MISSING:
        lines += (
            ["## Files not present when this report was built\n"]
            + [f"- `{m}`" for m in sorted(set(MISSING))]
            + [""]
        )
    (ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=str) + "\n"
    )
    print(f"Wrote {ROOT / 'RESULTS.md'} ({len(lines)} lines); missing inputs: {len(set(MISSING))}")


if __name__ == "__main__":
    main()
