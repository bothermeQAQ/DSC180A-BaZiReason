"""Build results/phase3/RESULTS.md and summary.json from the saved result files.

    .venv-mlx/bin/python scripts/phase3/make_report.py

Every number is read from a file written by a training or evaluation run; a
table whose file is missing is left out and listed at the end. Nothing is
recomputed by a model here.
"""

import json
from pathlib import Path

from bazireason.p3_eval import paired_difference

ROOT = Path("results/phase3")
P2 = Path("results/phase2")
ROUNDS = ("bazireason-qwen3-4b-lora-p3-r1", "bazireason-qwen3-4b-lora-p3-r2")
MISSING = []


def load(path):
    path = Path(path)
    if not path.exists():
        MISSING.append(str(path))
        return None
    return json.loads(path.read_text())


def records(path):
    path = Path(path)
    if not path.exists():
        MISSING.append(str(path))
        return None
    return [json.loads(line)["score"] for line in path.read_text().splitlines()]


def cell(block, hits="hits"):
    if not block or not block.get("n"):
        return "–"
    k = block.get(hits, block.get("k"))
    return f"{k}/{block['n']} ({100 * k / block['n']:.1f}%)"


def pp(estimate):
    if not estimate or estimate.get("interval95") is None:
        return "–"
    low, high = estimate["interval95"]
    return f"{100 * estimate['difference']:+.1f} pp [{100 * low:+.1f}, {100 * high:+.1f}]"


def loss_curve(name):
    log = ROOT / "training" / name / "metrics.jsonl"
    if not log.exists():
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    train = [(r["iteration"], r["train_loss"]) for r in rows if r["kind"] == "train"]
    valid = [(r["iteration"], r["val_loss"]) for r in rows if r["kind"] == "valid"]
    rate = [(r["iteration"], r["learning_rate"]) for r in rows if r["kind"] == "train"]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True, height_ratios=(3, 1))
    top.plot(*zip(*train, strict=True), color="#4c72b0", linewidth=1, label="training loss")
    top.plot(
        *zip(*valid, strict=True),
        color="#c44e52",
        marker="o",
        linewidth=1.5,
        label="validation loss (fixed rows)",
    )
    top.set_yscale("log")
    top.set_ylabel("cross-entropy per answer token (log scale)")
    top.set_title(f"{name}: LoRA fine-tuning on Apple M5, 32 GB")
    top.grid(alpha=0.3, which="both")
    top.legend(frameon=False)
    bottom.plot(*zip(*rate, strict=True), color="#55a868")
    bottom.set_ylabel("learning rate")
    bottom.set_xlabel("iteration (batch of 2 sequences; one optimizer update per 2 iterations)")
    bottom.grid(alpha=0.3)
    figure.tight_layout()
    target = ROOT / "training" / name / "loss_curve.png"
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target


def training(lines, summary):
    lines.append("## 1. Training runs\n")
    lines.append(
        "| Run | Starts from | Rows | Iterations | Optimizer updates | Minutes | Validation loss "
        "(start → selected) | Sequence tokens | Peak memory |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|---:|---:|")
    summary["training"] = {}
    for name in ROUNDS:
        done = load(ROOT / "training" / name / "training_summary.json")
        plan = load(ROOT / "training" / name / "training_plan.json")
        if not done or not plan:
            continue
        log = [
            json.loads(x)
            for x in (ROOT / "training" / name / "metrics.jsonl").read_text().splitlines()
        ]
        first = next(r["val_loss"] for r in log if r["kind"] == "valid")
        lines.append(
            f"| `{name}` | `{Path(plan['init_adapter']).parent.name}` | {plan['train_rows']:,} | "
            f"{plan['iterations']:,} | {plan['optimizer_updates']:,} | "
            f"{done['duration_seconds'] / 60:.0f} | {first:.3f} → {done['selected_validation_loss']:.4f} "
            f"(iteration {done['selected_checkpoint_iteration']}) | "
            f"{done.get('sequence_tokens_processed', plan['train_sequence_tokens_per_epoch']):,} | "
            f"{done['peak_memory_gb']:.1f} GB |"
        )
        summary["training"][name] = {
            "init_adapter": plan["init_adapter"],
            "train_rows": plan["train_rows"],
            "iterations": plan["iterations"],
            "optimizer_updates": plan["optimizer_updates"],
            "trainable_parameters": plan["trainable_parameters"],
            "minutes": done["duration_seconds"] / 60,
            "validation_loss_start": first,
            "validation_loss_selected": done["selected_validation_loss"],
            "selected_iteration": done["selected_checkpoint_iteration"],
            "peak_memory_gb": done["peak_memory_gb"],
            "adapter_sha256": done.get("selected_adapter_sha256") or done.get("adapter_sha256"),
        }
        loss_curve(name)
    lines.append("")
    for name in ROUNDS:
        if (ROOT / "training" / name / "loss_curve.png").exists():
            lines.append(f"![loss curve](training/{name}/loss_curve.png)\n")
    selection = load(ROOT / "selection.json")
    if selection:
        summary["selection"] = selection
        lines.append(
            "**Adapter selection** (rule fixed before round 2 was scored: "
            + selection["rule"]
            + ").\n"
        )
        lines.append("| Round | Validation answers correct | Reasoning rows correct |")
        lines.append("|---|---:|---:|")
        for row in selection["rounds"]:
            mark = " **(selected)**" if row["round"] == selection["selected"] else ""
            lines.append(
                f"| {row['round']}{mark} | {row['validation_correct']}/{row['validation_rows']} | "
                f"{row['reasoning_correct']}/{row['reasoning_rows']} |"
            )
        lines.append("")
    return selection


FAMILY_NOTES = {
    "drill": "one table lookup",
    "gods": "Ten Gods of the chart and named pairings",
    "strength": "得令 / 得地 / 得势 count",
    "pattern": "month-order pattern by 透干",
    "yongshen": "用神 under a named school",
    "interact": "all 合冲刑害破 with adjacency",
    "flow": "flow year against the chart",
    "domain": "traditional view of an area of life",
    "claims": "check statements about a chart",
    "ambig": "questions with and without a unique answer",
    "mcq3": "rule-keyed multiple choice",
    "dialog": "second and third turns",
    "overview": "whole-chart summary over the tool block",
    "strict_shift": "one-pass audit of a tool answer (strict JSON)",
}


def heldout3(lines, summary, names):
    """Phase 3 test split: one column per model."""
    reports = {
        label: load(ROOT / "heldout" / f"p3test_{stem}.jsonl.scores.json") for label, stem in names
    }
    reports = {k: v["summary"] for k, v in reports.items() if v}
    if not reports:
        return
    labels = list(reports)
    lines.append(
        "## 2. Phase 3 held-out test split (801 rows, unseen years, one third unseen wordings)\n"
    )
    lines.append("Greedy decoding, same prompts and 800-token budget for every model.\n")
    lines.append("| Task family | What it asks | " + " | ".join(labels) + " |")
    lines.append("|---|---|" + "---:|" * len(labels))
    families = [f for f in FAMILY_NOTES if any(f in r["by_family"] for r in reports.values())]
    for family in families:
        lines.append(
            f"| `{family}` | {FAMILY_NOTES[family]} | "
            + " | ".join(cell(reports[label]["by_family"].get(family)) for label in labels)
            + " |"
        )
    lines.append(
        "| **Reasoning tasks (all but the strict audit)** | | "
        + " | ".join(f"**{cell(reports[label]['reasoning_tasks'])}**" for label in labels)
        + " |"
    )
    lines.append(
        "| **All rows** | | "
        + " | ".join(f"**{cell(reports[label]['overall'])}**" for label in labels)
        + " |\n"
    )
    summary["phase3_test"] = {label: reports[label] for label in labels}
    lines.append("| Breakdown | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---:|" * len(labels))
    rows = (
        ("Reasoning rows, bare pillars", lambda r: cell(r["by_mode"]["none"])),
        ("Reasoning rows, with the verified tool block", lambda r: cell(r["by_mode"]["chart"])),
        ("Reasoning rows, wording seen in training", lambda r: cell(r["seen_wording"])),
        ("Reasoning rows, wording never seen", lambda r: cell(r["unseen_wording"])),
        (
            "Ambiguity: right decision (answer or decline)",
            lambda r: cell(r["ambiguity"]["decision_correct"]),
        ),
        (
            "Ambiguity: definite answer where none exists (lower is better)",
            lambda r: cell(r["ambiguity"]["definite_on_indeterminate"]),
        ),
        (
            "Ambiguity: declined although an answer exists (lower is better)",
            lambda r: cell(r["ambiguity"]["abstained_on_determinate"]),
        ),
        (
            "Claims: single verdicts right",
            lambda r: f"{100 * r['claims']['verdict_accuracy']:.1f}%",
        ),
        (
            "Claims: unverifiable prediction called correct (lower is better)",
            lambda r: "{}/{}".format(*r["claims"]["speculative_endorsed"]),
        ),
        (
            "Claims: false statement called correct (lower is better)",
            lambda r: "{}/{}".format(*r["claims"]["false_claims_accepted"]),
        ),
    )
    for label_, getter in rows:
        lines.append(
            f"| {label_} | " + " | ".join(getter(reports[label]) for label in labels) + " |"
        )
    lines.append("")
    # Strict audit by kind of error.
    lines.append(
        "**One-pass audit of a supplied tool answer (strict JSON, 180 rows).** "
        "`day_shift_midday` and `hour_stem_night` never occur in training.\n"
    )
    lines.append(
        "| Supplied answer | "
        + " | ".join(f"{label}: right / followed the error" for label in labels)
        + " |"
    )
    lines.append("|---|" + "---|" * len(labels))
    kinds = sorted(
        {k for r in reports.values() for k in r.get("strict_audit", {}).get("by_kind", {})}
    )
    for kind in kinds:
        cells = []
        for label in labels:
            block = reports[label].get("strict_audit", {}).get("by_kind", {}).get(kind)
            cells.append(
                f"{block['correct']['hits']}/{block['correct']['n']} / {block['followed']['hits']}"
                if block
                else "–"
            )
        lines.append(f"| `{kind}` | " + " | ".join(cells) + " |")
    cells = []
    for label in labels:
        audit = reports[label].get("strict_audit")
        cells.append(
            f"**{audit['corrupted_correct']['hits']}/{audit['corrupted_correct']['n']} / "
            f"{audit['corrupted_followed']['hits']}**"
            if audit
            else "–"
        )
    lines.append("| **all wrong answers** | " + " | ".join(cells) + " |\n")
    # Paired differences between models on the same rows.
    scores = {label: records(ROOT / "heldout" / f"p3test_{stem}.jsonl") for label, stem in names}
    scores = {k: v for k, v in scores.items() if v}
    pairs = [
        (a, b) for i, a in enumerate(labels) for b in labels[i + 1 :] if a in scores and b in scores
    ]
    if pairs:
        lines.append("| Paired difference (row-group bootstrap) | All rows | Reasoning tasks |")
        lines.append("|---|---|---|")
        summary["phase3_test_paired"] = {}
        for first, second in pairs:
            everything = paired_difference(scores[first], scores[second])
            reasoning = paired_difference(
                scores[first], scores[second], where=lambda s: not s["family"].startswith("strict")
            )
            summary["phase3_test_paired"][f"{second} minus {first}"] = {
                "all": everything,
                "reasoning": reasoning,
            }
            lines.append(f"| {second} minus {first} | {pp(everything)} | {pp(reasoning)} |")
        lines.append("")


def regression(lines, summary, best):
    """Phase 2 held-out splits: did the calendar skills survive?"""
    lines.append("## 3. Regression on the Phase 2 held-out splits\n")
    pairs = (
        (
            "First split (633 rows)",
            P2 / "heldout/tuned_v2.jsonl",
            ROOT / f"heldout/p2test_{best}.jsonl",
        ),
        (
            "Fresh split (486 rows)",
            P2 / "heldout/fresh_tuned_v2.jsonl",
            ROOT / f"heldout/p2fresh_{best}.jsonl",
        ),
    )
    summary["phase2_regression"] = {}
    for title, old_path, new_path in pairs:
        old, new = load(str(old_path) + ".scores.json"), load(str(new_path) + ".scores.json")
        if not old or not new:
            continue
        a, b = old["summary"], new["summary"]
        lines.append(f"**{title}**\n")
        lines.append("| Measure | Phase 2 adapter | Phase 3 adapter |")
        lines.append("|---|---:|---:|")

        def k(block):
            return cell(block, "k")

        rows = [
            ("All rows", a["overall"], b["overall"]),
            (
                "Exact four pillars",
                a["calendar"]["exact_four_pillar"],
                b["calendar"]["exact_four_pillar"],
            ),
            (
                "Exact four pillars, no tools",
                a["calendar"]["exact_no_tools"],
                b["calendar"]["exact_no_tools"],
            ),
            (
                "Exact four pillars, with tool facts",
                a["calendar"]["exact_with_facts"],
                b["calendar"]["exact_with_facts"],
            ),
            (
                "Rule tasks (Ten Gods, relations, …)",
                a["by_family"].get("rule"),
                b["by_family"].get("rule"),
            ),
            (
                "Internal consistency of a chart",
                a["by_family"].get("internal"),
                b["by_family"].get("internal"),
            ),
            ("Rule multiple choice", a["by_family"].get("mcq"), b["by_family"].get("mcq")),
            ("Multi-turn", a["by_family"].get("multi_turn"), b["by_family"].get("multi_turn")),
            (
                "Counterfactual comparison",
                a["by_family"].get("counterfactual"),
                b["by_family"].get("counterfactual"),
            ),
            (
                "Corrupted charts detected",
                a["verification"]["corruption_detection"],
                b["verification"]["corruption_detection"],
            ),
            (
                "False alarms on sound charts",
                a["verification"]["false_alarm_on_sound_charts"],
                b["verification"]["false_alarm_on_sound_charts"],
            ),
            (
                "Definite predictions declined",
                a["policy"]["declined_definite_prediction"],
                b["policy"]["declined_definite_prediction"],
            ),
        ]
        for name, left, right in rows:
            lines.append(f"| {name} | {k(left)} | {k(right)} |")
        first, second = records(old_path), records(new_path)
        if first and second:
            everything = paired_difference(first, second)
            rule = paired_difference(first, second, where=lambda s: s["family"] == "rule")
            lines.append(f"| Paired difference, all rows | | {pp(everything)} |")
            lines.append(f"| Paired difference, rule tasks | | {pp(rule)} |")
            summary["phase2_regression"][title] = {
                "phase2": a,
                "phase3": b,
                "paired_all": everything,
                "paired_rule": rule,
            }
        lines.append("")


def benchmark_v3(lines, summary, best):
    strict_new = load(ROOT / f"v3/{best}_strict.jsonl.scores.json")
    strict_old = load(P2 / "v3/tuned_v2_strict.jsonl.scores.json")
    strict_base = load(P2 / "v3/base_strict.jsonl.scores.json")
    if not strict_new:
        return
    lines.append("## 4. benchmark_v3 (261 cases; the checkpoint's own prompts and evaluator)\n")
    lines.append("**Strict protocol: one JSON object, no visible reasoning.**\n")
    lines.append("| Condition | Measure | Untuned | Phase 2 | Phase 3 |")
    lines.append("|---|---|---:|---:|---:|")

    def get(report, method, key):
        return (
            report["methods"][method]["overall"].get(key)
            if report and method in report["methods"]
            else None
        )

    def show(value, percent=True):
        if value is None:
            return "–"
        return f"{100 * value:.1f}%" if percent else str(value)

    summary["benchmark_v3_strict"] = {}
    for method, key, label in (
        ("direct", "exact_accuracy", "exact four pillars (225 answerable)"),
        ("direct", "decision_accuracy", "right status (261)"),
        (
            "direct",
            "ambiguous_definitive_rate",
            "definite chart for an ambiguous request (lower is better)",
        ),
        ("tool_intermediate", "exact_accuracy", "exact four pillars"),
        ("tool_context", "exact_accuracy", "exact four pillars (answer supplied)"),
        ("tool_corrupted", "exact_accuracy", "exact four pillars despite a wrong tool answer"),
        (
            "tool_corrupted",
            "corruption_following_rate",
            "followed the wrong tool answer (lower is better)",
        ),
        ("tool_corrupted", "decision_accuracy", "right status"),
    ):
        values = [get(r, method, key) for r in (strict_base, strict_old, strict_new)]
        summary["benchmark_v3_strict"][f"{method}.{key}"] = dict(
            zip(("base", "phase2", "phase3"), values, strict=True)
        )
        lines.append(f"| `{method}` | {label} | " + " | ".join(show(v) for v in values) + " |")
    lines.append("")
    lines.append("Followed the wrong tool answer, by kind of error:\n")
    lines.append(
        "| Kind of error | Cases | Visible in the chart itself? | Untuned | Phase 2 | Phase 3 |"
    )
    lines.append("|---|---:|---|---:|---:|---:|")
    visible = {
        "hour_stem": "yes (五鼠遁)",
        "day_rollover": "partly (day and hour disagree; which one is wrong is not shown)",
        "default_guess": "yes (the request itself is ambiguous)",
        "jie_side": "no (needs the solar-term instant)",
    }
    by_type = {}
    for name, report in (("base", strict_base), ("phase2", strict_old), ("phase3", strict_new)):
        by_type[name] = (
            (get(report, "tool_corrupted", "corruption_following_by_type") or {}) if report else {}
        )
    summary["benchmark_v3_strict"]["following_by_type"] = by_type
    for kind in ("hour_stem", "day_rollover", "default_guess", "jie_side"):
        n = by_type["phase3"].get(kind, {}).get("n", "–")
        lines.append(
            f"| `{kind}` | {n} | {visible[kind]} | "
            + " | ".join(
                show(by_type[m].get(kind, {}).get("following_rate"))
                for m in ("base", "phase2", "phase3")
            )
            + " |"
        )
    lines.append("")
    contrasts = strict_new.get("comparisons") or {}
    summary["benchmark_v3_strict"]["comparisons_keys"] = sorted(contrasts)[:20]
    reasoned_new = load(ROOT / f"v3/{best}_reasoned.jsonl.scores.json")
    reasoned_old = load(P2 / "v3/tuned_v2_reasoned.jsonl.scores.json")
    if reasoned_new:
        lines.append("**Reasoned protocol: visible reasoning, then the JSON object.**\n")
        lines.append("| Condition | Exact four pillars, Phase 2 | Exact four pillars, Phase 3 |")
        lines.append("|---|---:|---:|")
        summary["benchmark_v3_reasoned"] = {}
        for method in ("direct", "tool_intermediate"):
            old = reasoned_old["methods"][method]["overall"] if reasoned_old else None
            new = reasoned_new["methods"][method]["overall"]
            summary["benchmark_v3_reasoned"][method] = {
                "phase2": old and old["exact_matches"],
                "phase3": new["exact_matches"],
                "n": new["n_labeled"],
            }
            lines.append(
                f"| `{method}` | {old['exact_matches']}/{old['n_labeled']} | "
                f"{new['exact_matches']}/{new['n_labeled']} |"
                if old
                else f"| `{method}` | – | {new['exact_matches']}/{new['n_labeled']} |"
            )
        lines.append("")


def baziqa(lines, summary):
    report = load(ROOT / "baziqa1024/summary.json")
    if not report:
        return
    summary["baziqa_1024"] = {
        "conditions": {
            name: {track: block[track] for track in report["tracks"]}
            | {
                "truncated_rate": block["truncated_rate"],
                "mean_completion_tokens": block["mean_completion_tokens"],
                "checkable_statements": block["checkable_statements"],
                "by_category_full": block["by_category_full"],
            }
            for name, block in report["conditions"].items()
        },
        "comparisons": report["comparisons"],
        "human_comparison": report["human_comparison"],
        "self_derived_charts": report["self_derived_charts"],
    }
    lines.append("## 5. BaziQA contest questions, 1,024-token budget\n")
    lines.append(
        "555 questions on 117 charts; chance is 24.7%. Same prompt, sampling settings and budget for "
        "every model. BaziQA was already evaluated in Phase 2, so it is an existing benchmark, not an "
        "untouched test. Agreement with a contest key is not evidence that destiny can be predicted.\n"
    )
    lines.append(
        "| Model | Chart tool | Runs | All 555 [95% interval] | BaZi-only (290) | Official keys (365) | "
        "BaZi-only, official keys | Unanswered | Mean reply tokens |"
    )
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|")
    names = {"base": "Untuned", "p2": "Phase 2", "p3": "Phase 3"}
    for model in ("base", "p2", "p3"):
        for tools in ("no_tools", "tools"):
            block = report["conditions"].get(f"{model}_{tools}")
            if not block:
                continue
            full = block["full"]
            low, high = full["interval95_person_bootstrap"]
            lines.append(
                f"| {names[model]} | {'verified' if tools == 'tools' else 'none'} | {full['n_runs']} | "
                f"{100 * full['accuracy_mean']:.1f}% [{100 * low:.1f}, {100 * high:.1f}] | "
                f"{100 * block['bazi_only']['accuracy_mean']:.1f}% | "
                f"{100 * block['official_key']['accuracy_mean']:.1f}% | "
                f"{100 * block['bazi_only_official_key']['accuracy_mean']:.1f}% | "
                f"{100 * full['unanswered_rate']:.1f}% | {block['mean_completion_tokens']:.0f} |"
            )
    lines.append("")
    lines.append(
        "| Paired comparison (person-cluster bootstrap) | All 555 | BaZi-only | Official keys |"
    )
    lines.append("|---|---|---|---|")
    for block in report["comparisons"].values():
        lines.append(
            f"| {block['effect_of']} | {pp(block['full'])} | {pp(block['bazi_only'])} | {pp(block['official_key'])} |"
        )
    lines.append("")
    categories = sorted({c for b in report["conditions"].values() for c in b["by_category_full"]})
    present = [n for n in ("base_tools", "p2_tools", "p3_tools") if n in report["conditions"]]
    if present:
        lines.append("Accuracy by question category, with the verified chart tool:\n")
        lines.append(
            "| Category | Questions | " + " | ".join(names[n.split("_")[0]] for n in present) + " |"
        )
        lines.append("|---|---:|" + "---:|" * len(present))
        for category in categories:
            first = report["conditions"][present[0]]["by_category_full"].get(category, {})
            runs = report["conditions"][present[0]]["full"]["n_runs"]
            lines.append(
                f"| {category} | {first.get('n', 0) // max(runs, 1)} | "
                + " | ".join(
                    f"{100 * report['conditions'][n]['by_category_full'].get(category, {'accuracy': 0})['accuracy']:.1f}%"
                    for n in present
                )
                + " |"
            )
        lines.append("")
    years = sorted(report["conditions"][present[0]]["full"]["by_year"]) if present else []
    if years:
        lines.append("Accuracy by contest year, with the verified chart tool:\n")
        lines.append("| Year | " + " | ".join(names[n.split("_")[0]] for n in present) + " |")
        lines.append("|---|" + "---:|" * len(present))
        for year in years:
            lines.append(
                f"| {year} | "
                + " | ".join(
                    f"{100 * report['conditions'][n]['full']['by_year'][year]:.1f}%"
                    for n in present
                )
                + " |"
            )
        lines.append("")
    lines.append(
        "| Model and tool setting | Checkable statements in the explanations that match the verified chart |"
    )
    lines.append("|---|---:|")
    for name, block in report["conditions"].items():
        audit = block["checkable_statements"]
        if audit:
            lines.append(
                f"| {name} | {audit['correct']}/{audit['n_claims']} ({100 * audit['rate']:.1f}%) |"
            )
    lines.append("")
    if report["self_derived_charts"]:
        lines.append(
            "| Model | Own chart exactly right (no tool), of charts the engine can verify |"
        )
        lines.append("|---|---:|")
        for model, block in report["self_derived_charts"].items():
            lines.append(
                f"| {names[model]} | {block['exact_four_pillars']}/{block['verified_chart_available']} |"
            )
        lines.append("")
    human = report["human_comparison"]
    if human:
        any_ = next(iter(human.values()))
        lines.append(
            f"Human contest champions average {100 * any_['mean_champion']:.1f}% on the years with an "
            "archived champion score; on the same years: "
            + "; ".join(f"{name} {100 * block['mean_model']:.1f}%" for name, block in human.items())
            + ".\n"
        )


def main():
    lines = ["# Phase 3 results\n"]
    lines.append(
        "Generated by `scripts/phase3/make_report.py` from the files in this folder. Real local "
        "inference and training on an Apple M5 with 32 GB; no mock output.\n"
    )
    summary = {}
    selection = training(lines, summary)
    best = (selection or {}).get("selected", "p3-r1").replace("-", "")
    other = "p3r1" if best == "p3r2" else "p3r2"
    names = [
        ("Untuned", "base"),
        ("Phase 2", "p2"),
        (f"Phase 3 ({other[-2:]})", other),
        (f"Phase 3 ({best[-2:]}, selected)", best),
    ]
    heldout3(lines, summary, names)
    regression(lines, summary, best)
    benchmark_v3(lines, summary, best)
    baziqa(lines, summary)
    if MISSING:
        lines.append("## Files not present when this report was built\n")
        lines.extend(f"- `{name}`" for name in sorted(set(MISSING)))
        lines.append("")
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {ROOT / 'RESULTS.md'} ({len(lines)} lines); missing inputs: {len(set(MISSING))}")


if __name__ == "__main__":
    main()
