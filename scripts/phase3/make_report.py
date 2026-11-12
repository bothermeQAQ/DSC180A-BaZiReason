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
ROUNDS = (
    "bazireason-qwen3-4b-lora-p3-r1",
    "bazireason-qwen3-4b-lora-p3-r2",
    "bazireason-qwen3-4b-lora-p3-r3",
)
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


def cell(block):
    if not block or not block.get("n"):
        return "–"
    k = block.get("hits", block.get("k"))
    return f"{k}/{block['n']} ({100 * k / block['n']:.1f}%)"


def pp(estimate):
    if not estimate or estimate.get("interval95") is None:
        return "–"
    low, high = estimate["interval95"]
    return f"{100 * estimate['difference']:+.1f} pp [{100 * low:+.1f}, {100 * high:+.1f}]"


def pct(value):
    return "–" if value is None else f"{100 * value:.1f}%"


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
    lines.append("## 1. Training runs and adapter selection\n")
    lines.append(
        "| Run | Starts from | Rows | Iterations | Optimizer updates | Peak LR | Minutes | "
        "Validation loss (start → selected) | Sequence tokens | Peak memory |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---:|---:|")
    summary["training"] = {}
    for name in ROUNDS:
        folder = ROOT / "training" / name
        if not (folder / "training_summary.json").exists():
            continue
        done = load(folder / "training_summary.json")
        plan = load(folder / "training_plan.json")
        log = [json.loads(x) for x in (folder / "metrics.jsonl").read_text().splitlines()]
        first = next(r["val_loss"] for r in log if r["kind"] == "valid")
        peak = max(r["learning_rate"] for r in log if r["kind"] == "train")
        lines.append(
            f"| `{name}` | `{Path(plan['init_adapter']).parent.name}` | {plan['train_rows']:,} | "
            f"{plan['iterations']:,} | {plan['optimizer_updates']:,} | {peak:.0e} | "
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
    lines.append(
        "Validation losses of different rounds are not comparable: each round's validation rows "
        "come from its own dataset.\n"
    )
    for name in ROUNDS:
        if (ROOT / "training" / name / "loss_curve.png").exists():
            lines.append(f"![loss curve](training/{name}/loss_curve.png)\n")
    first = load(ROOT / "selection.json")
    if first:
        summary["selection_round1_round2"] = first
        lines.append(
            f"**Round 1 against round 2.** Rule fixed before round 2 was scored: {first['rule']}.\n"
        )
        lines.append("| Round | Phase 3 validation answers correct |")
        lines.append("|---|---:|")
        for row in first["rounds"]:
            mark = " (chosen)" if row["round"] == first["selected"] else ""
            lines.append(
                f"| {row['round']}{mark} | {row['validation_correct']}/{row['validation_rows']} |"
            )
        lines.append("")
    final = (
        load(ROOT / "selection_final.json") if (ROOT / "selection_final.json").exists() else None
    )
    if final:
        summary["selection_final"] = final
        lines.append(
            f"**Round 2 against round 3.** Rule fixed before round 3 was scored: {final['rule']}.\n"
        )
        lines.append("| Round | Phase 3 validation | Phase 2 validation | Total |")
        lines.append("|---|---:|---:|---:|")
        for row in final["candidates"]:
            mark = " **(final)**" if row["round"] == final["selected"] else ""
            lines.append(
                f"| {row['round']}{mark} | {row['phase3_valid'][0]}/{row['phase3_valid'][1]} | "
                f"{row['phase2_valid'][0]}/{row['phase2_valid'][1]} | {row['total']} |"
            )
        lines.append("")
    return (final or first or {}).get("selected", "p3-r2").replace("-", "")


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
    reports = {}
    for label, stem in names:
        report = load(ROOT / "heldout" / f"p3test_{stem}.jsonl.scores.json")
        if report:
            reports[label] = report["summary"]
    if not reports:
        return
    labels = list(reports)
    lines.append(
        "## 2. Phase 3 held-out test split (801 rows, unseen years, a third in unseen wordings)\n"
    )
    lines.append("Greedy decoding, same prompts and 800-token budget for every model.\n")
    lines.append("| Task family | What it asks | " + " | ".join(labels) + " |")
    lines.append("|---|---|" + "---:|" * len(labels))
    for family in FAMILY_NOTES:
        if any(family in r["by_family"] for r in reports.values()):
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
    summary["phase3_test"] = reports
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
        ("Claims: single verdicts right", lambda r: pct(r["claims"]["verdict_accuracy"])),
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
    lines.append(
        "**One-pass audit of a supplied tool answer (strict JSON, 180 rows): corrected / followed the "
        "error.** `day_shift_midday` and `hour_stem_night` never occur in training.\n"
    )
    lines.append("| Supplied answer | " + " | ".join(labels) + " |")
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
    scores = {label: records(ROOT / "heldout" / f"p3test_{stem}.jsonl") for label, stem in names}
    scores = {k: v for k, v in scores.items() if v}
    base = [label for label in labels if label in scores]
    pairs = [(a, b) for i, a in enumerate(base) for b in base[i + 1 :]]
    if pairs:
        lines.append(
            "| Paired difference on the same rows (row-group bootstrap) | All rows | Reasoning tasks |"
        )
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


def p2_rows(a):
    return [
        ("All rows", a["overall"]),
        ("Exact four pillars", a["calendar"]["exact_four_pillar"]),
        ("Exact four pillars, no tools", a["calendar"]["exact_no_tools"]),
        ("Exact four pillars, with tool facts", a["calendar"]["exact_with_facts"]),
        ("Rule tasks (Phase 2 prompts)", a["by_family"].get("rule")),
        ("Internal consistency of a chart", a["by_family"].get("internal")),
        ("Rule multiple choice", a["by_family"].get("mcq")),
        ("Multi-turn", a["by_family"].get("multi_turn")),
        ("Counterfactual comparison", a["by_family"].get("counterfactual")),
        ("Corrupted charts detected", a["verification"]["corruption_detection"]),
        (
            "False alarms on sound charts (lower is better)",
            a["verification"]["false_alarm_on_sound_charts"],
        ),
        ("Definite predictions declined", a["policy"]["declined_definite_prediction"]),
    ]


def regression(lines, summary, columns):
    """Phase 2-style splits: did the calendar skills survive?"""
    lines.append("## 3. Calendar and Phase 2 skills\n")
    splits = (
        (
            "Never-inspected check split (485 rows, generated after round 2)",
            {"Phase 2": ROOT / "heldout/p2check_p2.jsonl"}
            | {label: ROOT / f"heldout/p2check_{stem}.jsonl" for label, stem in columns},
        ),
        (
            "Phase 2 first test split (633 rows; read row by row while diagnosing round 2)",
            {"Phase 2": P2 / "heldout/tuned_v2.jsonl"}
            | {label: ROOT / f"heldout/p2test_{stem}.jsonl" for label, stem in columns},
        ),
        (
            "Phase 2 fresh test split (486 rows)",
            {"Phase 2": P2 / "heldout/fresh_tuned_v2.jsonl"}
            | {label: ROOT / f"heldout/p2fresh_{stem}.jsonl" for label, stem in columns},
        ),
    )
    summary["phase2_skills"] = {}
    for title, paths in splits:
        reports = {}
        for label, path in paths.items():
            if Path(str(path) + ".scores.json").exists():
                reports[label] = json.loads(Path(str(path) + ".scores.json").read_text())["summary"]
        if len(reports) < 2:
            continue
        labels = list(reports)
        lines.append(f"**{title}**\n")
        lines.append("| Measure | " + " | ".join(labels) + " |")
        lines.append("|---|" + "---:|" * len(labels))
        table = {label: p2_rows(reports[label]) for label in labels}
        for index, (name, _) in enumerate(table[labels[0]]):
            lines.append(
                f"| {name} | " + " | ".join(cell(table[label][index][1]) for label in labels) + " |"
            )
        first = records(paths["Phase 2"])
        diffs = {}
        for label in labels[1:]:
            second = records(paths[label])
            if first and second:
                diffs[label] = {
                    "all": paired_difference(first, second),
                    "calendar": paired_difference(
                        first, second, where=lambda s: s["task"] == "chart"
                    ),
                    "rule": paired_difference(first, second, where=lambda s: s["family"] == "rule"),
                }
        for key, name in (
            ("all", "all rows"),
            ("calendar", "calendar rows"),
            ("rule", "rule tasks"),
        ):
            lines.append(
                f"| Paired difference against Phase 2, {name} | – | "
                + " | ".join(pp(diffs.get(label, {}).get(key)) for label in labels[1:])
                + " |"
            )
        lines.append("")
        summary["phase2_skills"][title] = {"summaries": reports, "paired_vs_phase2": diffs}


def benchmark_v3(lines, summary, columns):
    strict = {
        "Untuned": load(P2 / "v3/base_strict.jsonl.scores.json"),
        "Phase 2": load(P2 / "v3/tuned_v2_strict.jsonl.scores.json"),
    }
    for label, stem in columns:
        strict[label] = load(ROOT / f"v3/{stem}_strict.jsonl.scores.json")
    strict = {k: v for k, v in strict.items() if v}
    if len(strict) < 3:
        return
    labels = list(strict)
    lines.append("## 4. benchmark_v3 (261 cases; the checkpoint's own prompts and evaluator)\n")
    lines.append("**Strict protocol: one JSON object, no visible reasoning.**\n")
    lines.append("| Condition | Measure | " + " | ".join(labels) + " |")
    lines.append("|---|---|" + "---:|" * len(labels))

    def get(report, method, key):
        return (
            report["methods"][method]["overall"].get(key) if method in report["methods"] else None
        )

    summary["benchmark_v3_strict"] = {}
    for method, key, text in (
        ("direct", "exact_accuracy", "exact four pillars (225 answerable)"),
        ("direct", "decision_accuracy", "right status (261)"),
        ("direct", "false_refusal_rate", "answerable request refused (lower is better)"),
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
    ):
        values = [get(strict[label], method, key) for label in labels]
        summary["benchmark_v3_strict"][f"{method}.{key}"] = dict(zip(labels, values, strict=True))
        lines.append(f"| `{method}` | {text} | " + " | ".join(pct(v) for v in values) + " |")
    lines.append("")
    lines.append("Followed the wrong tool answer, by kind of error:\n")
    lines.append(
        "| Kind of error | Cases | Visible in the supplied chart itself? | "
        + " | ".join(labels)
        + " |"
    )
    lines.append("|---|---:|---|" + "---:|" * len(labels))
    visible = {
        "hour_stem": "yes (五鼠遁)",
        "day_rollover": "partly (day and hour disagree; which is wrong is not shown)",
        "default_guess": "yes (the request itself is ambiguous)",
        "jie_side": "no (needs the solar-term instant)",
    }
    by_type = {
        label: get(strict[label], "tool_corrupted", "corruption_following_by_type") or {}
        for label in labels
    }
    summary["benchmark_v3_strict"]["following_by_type"] = by_type
    for kind in visible:
        n = by_type[labels[-1]].get(kind, {}).get("n", "–")
        lines.append(
            f"| `{kind}` | {n} | {visible[kind]} | "
            + " | ".join(
                pct(by_type[label].get(kind, {}).get("following_rate")) for label in labels
            )
            + " |"
        )
    lines.append("")
    reasoned = {"Phase 2": load(P2 / "v3/tuned_v2_reasoned.jsonl.scores.json")}
    for label, stem in columns:
        reasoned[label] = load(ROOT / f"v3/{stem}_reasoned.jsonl.scores.json")
    reasoned = {k: v for k, v in reasoned.items() if v}
    if len(reasoned) >= 2:
        names = list(reasoned)
        lines.append(
            "**Reasoned protocol: visible reasoning, then the JSON object. Exact four pillars.**\n"
        )
        lines.append("| Condition | " + " | ".join(names) + " |")
        lines.append("|---|" + "---:|" * len(names))
        summary["benchmark_v3_reasoned"] = {}
        for method in ("direct", "tool_intermediate"):
            values = [reasoned[n]["methods"][method]["overall"] for n in names]
            summary["benchmark_v3_reasoned"][method] = {
                n: [v["exact_matches"], v["n_labeled"]] for n, v in zip(names, values, strict=True)
            }
            lines.append(
                f"| `{method}` | "
                + " | ".join(f"{v['exact_matches']}/{v['n_labeled']}" for v in values)
                + " |"
            )
        lines.append("")


def baziqa(lines, summary):
    report = load(ROOT / "baziqa1024/summary.json")
    if not report:
        return
    loops = load(ROOT / "baziqa1024/citations_and_loops.json")
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
        "citations_and_loops": loops,
    }
    names = {"base": "Untuned", "p2": "Phase 2", "p3": "Phase 3 round 2", "p3r3": "Phase 3 round 3"}
    lines.append("## 5. BaziQA contest questions, 1,024-token budget\n")
    lines.append(
        "555 questions on 117 charts; chance is 24.7%. Same prompt, sampling settings (temperature 0.7) "
        "and budget for every model. BaziQA was already evaluated in Phase 2, so it is an existing "
        "benchmark, not an untouched test, and nothing was tuned on it. Agreement with a contest key "
        "is not evidence that destiny can be predicted.\n"
    )
    lines.append(
        "| Model | Chart tool | Runs | All 555 [95% interval] | BaZi-only (290) | Official keys (365) | "
        "BaZi-only, official keys | Unanswered | Mean reply tokens |"
    )
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|")
    for model in names:
        for tools in ("no_tools", "tools"):
            block = report["conditions"].get(f"{model}_{tools}")
            if not block:
                continue
            full = block["full"]
            low, high = full["interval95_person_bootstrap"]
            lines.append(
                f"| {names[model]} | {'verified' if tools == 'tools' else 'none'} | {full['n_runs']} | "
                f"{100 * full['accuracy_mean']:.1f}% [{100 * low:.1f}, {100 * high:.1f}] | "
                f"{pct(block['bazi_only']['accuracy_mean'])} | {pct(block['official_key']['accuracy_mean'])} | "
                f"{pct(block['bazi_only_official_key']['accuracy_mean'])} | "
                f"{pct(full['unanswered_rate'])} | {block['mean_completion_tokens']:.0f} |"
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
    present = [
        n for n in ("base_tools", "p2_tools", "p3_tools", "p3r3_tools") if n in report["conditions"]
    ]
    label = {n: names[n.rsplit("_", 1)[0]] for n in present}
    if present:
        categories = sorted(
            {c for n in present for c in report["conditions"][n]["by_category_full"]}
        )
        lines.append("Accuracy by question category, with the verified chart tool:\n")
        lines.append("| Category | Questions | " + " | ".join(label[n] for n in present) + " |")
        lines.append("|---|---:|" + "---:|" * len(present))
        for category in categories:
            first = report["conditions"][present[0]]["by_category_full"].get(category, {})
            runs = report["conditions"][present[0]]["full"]["n_runs"]
            lines.append(
                f"| {category} | {first.get('n', 0) // max(runs, 1)} | "
                + " | ".join(
                    pct(
                        report["conditions"][n]["by_category_full"]
                        .get(category, {})
                        .get("accuracy")
                    )
                    for n in present
                )
                + " |"
            )
        lines.append("")
        years = sorted(report["conditions"][present[0]]["full"]["by_year"])
        lines.append("Accuracy by contest year, with the verified chart tool:\n")
        lines.append("| Year | " + " | ".join(label[n] for n in present) + " |")
        lines.append("|---|" + "---:|" * len(present))
        for year in years:
            lines.append(
                f"| {year} | "
                + " | ".join(
                    pct(report["conditions"][n]["full"]["by_year"].get(year)) for n in present
                )
                + " |"
            )
        lines.append("")
    lines.append(
        "| Model and tool setting | Checkable statements matching the verified chart | Replies naming a book "
        "| Naming a book not in a list of real classics | Repetition loops |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for name, block in report["conditions"].items():
        audit = block["checkable_statements"] or {}
        extra = (loops or {}).get(name, {})
        lines.append(
            f"| {name} | "
            + (
                f"{audit['correct']}/{audit['n_claims']} ({100 * audit['rate']:.1f}%)"
                if audit
                else "–"
            )
            + f" | {extra.get('names_a_book', '–')}/{extra.get('replies', '–')} | "
            f"{extra.get('names_unknown_book', '–')} | {extra.get('repetition_loop', '–')} |"
        )
    lines.append("")
    if report["self_derived_charts"]:
        lines.append(
            "| Model | Own chart exactly right without the tool (of charts the engine can verify) |"
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
    final = training(lines, summary)
    rounds = [("Round 1", "p3r1"), ("Round 2", "p3r2"), ("Round 3", "p3r3")]
    rounds = [(f"{name} (final)" if stem == final else name, stem) for name, stem in rounds]
    heldout3(lines, summary, [("Untuned", "base"), ("Phase 2", "p2"), *rounds])
    regression(lines, summary, rounds[1:])
    benchmark_v3(lines, summary, rounds[1:])
    baziqa(lines, summary)
    summary["final_round"] = final
    if MISSING:
        lines.append("## Files not present when this report was built\n")
        lines.extend(f"- `{name}`" for name in sorted(set(MISSING)))
        lines.append("")
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {ROOT / 'RESULTS.md'} ({len(lines)} lines); missing inputs: {len(set(MISSING))}")


if __name__ == "__main__":
    main()
