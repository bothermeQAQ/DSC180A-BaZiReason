"""Collect every Phase 2 result into ``results/phase2/RESULTS.md`` and ``summary.json``.

    python scripts/phase2/make_report.py

The script only reads result files that exist, so it can be run while experiments
are still in progress; missing parts are listed as "not run". It never edits a
prediction file. Differences between models on the same items come with a
cluster bootstrap interval (groups for the held-out split, persons for BaziQA).
"""

import json
from pathlib import Path

from bazireason.p2_eval import paired_difference, summarize

ROOT = Path("results/phase2")
RUNS = {
    "v1": "bazireason-qwen3-4b-lora-v1",
    "e2": "bazireason-qwen3-4b-lora-v1-e2",
}


def load(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def scores_of(path):
    path = Path(path)
    if not path.exists():
        return None
    return [json.loads(line)["score"] for line in path.read_text(encoding="utf-8").splitlines()]


def pct(value, digits=1):
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"


def frac(cell):
    if not cell or not cell.get("n"):
        return "n/a"
    return f"{cell['k']}/{cell['n']} ({pct(cell['rate'])})" if "k" in cell else pct(cell["rate"])


def interval(estimate):
    if not estimate or estimate.get("interval95") is None:
        return ""
    low, high = estimate["interval95"]
    return f"{100 * estimate['difference']:+.1f} pp [{100 * low:+.1f}, {100 * high:+.1f}]"


def loss_curve(name, lines):
    log = ROOT / "training" / name / "metrics.jsonl"
    if not log.exists():
        return None
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        lines.append("_matplotlib is not installed; loss curve not drawn._")
        return None
    train = [(r["iteration"], r["train_loss"]) for r in rows if r["kind"] == "train"]
    valid = [(r["iteration"], r["val_loss"]) for r in rows if r["kind"] == "valid"]
    rate = [(r["iteration"], r["learning_rate"]) for r in rows if r["kind"] == "train"]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True, height_ratios=(3, 1))
    top.plot(
        *zip(*train, strict=True),
        color="#4c72b0",
        linewidth=1,
        label="training loss (mean of 20 iterations)",
    )
    top.plot(
        *zip(*valid, strict=True),
        color="#c44e52",
        marker="o",
        linewidth=1.5,
        label="validation loss (120 fixed rows)",
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


def training_section(lines, summary):
    lines.append("## 1. Training\n")
    for key, name in RUNS.items():
        info = load(ROOT / "training" / name / "training_summary.json")
        if not info:
            lines.append(f"- `{name}`: not run or not finished.\n")
            continue
        summary["training"][key] = info
        curve = loss_curve(name, lines)
        val = info["validation_loss_by_iteration"]
        lines += [
            f"### `{name}`\n",
            "| Quantity | Value |",
            "|---|---|",
            f"| Training rows / validation rows used for selection | {info['train_rows']} / {info['valid_rows_for_selection']} |",
            f"| Iterations in this run (batch {info['batch_size']}) | {info['iterations'] - info['start_iteration']} (global {info['start_iteration'] + 1}-{info['iterations']}) |",
            f"| Gradient accumulation / optimizer updates in the schedule | {info['grad_accumulation_steps']} / {info['optimizer_updates']} |",
            f"| Trainable LoRA parameters | {info['trainable_parameters']:,} |",
            f"| Sequence tokens processed | {info['sequence_tokens_processed']:,} |",
            f"| Answer tokens in the loss | {info['loss_tokens_total']:,} |",
            f"| Wall time | {info['duration_seconds'] / 60:.1f} min |",
            f"| Peak memory | {info['peak_memory_gb']:.1f} GB |",
            f"| Validation loss by iteration | {', '.join(f'{k}: {v:.4f}' for k, v in val.items())} |",
            f"| Selected checkpoint (lowest validation loss) | iteration {info['selected_checkpoint_iteration']}, loss {info['selected_validation_loss']:.4f} |",
            f"| Selected adapter | `{info['selected_adapter']}` |",
            f"| Adapter SHA-256 | `{info['selected_adapter_sha256']}` |",
            "",
        ]
        if curve:
            lines.append(f"![loss curve](training/{name}/loss_curve.png)\n")


HELDOUT_ROWS = (
    ("All 633 held-out items correct", ("overall",)),
    ("Reply ends with a parseable answer", ("parsed",)),
    ("Exact four pillars, all answerable requests", ("calendar", "exact_four_pillar")),
    ("Exact four pillars, with tool facts", ("calendar", "exact_with_facts")),
    ("Exact four pillars, no tools", ("calendar", "exact_no_tools")),
    ("Boundary cases exact, with tool facts", ("calendar", "boundary_exact_with_facts")),
    ("Boundary cases exact, no tools", ("calendar", "boundary_exact_no_tools")),
    ("Status decision correct", ("calendar", "status_decision")),
    ("Clarification decision (expected ambiguous)", ("calendar", "clarification_decision")),
    ("Needs-tool decision", ("calendar", "needs_tool_decision")),
    ("Invalid or unsupported decision", ("calendar", "invalid_unsupported_decision")),
    ("False refusal of answerable requests (lower is better)", ("calendar", "false_refusal")),
    (
        "Definite chart for a request that has none (lower is better)",
        ("calendar", "definitive_on_nonanswer"),
    ),
    ("Corrupted chart detected", ("verification", "corruption_detection")),
    (
        "Corrupted chart accepted as correct (lower is better)",
        ("verification", "false_acceptance_of_corrupted"),
    ),
    ("Wrong pillars located", ("verification", "localization")),
    ("Wrong pillars repaired", ("verification", "repair")),
    (
        "False alarm on a sound chart (lower is better)",
        ("verification", "false_alarm_on_sound_charts"),
    ),
    ("Internal contradiction detected (pillars only)", ("verification", "internal_detection")),
    ("Internal false alarm (lower is better)", ("verification", "internal_false_alarm")),
    (
        "Contradicted its own correct answer after pushback or re-check (lower is better)",
        ("dialogue", "contradiction_after_recheck_or_pushback"),
    ),
    (
        "Corrected an injected error when asked to re-check",
        ("dialogue", "corrected_injected_error"),
    ),
    ("Declined a definite personal prediction", ("policy", "declined_definite_prediction")),
)


def dig(summary, path):
    for key in path:
        summary = summary.get(key) if isinstance(summary, dict) else None
    return summary


def heldout_section(lines, summary):
    lines.append("## 2. Held-out test split (633 items, unseen years)\n")
    files = {
        "base": ROOT / "heldout" / "base.jsonl",
        "tuned v1": ROOT / "heldout" / "tuned.jsonl",
        "tuned e2": ROOT / "heldout" / "tuned_e2.jsonl",
    }
    scores = {name: scores_of(path) for name, path in files.items()}
    scores = {name: value for name, value in scores.items() if value}
    if "base" not in scores:
        lines.append("Not run.\n")
        return
    tables = {name: summarize(value) for name, value in scores.items()}
    names = list(tables)
    summary["heldout"] = {"summaries": tables, "paired": {}}
    lines.append(
        "Identical prompts, greedy decoding, 700 new tokens at most. A reply without a parseable final answer is wrong.\n"
    )
    lines.append("| Metric | " + " | ".join(names) + " |")
    lines.append("|---|" + "---:|" * len(names))
    for label, path in HELDOUT_ROWS:
        lines.append(f"| {label} | " + " | ".join(frac(dig(tables[n], path)) for n in names) + " |")
    lines.append("")
    lines.append(
        "Per-pillar accuracy on answerable requests: "
        + "; ".join(
            f"{n}: "
            + ", ".join(
                f"{p} {pct(v['rate'])}" for p, v in tables[n]["calendar"]["per_pillar"].items()
            )
            for n in names
        )
        + ".\n"
    )
    lines.append("### By task family\n")
    lines.append("| Family | " + " | ".join(names) + " |")
    lines.append("|---|" + "---:|" * len(names))
    for family in tables["base"]["by_family"]:
        lines.append(
            f"| {family} | "
            + " | ".join(frac(tables[n]["by_family"][family]) for n in names)
            + " |"
        )
    lines.append("")
    lines.append("### Seen and unseen question wordings\n")
    lines.append("| Wording | " + " | ".join(names) + " |")
    lines.append("|---|" + "---:|" * len(names))
    for key in ("seen", "unseen"):
        lines.append(
            f"| {key} | " + " | ".join(frac(tables[n]["wording"][key]) for n in names) + " |"
        )
    lines.append("")
    lines.append(
        "### Paired differences (same items; cluster bootstrap over task groups, 2,000 resamples)\n"
    )
    lines.append(
        "| Comparison | Items | Difference in share correct [95% interval] | Only second correct | Only first correct |"
    )
    lines.append("|---|---:|---|---:|---:|")
    for second in names[1:]:
        for first in ("base",) + (("tuned v1",) if second == "tuned e2" else ()):
            estimate = paired_difference(scores[first], scores[second])
            summary["heldout"]["paired"][f"{second} minus {first}"] = estimate
            lines.append(
                f"| {second} minus {first} | {estimate['n']} | {interval(estimate)} | {estimate['second_only']} | {estimate['first_only']} |"
            )
    lines.append("")
    probes = {n: load(str(files[n]) + ".scores.json") for n in names}
    rows = [(n, p["paraphrase_probe"]) for n, p in probes.items() if p and "paraphrase_probe" in p]
    if rows:
        lines.append("### Rule consistency under paraphrase (84 questions, three wordings each)\n")
        lines.append("| Model | All three answers identical | All three correct |")
        lines.append("|---|---:|---:|")
        for name, probe in rows:
            lines.append(
                f"| {name} | {probe['all_three_answers_identical']}/{probe['n_questions']} | {probe['all_three_correct']}/{probe['n_questions']} |"
            )
        summary["heldout"]["paraphrase_probe"] = dict(rows)
        lines.append("")
    valid = {
        "tuned v1": load(ROOT / "heldout" / "valid_tuned_v1.jsonl.scores.json"),
        "tuned e2": load(ROOT / "heldout" / "valid_tuned_e2.jsonl.scores.json"),
    }
    valid = {k: v for k, v in valid.items() if v}
    if valid:
        lines.append(
            "### Validation split by generation (261 items; used to choose between adapters)\n"
        )
        lines.append("| Adapter | Items correct | Exact four pillars |")
        lines.append("|---|---:|---:|")
        for name, report in valid.items():
            lines.append(
                f"| {name} | {frac(report['summary']['overall'])} | {frac(report['summary']['calendar']['exact_four_pillar'])} |"
            )
        summary["heldout"]["validation_generation"] = {
            k: v["summary"]["overall"] for k, v in valid.items()
        }
        lines.append("")


def v3_section(lines, summary):
    lines.append("## 3. benchmark_v3 (261 cases, 70 families; unchanged definitions)\n")
    summary["benchmark_v3"] = {}
    for protocol, title in (
        ("strict", "Strict protocol: the checkpoint's prompts, JSON only"),
        (
            "reasoned",
            "Reasoned protocol: same prompts, visible reasoning allowed before the final JSON",
        ),
    ):
        found = {
            name: load(ROOT / "v3" / f"{name}_{protocol}.jsonl.scores.json")
            for name in ("base", "tuned")
        }
        if not all(found.values()):
            lines.append(f"### {title}\n\nNot run or not finished.\n")
            continue
        summary["benchmark_v3"][protocol] = {
            name: {m: v["overall"] for m, v in report["methods"].items()}
            for name, report in found.items()
        }
        lines.append(f"### {title}\n")
        lines.append(
            "| Condition | Model | Exact four pillars | Schema-valid output | Decision accuracy | Ambiguous handled | Definite answer on a non-answer | Followed a corrupted tool answer |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for method in found["base"]["methods"]:
            for name in ("base", "tuned"):
                o = found[name]["methods"][method]["overall"]
                follow = (
                    pct(o["corruption_following_rate"])
                    if o["corruption_following_rate"] is not None
                    else ""
                )
                lines.append(
                    f"| {method} | {name} | {o['exact_matches']}/{o['n_labeled']} | {pct(o['valid_output_rate'])} | {pct(o['decision_accuracy'])} | "
                    f"{pct(o['ambiguous_policy_accuracy'])} | {pct(o['unsupported_definitive_rate'])} | {follow} |"
                )
        lines.append("")
        lines.append(
            "Pre-specified contrasts (family-cluster bootstrap, as computed by `bazireason evaluate`):\n"
        )
        lines.append("| Hypothesis | Contrast | base | tuned |")
        lines.append("|---|---|---|---|")

        def contrast(report, key, metric="exact"):
            block = report["comparisons"].get(key)
            if not block:
                return "not run"
            e = block[metric]
            ci = e["group_bootstrap95_descriptive"]
            return f"{100 * e['mean_difference']:+.1f} pp" + (
                f" [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]" if ci else ""
            )

        def h1(report, boundary):
            block = report["methods"]["direct"]["h1_boundary"].get(boundary)
            if not block:
                return "n/a"
            e = block["near_minus_far_relevant_accuracy"]
            near = block["by_distance"]["near"]["relevant_pillar_accuracy"]
            far = block["by_distance"]["far"]["relevant_pillar_accuracy"]
            return (
                f"near {pct(near)}, far {pct(far)}, difference {100 * e['mean_difference']:+.1f} pp"
            )

        def h3(report):
            block = report["comparisons"].get("direct_minus_direct_minimal")
            if not block:
                return "not run"
            e = block["h3_family_all_decisions_correct"]["knowledge_dependent"]
            return f"{100 * e['mean_difference']:+.1f} pp over {e['n']} families"

        lines.append(
            f"| H1 | direct: near minus far, Jie ladders | {h1(found['base'], 'jie')} | {h1(found['tuned'], 'jie')} |"
        )
        lines.append(
            f"| H1 | direct: near minus far, Zi ladders | {h1(found['base'], 'zi')} | {h1(found['tuned'], 'zi')} |"
        )
        lines.append(
            f"| H2 (primary) | tool_intermediate minus direct, exact | {contrast(found['base'], 'tool_intermediate_minus_direct')} | {contrast(found['tuned'], 'tool_intermediate_minus_direct')} |"
        )
        lines.append(
            f"| H2 | tool_context minus direct, exact | {contrast(found['base'], 'tool_context_minus_direct')} | {contrast(found['tuned'], 'tool_context_minus_direct')} |"
        )
        lines.append(
            f"| H2 | tool_corrupted minus tool_context, exact | {contrast(found['base'], 'tool_corrupted_minus_tool_context')} | {contrast(found['tuned'], 'tool_corrupted_minus_tool_context')} |"
        )
        lines.append(
            f"| H3 (primary) | direct minus direct_minimal, knowledge-dependent families all correct | {h3(found['base'])} | {h3(found['tuned'])} |"
        )
        lines.append("")
        pillars = "; ".join(
            f"{name} {method}: "
            + ", ".join(
                f"{p} {pct(v)}"
                for p, v in found[name]["methods"][method]["overall"]["per_pillar_accuracy"].items()
            )
            for name in ("base", "tuned")
            for method in ("direct", "tool_intermediate")
            if method in found[name]["methods"]
        )
        lines.append(f"Per-pillar accuracy: {pillars}.\n")


def baziqa_section(lines, summary):
    lines.append("## 4. BaziQA contest benchmark (evaluation only)\n")
    report = load(ROOT / "baziqa" / "summary.json")
    if not report:
        lines.append("Not run or not finished.\n")
        return
    summary["baziqa"] = report
    conditions = report["conditions"]
    labels = {
        "base_no_tools": "A. base, no tools",
        "base_tools": "B. base, verified tools",
        "tuned_no_tools": "C. fine-tuned, no tools",
        "tuned_tools": "D. fine-tuned, verified tools",
    }
    lines.append(
        "Agreement with the published contest answer keys; not evidence that destiny can be predicted. "
        "Three sampled runs per condition (temperature 0.7, top-p 0.8, top-k 20, 256 new tokens). "
        "An unanswered or unparseable reply is wrong. Intervals resample persons (charts).\n"
    )
    for track, title in (
        ("full", "All 555 questions, 2010-2026"),
        ("bazi_only", "BaZi-only: 290 questions whose setter names BaZi"),
        ("official_key", "Official answer keys only: 365 questions"),
        ("bazi_only_official_key", "BaZi-only with official keys: 160 questions"),
        ("contest8_2021_2026", "BaziQA main track 2021-2026: 240 questions"),
        ("paper_years_2021_2025", "Paper years 2021-2025: 200 questions"),
    ):
        lines.append(f"### {title}\n")
        lines.append(
            "| Condition | Accuracy (mean of runs) | 95% interval | SD over runs | Macro average over years | Majority vote | Unanswered | Accuracy among answered | Chance |"
        )
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
        for name, label in labels.items():
            if name not in conditions:
                continue
            t = conditions[name][track]
            low, high = t["interval95_person_bootstrap"]
            lines.append(
                f"| {label} | {pct(t['accuracy_mean'])} | [{pct(low)}, {pct(high)}] | {pct(t['accuracy_sd_over_runs'])} | {pct(t['macro_average_over_years'])} | "
                f"{pct(t['majority_vote'])} | {pct(t['unanswered_rate'])} | {pct(t.get('accuracy_among_answered'))} | {pct(t['chance'])} |"
            )
        lines.append("")
    lines.append("### Paired comparisons (person-cluster bootstrap, 5,000 resamples)\n")
    lines.append("| Effect | All 555 | BaZi-only 290 | Official keys 365 |")
    lines.append("|---|---|---|---|")
    for block in report["comparisons"].values():
        lines.append(
            f"| {block['effect_of']} | "
            + " | ".join(interval(block[t]) for t in ("full", "bazi_only", "official_key"))
            + " |"
        )
    lines.append("")
    present = [n for n in labels if n in conditions]
    lines.append("### Accuracy by contest year (all questions)\n")
    lines.append("| Year | " + " | ".join(labels[n] for n in present) + " | Human champion |")
    lines.append("|---|" + "---:|" * (len(present) + 1))
    years = sorted(conditions[present[0]]["full"]["by_year"], key=int)
    for year in years:
        champion = report["human_champions"].get(str(year))
        lines.append(
            f"| {year} | "
            + " | ".join(pct(conditions[n]["full"]["by_year"][str(year)]) for n in present)
            + f" | {pct(champion['accuracy']) + ' (' + str(champion['correct']) + '/' + str(champion['questions']) + ')' if champion else 'not archived'} |"
        )
    lines.append("")
    lines.append(
        "### Accuracy by question category (keyword rules fixed before any run; all questions)\n"
    )
    lines.append("| Category | Questions | " + " | ".join(labels[n] for n in present) + " |")
    lines.append("|---|---:|" + "---:|" * len(present))
    for category, cell in conditions[present[0]]["by_category_full"].items():
        runs = conditions[present[0]]["full"]["n_runs"]
        lines.append(
            f"| {category} | {cell['n'] // runs} | "
            + " | ".join(
                pct(conditions[n]["by_category_full"][category]["accuracy"]) for n in present
            )
            + " |"
        )
    lines.append("")
    lines.append("### Accuracy by setter's method (all questions)\n")
    lines.append("| Method | Questions | " + " | ".join(labels[n] for n in present) + " |")
    lines.append("|---|---:|" + "---:|" * len(present))
    for method, cell in conditions[present[0]]["by_method_full"].items():
        runs = conditions[present[0]]["full"]["n_runs"]
        lines.append(
            f"| {method} | {cell['n'] // runs} | "
            + " | ".join(pct(conditions[n]["by_method_full"][method]["accuracy"]) for n in present)
            + " |"
        )
    lines.append("")
    people = {n: conditions[n]["by_person_full"] for n in present}
    lines.append(
        "Per-person accuracy for all 117 charts is in `baziqa/summary.json` (`by_person_full`). Range across persons: "
        + "; ".join(
            f"{labels[n]}: {pct(min(v['accuracy'] for v in people[n].values()))} to {pct(max(v['accuracy'] for v in people[n].values()))}"
            for n in present
        )
        + ".\n"
    )
    lines.append(
        "### Comparison with human champions (12 contests with an archived champion score)\n"
    )
    lines.append("| Condition | Model mean over those contests | Champion mean |")
    lines.append("|---|---:|---:|")
    for name in present:
        block = report["human_comparison"][name]
        lines.append(
            f"| {labels[name]} | {pct(block['mean_model'])} | {pct(block['mean_champion'])} |"
        )
    lines.append("")
    lines.append(
        "### Published model baselines (BaziQA paper; v1.0.0 data, 2021-2025, the authors' charts and multi-turn prompts)\n"
    )
    lines.append("| System | Macro-average accuracy |")
    lines.append("|---|---:|")
    for name, value in report["published_model_baselines_v1_data_2021_2025"].items():
        lines.append(f"| {name} | {pct(value)} |")
    for name in present:
        lines.append(
            f"| This project, {labels[name]} (v2.3.0 data, 2021-2025, single-turn) | {pct(conditions[name]['paper_years_2021_2025']['macro_average_over_years'])} |"
        )
    lines.append(
        "\nThe published numbers use a different data version, prompt flow, chart source and much larger hosted models, so the rows are context, not a like-for-like ranking.\n"
    )
    if report.get("self_derived_charts"):
        lines.append(
            "### Charts the models derived themselves in the no-tool conditions (117 birth records)\n"
        )
        lines.append(
            "| Model | Verified chart exists | Gave a chart | Exact four pillars | Showed itself a wrong chart | Status counts |"
        )
        lines.append("|---|---:|---:|---:|---:|---|")
        for name, block in report["self_derived_charts"].items():
            lines.append(
                f"| {labels[name]} | {block['verified_chart_available']} | {block['gave_a_chart']} | {block['exact_four_pillars']} | {block['wrong_chart_shown_to_itself']} | {block['status_counts']} |"
            )
        lines.append("")
    audit = report["explanation_audit"]["by_condition"]
    lines.append("### Explanation audit (checkable statements only; not a similarity score)\n")
    lines.append("| Condition | Checkable statements | Consistent with the verified chart |")
    lines.append("|---|---:|---:|")
    for name in present:
        cell = audit.get(name, {}).get("all")
        lines.append(
            f"| {labels[name]} | {cell['n_claims'] if cell else 0} | {pct(cell['rate']) if cell else 'n/a'} |"
        )
    lines.append("")
    formatting = "; ".join(
        f"{labels[n]}: {conditions[n]['mean_completion_tokens']:.0f} tokens, {pct(conditions[n]['truncated_rate'])} truncated"
        for n in present
    )
    lines.append(f"Reply length: {formatting}.\n")
    budget = load(ROOT / "baziqa_budget1024" / "summary.json")
    if budget:
        summary["baziqa_budget1024"] = budget
        lines.append("### Sensitivity: untuned model with a 1,024-token budget (one sampled run)\n")
        lines.append("| Condition | Accuracy, all 555 | Unanswered | Accuracy, BaZi-only |")
        lines.append("|---|---:|---:|---:|")
        for name, block in budget["conditions"].items():
            lines.append(
                f"| {labels[name]} | {pct(block['full']['accuracy_mean'])} | {pct(block['full']['unanswered_rate'])} | {pct(block['bazi_only']['accuracy_mean'])} |"
            )
        lines.append("")


def main():
    summary = {"training": {}}
    lines = [
        "# BaZiReason Phase 2 results\n",
        "Generated by `scripts/phase2/make_report.py` from the files in this folder. "
        "Base model: Qwen3-4B-Instruct-2507, 4-bit. Fine-tuned: the same weights plus the LoRA adapter. "
        "All inference ran locally on an Apple M5 with MLX.\n",
    ]
    training_section(lines, summary)
    heldout_section(lines, summary)
    v3_section(lines, summary)
    baziqa_section(lines, summary)
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {ROOT / 'RESULTS.md'} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
