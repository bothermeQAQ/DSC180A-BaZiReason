"""Score the Phase 3 BaziQA comparison: base, Phase 2 and Phase 3, with and without tools.

    python scripts/phase3/score_baziqa3.py

Every model was run with the same prompts, the same sampling settings and a
1,024-token budget, so the Phase 2 token-limit confound does not apply. BaziQA
was already evaluated in Phase 2: it is an existing benchmark here, not an
untouched test set, and no prompt or checkpoint was tuned on its answers.
Agreement with contest answer keys is not evidence that destiny can be
predicted. Unanswered or unparseable replies count as wrong.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from bazireason import baziqa as B

MODELS = (("base", "base"), ("p2", "tuned"), ("p3", "tuned"))
TOOLS = ("no_tools", "tools")
REPORT_TRACKS = ("full", "bazi_only", "official_key", "bazi_only_official_key", "paper_years_2021_2025")


def read(path):
    runs, records = defaultdict(dict), []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        runs[record["run"]][record["question_id"]] = record["answer"]
        records.append(record)
    return dict(runs), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="results/phase3/baziqa1024")
    parser.add_argument("--root", default="external/BaziQA")
    parser.add_argument("--output")
    args = parser.parse_args()
    folder = Path(args.input_dir)
    persons = B.load_contest(args.root)
    tracks = B.tracks(persons)
    charts = {}
    for person in persons:
        chart, _ = B.tool_chart(person)
        if chart["status"] == "ok":
            charts[person["person_id"]] = chart["pillars"]
    conditions, records = {}, {}
    for model, prefix in MODELS:
        for tools in TOOLS:
            path = folder / model / f"{prefix}_{tools}.jsonl"
            if path.exists():
                conditions[f"{model}_{tools}"], records[f"{model}_{tools}"] = read(path)
    expected = {q["question_id"] for p in persons for q in p["questions"]}
    summary = {
        "benchmark": "BaziQA",
        "benchmark_commit": B.PINNED_COMMIT,
        "note": "Agreement with published contest answer keys under a 1,024-token budget. Not "
        "evidence that destiny can be predicted. Unanswered replies are wrong. BaziQA was "
        "already evaluated in Phase 2 and is not an untouched test set.",
        "tracks": {k: len(tracks[k]) for k in REPORT_TRACKS},
        "completeness": {
            name: {str(run): len(expected & set(answers)) for run, answers in sorted(runs.items())}
            for name, runs in conditions.items()
        },
        "conditions": {},
        "comparisons": {},
    }
    for name, runs in conditions.items():
        block = {}
        for track in REPORT_TRACKS:
            ids = tracks[track]
            table = B.accuracy_table(persons, runs, ids)
            table["interval95_person_bootstrap"] = B.accuracy_interval(persons, runs, ids)
            block[track] = table
        block["by_category_full"] = B.breakdown(persons, runs, tracks["full"], "category")
        block["by_category_bazi_only"] = B.breakdown(persons, runs, tracks["bazi_only"], "category")
        block["by_person_full"] = B.breakdown(persons, runs, tracks["full"], "person")
        mine = records[name]
        block["parseable_rate"] = sum(r["parseable"] for r in mine) / len(mine)
        block["truncated_rate"] = sum(r["truncated"] for r in mine) / len(mine)
        block["mean_completion_tokens"] = sum(r["completion_tokens"] for r in mine) / len(mine)
        letters = defaultdict(int)
        for r in mine:
            letters[r["answer"] or "none"] += 1
        block["answer_letter_share"] = {k: v / len(mine) for k, v in sorted(letters.items())}
        audit = B.audit_summary([dict(r, condition=name) for r in mine], charts).get(name, {})
        block["checkable_statements"] = audit.get("all")
        summary["conditions"][name] = block
    pairs = []
    for tools in TOOLS:
        pairs += [
            (f"p3_{tools}", f"p2_{tools}", f"Phase 3 training beyond Phase 2 ({tools})"),
            (f"p3_{tools}", f"base_{tools}", f"Phase 2 and 3 training together ({tools})"),
            (f"p2_{tools}", f"base_{tools}", f"Phase 2 training ({tools})"),
        ]
    for model, _ in MODELS:
        pairs.append((f"{model}_tools", f"{model}_no_tools", f"verified chart tool, {model} model"))
    for second, first, label in pairs:
        if first in conditions and second in conditions:
            summary["comparisons"][f"{second}_minus_{first}"] = {
                "effect_of": label,
                **{
                    track: B.paired_by_person(persons, conditions[first], conditions[second], tracks[track])
                    for track in REPORT_TRACKS
                },
            }
    human = {}
    for name, runs in conditions.items():
        rows = {}
        for year, (k, n) in B.HUMAN_CHAMPIONS.items():
            ids = {q["question_id"] for p in persons if p["year"] == year for q in p["questions"]}
            rows[str(year)] = {
                "model_accuracy": B.accuracy_table(persons, runs, ids)["accuracy_mean"],
                "champion_accuracy": k / n,
            }
        human[name] = {
            "mean_model": sum(r["model_accuracy"] for r in rows.values()) / len(rows),
            "mean_champion": sum(r["champion_accuracy"] for r in rows.values()) / len(rows),
        }
    summary["human_comparison"] = human
    summary["published_model_baselines_v1_data_2021_2025"] = B.PUBLISHED_BASELINES
    own = {}
    for model, prefix in MODELS:
        path = folder / model / f"{prefix}_no_tools.self_charts.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        answerable = [r for r in rows if r["verified_status"] == "ok"]
        gave = [r for r in answerable if r["block"]]
        exact = [
            r
            for r in gave
            if isinstance(r["answer"], dict) and r["answer"].get("pillars") == r["verified_pillars"]
        ]
        own[model] = {
            "persons": len(rows),
            "verified_chart_available": len(answerable),
            "gave_a_chart": len(gave),
            "exact_four_pillars": len(exact),
            "wrong_chart_shown_to_itself": len(gave) - len(exact),
        }
    summary["self_derived_charts"] = own
    output = Path(args.output or folder / "summary.json")
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, block in summary["conditions"].items():
        full, bazi = block["full"], block["bazi_only"]
        low, high = full["interval95_person_bootstrap"]
        print(
            f"{name:14s} runs {full['n_runs']} full {full['accuracy_mean']:.3f} [{low:.3f}, {high:.3f}] "
            f"bazi-only {bazi['accuracy_mean']:.3f} official {block['official_key']['accuracy_mean']:.3f} "
            f"unanswered {full['unanswered_rate']:.3f} tokens {block['mean_completion_tokens']:.0f}"
        )
    for name, block in summary["comparisons"].items():
        full = block["full"]
        print(f"{name:34s} {full['difference']:+.3f} [{full['interval95'][0]:+.3f}, {full['interval95'][1]:+.3f}]")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
