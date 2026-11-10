"""Score the BaziQA conditions and write one summary with every requested breakdown.

    python scripts/phase2/score_baziqa.py --input-dir results/phase2/baziqa

Reported for each condition and track: mean accuracy over runs, the BaziQA
headline macro average over contest years, majority vote, the unanswered rate,
chance level, a person-cluster bootstrap interval, and accuracy by year,
category, person and setter method. Conditions are compared pairwise with a
person-cluster bootstrap, because questions about one chart are correlated.
Unanswered or unparseable replies count as wrong everywhere.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from bazireason import baziqa as B


def read_condition(path):
    runs, records = defaultdict(dict), []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        runs[record["run"]][record["question_id"]] = record["answer"]
        records.append(record)
    return dict(runs), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="results/phase2/baziqa")
    parser.add_argument("--root", default="external/BaziQA")
    parser.add_argument("--output")
    args = parser.parse_args()
    folder = Path(args.input_dir)
    persons = B.load_contest(args.root)
    tracks = B.tracks(persons)
    conditions, records = {}, []
    for name in B.CONDITIONS:
        path = folder / f"{name}.jsonl"
        if path.exists():
            conditions[name], rows = read_condition(path)
            records.extend(rows)
    expected = {q["question_id"] for p in persons for q in p["questions"]}
    summary = {
        "benchmark": "BaziQA",
        "benchmark_commit": B.PINNED_COMMIT,
        "note": "Agreement with published contest answer keys. Not evidence that destiny can "
        "be predicted. Unanswered replies are wrong.",
        "tracks": {k: len(v) for k, v in tracks.items()},
        "completeness": {
            name: {str(run): len(expected & set(answers)) for run, answers in sorted(runs.items())}
            for name, runs in conditions.items()
        },
        "conditions": {},
        "comparisons": {},
        "human_champions": {
            str(year): {"correct": k, "questions": n, "accuracy": k / n}
            for year, (k, n) in B.HUMAN_CHAMPIONS.items()
        },
        "published_model_baselines_v1_data_2021_2025": B.PUBLISHED_BASELINES,
    }
    for name, runs in conditions.items():
        block = {}
        for track, ids in tracks.items():
            table = B.accuracy_table(persons, runs, ids)
            table["interval95_person_bootstrap"] = B.accuracy_interval(persons, runs, ids)
            table["above_chance"] = table["accuracy_mean"] - table["chance"]
            block[track] = table
        block["by_category_full"] = B.breakdown(persons, runs, tracks["full"], "category")
        block["by_category_bazi_only"] = B.breakdown(persons, runs, tracks["bazi_only"], "category")
        block["by_method_full"] = B.breakdown(persons, runs, tracks["full"], "method")
        block["by_answer_source_full"] = B.breakdown(persons, runs, tracks["full"], "answer_source")
        block["by_person_full"] = B.breakdown(persons, runs, tracks["full"], "person")
        mine = [r for r in records if r["condition"] == name]
        block["parseable_rate"] = sum(r["parseable"] for r in mine) / len(mine)
        block["truncated_rate"] = sum(r["truncated"] for r in mine) / len(mine)
        block["mean_completion_tokens"] = sum(r["completion_tokens"] for r in mine) / len(mine)
        letters = defaultdict(int)
        for r in mine:
            letters[r["answer"] or "none"] += 1
        block["answer_letter_share"] = {k: v / len(mine) for k, v in sorted(letters.items())}
        summary["conditions"][name] = block
    pairs = (
        ("tuned_no_tools", "base_no_tools", "fine-tuning, no tools"),
        ("tuned_tools", "base_tools", "fine-tuning, with tools"),
        ("base_tools", "base_no_tools", "tools, base model"),
        ("tuned_tools", "tuned_no_tools", "tools, fine-tuned model"),
        ("tuned_tools", "base_no_tools", "fine-tuning and tools together"),
    )
    for second, first, label in pairs:
        if first in conditions and second in conditions:
            summary["comparisons"][f"{second}_minus_{first}"] = {
                "effect_of": label,
                **{
                    track: B.paired_by_person(persons, conditions[first], conditions[second], ids)
                    for track, ids in tracks.items()
                },
            }
    # Human comparison on the years with an archived champion score (same question sets).
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
            "by_year": rows,
            "mean_model": sum(r["model_accuracy"] for r in rows.values()) / len(rows),
            "mean_champion": sum(r["champion_accuracy"] for r in rows.values()) / len(rows),
        }
    summary["human_comparison"] = human
    # Auditable check of the saved explanations against the verified charts.
    charts = {}
    for person in persons:
        chart, _assumed = B.tool_chart(person)
        if chart["status"] == "ok":
            charts[person["person_id"]] = chart["pillars"]
    summary["explanation_audit"] = {
        "method": "share of fixed-form, checkable statements (a named pillar, the day master, "
        "the pillar of a calendar year, the Ten God of a named stem) that match the verified "
        "chart; not a similarity score and not a judgement of the interpretation",
        "by_condition": B.audit_summary(records, charts),
    }
    # Chart quality of the no-tool conditions' own charts.
    own = {}
    for name in ("base_no_tools", "tuned_no_tools"):
        path = folder / f"{name}.self_charts.jsonl"
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
        statuses = defaultdict(int)
        for r in rows:
            statuses[
                (r["answer"] or {}).get("status") if isinstance(r["answer"], dict) else "unparsed"
            ] += 1
        own[name] = {
            "persons": len(rows),
            "verified_chart_available": len(answerable),
            "gave_a_chart": len(gave),
            "exact_four_pillars": len(exact),
            "exact_rate_over_verifiable": len(exact) / len(answerable),
            "wrong_chart_shown_to_itself": len(gave) - len(exact),
            "status_counts": dict(statuses),
        }
    summary["self_derived_charts"] = own
    output = Path(args.output or folder / "summary.json")
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, block in summary["conditions"].items():
        full, bazi = block["full"], block["bazi_only"]
        print(
            f"{name:16s} full {full['accuracy_mean']:.3f} (macro {full['macro_average_over_years']:.3f}, "
            f"vote {full['majority_vote']:.3f}, chance {full['chance']:.3f}) | bazi-only "
            f"{bazi['accuracy_mean']:.3f} | unanswered {full['unanswered_rate']:.3f}"
        )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
