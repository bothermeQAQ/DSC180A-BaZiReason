"""Choose the final Phase 3 adapter by generation accuracy on the validation split.

Pre-specified rule: the round with more correct answers on
``data/phase3/train_v1/valid.jsonl`` (validation years, greedy decoding) is
selected; a tie keeps the earlier round. The test split and the benchmarks are
not consulted. Prints the selected adapter folder.
"""

import json
from pathlib import Path

ROUNDS = (
    ("p3-r1", "artifacts/adapters/bazireason-qwen3-4b-lora-p3-r1", "valid_p3r1"),
    ("p3-r2", "artifacts/adapters/bazireason-qwen3-4b-lora-p3-r2", "valid_p3r2"),
)


def main():
    table = []
    for name, adapter, stem in ROUNDS:
        path = Path(f"results/phase3/heldout/{stem}.jsonl.scores.json")
        if path.exists():
            summary = json.loads(path.read_text())["summary"]
            table.append(
                {
                    "round": name,
                    "adapter": adapter,
                    "validation_correct": summary["overall"]["hits"],
                    "validation_rows": summary["overall"]["n"],
                    "reasoning_correct": summary["reasoning_tasks"]["hits"],
                    "reasoning_rows": summary["reasoning_tasks"]["n"],
                }
            )
    best = max(table, key=lambda row: row["validation_correct"])
    best = next(row for row in table if row["validation_correct"] == best["validation_correct"])
    Path("results/phase3/selection.json").write_text(
        json.dumps(
            {
                "rule": "most correct answers on data/phase3/train_v1/valid.jsonl; tie keeps "
                "the earlier round; test split and benchmarks not consulted",
                "rounds": table,
                "selected": best["round"],
                "selected_adapter": best["adapter"],
            },
            indent=2,
        )
        + "\n"
    )
    print(best["adapter"])


if __name__ == "__main__":
    main()
