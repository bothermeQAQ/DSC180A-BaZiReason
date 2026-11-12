"""Choose between Phase 3 round 2 and the corrective round 3 (rule fixed before round 3 was scored).

Pre-specified rule: the adapter with more correct generated answers on the two
validation splits together wins, so neither the new reasoning skills nor the
Phase 2 calendar skills can be traded away unseen:

* ``data/phase3/train_v1/valid.jsonl`` (Phase 3 tasks, validation years), and
* ``data/phase2/train_v2/valid.jsonl`` (Phase 2 tasks, validation years).

A tie keeps round 2. No test split and no benchmark is consulted. Writes
``results/phase3/selection_final.json`` and prints the selected adapter folder.
"""

import json
from pathlib import Path

H = Path("results/phase3/heldout")
CANDIDATES = (
    ("p3-r2", "artifacts/adapters/bazireason-qwen3-4b-lora-p3-r2", "valid_p3r2", "p2valid_p3r2"),
    ("p3-r3", "artifacts/adapters/bazireason-qwen3-4b-lora-p3-r3", "valid_p3r3", "p2valid_p3r3"),
)


def hits(stem):
    summary = json.loads((H / f"{stem}.jsonl.scores.json").read_text())["summary"]
    overall = summary["overall"]
    return overall.get("hits", overall.get("k")), overall["n"]


def main():
    table = []
    for name, adapter, phase3, phase2 in CANDIDATES:
        (a, n_a), (b, n_b) = hits(phase3), hits(phase2)
        table.append(
            {
                "round": name,
                "adapter": adapter,
                "phase3_valid": [a, n_a],
                "phase2_valid": [b, n_b],
                "total": a + b,
            }
        )
    best = table[0] if table[0]["total"] >= table[1]["total"] else table[1]
    (Path("results/phase3") / "selection_final.json").write_text(
        json.dumps(
            {
                "rule": "most correct answers on data/phase3/train_v1/valid.jsonl plus "
                "data/phase2/train_v2/valid.jsonl; a tie keeps round 2; no test split or "
                "benchmark consulted; fixed before round 3 was scored",
                "candidates": table,
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
