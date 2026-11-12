"""Frozen Phase 4 adapter selection (docs/phase4_preregistration.md, section 4).

    python scripts/phase4/select.py

Reads only W0 runs on data/phase4/valid (results/phase4/valid/<label>_<part>.jsonl) and
writes results/phase4/selection.json. The final test split is never read.
"""

import json
from pathlib import Path

from bazireason.p2_eval import paired_difference

ROOT = Path("results/phase4")
ADAPTERS = {
    "p2": "artifacts/adapters/bazireason-qwen3-4b-lora-v2",
    "p3r2": "artifacts/adapters/bazireason-qwen3-4b-lora-p3-r2",
    "p3r3": "artifacts/adapters/bazireason-qwen3-4b-lora-p3-r3",
    "p4a1": "artifacts/adapters/bazireason-qwen3-4b-lora-p4-a1",
    "p4a2": "artifacts/adapters/bazireason-qwen3-4b-lora-p4-a2",
}
PARTS = ("p3", "p2", "indep")
MARGIN = -0.05
INCUMBENT = "p3r3"


def scores(label, part):
    path = ROOT / "valid" / f"{label}_{part}.jsonl"
    if not path.exists():
        return None
    return [json.loads(line)["score"] for line in path.read_text().splitlines()]


def main():
    reference = scores("p2", "p2")
    if reference is None:
        raise SystemExit("The P2 validation run is missing")
    rows = []
    for label, adapter in ADAPTERS.items():
        parts = {part: scores(label, part) for part in PARTS}
        if any(v is None for v in parts.values()):
            continue
        calendar = paired_difference(reference, parts["p2"], where=lambda s: s["task"] == "chart")
        low = calendar["interval95"][0] if calendar["interval95"] else 0.0
        eligible = label == "p2" or low >= MARGIN
        rows.append(
            {
                "label": label,
                "adapter": adapter,
                "correct": {p: sum(bool(s["correct"]) for s in v) for p, v in parts.items()},
                "rows": {p: len(v) for p, v in parts.items()},
                "total_correct": sum(sum(bool(s["correct"]) for s in v) for v in parts.values()),
                "calendar_vs_p2": calendar,
                "eligible": eligible,
            }
        )
    eligible = [r for r in rows if r["eligible"]]
    best = max((r["total_correct"] for r in eligible), default=None)
    winners = [r["label"] for r in eligible if r["total_correct"] == best]
    selected = INCUMBENT if (not winners or INCUMBENT in winners) else winners[0]
    record = {
        "rule": (
            "eligible if calendar rows (valid/p2, task == chart) minus P2 have a 95% lower bound "
            ">= -5.0 points; among eligible, most correct over valid p3 + p2 + indep; ties and "
            "no eligible adapter keep p3r3. Written before any candidate was scored."
        ),
        "candidates": rows,
        "selected": selected,
        "selected_adapter": ADAPTERS[selected],
        "no_candidate_eligible": not eligible,
    }
    (ROOT / "selection.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({r["label"]: (r["total_correct"], r["eligible"]) for r in rows}), selected)


if __name__ == "__main__":
    main()
