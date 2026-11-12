"""Recompute the score file of a saved Phase 3 evaluation from its raw outputs (no inference).

python scripts/phase3/rescore.py --data data/phase3/train_v1/valid.jsonl --output <file.jsonl>
"""

import argparse
import json
from pathlib import Path

from bazireason.p3_eval import score3, summarize3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = {
        row["id"]: row
        for row in map(json.loads, Path(args.data).read_text(encoding="utf-8").splitlines())
    }
    records = [
        json.loads(line) for line in Path(args.output).read_text(encoding="utf-8").splitlines()
    ]
    scores = [score3(rows[r["id"]], r["raw_output"]) for r in records]
    report = {"model": records[0]["model"], "n_rows": len(records), "summary": summarize3(scores)}
    Path(args.output + ".scores.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    summary = report["summary"]
    print(
        json.dumps(
            {
                "model": report["model"],
                "overall": summary["overall"],
                "reasoning": summary["reasoning_tasks"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
