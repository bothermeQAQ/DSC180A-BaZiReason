"""Run a declared representative pilot and score its retained manifest.

Mock is the default. API access must be selected explicitly and configured in
BAZIREASON_* variables; this wrapper does not discover or spend credentials.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

DESIGNS = {
    "h2": [
        "direct",
        "tool_facts",
        "tool_intermediate",
        "tool_context",
        "tool_corrupted",
        "copy_context",
    ],
    "h3": ["direct", "direct_minimal", "tool_context", "tool_context_minimal", "heuristic_surface"],
    "combined": [
        "direct",
        "direct_minimal",
        "tool_facts",
        "tool_intermediate",
        "tool_intermediate_minimal",
        "tool_context",
        "tool_context_minimal",
        "tool_corrupted",
        "copy_context",
        "heuristic_surface",
    ],
    # Zero model calls: answer-copy ceiling and three trivial floors.
    "baselines": ["copy_context", "heuristic_surface", "always_abstain", "always_answer"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mock", "api"], default="mock")
    parser.add_argument("--design", choices=DESIGNS, default="h3")
    parser.add_argument("--data", default="data/synthetic/benchmark_v3.jsonl")
    parser.add_argument("--ids", default="data/synthetic/benchmark_v3_pilot_ids.json")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-calls", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ids = json.loads(Path(args.ids).read_text())
    if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids):
        parser.error("ID file must be a nonempty string array")
    command = [sys.executable, "-c", "from bazireason.cli import main; main()"]
    subprocess.run(
        command
        + [
            "run",
            "--data",
            args.data,
            "--case-ids",
            ",".join(ids),
            "--methods",
            *DESIGNS[args.design],
            "--backend",
            args.backend,
            "--trials",
            str(args.trials),
            "--max-tokens",
            str(args.max_tokens),
            "--max-calls",
            str(args.max_calls),
            "--output",
            args.output,
        ],
        check=True,
    )
    subprocess.run(
        command
        + [
            "evaluate",
            "--data",
            args.data,
            "--predictions",
            args.output,
            "--output",
            args.output + ".scores.json",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
