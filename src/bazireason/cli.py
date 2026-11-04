"""CLI entry point; defaults are offline, and live runs require explicit bounds."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from bazireason.baselines import run_case
from bazireason.dataset_generator import generate_dataset, read_jsonl, write_jsonl
from bazireason.evaluator import evaluate
from bazireason.llm_clients import OpenAICompatibleClient, SyntheticMockClient


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--seed", type=int, default=1802026)
    gen.add_argument("--output", default="data/synthetic/checkpoint.jsonl")
    diag = sub.add_parser("diagnostics")
    diag.add_argument("--data", default="data/synthetic/checkpoint.jsonl")
    diag.add_argument("--output", default="results/calendar_diagnostics.json")
    run = sub.add_parser("run")
    run.add_argument("--data", default="data/synthetic/checkpoint.jsonl")
    run.add_argument("--output", required=True)
    run.add_argument("--backend", choices=["mock", "api"], default="mock")
    run.add_argument("--model", default=os.environ.get("BAZIREASON_MODEL"))
    run.add_argument("--base-url", default=os.environ.get("BAZIREASON_BASE_URL"))
    run.add_argument("--limit", type=int, default=4)
    run.add_argument("--trials", type=int, default=1)
    run.add_argument("--max-calls", type=int, default=8)
    run.add_argument("--temperature", type=float, default=0)
    run.add_argument("--max-tokens", type=int, default=512)
    score = sub.add_parser("evaluate")
    score.add_argument("--data", default="data/synthetic/checkpoint.jsonl")
    score.add_argument("--predictions", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--limit", type=int, default=None)
    score.add_argument("--trials", type=int, default=1)
    args = parser.parse_args()
    if args.command == "generate":
        cases = generate_dataset(args.seed)
        write_jsonl(args.output, cases)
        print(f"Wrote {len(cases)} cases to {args.output}")
        return
    cases = read_jsonl(args.data)
    if args.command == "diagnostics":
        from bazireason.analysis import diagnostics

        output = diagnostics(cases)
    elif args.command == "evaluate":
        if args.limit is not None:
            cases = cases[: args.limit]
        output = evaluate(cases, read_jsonl(args.predictions), trials=args.trials)
    else:
        if min(args.limit, args.trials, args.max_calls, args.max_tokens) < 1:
            parser.error("Run bounds must be positive")
        cases = cases[: args.limit]
        calls = 2 * len(cases) * args.trials
        if calls > args.max_calls:
            parser.error(f"{calls} requested calls exceeds --max-calls")
        if Path(args.output).exists():
            parser.error("Prediction file exists; choose a new output path")
        if args.backend == "api":
            if not args.model or not args.base_url:
                parser.error("API requires --model and --base-url")
            client = OpenAICompatibleClient(args.model, args.base_url)
        else:
            client = SyntheticMockClient()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Flush each observation, preserving completed calls if interrupted.
        with output_path.open("x", encoding="utf-8") as stream:
            for trial in range(args.trials):
                for i, case in enumerate(cases):
                    methods = (
                        ("direct", "tool_context")
                        if (i + trial) % 2 == 0
                        else ("tool_context", "direct")
                    )
                    for method in methods:
                        prediction = run_case(
                            case, client, method, trial, args.temperature, args.max_tokens
                        )
                        stream.write(
                            json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n"
                        )
                        stream.flush()
        manifest = {
            "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            "case_ids": [c["case_id"] for c in cases],
            "trials": args.trials,
            "n_calls": calls,
            "synthetic": args.backend == "mock",
            "backend": args.backend,
            "model": client.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "attempts_per_call": 1,
            "method_order": "alternating_by_case_and_trial",
            "cost": "not measured; call and output-token caps are not dollar caps",
        }
        Path(args.output + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote {calls} observations ({args.backend})")
        return
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {path}")
