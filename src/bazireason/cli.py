"""CLI entry point; defaults are offline, and live runs require explicit bounds."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from bazireason.baselines import run_case
from bazireason.dataset_generator import generate_dataset, read_jsonl, write_jsonl
from bazireason.evaluator import evaluate
from bazireason.llm_clients import OpenAICompatibleClient, SyntheticMockClient
from bazireason.prompts import METHODS, PROMPT_VERSION, condition


def select_cases(cases, limit, case_ids, parser):
    if case_ids is not None:
        ids = [x.strip() for x in case_ids.split(",")]
        if any(not x for x in ids) or len(set(ids)) != len(ids):
            parser.error("--case-ids must contain nonempty unique IDs")
        indexed = {c["case_id"]: c for c in cases}
        if any(i not in indexed for i in ids):
            parser.error("--case-ids includes unknown IDs")
        return [indexed[i] for i in ids]
    if limit is not None and limit < 1:
        parser.error("--limit must be positive")
    return cases[:limit]


def evaluation_design(args, cases, parser):
    """Bind interrupted/complete artifact scoring to its declared pre-run manifest."""
    manifest_path = Path(args.predictions + ".manifest.json")
    if not manifest_path.exists():
        return (
            select_cases(cases, args.limit, args.case_ids, parser),
            args.methods or ["direct", "tool_context"],
            1 if args.trials is None else args.trials,
            None,
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("dataset_sha256") != hashlib.sha256(Path(args.data).read_bytes()).hexdigest():
        parser.error("Dataset hash differs from planned manifest")
    methods = manifest.get("methods", ["direct", "tool_context"])
    trials = manifest.get("trials")
    ids = manifest.get("case_ids")
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(i, str) for i in ids)
        or len(set(ids)) != len(ids)
        or type(trials) is not int
        or trials < 1
    ):
        parser.error("Invalid experiment manifest design")
    selected = select_cases(cases, None, ",".join(ids), parser)
    if args.case_ids is not None or args.limit is not None:
        requested = select_cases(cases, args.limit, args.case_ids, parser)
        if [c["case_id"] for c in requested] != ids:
            parser.error("Evaluation selection differs from planned manifest")
    if args.methods is not None and args.methods != methods:
        parser.error("Evaluation methods differ from planned manifest")
    if args.trials is not None and args.trials != trials:
        parser.error("Evaluation trials differ from planned manifest")
    return selected, methods, trials, manifest


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--seed", type=int, default=1802026)
    gen.add_argument("--output")
    gen.add_argument(
        "--dataset-version", choices=["checkpoint_v1", "audit_v2"], default="checkpoint_v1"
    )
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
    run.add_argument("--case-ids", help="Comma-separated exact IDs; overrides --limit")
    run.add_argument("--methods", nargs="+", choices=METHODS, default=["direct", "tool_context"])
    run.add_argument("--trials", type=int, default=1)
    run.add_argument("--max-calls", type=int, default=8)
    run.add_argument("--temperature", type=float, default=0)
    run.add_argument("--max-tokens", type=int, default=512)
    score = sub.add_parser("evaluate")
    score.add_argument("--data", default="data/synthetic/checkpoint.jsonl")
    score.add_argument("--predictions", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--limit", type=int, default=None)
    score.add_argument("--case-ids", help="Same declared IDs as inference; overrides --limit")
    score.add_argument("--methods", nargs="+", choices=METHODS)
    score.add_argument("--trials", type=int)
    score.add_argument("--bootstrap-samples", type=int, default=1000)
    score.add_argument("--bootstrap-seed", type=int, default=1802026)
    args = parser.parse_args()
    if args.command == "generate":
        if args.dataset_version == "audit_v2":
            from bazireason.dataset_generator import generate_audit_dataset

            cases = generate_audit_dataset(args.seed)
        else:
            cases = generate_dataset(args.seed)
        args.output = args.output or (
            "data/synthetic/audit_v2.jsonl"
            if args.dataset_version == "audit_v2"
            else "data/synthetic/checkpoint.jsonl"
        )
        write_jsonl(args.output, cases)
        print(f"Wrote {len(cases)} cases to {args.output}")
        return
    cases = read_jsonl(args.data)
    if args.command == "diagnostics":
        from bazireason.analysis import diagnostics

        output = diagnostics(cases)
    elif args.command == "evaluate":
        cases, methods, trials, manifest = evaluation_design(args, cases, parser)
        output = evaluate(
            cases,
            read_jsonl(args.predictions),
            methods=methods,
            trials=trials,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        output["evaluation_design_source"] = (
            "planned_manifest" if manifest else "explicit_or_legacy"
        )
        output["declared_synthetic"] = manifest.get("synthetic") if manifest else None
    else:
        if min(args.limit, args.trials, args.max_tokens) < 1 or args.max_calls < 0:
            parser.error("Run bounds must be positive")
        if not math.isfinite(args.temperature) or args.temperature < 0:
            parser.error("--temperature must be finite and nonnegative")
        if len(set(args.methods)) != len(args.methods):
            parser.error("--methods must be unique")
        cases = select_cases(cases, args.limit, args.case_ids, parser)
        if not cases:
            parser.error("Dataset selection must be nonempty")
        model_methods = [m for m in args.methods if condition(m)[0] != "copy_context"]
        calls = len(model_methods) * len(cases) * args.trials
        observations = len(args.methods) * len(cases) * args.trials
        if calls > args.max_calls:
            parser.error(f"{calls} requested calls exceeds --max-calls")
        manifest_path = Path(args.output + ".manifest.json")
        if Path(args.output).exists() or manifest_path.exists():
            parser.error("Prediction file exists; choose a new output path")
        if args.backend == "api" and model_methods:
            if not args.model or not args.base_url:
                parser.error("API requires --model and --base-url")
            client = OpenAICompatibleClient(args.model, args.base_url)
        else:
            client = SyntheticMockClient()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "status": "planned",
            "prompt_version": PROMPT_VERSION,
            "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            "case_ids": [c["case_id"] for c in cases],
            "methods": args.methods,
            "trials": args.trials,
            "n_calls": calls,
            "n_observations": observations,
            "synthetic": args.backend == "mock" and bool(model_methods),
            "backend": args.backend if model_methods else "deterministic_control",
            "model": client.model if model_methods else None,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "attempts_per_model_call": 1,
            "method_order": "rotating_by_case_and_trial",
            "cost": "not measured; call and output-token caps are not dollar caps",
        }
        # Persist the declared denominator BEFORE inference; interruption retains it.
        with manifest_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2) + "\n")
        # Flush each observation, preserving completed calls if interrupted.
        with output_path.open("x", encoding="utf-8") as stream:
            for trial in range(args.trials):
                for i, case in enumerate(cases):
                    shift = (i + trial) % len(args.methods)
                    methods = args.methods[shift:] + args.methods[:shift]
                    for method in methods:
                        prediction = run_case(
                            case, client, method, trial, args.temperature, args.max_tokens
                        )
                        stream.write(
                            json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n"
                        )
                        stream.flush()
        manifest["status"] = "complete"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"Wrote {observations} observations, {calls} completion invocations "
            f"({manifest['backend']})"
        )
        return
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {path}")
