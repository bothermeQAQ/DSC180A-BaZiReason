"""CLI entry point; defaults are offline, and live runs require explicit bounds."""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

from bazireason.baselines import run_case
from bazireason.dataset_generator import (
    dataset_bytes,
    generate_audit_dataset,
    generate_dataset,
    read_jsonl,
)
from bazireason.dataset_v3 import generate_benchmark_v3
from bazireason.evaluator import evaluate
from bazireason.llm_clients import OpenAICompatibleClient, SyntheticMockClient
from bazireason.prompts import METHODS, PROMPT_VERSION, condition

CURRENT_DATA = "data/synthetic/benchmark_v3.jsonl"
DATASETS = {
    "checkpoint_v1": (generate_dataset, "data/synthetic/checkpoint.jsonl"),
    "audit_v2": (generate_audit_dataset, "data/synthetic/audit_v2.jsonl"),
    "benchmark_v3": (generate_benchmark_v3, CURRENT_DATA),
}
LEGACY_PROMPT_VERSIONS = {"v2-policy-controls": "v2"}


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


def _ids_from_file(path, parser):
    ids = json.loads(Path(path).read_text())
    if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids):
        parser.error("ID file must be a nonempty JSON array of strings")
    return ",".join(ids)


def read_predictions(path):
    """Read JSONL predictions, tolerating only an interrupted final write.

    A final line without its newline that does not parse is the trace of an
    interruption: it is dropped, reported, and its observation stays missing.
    Any other malformed line is an error.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = text.split("\n")
    records, truncated = [], False
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            if number == len(lines) and not text.endswith("\n"):
                truncated = True
                continue
            raise ValueError(f"Malformed prediction line {number}") from error
    return records, truncated


def check_records(records, manifest, parser):
    """Every model record must match the manifest's declared run settings."""
    for record in records:
        method = record.get("method")
        if method not in METHODS or condition(method)[1] is None:
            continue
        declared = (
            manifest.get("prompt_version"),
            manifest.get("model"),
            manifest.get("temperature"),
            manifest.get("max_tokens"),
            bool(manifest.get("synthetic")),
        )
        actual = (
            record.get("prompt_version"),
            record.get("requested_model"),
            record.get("temperature"),
            record.get("max_tokens"),
            bool(record.get("synthetic")),
        )
        if actual != declared:
            parser.error(
                f"Record {record.get('case_id')}/{method} differs from the manifest "
                "(prompt version, model, decoding settings or synthetic flag)"
            )


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


def _refuse_existing(path, force, parser):
    if Path(path).exists() and not force:
        parser.error(f"{path} exists; choose a new path or pass --force")


def _generate(args, parser):
    generator, default_path = DATASETS[args.dataset_version]
    payload = dataset_bytes(generator(args.seed))
    if args.check:
        same = Path(args.check).read_bytes() == payload
        digest = hashlib.sha256(payload).hexdigest()
        print(f"{'PASS' if same else 'FAIL'}: regenerated {args.dataset_version} sha256 {digest}")
        if not same:
            sys.exit(1)
        return
    output = args.output or default_path
    historical = {Path(DATASETS[v][1]).resolve() for v in ("checkpoint_v1", "audit_v2")}
    if Path(output).resolve() in historical:
        parser.error("Historical datasets are write-protected; use --check to verify them")
    _refuse_existing(output, args.force, parser)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_bytes(payload)
    count = payload.count(b"\n")
    print(f"Wrote {count} cases to {output}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--seed", type=int, default=1802026)
    gen.add_argument("--output")
    gen.add_argument("--dataset-version", choices=list(DATASETS), default="benchmark_v3")
    gen.add_argument("--check", metavar="PATH", help="Regenerate and compare bytes; write nothing")
    gen.add_argument("--force", action="store_true", help="Allow replacing an existing output")
    diag = sub.add_parser("diagnostics")
    diag.add_argument("--data", default=CURRENT_DATA)
    diag.add_argument("--output", required=True)
    diag.add_argument("--force", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--data", default=CURRENT_DATA)
    run.add_argument("--output", required=True)
    run.add_argument("--backend", choices=["mock", "api"], default="mock")
    run.add_argument("--model", default=os.environ.get("BAZIREASON_MODEL"))
    run.add_argument("--base-url", default=os.environ.get("BAZIREASON_BASE_URL"))
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--case-ids", help="Comma-separated exact IDs; overrides --limit")
    run.add_argument("--case-ids-file", help="JSON array of exact IDs, e.g. the pilot file")
    run.add_argument("--methods", nargs="+", choices=METHODS, default=["direct", "tool_context"])
    run.add_argument("--trials", type=int, default=1)
    run.add_argument("--max-calls", type=int, default=8)
    run.add_argument("--temperature", type=float, default=0)
    run.add_argument("--max-tokens", type=int, default=512)
    score = sub.add_parser("evaluate")
    score.add_argument("--data", default=CURRENT_DATA)
    score.add_argument("--predictions", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--force", action="store_true")
    score.add_argument("--limit", type=int, default=None)
    score.add_argument("--case-ids", help="Same declared IDs as inference; overrides --limit")
    score.add_argument("--methods", nargs="+", choices=METHODS)
    score.add_argument("--trials", type=int)
    score.add_argument("--bootstrap-samples", type=int, default=1000)
    score.add_argument("--bootstrap-seed", type=int, default=1802026)
    args = parser.parse_args()
    if args.command == "generate":
        _generate(args, parser)
        return
    cases = read_jsonl(args.data)
    if args.command == "diagnostics":
        from bazireason.analysis import diagnostics

        _refuse_existing(args.output, args.force, parser)
        output = diagnostics(cases)
    elif args.command == "evaluate":
        _refuse_existing(args.output, args.force, parser)
        cases, methods, trials, manifest = evaluation_design(args, cases, parser)
        try:
            records, truncated = read_predictions(args.predictions)
        except ValueError as error:
            parser.error(str(error))
        versions = {r.get("prompt_version") for r in records if r.get("prompt_version")}
        if manifest:
            check_records(records, manifest, parser)
            versions = {manifest.get("prompt_version")}
        scheme = LEGACY_PROMPT_VERSIONS.get(next(iter(versions))) if len(versions) == 1 else None
        output = evaluate(
            cases,
            records,
            methods=methods,
            trials=trials,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            corruption_scheme=scheme or "v3",
        )
        output["evaluation_design_source"] = (
            "planned_manifest" if manifest else "explicit_or_legacy"
        )
        output["declared_synthetic"] = manifest.get("synthetic") if manifest else None
        output["manifest_status"] = manifest.get("status") if manifest else None
        output["truncated_final_line_dropped"] = truncated
    else:
        if args.case_ids_file is not None:
            if args.case_ids is not None:
                parser.error("Use either --case-ids or --case-ids-file")
            args.case_ids = _ids_from_file(args.case_ids_file, parser)
        if args.limit is not None and args.limit < 1:
            parser.error("Run bounds must be positive")
        if min(args.trials, args.max_tokens) < 1 or args.max_calls < 0:
            parser.error("Run bounds must be positive")
        if not math.isfinite(args.temperature) or args.temperature < 0:
            parser.error("--temperature must be finite and nonnegative")
        if len(set(args.methods)) != len(args.methods):
            parser.error("--methods must be unique")
        cases = select_cases(cases, args.limit, args.case_ids, parser)
        if not cases:
            parser.error("Dataset selection must be nonempty")
        model_methods = [m for m in args.methods if condition(m)[1] is not None]
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
