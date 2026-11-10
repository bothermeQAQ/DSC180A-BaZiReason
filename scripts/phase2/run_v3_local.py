"""Run benchmark_v3 conditions on a local MLX model (base or base plus adapter).

Two protocols share the benchmark's cases, methods and scoring:

* ``strict``: the unchanged ``v3-information-ladder`` prompts and the unchanged
  ``run_case`` record builder. The whole output must be one JSON object.
* ``reasoned``: the same prompts plus one sentence allowing visible reasoning
  before the final JSON object (the planned addition in docs/research_design.md).
  The answer object is taken by the pre-specified rule in
  ``bazireason.p2_prompts.extract_final_json``.

A manifest declaring every planned observation is written before inference, so
``bazireason evaluate --predictions FILE --output FILE.scores.json`` binds to it.

    python scripts/phase2/run_v3_local.py --label base --protocol strict \
        --methods direct tool_context --ids data/synthetic/benchmark_v3_pilot_ids.json \
        --output results/phase2/v3/base_strict_pilot.jsonl
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from bazireason.baselines import parse_strict, recover_json, run_case
from bazireason.calendar_engine import compute_chart
from bazireason.controls import corrupt_context
from bazireason.dataset_generator import read_jsonl
from bazireason.mlx_backend import MLXChatModel, ReplayClient, file_sha256, messages_sha256
from bazireason.p2_prompts import REASONED_NOTE, REASONED_PROMPT_VERSION, extract_final_json
from bazireason.prompts import (
    MODEL_METHODS,
    PROMPT_VERSION,
    build_messages,
    condition,
    context_for_method,
)
from bazireason.schema import valid_response

DATA = "data/synthetic/benchmark_v3.jsonl"
DEFAULT_MODEL = "models/qwen3-4b-instruct-2507-mlx-4bit"


def reasoned_messages(case, method):
    base, _policy = condition(method)
    needs_engine = base not in ("direct", "tool_facts", "tool_intermediate")
    context = compute_chart(case["input"], case["convention"]) if needs_engine else None
    messages = build_messages(case, method, context)
    messages[0] = {"role": "system", "content": messages[0]["content"] + REASONED_NOTE}
    return messages, context


def reasoned_record(case, method, chat, text, latency, tokens, max_tokens, messages, context):
    base, policy = condition(method)
    response = extract_final_json(text)
    try:
        whole = parse_strict(text)
    except (ValueError, TypeError):
        whole = None
    record = {
        "case_id": case["case_id"],
        "method": method,
        "trial": 0,
        "requested_model": chat.model,
        "synthetic": False,
        "record_kind": "llm",
        "is_llm": True,
        "context_condition": base,
        "policy": policy,
        "temperature": 0,
        "max_tokens": max_tokens,
        "prompt_sha256": messages_sha256(messages),
        "prompt_version": REASONED_PROMPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "attempts": 1,
        "monetary_cost": None,
        "raw_output": text,
        "served_model": chat.model,
        "latency_seconds": latency,
        "usage": {"completion_tokens": tokens},
        "finish_reason": "length" if tokens >= max_tokens else "stop",
        "system_fingerprint": None,
        "response": response,
        "error": None,
        "answer_extraction": "whole_output" if whole is not None else "final_json_rule",
    }
    if response is None:
        record["error"] = "invalid_json"
        record["recovered_response"], record["recovery"] = recover_json(text)
    elif not valid_response(response):
        record["error"] = "invalid_schema"
    if base != "direct":
        record["supplied_context"] = context_for_method(context, method, case)
    if base == "tool_corrupted":
        record["corruption"] = corrupt_context(case, context)[1]
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter")
    parser.add_argument("--label", required=True, help="Name recorded as the model")
    parser.add_argument("--protocol", choices=("strict", "reasoned"), default="strict")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--ids", help="JSON array of case IDs; default is every case")
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    allowed = MODEL_METHODS + tuple(m + "_minimal" for m in MODEL_METHODS)
    if any(m not in allowed for m in args.methods) or len(set(args.methods)) != len(args.methods):
        parser.error("--methods must be unique model methods")
    max_tokens = args.max_tokens or (512 if args.protocol == "strict" else 900)
    cases = read_jsonl(args.data)
    if args.ids:
        wanted = json.loads(Path(args.ids).read_text())
        indexed = {c["case_id"]: c for c in cases}
        cases = [indexed[i] for i in wanted]
    output, manifest_path = Path(args.output), Path(args.output + ".manifest.json")
    if output.exists() or manifest_path.exists():
        parser.error("Prediction file exists; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    chat = MLXChatModel(args.model, args.adapter, label=args.label)
    adapter_file = Path(args.adapter) / "adapters.safetensors" if args.adapter else None
    manifest = {
        "status": "planned",
        "prompt_version": PROMPT_VERSION if args.protocol == "strict" else REASONED_PROMPT_VERSION,
        "protocol": args.protocol,
        "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "case_ids": [c["case_id"] for c in cases],
        "methods": args.methods,
        "trials": 1,
        "n_calls": len(cases) * len(args.methods),
        "n_observations": len(cases) * len(args.methods),
        "synthetic": False,
        "backend": "mlx_local",
        "model": chat.model,
        "model_path": args.model,
        "model_weights_sha256": file_sha256(Path(args.model) / "model.safetensors"),
        "adapter_path": args.adapter,
        "adapter_sha256": file_sha256(adapter_file) if adapter_file else None,
        "temperature": 0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "attempts_per_model_call": 1,
        "decoding": "greedy; one trial because greedy decoding is deterministic",
        "cost": "local inference; no paid calls",
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    started = time.perf_counter()
    records = []
    if args.protocol == "strict":
        client = ReplayClient(chat.model)
        for case in cases:
            for method in args.methods:
                run_case(case, client, method, 0, 0, max_tokens)  # collects the prompt
        totals = client.run_pending(chat, max_tokens, batch=args.batch)
        for case in cases:
            for method in args.methods:
                records.append(run_case(case, client, method, 0, 0, max_tokens))
    else:
        plan = [
            (case, method, *reasoned_messages(case, method))
            for case in cases
            for method in args.methods
        ]
        texts, totals = chat.generate_batch(
            [p[2] for p in plan], max_tokens, batch=args.batch, share_prefix=True
        )
        latency = totals["seconds"] / max(1, len(plan))
        for (case, method, messages, context), text in zip(plan, texts, strict=True):
            tokens = len(chat.tokenizer.encode(text, add_special_tokens=False))
            records.append(
                reasoned_record(
                    case, method, chat, text, latency, tokens, max_tokens, messages, context
                )
            )
    with output.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    manifest["status"] = "complete"
    manifest["wall_seconds"] = round(time.perf_counter() - started, 1)
    manifest["generation_totals"] = {
        k: (round(v, 2) if isinstance(v, float) else v) for k, v in totals.items()
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(records)} observations to {output} in {manifest['wall_seconds']} s")


if __name__ == "__main__":
    main()
