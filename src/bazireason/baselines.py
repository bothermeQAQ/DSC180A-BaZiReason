"""Matched inference conditions with raw output and operational provenance."""

import hashlib
import json
from datetime import datetime, timezone

from bazireason.calendar_engine import compute_chart
from bazireason.prompts import PROMPT_VERSION, build_messages, condition, context_for_method
from bazireason.schema import model_view, valid_response


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Duplicate JSON key")
        obj[key] = value
    return obj


def run_case(case, client, method, trial=0, temperature=0, max_tokens=512):
    base, policy = condition(method)
    context = compute_chart(case["input"], case["convention"]) if base != "direct" else None
    deterministic = base == "copy_context"
    messages = None if deterministic else build_messages(case, method, context)
    prompt_sha = (
        hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        if messages is not None
        else None
    )
    synthetic = not deterministic and client.model == "SYNTHETIC-MOCK-NOT-AN-LLM"
    result = {
        "case_id": case["case_id"],
        "method": method,
        "trial": trial,
        "requested_model": None if deterministic else client.model,
        "synthetic": synthetic,
        "record_kind": "deterministic_control" if deterministic else "mock" if synthetic else "llm",
        "is_llm": not deterministic and not synthetic,
        "context_condition": base,
        "policy": policy,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_sha256": prompt_sha,
        "prompt_version": PROMPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "attempts": 0 if deterministic else 1,
        "monetary_cost": None,
    }
    if context is not None:
        supplied = context_for_method(context, method)
        result["supplied_context"] = supplied
        result["context_corrupted"] = base == "tool_corrupted" and context["status"] == "ok"
    if deterministic:
        result.update(
            {
                "response": model_view(context),
                "raw_output": json.dumps(model_view(context), ensure_ascii=False, sort_keys=True),
                "served_model": None,
                "error": None,
                "latency_seconds": 0.0,
                "usage": None,
                "finish_reason": None,
                "system_fingerprint": None,
            }
        )
        return result
    try:
        completion = client.complete(messages, temperature=temperature, max_tokens=max_tokens)
        result.update(
            {
                "raw_output": completion.text,
                "served_model": completion.model,
                "latency_seconds": completion.latency_seconds,
                "usage": completion.usage,
                "finish_reason": completion.finish_reason,
                "system_fingerprint": completion.system_fingerprint,
                "error": None,
            }
        )
        try:
            result["response"] = json.loads(completion.text, object_pairs_hook=_unique_object)
            if not valid_response(result["response"]):
                result["error"] = "invalid_schema"
        except (ValueError, TypeError):
            result["response"] = None
            result["error"] = "invalid_json"
    except Exception as e:
        result.update(
            {
                "raw_output": None,
                "response": None,
                "error": type(e).__name__,
                "latency_seconds": None,
                "usage": None,
            }
        )
    return result
