"""Matched inference conditions with raw output and operational provenance."""

import hashlib
import json
import re
from datetime import datetime, timezone

from bazireason.calendar_engine import compute_chart
from bazireason.controls import BASELINES, corrupt_context
from bazireason.prompts import PROMPT_VERSION, build_messages, condition, context_for_method
from bazireason.schema import model_view, valid_response

_FENCE = re.compile(r"\A\s*```[A-Za-z]*[ \t]*\n(.*?)\n?[ \t]*```\s*\Z", re.DOTALL)


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Duplicate JSON key")
        obj[key] = value
    return obj


def parse_strict(text):
    """The primary scoring parse: the whole output must be one JSON object."""
    return json.loads(text, object_pairs_hook=_unique_object)


def recover_json(text):
    """Optional secondary recovery, never used for primary scores.

    Accepts exactly one Markdown code fence around the output, or else the span
    from the first "{" to the last "}". Duplicate keys are still rejected.
    Returns (object, method) or (None, None).
    """
    if not isinstance(text, str):
        return None, None
    fence = _FENCE.match(text)
    if fence:
        candidate, method = fence.group(1), "code_fence"
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None, None
        candidate, method = text[start : end + 1], "outer_braces"
    try:
        value = parse_strict(candidate)
    except (ValueError, TypeError):
        return None, None
    return (value, method) if isinstance(value, dict) else (None, None)


def _deterministic(result, response):
    result.update(
        {
            "response": response,
            "raw_output": json.dumps(response, ensure_ascii=False, sort_keys=True),
            "served_model": None,
            "error": None,
            "latency_seconds": 0.0,
            "usage": None,
            "finish_reason": None,
            "system_fingerprint": None,
        }
    )
    return result


def run_case(case, client, method, trial=0, temperature=0, max_tokens=512):
    base, policy = condition(method)
    deterministic = policy is None
    needs_engine = (
        base not in ("direct", "tool_facts", "tool_intermediate") and base not in BASELINES
    )
    context = compute_chart(case["input"], case["convention"]) if needs_engine else None
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
    if base in BASELINES:
        return _deterministic(result, BASELINES[base](case))
    if base == "copy_context":
        return _deterministic(result, model_view(context))
    if base != "direct":
        result["supplied_context"] = context_for_method(context, method, case)
    if base == "tool_corrupted":
        # Informational only: scoring recomputes the corruption from the design.
        result["corruption"] = corrupt_context(case, context)[1]
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
            result["response"] = parse_strict(completion.text)
            if not valid_response(result["response"]):
                result["error"] = "invalid_schema"
        except (ValueError, TypeError):
            result["response"] = None
            result["error"] = "invalid_json"
            recovered, how = recover_json(completion.text)
            result["recovered_response"], result["recovery"] = recovered, how
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
