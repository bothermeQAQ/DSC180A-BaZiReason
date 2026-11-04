"""Matched inference conditions with raw output and operational provenance."""

import hashlib
import json
from datetime import datetime, timezone

from bazireason.calendar_engine import compute_chart
from bazireason.prompts import build_messages


def run_case(case, client, method, trial=0, temperature=0, max_tokens=512):
    context = compute_chart(case["input"], case["convention"]) if method == "tool_context" else None
    messages = build_messages(case, method, context)
    prompt_sha = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
    result = {
        "case_id": case["case_id"],
        "method": method,
        "trial": trial,
        "requested_model": client.model,
        "synthetic": client.model == "SYNTHETIC-MOCK-NOT-AN-LLM",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_sha256": prompt_sha,
        "prompt_version": "v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "attempts": 1,
        "monetary_cost": None,
    }
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
            result["response"] = json.loads(completion.text)
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
