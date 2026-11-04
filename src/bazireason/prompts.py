"""Controlled prompts; full-answer context is an assisted-copy condition."""

import copy
import json

from bazireason.conventions import CONVENTION_SPEC, CYCLE
from bazireason.schema import model_view

MODEL_METHODS = ("direct", "tool_context", "tool_facts", "tool_corrupted")
METHODS = MODEL_METHODS + tuple(m + "_minimal" for m in MODEL_METHODS) + ("copy_context",)
PROMPT_VERSION = "v2-policy-controls"

# Same computation/range rules in both policy levels; only nonanswer instructions differ.
CALENDAR_SPEC, _separator, _legacy_policy = CONVENTION_SPEC.partition("Missing required fields")
if not _separator:
    raise RuntimeError("Calendar/policy specification separator is missing")
RESPONSE_SPEC = (
    "You are a calendar computation assistant. Return only a JSON object with exactly "
    "status, pillars, reason_codes, conventions. status is ok, ambiguous, unsupported, "
    "or invalid. For ok, pillars has exactly year, month, day, hour (each a valid Chinese "
    "sexagenary pair), reason_codes=[], and conventions repeats the complete input "
    "convention. Otherwise pillars=null and reason_codes is a nonempty list of concise "
    "machine-readable reasons. conventions repeats the selected convention, or null if "
    "one is missing. Do not make destiny predictions. "
)
EXPLICIT_POLICY = (
    "Missing required fields or an unspecified sect require ambiguous status and "
    "clarification; never guess a missing field or convention. A repeated DST wall "
    "time requires fold=0 or fold=1. Nonexistent wall times and invalid dates/zones "
    "require invalid status. Unsupported profiles or reference years require "
    "unsupported status. Never give definitive pillars for these nonanswer cases. "
)
SYSTEM = RESPONSE_SPEC + CALENDAR_SPEC + EXPLICIT_POLICY


def condition(method):
    if method not in METHODS:
        raise ValueError("Unknown method")
    return (
        method.removesuffix("_minimal"),
        "minimal" if method.endswith("_minimal") else "explicit",
    )


def context_for_method(context, method):
    """Return supplied context without mutating the reference computation."""
    base, _policy = condition(method)
    if context is None:
        raise ValueError("Tool context required")
    if base == "tool_facts":
        # No result status, reason codes or pillars: a preprocessing-only control.
        return {
            "normalized_datetime": context.get("normalized_datetime"),
            "provenance": context.get("provenance"),
        }
    supplied = copy.deepcopy(model_view(context))
    if base == "tool_corrupted" and supplied["status"] == "ok":
        hour = supplied["pillars"]["hour"]
        # Preserve the hour branch; change the stem to another valid cycle pair.
        supplied["pillars"]["hour"] = CYCLE[(CYCLE.index(hour) + 12) % len(CYCLE)]
    return supplied


def build_messages(case, method, context=None):
    base, policy = condition(method)
    if base == "copy_context":
        raise ValueError("copy_context is a deterministic baseline, not a model prompt")
    public = {"input": case["input"], "convention": case["convention"]}
    prompt = "Compute the calendar result under the specified profile.\n"
    prompt += json.dumps(public, ensure_ascii=False, sort_keys=True)
    if base != "direct":
        supplied = context_for_method(context, method)
        label = "Normalized timestamp context" if base == "tool_facts" else "Calendar tool context"
        # Correct/corrupted conditions use identical labels and instructions.
        prompt += f"\n{label}:\n" + json.dumps(supplied, ensure_ascii=False, sort_keys=True)
    system = RESPONSE_SPEC + CALENDAR_SPEC + (EXPLICIT_POLICY if policy == "explicit" else "")
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
