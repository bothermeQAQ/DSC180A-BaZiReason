"""Paired prompts: only the treatment contains computed reference answers."""

import json

from bazireason.conventions import CONVENTION_SPEC
from bazireason.schema import model_view

SYSTEM = (
    "You are a calendar computation assistant. Return only a JSON object with exactly "
    "status, pillars, reason_codes, conventions. status is ok, ambiguous, unsupported, "
    "or invalid. For ok, pillars has exactly year, month, day, hour (each a valid Chinese "
    "sexagenary pair), reason_codes=[], and conventions repeats the complete input "
    "convention. Otherwise pillars=null and reason_codes is a nonempty list of concise "
    "machine-readable reasons. conventions repeats the selected convention, or null if "
    "one is missing. Do not make destiny predictions. " + CONVENTION_SPEC
)


def build_messages(case, method, context=None):
    if method not in {"direct", "tool_context"}:
        raise ValueError("Unknown method")
    public = {"input": case["input"], "convention": case["convention"]}
    prompt = "Compute the calendar result under the specified profile.\n"
    prompt += json.dumps(public, ensure_ascii=False, sort_keys=True)
    if method == "tool_context":
        if context is None:
            raise ValueError("Tool context required")
        prompt += "\nDeterministic context (use faithfully):\n"
        prompt += json.dumps(model_view(context), ensure_ascii=False, sort_keys=True)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
