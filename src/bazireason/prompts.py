"""Controlled prompts. Context conditions form an information ladder.

direct < tool_facts (normalized instants) < tool_intermediate (instants, Jie
and day count, but no pillars) < tool_context (the computed answer). The
tool_corrupted condition supplies a computed answer containing one plausible,
deterministic error. Only the explicit/minimal policy factor changes the
nonanswer instructions.
"""

import json

from bazireason.calendar_engine import temporal_facts
from bazireason.controls import corrupt_context
from bazireason.schema import model_view

MODEL_METHODS = ("direct", "tool_facts", "tool_intermediate", "tool_context", "tool_corrupted")
DETERMINISTIC_METHODS = ("copy_context", "heuristic_surface", "always_abstain", "always_answer")
METHODS = MODEL_METHODS + tuple(m + "_minimal" for m in MODEL_METHODS) + DETERMINISTIC_METHODS
PROMPT_VERSION = "v3-information-ladder"

RESPONSE_SPEC = (
    "You are a calendar computation assistant. Return only a JSON object with exactly "
    "these keys: status, pillars, reason_codes, conventions. status is ok, ambiguous, "
    "unsupported, or invalid. For ok: pillars has exactly year, month, day and hour, each "
    "a Chinese sexagenary pair such as 甲子; reason_codes is []; conventions is the complete "
    "profile used, with sect null if the input gave no sect. Otherwise: pillars is null, "
    "reason_codes is a nonempty list of short machine-readable reasons, and conventions is "
    "the input profile if it was fully specified, else null. Do not make destiny "
    "predictions. "
)
# Shared by every condition, so tools add facts and computation, not rule knowledge.
CALENDAR_SPEC = (
    "Profile rules. Convert the civil instant to fixed UTC+08:00 and use that reference "
    "clock for all four pillars, with no longitude or true-solar-time correction; this is "
    "an experimental reference clock, not a local birthplace chart. Supported reference "
    "years are 1901-2099. Year: the pillar changes at the 立春 instant; BaZi year Y has "
    "cycle index (Y - 4) mod 60, where 甲子 is index 0, so 1984 is 甲子. Month: the "
    "pillar changes at the instants of the twelve Jie: 立春 starts 寅, then 惊蛰 卯, 清明 "
    "辰, 立夏 巳, 芒种 午, 小暑 未, 立秋 申, 白露 酉, 寒露 戌, 立冬 亥, 大雪 子, 小寒 丑. "
    "The 寅-month stem is 丙 in 甲 or 己 years, 戊 in 乙 or 庚, 庚 in 丙 or 辛, 壬 in 丁 "
    "or 壬, and 甲 in 戊 or 癸 years; later months continue the stem sequence. Day: a "
    "continuous 60-day cycle in which 2000-01-01 is 戊午. Sect 1 advances the day pillar "
    "at 23:00; sect 2 advances it at 00:00. Hour: two-hour branches with 子 from 23:00 to "
    "00:59, 丑 from 01:00, and so on. The 子-hour stem is 甲 on 甲 or 己 days, 丙 on 乙 or "
    "庚, 戊 on 丙 or 辛, 庚 on 丁 or 壬, and 壬 on 戊 or 癸 days; later hours continue the "
    "stem sequence. For both sects, hours from 23:00 take their stem from the next day. "
)
EXPLICIT_POLICY = (
    "Answer whenever the supplied information determines a single chart under this "
    "profile, even if a field is absent. If missing or imprecise information could change "
    "any pillar, return ambiguous and say what is needed instead of guessing. For example, "
    "an unspecified sect matters only from 23:00 to 23:59:59, and a repeated "
    "daylight-saving wall time without fold matters only if its two instants give "
    "different pillars. A date without a time of day is ambiguous. A wall time that does "
    "not exist, or an invalid date or zone key, is invalid. An unsupported profile or a "
    "reference year outside 1901-2099 is unsupported. Never give pillars for a nonanswer. "
)
SYSTEM = RESPONSE_SPEC + CALENDAR_SPEC + EXPLICIT_POLICY
CONTEXT_LABELS = {
    "tool_facts": "Normalized time tool facts",
    "tool_intermediate": "Temporal tool facts (intermediate values, not pillars)",
    # Correct and corrupted answers use the same label and instructions.
    "tool_context": "Calendar tool context",
    "tool_corrupted": "Calendar tool context",
}


def condition(method):
    """(context condition, policy); deterministic baselines have no prompt policy."""
    if method not in METHODS:
        raise ValueError("Unknown method")
    if method in DETERMINISTIC_METHODS:
        return method, None
    return method.removesuffix("_minimal"), "minimal" if method.endswith("_minimal") else "explicit"


def context_for_method(context, method, case=None):
    """Supplied context for a tool condition, without mutating the reference result.

    `context` is the engine result; fact conditions are computed from the case
    input alone, so they never depend on the profile or on a stored label.
    """
    base, _policy = condition(method)
    if base in ("tool_facts", "tool_intermediate"):
        if case is None:
            raise ValueError("Fact conditions require the case input")
        facts = temporal_facts(case["input"])
        if base == "tool_facts":
            keep = ("utc8_start", "utc8_end", "fold")
            facts["candidates"] = [
                {k: v for k, v in entry.items() if k in keep} for entry in facts["candidates"]
            ]
        return facts
    if context is None:
        raise ValueError("Tool context required")
    if base == "tool_corrupted":
        if case is None:
            raise ValueError("Corrupted context requires the case input")
        return corrupt_context(case, context)[0]
    return model_view(context)


def build_messages(case, method, context=None):
    base, policy = condition(method)
    if policy is None:
        raise ValueError(f"{method} is a deterministic baseline, not a model prompt")
    public = {"input": case["input"], "convention": case["convention"]}
    prompt = "Compute the calendar result under the specified profile.\n"
    prompt += json.dumps(public, ensure_ascii=False, sort_keys=True)
    if base != "direct":
        supplied = context_for_method(context, method, case)
        prompt += f"\n{CONTEXT_LABELS[base]}:\n" + json.dumps(
            supplied, ensure_ascii=False, sort_keys=True
        )
    system = RESPONSE_SPEC + CALENDAR_SPEC + (EXPLICIT_POLICY if policy == "explicit" else "")
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
