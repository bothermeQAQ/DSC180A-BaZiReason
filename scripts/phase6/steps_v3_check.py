"""Derivation version 3 against version 2 (CPU only; preregistration 12.14): same answers, no
wrong statement, the calendar decisions of no_tool_expected, and lengths.

    PYTHONPATH=src .venv-mlx/bin/python scripts/phase6/steps_v3_check.py
        -> results/phase6/steps_v3_check.json

Rule items: for random charts, the two question types whose derivation changed (relations and
areas of life: every domain, both sexes, years 1915-2150), the returned answer equals version 2's,
the final JSON equals it (relation lists as sets, as scoring compares them), and the claim checker
finds no wrong statement. Calendar: for random requests without tool facts, the answer equals
no_tool_expected's, the 时区 step concludes as known_offset, the 交节 step as jie_window_of, and a
refusal is written exactly when the tool is needed. Reply lengths in tokens, the final JSON line
included: replies are capped at 1,100 tokens for rule items and 700 for calendar requests
(run6.RULE_TOKENS, run6.MAX_TOKENS); "over_cap" counts training answers longer than that.
"""

import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

from bazireason import splits6 as S
from bazireason import steps6 as S6
from bazireason import train_data6 as T6
from bazireason import worked as W
from bazireason.calendar_engine import parse_wall_time
from bazireason.grounded import claim_failures
from bazireason.p2_prompts import extract_final_json, final_line

OUT = Path("results/phase6/steps_v3_check.json")
CHARTS = 1500
CAPS = {"interact": 1100, "domain": 1100}  # calendar: 700
REQUESTS = 20000
DOMAINS = ("marriage", "career", "wealth", "study", "children")
ZONES = ("Asia/Shanghai",) * 4 + ("UTC", "Asia/Tokyo", "Asia/Kolkata", "Asia/Hong_Kong",
         "Asia/Singapore", "America/New_York", "America/Los_Angeles", "Europe/London",
         "Australia/Sydney", "Europe/Paris", "America/Chicago")  # fmt: skip


def length(tok, text):
    return len(tok.encode(text, add_special_tokens=False).ids)


def cell(out, kind):
    return out["tokens"].setdefault(
        kind, {"v2_max": 0, "v3_max": 0, "v2_sum": 0, "v3_sum": 0, "n": 0, "v3_over_cap": 0}
    )


def rule_items(out, tok, rng):
    for _ in range(CHARTS):
        pillars, gender = S.random_pillars(rng), rng.choice(("male", "female"))
        intents = ["interact", f"domain:{rng.choice(DOMAINS)}:{rng.randrange(1915, 2151)}"]
        for intent in intents:
            texts = {}
            for version in (2, 3):
                S6.set_steps(version)
                try:
                    texts[version] = S6.derive(intent, pillars, gender)
                finally:
                    S6.set_steps(1)
            (second, reference), (third, same) = texts[2], texts[3]
            shown = extract_final_json(third)
            sets = {k: sorted(v) if isinstance(v, list) else v for k, v in shown.items()}
            want = {k: sorted(v) if isinstance(v, list) else v for k, v in reference.items()}
            out["derivations"] += 1
            out["answers_differ"] += same != reference or sets != want
            out["claim_failures"] += bool(claim_failures(third, pillars, gender))
            c = cell(out, intent.split(":")[0])
            for version, text in ((2, second), (3, third)):
                k = length(tok, text)
                c[f"v{version}_max"] = max(c[f"v{version}_max"], k)
                c[f"v{version}_sum"] += k
                c["v3_over_cap"] += version == 3 and k > CAPS[intent.split(":")[0]]
            c["n"] += 1


def request(rng):
    wall = datetime(rng.randrange(1901, 2100), 1, 1) + timedelta(minutes=rng.randrange(366 * 24 * 60))
    if rng.random() < 0.15:
        wall = wall.replace(hour=23)
    text = wall.strftime("%Y-%m-%dT%H:%M:%S" if rng.random() < 0.7 else "%Y-%m-%dT%H:%M")
    return {"datetime": text, "timezone": rng.choice(ZONES)}, (
        {"sect": rng.choice((1, 2))} if rng.random() < 0.8 else {}
    )


def calendar(out, tok, rng):
    c = out["calendar"]
    for _ in range(REQUESTS):
        inputs, convention = request(rng)
        before = W.explain_request(inputs, convention, "none")
        W.set_check_first(True)
        try:
            after = W.explain_request(inputs, convention, "none")
        finally:
            W.set_check_first(False)
        expected = W.no_tool_expected(inputs, convention)
        c["requests"] += 1
        c["response_differs"] += not (after["response"] == before["response"] == expected)
        steps = after["steps"]
        if not steps[0].startswith("时区："):
            c["decided_before_the_offset"] += 1
            c["unchanged_text_differs"] += steps != before["steps"]
            continue
        c["checked"] += 1
        status = expected["status"]
        c["by_status"][status] = c["by_status"].get(status, 0) + 1
        wall = parse_wall_time(inputs["datetime"])[1]
        offset = W.known_offset(inputs["timezone"], wall)
        c["zone_conclusion_wrong"] += ("→ 偏移需工具。" in steps[0]) != (offset is None)
        refused = steps[-1] == W.OFFSET_REFUSAL or steps[-1].startswith("没有该年")
        c["refusal_wrong"] += refused != (status == W.NEEDS_TOOL)
        if offset is not None:
            reference = wall + timedelta(minutes=480 - offset)
            for s in steps:
                if s.startswith("交节："):
                    inside = W.jie_window_of(reference.date()) is not None
                    c["window_conclusion_wrong"] += ("窗口内，需工具" in s) != inside
        k = cell(out, f"calendar:{status}")
        tail = "\n" + final_line(expected)
        for version, text in ((2, W.numbered(before["steps"]) + tail), (3, W.numbered(steps) + tail)):
            n = length(tok, text)
            k[f"v{version}_max"] = max(k[f"v{version}_max"], n)
            k[f"v{version}_sum"] += n
            k["v3_over_cap"] += version == 3 and n > 700
        k["n"] += 1


def main():
    tok = T6._tokenizer()
    if tok is None:
        sys.exit("needs the tokenizers package (.venv-mlx)")
    rng = random.Random(20261118)
    out = {"charts": CHARTS, "derivations": 0, "answers_differ": 0, "claim_failures": 0,
           "calendar": {"requests": 0, "response_differs": 0, "decided_before_the_offset": 0,
                        "unchanged_text_differs": 0, "checked": 0, "by_status": {},
                        "zone_conclusion_wrong": 0, "window_conclusion_wrong": 0, "refusal_wrong": 0},
           "tokens": {}}  # fmt: skip
    rule_items(out, tok, rng)
    calendar(out, tok, rng)
    for c in out["tokens"].values():
        c["v2_mean"] = round(c.pop("v2_sum") / c["n"])
        c["v3_mean"] = round(c.pop("v3_sum") / c["n"])
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
