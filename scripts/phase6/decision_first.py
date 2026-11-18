"""Phase 6: does a calendar reply decide before it checks? (docs/phase6_log.md, 17 Nov 2026)

In the training answers without tool facts (bazireason.worked.explain_request) the first sentence
already carries the decision: an answer starts "参考时刻：{zone} 在该日期为 …", a refusal starts with
the solar-term window sentence or the unknown-offset sentence. This script reads saved replies and
counts, per run, the refusals whose first sentence contradicts the check it names:

  window   "参考日期 M 月 D 日落在{节}可能的交节日期（约 …）内" although that date lies outside the
           window of that term (bazireason.worked.jie_windows); also whether the stated window is right;
  offset   "不能凭记忆确定 {zone} … 的 UTC 偏移" although bazireason.worked.known_offset gives the offset.

Version-3 replies (preregistration 12.14) write the checks first: a 时区 line ending "→ UTC…" or
"→ 偏移需工具" and a 交节 line ending "→ 不在交节窗口内" or "→ 在{节}窗口内，需工具". For those,
each conclusion is compared with the rule for the date the reply itself states:

  v3 zone     the 时区 conclusion against known_offset for the request's zone and date;
  v3 window   the 交节 conclusion against jie_window_of for the reference date the reply states.

Usage (reads only U*'s fit samples or dev6, never valid6 or sealed6):
  PYTHONPATH=src .venv/bin/python scripts/phase6/decision_first.py fit p6u4 p6u5
  PYTHONPATH=src .venv/bin/python scripts/phase6/decision_first.py dev6 p6u3v p6u4 p6u5
"""

import ast
import json
import re
import sys
from collections import Counter
from datetime import date

from bazireason import worked as W
from bazireason.calendar_engine import parse_wall_time

WINDOW = re.compile(r"参考日期 (\d+) 月 (\d+) 日落在(\S+?)可能的交节日期（约 (.+?)）内")
V3_WINDOW = re.compile(r"交节：参考日期 (\d+) 月 (\d+) 日。.*?→ (不在交节窗口内|在(\S+?)窗口内)")


def fit_rows(label):
    for r in map(json.loads, open(f"results/phase6/own_replies_{label}_c.jsonl", encoding="utf-8")):
        yield r["row"]["meta"], r["reply"], bool(r["correct"])


def dev6_rows(label):
    items = {
        r["id"]: r for r in map(json.loads, open("data/phase6/dev6/calendar.jsonl", encoding="utf-8"))
    }
    path = f"results/phase6/dev6/{label}/calendar_L0.jsonl"
    for r in map(json.loads, open(path, encoding="utf-8")):
        score = r["score"] if isinstance(r["score"], dict) else ast.literal_eval(r["score"])
        yield items[r["id"]]["meta"], r["raw_output"], bool(score.get("correct"))


def classify_v3(meta, reply):
    """Keys for the 时区 and 交节 conclusions of a version-3 reply (none for other replies)."""
    keys, expected = [], meta["expected"]["status"]
    lines = [re.sub(r"^\d+\.\s*", "", x) for x in reply.strip().splitlines()]
    zone = next((x for x in lines if x.startswith("时区：")), None)
    if zone:
        inputs = meta.get("inputs", {})
        try:
            known = W.known_offset(inputs["timezone"], parse_wall_time(inputs["datetime"])[1])
        except (KeyError, TypeError, ValueError):
            known = "unparsable"
        said_tool = "→ 偏移需工具" in zone
        right = known != "unparsable" and said_tool == (known is None)
        keys.append(("v3 zone", "conclusion right" if right else "conclusion WRONG", f"expected {expected}"))
    for x in lines:
        match = V3_WINDOW.search(x)
        if match:
            month, day = int(match[1]), int(match[2])
            try:
                inside = W.jie_window_of(date(2001, month, day)) is not None
            except ValueError:
                inside = None
            said_inside = match[3] != "不在交节窗口内"
            right = inside is not None and said_inside == inside
            keys.append(("v3 window", "conclusion right" if right else "conclusion WRONG", f"expected {expected}"))
    return keys


def classify(meta, reply):
    first = re.sub(r"^\d+\.\s*", "", reply.strip().splitlines()[0]) if reply.strip() else ""
    expected = meta["expected"]["status"]
    match = WINDOW.search(first)
    if match:
        month, day, name, stated = int(match[1]), int(match[2]), match[3], match[4]
        if name not in W.jie_windows():
            return ("window", "unknown term", f"expected {expected}")
        low, high = W.jie_windows()[name]
        where = "date inside" if low <= (month, day) <= high else "date OUTSIDE"
        text = "window right" if stated == W.window_text(name) else "window WRONG"
        return ("window", where, text, f"expected {expected}")
    if "不能凭记忆确定" in first:
        inputs = meta.get("inputs", {})
        try:
            known = W.known_offset(inputs["timezone"], parse_wall_time(inputs["datetime"])[1])
        except (KeyError, TypeError, ValueError):
            return ("offset", "input not parsable", f"expected {expected}")
        return ("offset", "offset KNOWABLE" if known is not None else "offset not knowable",
                f"expected {expected}")
    return None


def main(argv):
    if len(argv) < 2 or argv[0] not in ("fit", "dev6"):
        raise SystemExit(__doc__)
    source = fit_rows if argv[0] == "fit" else dev6_rows
    for label in argv[1:]:
        counts, n, wrong = Counter(), 0, 0
        for meta, reply, correct in source(label):
            n += 1
            wrong += not correct
            kind = classify(meta, reply or "")
            if kind:
                counts[kind + ("right" if correct else "wrong",)] += 1
            for key in classify_v3(meta, reply or ""):
                counts[key + ("right" if correct else "wrong",)] += 1
        print(f"== {argv[0]} {label}: {n} replies, {wrong} wrong")
        for key, value in sorted(counts.items()):
            print("   " + " | ".join(key) + f": {value}")


if __name__ == "__main__":
    main(sys.argv[1:])
