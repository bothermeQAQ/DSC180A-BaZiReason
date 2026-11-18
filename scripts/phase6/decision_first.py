"""Phase 6: does a calendar reply decide before it checks? (docs/phase6_log.md, 17 Nov 2026)

In the training answers without tool facts (bazireason.worked.explain_request) the first sentence
already carries the decision: an answer starts "参考时刻：{zone} 在该日期为 …", a refusal starts with
the solar-term window sentence or the unknown-offset sentence. This script reads saved replies and
counts, per run, the refusals whose first sentence contradicts the check it names:

  window   "参考日期 M 月 D 日落在{节}可能的交节日期（约 …）内" although that date lies outside the
           window of that term (bazireason.worked.jie_windows); also whether the stated window is right;
  offset   "不能凭记忆确定 {zone} … 的 UTC 偏移" although bazireason.worked.known_offset gives the offset.

Usage (reads only U*'s fit samples or dev6, never valid6 or sealed6):
  PYTHONPATH=src .venv/bin/python scripts/phase6/decision_first.py fit p6u4 p6u5
  PYTHONPATH=src .venv/bin/python scripts/phase6/decision_first.py dev6 p6u3v p6u4 p6u5
"""

import ast
import json
import re
import sys
from collections import Counter

from bazireason import worked as W
from bazireason.calendar_engine import parse_wall_time

WINDOW = re.compile(r"参考日期 (\d+) 月 (\d+) 日落在(\S+?)可能的交节日期（约 (.+?)）内")


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
        print(f"== {argv[0]} {label}: {n} replies, {wrong} wrong")
        for key, value in sorted(counts.items()):
            print("   " + " | ".join(key) + f": {value}")


if __name__ == "__main__":
    main(sys.argv[1:])
