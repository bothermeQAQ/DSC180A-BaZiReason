"""Score BaziQA for Phase 4: training (A1) against tools (W3), next to P2 and Phase 3.

    python scripts/phase4/score_baziqa4.py     # writes results/phase4/baziqa/summary.json

BaziQA was inspected in Phases 2 and 3: it is an existing benchmark here, not an untouched
test, and nothing was tuned on its answer keys. Accuracy is agreement with contest keys,
not evidence that destiny can be predicted. Unanswered replies count as wrong.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from bazireason import baziqa as B

P3 = Path("results/phase3/baziqa1024")
P4 = Path("results/phase4/baziqa")
CONDITIONS = {
    "base_no_tools": P3 / "base/base_no_tools.jsonl",
    "base_tools": P3 / "base/base_tools.jsonl",
    "p2_no_tools": P3 / "p2/tuned_no_tools.jsonl",
    "p2_tools": P3 / "p2/tuned_tools.jsonl",
    "p3r2_no_tools": P3 / "p3/tuned_no_tools.jsonl",
    "p3r2_tools": P3 / "p3/tuned_tools.jsonl",
    "p3r3_no_tools": P3 / "p3r3/tuned_no_tools.jsonl",
    "p3r3_tools": P3 / "p3r3/tuned_tools.jsonl",
    "p2_w3": P4 / "p2/tuned_tools_w3.jsonl",
    "p3r3_w3": P4 / "p3r3/tuned_tools_w3.jsonl",
    "p4a1_no_tools": P4 / "p4a1/tuned_no_tools.jsonl",
    "p4a1_tools": P4 / "p4a1/tuned_tools.jsonl",
    "p4a1_w3": P4 / "p4a1/tuned_tools_w3.jsonl",
}
PAIRS = (
    ("p4a1_no_tools", "p3r3_no_tools", "approach A training, no tools"),
    ("p4a1_tools", "p3r3_tools", "approach A training, verified chart"),
    ("p2_w3", "p2_tools", "approach B contest guard on Phase 2 (the selected adapter)"),
    ("p3r3_w3", "p3r3_tools", "approach B contest guard on R3"),
    ("p4a1_w3", "p4a1_tools", "approach B contest guard on A1"),
    ("p4a1_w3", "p3r3_tools", "A and B together, against R3 with the verified chart"),
    ("p4a1_no_tools", "p2_no_tools", "A1 against Phase 2, no tools"),
    ("p4a1_tools", "p2_tools", "A1 against Phase 2, verified chart"),
    ("p4a1_w3", "p2_tools", "A1 with W3 against Phase 2 with the verified chart"),
    ("p3r3_w3", "p2_tools", "R3 with W3 against Phase 2 with the verified chart"),
)
TRACKS = ("full", "bazi_only", "official_key")
LOOP = re.compile(r"(.{20,}?)\1\1", re.S)
TITLE = re.compile(r"《([^》]{1,12})》")
CARD_WORKS = {"渊海子平", "三命通会", "滴天髓", "穷通宝鉴", "子平真诠"}


def read(path):
    runs, records = defaultdict(dict), []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        runs[record["run"]][record["question_id"]] = record["answer"]
        records.append(record)
    return dict(runs), records


def main():
    persons = B.load_contest("external/BaziQA")
    tracks = B.tracks(persons)
    runs, records = {}, {}
    for name, path in CONDITIONS.items():
        if path.exists():
            runs[name], records[name] = read(path)
    out = {"note": __doc__.strip().splitlines()[-3:], "conditions": {}, "comparisons": {}}
    for name in runs:
        block = {}
        for track in TRACKS:
            table = B.accuracy_table(persons, runs[name], tracks[track])
            block[track] = {
                "accuracy_mean": table["accuracy_mean"],
                "interval95": B.accuracy_interval(persons, runs[name], tracks[track]),
                "n_runs": len(runs[name]),
            }
        rows = records[name]
        texts = [r["raw_output"] or "" for r in rows]
        block["replies"] = len(rows)
        block["unanswered"] = sum(not r["parseable"] for r in rows)
        block["loops"] = sum(bool(LOOP.search(t)) for t in texts)
        block["loop_guard_used"] = sum(bool(r.get("loop_guard_used")) for r in rows)
        block["names_a_book"] = sum(bool(TITLE.search(t)) for t in texts)
        block["names_a_book_not_in_cards"] = sum(
            any(x not in CARD_WORKS for x in TITLE.findall(t)) for t in texts
        )
        block["mean_tokens"] = round(sum(r["completion_tokens"] for r in rows) / len(rows))
        out["conditions"][name] = block
    for second, first, label in PAIRS:
        if first in runs and second in runs:
            out["comparisons"][f"{second}_minus_{first}"] = {
                "effect_of": label,
                **{
                    t: B.paired_by_person(persons, runs[first], runs[second], tracks[t])
                    for t in TRACKS
                },
            }
    P4.mkdir(parents=True, exist_ok=True)
    (P4 / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    for name, block in out["conditions"].items():
        print(
            f"{name:15s} {block['full']['accuracy_mean']:.3f} "
            f"unanswered {block['unanswered']} loops {block['loops']} "
            f"books {block['names_a_book']}/{block['names_a_book_not_in_cards']}"
        )
    for key, value in out["comparisons"].items():
        full = value["full"]
        print(key, round(full["difference"], 3), [round(x, 3) for x in full["interval95"]])


if __name__ == "__main__":
    main()
