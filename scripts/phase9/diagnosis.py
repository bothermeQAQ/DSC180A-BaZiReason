"""Summarize archived Phase 7 training inputs and Phase 8 replies before the Phase 9 experiment."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/phase7"))
sys.path.insert(0, str(ROOT / "src"))
from graphs import BRANCHES, OPS, R, calc, parse, rows


STEMS = "甲乙丙丁戊己庚辛壬癸"
GODS = {R.ten_god(a, b) for a in STEMS for b in STEMS}
DERIVED = re.compile(r"(?:可得|所得地支=)([" + "".join(BRANCHES) + "])")
DAY = re.compile(r"日主([" + STEMS + "])")


def training_support():
    result = defaultdict(Counter)
    for record in rows(ROOT / "data/phase7/g/train.jsonl"):
        if record.get("op") in {"stage", "god"}:
            question, answer = record["messages"][1:]
            day = DAY.search(question["content"])[1]
            branch = DERIVED.search(answer["content"])[1]
            result[record["op"]][day, branch] += 1
    return result


def audit(model, cases, support):
    observations = {r["id"]: r for r in rows(ROOT / f"results/phase8/{model}_transfer.jsonl")}
    assert len(cases) == len(observations) == 640
    misses = defaultdict(Counter)
    operand_exposure = defaultdict(Counter)
    for row in cases:
        observation = observations[row["id"]]
        condition, operation = row["condition"], row["op"]
        parsed = parse(observation["raw"])
        if operation in support:
            seen = (row["chart"]["day"][0], row["branch"]) in support[operation]
            key = f"{condition}/{operation}/" + ("previously_seen" if seen else "absent")
            operand_exposure[key].update(total=1, right=int(observation["correct"]))
        if observation["correct"] or condition not in ("familiar", "novel"):
            continue
        steps = misses[condition]
        first = DERIVED.search(observation["raw"].splitlines()[0])
        candidates = []
        for alternative in OPS:
            if alternative != operation:
                value = calc(alternative, row["chart"], row["branch"])
                if type(value) is type(parsed) and value == parsed:
                    candidates.append(alternative)
        steps.update(total=1, licensed_other_operation_matches=int(bool(candidates)),
                     strict_parse_miss=int(parsed is None), first_recognized=int(bool(first)),
                     first_correct=int(bool(first and first[1] == row["branch"])))
        if operation in support:
            in_domain = (isinstance(parsed, str) and
                         parsed in (R.LIFE_STAGES if operation == "stage" else GODS))
            steps[f"{operation}_wrong_in_requested_domain"] += int(in_domain)
            steps[f"{operation}_misses"] += 1
    return {"misses": misses, "operand_exposure": operand_exposure}


if __name__ == "__main__":
    support = training_support()
    cases = list(rows(ROOT / "data/phase8/transfer.jsonl"))
    result = {
        "training_support": {op: {"cells": len(c), "rows": sum(c.values())} for op, c in support.items()},
        "models": {m: audit(m, cases, support) for m in ("g", "t", "a", "b")}}
    print(json.dumps(result, ensure_ascii=False, indent=2))
