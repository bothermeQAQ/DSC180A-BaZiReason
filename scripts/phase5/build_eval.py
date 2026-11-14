"""Build the Phase 5 validation and sealed splits (rules, independent questions, calendar).

    python scripts/phase5/build_eval.py            # write both (refuses to overwrite)
    python scripts/phase5/build_eval.py --check    # regenerate in memory, compare hashes

Parts (the blind-wording part is made by gen_blind.py, conversations by build_convo.py):
  rules.jsonl     Phase 3 rule families that have an exact oracle plan (p5_eval.template_intent)
  indep.jsonl     rule questions labelled by lunar-python (bazireason.indep_rules)
  calendar.jsonl  Phase 2 calendar requests (task == "chart"), for the model-alone track

Every prompt used by any earlier split (training, validation, test, development, Phase 4
validation and final test) is excluded. For the sealed split only counts and hashes are
printed. Sizes follow docs/phase5_preregistration.md (power for the calendar margin).
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from bazireason import indep_rules as I
from bazireason import train_data as TD
from bazireason import train_data3 as P3
from bazireason.p5_eval import template_intent

OUT = Path("data/phase5")
PLAN = {
    "valid": {"split": "valid", "rules": (2026111511, 2.0), "calendar": (2026111512, 3.6), "indep": (2026111513, 140)},
    "sealed": {"split": "test", "rules": (2026111521, 1.4), "calendar": (2026111522, 3.3), "indep": (2026111523, 280)},
}  # fmt: skip
USED = sorted(
    {str(p) for p in Path("data").glob("phase[234]/*/*.jsonl") if "knowledge" not in str(p)}
)


def first_user(row):
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def used_prompts():
    seen = set()
    for path in USED:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            seen.add(first_user(json.loads(line)))
    return seen


def build(name, seen):
    plan = PLAN[name]
    split = plan["split"]
    blocked = TD.benchmark_exclusions() | TD.contest_exclusions()
    tag = name[0].upper()
    seed, scale = plan["rules"]
    rules = []
    for row in P3.generate_split(split, seed=seed, blocked=blocked, scale=scale, version=1):
        if row["family"].startswith("replay_") or first_user(row) in seen:
            continue
        intent = template_intent(row)
        if intent:
            row["meta"] = dict(row["meta"], intent=intent)
            rules.append(row)
    seed, scale = plan["calendar"]
    calendar = [
        r
        for r in TD.generate_split(split, seed, blocked, scale, 2)
        if r["task"] == "chart" and "expected" in r["meta"] and first_user(r) not in seen
    ]
    seed, count = plan["indep"]
    indep = I.build(split, count, seed, blocked, frozenset(seen))
    for part, rows in (("rules", rules), ("calendar", calendar)):
        for index, row in enumerate(rows, 1):
            row["id"] = f"P5{tag}-{part}-{index:05d}"
    for row in indep:
        row["id"] = row["id"].replace("IND-", f"P5{tag}-ind-")
    return {"rules": rules, "calendar": calendar, "indep": indep}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--splits", nargs="+", default=list(PLAN))
    args = parser.parse_args()
    seen = used_prompts()
    bad = False
    for name in args.splits:
        parts = build(name, seen)
        folder = OUT / name
        manifest = {
            "split_years": PLAN[name]["split"],
            "plan": PLAN[name],
            "excluded_prompts_from": USED,
            "files": {},
        }
        for part, rows in parts.items():
            payload = TD.dataset_bytes(rows) if part != "indep" else I.dumps(rows).encode()
            digest = hashlib.sha256(payload).hexdigest()
            path = folder / f"{part}.jsonl"
            key = "family" if part != "calendar" else "family"
            manifest["files"][f"{part}.jsonl"] = {
                "rows": len(rows),
                "sha256": digest,
                "families": dict(sorted(Counter(r[key] for r in rows).items())),
            }
            if args.check:
                same = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
                bad |= not same
                print(f"{path}: {'ok' if same else 'MISMATCH'}")
                continue
            if path.exists():
                parser.error(f"{path} exists; Phase 5 splits are never rewritten")
            folder.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            print(f"{path}: {len(rows)} rows sha256 {digest[:16]}")
        if not args.check:
            (folder / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
            )
    return int(bad)


if __name__ == "__main__":
    raise SystemExit(main())
