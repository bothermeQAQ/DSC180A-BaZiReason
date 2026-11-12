"""Build the Phase 4 validation split and the sealed, untouched final test split.

    python scripts/phase4/build_splits.py            # write both (refuses to overwrite)
    python scripts/phase4/build_splits.py --check    # regenerate in memory, compare hashes

Each split has three parts, all from the split's own year block and with new seeds:
  p3.jsonl     Phase 3 rule-reasoning families (the Phase 3 test generator, version 1)
  p2.jsonl     Phase 2 calendar families (the Phase 2 generator, version 2)
  indep.jsonl  rule questions labelled by lunar-python in wordings never used in training
Prompts identical to any earlier inspected row are dropped. For the final test only
counts and hashes are printed; its rows are scored once, after the selection in
docs/phase4_preregistration.md, by scripts/phase4/run_eval.py --final.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from bazireason import indep_rules as I
from bazireason import train_data as TD
from bazireason import train_data3 as P3

OUT = Path("data/phase4")
PLAN = {
    "valid": {"split": "valid", "p3": (2026111401, 1.0), "p2": (2026111402, 1.5), "indep": (2026111403, 140)},
    "final_test": {"split": "test", "p3": (2026111411, 1.0), "p2": (2026111412, 1.0), "indep": (2026111413, 280)},
}  # fmt: skip
INSPECTED = {
    "valid": (
        "data/phase3/train_v1/valid.jsonl",
        "data/phase3/train_v2/valid.jsonl",
        "data/phase3/train_v3/valid.jsonl",
        "data/phase3/dev_v1/dev.jsonl",
        "data/phase2/train_v1/valid.jsonl",
        "data/phase2/train_v2/valid.jsonl",
    ),
    "final_test": (
        "data/phase3/train_v1/test.jsonl",
        "data/phase2/train_v1/test.jsonl",
        "data/phase2/train_v2/test.jsonl",
        "data/phase3/p2check_v1/test.jsonl",
    ),
}


def user_prompts(path):
    prompts = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        prompts.add(next(m["content"] for m in row["messages"] if m["role"] == "user"))
    return prompts


def first_user(row):
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def build(name):
    plan = PLAN[name]
    split = plan["split"]
    blocked = TD.benchmark_exclusions() | TD.contest_exclusions()
    seen = set().union(*(user_prompts(p) for p in INSPECTED[name] if Path(p).exists()))
    seed, scale = plan["p3"]
    p3 = [
        r
        for r in P3.generate_split(split, seed=seed, blocked=blocked, scale=scale, version=1)
        if not r["family"].startswith("replay_") and first_user(r) not in seen
    ]
    seed, scale = plan["p2"]
    p2 = [
        r
        for r in TD.generate_split(split, seed, blocked, scale, 2)
        if first_user(r) not in seen and "expected" in r["meta"]
    ]
    seed, count = plan["indep"]
    indep = I.build(split, count, seed, blocked, frozenset(seen))
    for part, rows in (("p3", p3), ("p2", p2)):
        for index, row in enumerate(rows, 1):
            row["id"] = f"P4{name[0].upper()}-{part}-{index:05d}"
    for row in indep:
        row["id"] = row["id"].replace("IND-", f"P4{name[0].upper()}-ind-")
    return {"p3": p3, "p2": p2, "indep": indep}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    bad = False
    for name in PLAN:
        parts = build(name)
        folder = OUT / name
        manifest = {"split_years": PLAN[name]["split"], "plan": PLAN[name], "files": {}}
        for part, rows in parts.items():
            payload = TD.dataset_bytes(rows) if part != "indep" else I.dumps(rows).encode()
            digest = hashlib.sha256(payload).hexdigest()
            path = folder / f"{part}.jsonl"
            manifest["files"][f"{part}.jsonl"] = {
                "rows": len(rows),
                "sha256": digest,
                "families": dict(sorted(Counter(r["family"] for r in rows).items())),
            }
            if args.check:
                same = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
                bad |= not same
                print(f"{path}: {'ok' if same else 'MISMATCH'}")
                continue
            if path.exists():
                parser.error(f"{path} exists; Phase 4 splits are never rewritten")
            folder.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            print(f"{path}: {len(rows)} rows sha256 {digest[:16]}")
        if not args.check:
            manifest["inspected_prompts_excluded_from"] = INSPECTED[name]
            (folder / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
            )
    return int(bad)


if __name__ == "__main__":
    raise SystemExit(main())
