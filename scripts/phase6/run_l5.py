"""L5: the Phase 5 hybrid system (router, supplied conclusion, value repair, engine fallback)
on the Phase 6 items, for a given generator. Always reported apart from L0-L4.

    .venv-mlx/bin/python scripts/phase6/run_l5.py --split sealed6 --labels p5g1 p6u1

The system code is Phase 5's, unchanged (scripts/phase5/run5.run_condition, condition grv,
prompt version 2); only the items and the reply limit (1,100 tokens, as at L0-L4) differ.
Writes results/phase6/<split>/<label>/<part>_L5.jsonl. Combination and counterfactual items
are not run: the Phase 5 router has no intent for combinations.
"""

import argparse
import json
import sys
from datetime import UTC, datetime

sys.path.insert(0, "scripts/phase5")
sys.path.insert(0, "scripts/phase6")
import run5  # noqa: E402
import run6  # noqa: E402

from bazireason import grounded as G  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("dev6", "valid6", "sealed6"))
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--parts", nargs="+", default=["rules", "indep", "blind"])
    args = parser.parse_args()
    if args.split == "sealed6":
        if not (run6.ROOT / "selection.json").exists():
            raise SystemExit(
                "The sealed split stays closed until results/phase6/selection.json exists"
            )
        with (run6.ROOT / "sealed_access.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"utc": datetime.now(UTC).isoformat(timespec="seconds"), "labels": args.labels,
                                     "parts": args.parts, "levels": ["L5"]}) + "\n")  # fmt: skip
    G.PROMPT_VERSION["value"] = 2
    for part in ("rules", "indep", "blind"):
        run5.MAX_TOKENS[part] = run6.RULE_TOKENS
    bank = run6.Bank()
    for label in args.labels:
        for part in args.parts:
            out = run6.ROOT / args.split / label / f"{part}_L5.jsonl"
            if out.exists():
                print(f"skip {out} (exists)")
                continue
            items = run6.load_items(args.split, part)
            records, stats = run5.run_condition(
                bank.get(label), run6.Cache6(label), label, args.split, part, "grv", items, 32
            )
            for r in records:
                r["level"] = "L5"
            run6.write(out, records)
            print(
                json.dumps(
                    {"label": label, "part": part, **run5.summary(records, "grv"), "stats": stats},
                    ensure_ascii=False,
                )[:600],
                flush=True,
            )


if __name__ == "__main__":
    main()
