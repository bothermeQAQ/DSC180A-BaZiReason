"""Recompute every stored Phase 5 score from the saved raw replies with the current scorers.

    .venv/bin/python scripts/phase5/rescore5.py [--split dev]

Runs no model. Used once after the scoring rules were final (docs/phase5_log.md), so that
records written before a scorer change use the same rules as later ones. Summaries are
rewritten next to each record file.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run5  # noqa: E402

from bazireason import grounded as G  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", nargs="+", default=["dev", "valid", "sealed"])
    args = parser.parse_args()
    for split in args.split:
        folder = run5.ROOT / split
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*/*.jsonl")):
            part = path.stem.split("_")[0]
            if part not in run5.MAX_TOKENS or path.name.startswith("convo"):
                continue
            rows = {r["id"]: r for r in run5.load_items(split, part)}
            records = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
            changed = 0
            for rec in records:
                row = rows[rec["id"]]
                new = run5.score_item(part, row, rec["raw_output"])
                changed += bool(new["correct"]) != bool(rec["score"]["correct"])
                rec["score"] = new
                if "final_text" in rec:
                    rec["final_score"] = run5.score_item(part, row, rec["final_text"])
                if "check" in rec and part != "calendar":
                    plan = G.router_plan(row) if rec["condition"] in ("gr", "grv") else G.oracle_plan(row)
                    if rec["condition"] in ("go",) or rec["condition"].startswith("go"):
                        plan = G.oracle_plan(row)
                    m = row["meta"]
                    rec["check"] = G.verify(plan, rec["raw_output"], m["pillars"], m.get("gender"))
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
            info_path = Path(str(path) + ".summary.json")
            if info_path.exists():
                info = json.loads(info_path.read_text())
                info["summary"] = run5.summary(records, records[0]["condition"] if records else "w0")
                info["rescored"] = True
                info_path.write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n")
            if changed:
                print(f"{path}: {changed} scores changed")


if __name__ == "__main__":
    main()
