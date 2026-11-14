"""Addendum to Phase 4: the independent questions rescored with a content-lenient 旬空 form.

    .venv/bin/python scripts/phase5/addendum_indep.py   # writes results/phase5/addenda/indep_rescore.json

Reads saved replies only. The Phase 4 scorer (indep_rules.score_indep) accepted a 旬空 or
hidden-stem answer only as a JSON list; replies that wrote the same two branches as one
string ("寅卯") were marked wrong. Phase 4's numbers are not edited; this file adds both.
"""

import json
from pathlib import Path

from bazireason.p5_eval import indep_item, score5

OUT = Path("results/phase5/addenda/indep_rescore.json")


def main():
    out = {"note": __doc__.strip().splitlines()[2:6], "splits": {}}
    for split in ("valid", "final_test"):
        rows = {
            json.loads(x)["id"]: indep_item(json.loads(x))
            for x in Path(f"data/phase4/{split}/indep.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        }
        block = {}
        for label in ("base", "p2", "p3r2", "p3r3", "p4a1"):
            path = Path(f"results/phase4/{split}/{label}_indep.jsonl")
            if not path.exists():
                continue
            recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
            strict = sum(bool(r["score"]["correct"]) for r in recs)
            lenient = sum(bool(score5(rows[r["id"]], r["raw_output"])["correct"]) for r in recs)
            xk = [r for r in recs if r["score"]["kind"] == "xun_kong"]
            block[label] = {"n": len(recs), "strict_phase4": strict, "content_lenient": lenient,
                            "xun_kong_strict": sum(bool(r["score"]["correct"]) for r in xk),
                            "xun_kong_lenient": sum(bool(score5(rows[r["id"]], r["raw_output"])["correct"]) for r in xk)}  # fmt: skip
        out["splits"][split] = block
    out["status"] = ("data/phase4/final_test was read during the Phase 4 review (13 November 2026) and is "
                     "development material in Phase 5; no untouched Phase 4 split remains.")  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(out["splits"], ensure_ascii=False))


if __name__ == "__main__":
    main()
