"""Seed the Phase 5 reply cache with the Phase 4 final-test replies (now development data).

    .venv/bin/python scripts/phase5/seed_cache.py

The Phase 4 W0 runs used the same adapters, greedy decoding and byte-identical prompts
(``workflow.prompt_messages``), so their replies are reused instead of regenerated.
"""

import json
from pathlib import Path

from bazireason import workflow as W

sys_path = Path("scripts/phase5")
import sys  # noqa: E402

sys.path.insert(0, str(sys_path))
from run5 import ROOT, SAFE, sha  # noqa: E402

PARTS = {
    "p3": ("data/phase4/final_test/p3.jsonl", 800),
    "indep": ("data/phase4/final_test/indep.jsonl", 600),
    "p2": ("data/phase4/final_test/p2.jsonl", 700),
}


def main():
    for label in ("base", "p2", "p3r2", "p3r3", "p4a1"):
        path = ROOT / "cache" / f"{label.translate(SAFE)}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        have = set()
        if path.exists():
            have = {(r["sha"], r["max_tokens"]) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
        added = 0
        with path.open("a", encoding="utf-8") as stream:
            for part, (data, max_tokens) in PARTS.items():
                rows = {r["id"]: r for r in map(json.loads, Path(data).read_text(encoding="utf-8").splitlines())}
                out = Path(f"results/phase4/final_test/{label}_{part}.jsonl")
                for rec in map(json.loads, out.read_text(encoding="utf-8").splitlines()):
                    key = (sha(W.prompt_messages(rows[rec["id"]])), max_tokens)
                    if key in have:
                        continue
                    have.add(key)
                    added += 1
                    stream.write(json.dumps({"sha": key[0], "max_tokens": max_tokens, "text": rec["raw_output"],
                                             "from": str(out)}, ensure_ascii=False) + "\n")  # fmt: skip
        print(label, "added", added)


if __name__ == "__main__":
    main()
