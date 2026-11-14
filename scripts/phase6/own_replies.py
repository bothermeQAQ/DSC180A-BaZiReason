"""A model's own L0 replies on fresh training-distribution questions.

    .venv-mlx/bin/python scripts/phase6/own_replies.py --label p6s1 --n 2400
        -> results/phase6/own_replies_p6s1.jsonl   (row, reply, correct: the input of S3's self-check rows)
        -> results/phase6/fit_p6s1.json            (accuracy by question type: under-fitting check)

The questions come from the Phase 6 procedure generator (template wordings and the training
writer's paraphrases) on new training-year charts, with a seed no training set uses. A
reply is right when its final JSON matches the rule code (the Phase 5 scorer).
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/phase5")
sys.path.insert(0, "scripts/phase6")
import build_train6  # noqa: E402
import run6  # noqa: E402

from bazireason import train_data6 as T6  # noqa: E402
from bazireason.intent import plan_for  # noqa: E402
from bazireason.ladder import score  # noqa: E402
from bazireason.p5_eval import make_item  # noqa: E402

SEED = 2026111790


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--n", type=int, default=2400)
    args = parser.parse_args()
    out = Path("results/phase6") / f"own_replies_{args.label}.jsonl"
    fit = Path("results/phase6") / f"fit_{args.label}.json"
    if out.exists():
        sys.exit(f"{out} exists")
    rng = random.Random(f"own:{SEED}")
    charts = T6.Charts(SEED, build_train6.exclusions())
    paraphrases = [json.loads(x) for x in Path("data/phase6/blind/train6.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = []
    while len(rows) < args.n:
        rows += T6.procedure_rows(200_000, charts, rng, paraphrases, 0.45, styled=True)
    rows = rows[: args.n]
    cache = run6.Cache6(args.label)
    texts, stats = cache.generate(run6.Bank().get(args.label), [r["messages"][:2] for r in rows], run6.RULE_TOKENS, 32)
    by, tot = Counter(), Counter()
    with out.open("x", encoding="utf-8") as stream:
        for i, (r, text) in enumerate(zip(rows, texts, strict=True)):
            meta = r["meta"]
            plan = plan_for(meta["intent"], meta["pillars"], meta["gender"])
            item = make_item(f"own-{i}", "train", meta.get("source", "template"), meta["pillars"], meta["gender"],
                             meta["question"], meta["intent"], plan, f"own-{i}")  # fmt: skip
            ok = bool(score(item, text)["correct"])
            head = meta["intent"].split(":")[0] + (":" + meta["intent"].split(":")[1] if meta["intent"].startswith("fact") else "")
            by[head] += ok
            tot[head] += 1
            stream.write(json.dumps({"row": {k: v for k, v in r.items() if k != "_tokens"}, "reply": text, "correct": ok},
                                    ensure_ascii=False) + "\n")  # fmt: skip
    result = {"label": args.label, "n": len(rows), "correct": sum(by.values()), "share": round(sum(by.values()) / len(rows), 4),
              "by_type": {k: [by[k], tot[k]] for k in sorted(tot)}, "stats": stats}  # fmt: skip
    fit.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
