"""Under-fitting or coverage? A1 alone on its own training distribution (audit section 3.9).

    .venv-mlx/bin/python scripts/phase6/diag_fit.py     # results/phase6/diag_fit.json

Three samples of Phase 3 rule questions with an exact oracle plan, scored the same way:
  literal  rows of A1's own training file (data/phase4/train_v1), with their exact prompts
  fresh    new rows from the same generator (train_data4, training years, a new seed)
  dev6     the Phase 6 development rule rows (validation years, includes held-out wordings)
Near-perfect on literal and fresh with a low dev6 score means a coverage problem; a low
score on fresh rows means the model under-fits its own training distribution.
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/phase5")
sys.path.insert(0, "scripts/phase6")
import run6  # noqa: E402

from bazireason import train_data4 as T4  # noqa: E402
from bazireason import workflow as W  # noqa: E402
from bazireason.p5_eval import score5, template_intent  # noqa: E402

OUT = Path("results/phase6/diag_fit.json")
N = 240


def sample(rows, rng):
    out = []
    for row in rows:
        intent = template_intent(row)
        if intent and not row["family"].startswith("replay"):
            row["meta"] = dict(row["meta"], intent=intent)
            out.append(row)
    rng.shuffle(out)
    return out[:N]


def main():
    rng = random.Random(20261116)
    literal = sample(
        [
            json.loads(x)
            for x in Path("data/phase4/train_v1/train.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ],
        rng,
    )
    fresh = sample(T4.generate_split("train", seed=2026111699), rng)
    seen_literal = {W.prompt_messages(r)[-1]["content"] for r in literal}
    fresh = [r for r in fresh if W.prompt_messages(r)[-1]["content"] not in seen_literal]
    dev = run6.load_items("dev6", "rules")
    bank = run6.Bank()
    cache = run6.Cache6("p4a1")
    result = {}
    for name, items, prompt in (("literal", literal, W.prompt_messages), ("fresh", fresh, W.prompt_messages),
                                ("dev6", dev, lambda it: run6.prompt_of(it, "L0"))):  # fmt: skip
        texts, stats = cache.generate(
            bank.get("p4a1"), [prompt(it) for it in items], run6.RULE_TOKENS, 32
        )
        scores = [score5(it, t) for it, t in zip(items, texts, strict=True)]
        fam = Counter()
        tot = Counter()
        for it, s in zip(items, scores, strict=True):
            f = it["meta"]["intent"].split(":")[0]
            tot[f] += 1
            fam[f] += bool(s["correct"])
        result[name] = {"n": len(items), "correct": sum(bool(s["correct"]) for s in scores),
                        "by_family": {f: [fam[f], tot[f]] for f in sorted(tot)}, "stats": stats}  # fmt: skip
        print(name, result[name]["correct"], "/", len(items), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
