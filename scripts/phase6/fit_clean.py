"""Fit share on the fit questions a model never trained on (CPU only; reads saved replies).

    PYTHONPATH=src .venv/bin/python scripts/phase6/fit_clean.py --label p6u1
        -> results/phase6/fit_p6u1_clean.json

Fit sample A (own_replies.py) is not all fresh for S2, S3 and U1: about 45% of it are
training-writer questions on their original charts, which these runs train on, and S1's
replies to its first part are the input of the self-check rows. This script scores only the
questions whose chart is in none of the model's training files (its own and its ancestors').
The U1-hi rule reads this share (pre-registration, section 9). Sample B is scored the same
way by own_replies.py --sample b.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/phase6")
import build_train6  # noqa: E402

from bazireason import prompts6 as P6  # noqa: E402
from bazireason import train_data6 as T6  # noqa: E402

ROOT = Path("results/phase6")


def clean(label, sample="a"):
    suffix = "" if sample == "a" else "_b"
    recipe = P6.recipe_of(label)
    seen = build_train6.training_charts(recipe) if recipe else set()
    right, total = Counter(), Counter()
    left_out = 0
    path = ROOT / f"own_replies_{label}{suffix}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if T6.chart_of(r["row"]) in seen:
            left_out += 1
            continue
        meta = r["row"]["meta"]
        head = meta.get("head") or T6.head_of(meta["intent"]) or meta["intent"].split(":")[0]
        total[head] += 1
        right[head] += bool(r["correct"])
    n = sum(total.values())
    return {
        "label": label,
        "sample": sample,
        "n": n,
        "correct": sum(right.values()),
        "share": round(sum(right.values()) / n, 4) if n else None,
        "left_out_chart_in_training": left_out,
        "by_type": {k: [right[k], total[k]] for k in sorted(total)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    out = ROOT / f"fit_{args.label}_clean.json"
    if not out.exists():
        out.write_text(json.dumps(clean(args.label), ensure_ascii=False, indent=1) + "\n")
    print(out.read_text().strip())


if __name__ == "__main__":
    main()
