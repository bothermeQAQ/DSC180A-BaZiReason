"""Replies that reached the token cap, for every saved Phase 6 record file.

    PYTHONPATH=src .venv-mlx/bin/python scripts/phase6/truncation6.py   -> results/phase6/truncation.json

A reply counts as cut off when its text re-encodes to at least the cap minus 5 tokens (caps:
1,100 for rule items at every level, 700 for calendar rows). Re-encoding a decoded reply can
differ from the generated count by a token or two, hence the margin. Uses the tokenizers
package (in .venv-mlx); make_report6.py reads the result.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/phase6")
from bazireason import train_data6 as T6  # noqa: E402

ROOT = Path("results/phase6")
CAPS = {"calendar": 700}
RULE_CAP = 1100
MARGIN = 5


def main():
    tok = T6._tokenizer()
    if tok is None:
        sys.exit("needs the tokenizers package (.venv-mlx)")
    out = {}
    for path in sorted(ROOT.glob("*6/*/*_L*.jsonl")):
        split, label = path.parts[-3], path.parts[-2]
        part, level = path.stem.rsplit("_", 1)
        cap = CAPS.get(part, RULE_CAP)
        recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        lengths = [len(tok.encode(r["raw_output"], add_special_tokens=False).ids) for r in recs]
        cut = sum(n >= cap - MARGIN for n in lengths)
        out.setdefault(split, {}).setdefault(label, {})[f"{part}_{level}"] = {
            "n": len(recs),
            "cut_off": cut,
            "cut_off_and_wrong": sum(
                n >= cap - MARGIN and not r["score"]["correct"]
                for n, r in zip(lengths, recs, strict=True)
            ),
            "mean_tokens": round(sum(lengths) / max(len(lengths), 1), 1),
            "cap": cap,
        }
    (ROOT / "truncation.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for split, labs in out.items():
        for label, parts in labs.items():
            cut = sum(v["cut_off"] for v in parts.values())
            n = sum(v["n"] for v in parts.values())
            print(f"{split} {label}: {cut}/{n} cut off")


if __name__ == "__main__":
    main()
