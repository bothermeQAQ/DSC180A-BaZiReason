"""Retention panel: a fixed small dev6 panel scored at saved checkpoints of a training run.

    .venv-mlx/bin/python scripts/phase6/panel6.py --label p6u1 --every 2000

The panel: the first 4 dev6 items (by id) of every question type over the rule parts, and 40
calendar rows (fixed seed), all at L0 with the same prompts and caps as run6.py. Checkpoints:
every iteration that is a multiple of --every, and the last. Writes
results/phase6/panel/<label>.jsonl (one line per checkpoint: accuracy by skill). Diagnostic
only: never used for selection.
"""

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/phase5")
sys.path.insert(0, "scripts/phase6")
import run6  # noqa: E402

from bazireason import ladder as L  # noqa: E402
from bazireason.adapter_bank import ADAPTER_ROOT  # noqa: E402

PANEL = Path("artifacts/panel")


def skill(item):
    if item.get("part") == "calendar":
        return "calendar"
    intent = item["meta"].get("intent") or item.get("family") or ""
    head = intent.split(":")[0]
    return f"fact:{intent.split(':')[1]}" if head == "fact" else head


def panel_items():
    items, seen = [], Counter()
    pool = [
        it for p in ("rules", "indep", "blind", "combo", "cf") for it in run6.load_items("dev6", p)
    ]
    for it in sorted(pool, key=lambda x: x["id"]):
        k = skill(it)
        if seen[k] < 4:
            seen[k] += 1
            items.append(it)
    cal = run6.load_items("dev6", "calendar")
    items += random.Random("panel:2026111800").sample(cal, 40)
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--every", type=int, default=1000)
    args = parser.parse_args()
    folder = ADAPTER_ROOT / run6.FOLDERS[args.label]
    numbered = sorted(folder.glob("[0-9]*_adapters.safetensors"))
    last = int(numbered[-1].name.split("_")[0])
    chosen = [
        p
        for p in numbered
        if int(p.name.split("_")[0]) % args.every == 0 or int(p.name.split("_")[0]) == last
    ]
    out = run6.ROOT / "panel" / f"{args.label}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = (
        {json.loads(x)["iteration"] for x in out.read_text().splitlines()}
        if out.exists()
        else set()
    )
    items = panel_items()
    bank = run6.Bank()
    for path in chosen:
        it_no = int(path.name.split("_")[0])
        if it_no in done:
            continue
        work = PANEL / f"{args.label}-{it_no:07d}"
        work.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / "adapter_config.json", work / "adapter_config.json")
        link = work / "adapters.safetensors"
        if not link.exists():
            link.symlink_to(path.resolve())
        name = str(work.resolve())
        by, tot = Counter(), Counter()
        for cap, group in ((run6.RULE_TOKENS, [i for i in items if skill(i) != "calendar"]),
                           (run6.MAX_TOKENS["calendar"], [i for i in items if skill(i) == "calendar"])):  # fmt: skip
            texts, _ = run6.Cache6(f"panel-{args.label}-{it_no}").generate(
                bank.get(name), [run6.prompt_of(i, "L0") for i in group], cap, 32
            )
            for i, text in zip(group, texts, strict=True):
                k = skill(i)
                tot[k] += 1
                by[k] += bool(L.score(i, text)["correct"])
        rule_n = sum(v for k, v in tot.items() if k != "calendar")
        rule_k = sum(v for k, v in by.items() if k != "calendar")
        line = {"label": args.label, "iteration": it_no, "rule": [rule_k, rule_n], "calendar": [by["calendar"], tot["calendar"]],
                "by_skill": {k: [by[k], tot[k]] for k in sorted(tot)}}  # fmt: skip
        with out.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(json.dumps(line, ensure_ascii=False)[:300], flush=True)


if __name__ == "__main__":
    main()
