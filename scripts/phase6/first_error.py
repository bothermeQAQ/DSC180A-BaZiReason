"""Where a small-step reply first leaves the rule code's derivation (CPU only; saved replies).

    PYTHONPATH=src .venv/bin/python scripts/phase6/first_error.py --split dev6 --labels p6s1
        -> results/phase6/first_error/<split>_<label>.json

For every rule item with a reference derivation (bazireason.steps6, in the derivation version
the model was trained on), the reply is compared with the derivation step by step. A step is
named by the text before its first colon. Reported for the wrong answers, split by wording
(training wording, held-out template wording, natural "blind" wording):

  * the first step whose text differs from the derivation;
  * replies that do not even start with the derivation's first step (another procedure: the
    question was read as another type, or the reply is improvised);
  * replies without a final JSON (cut off, usually a loop);
  * which fields of the final JSON are wrong.

A diagnostic for models trained on small steps; it changes no score.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from bazireason import prompts6 as P6
from bazireason import steps6 as S6
from bazireason.p2_prompts import extract_final_json

ROOT = Path("results/phase6")
PARTS = ("rules", "indep", "blind", "combo", "cf")


def lines_of(text):
    return [x.strip() for x in text.split("\n") if x.strip()]


def label(line):
    """A step's name: the text before its first colon, without the step number."""
    body = re.sub(r"^\d+\.\s*", "", line)
    if body.startswith("结论JSON"):
        return "结论JSON"
    return re.split(r"[：:（(]", body, maxsplit=1)[0][:14]


def kind(intent):
    parts = intent.split(":")
    return ":".join(parts[:2]) if parts[0] in ("fact", "combo", "ambig") else parts[0]


def body(line):
    return re.sub(r"^\d+\.\s*", "", line)


def first_difference(reply, truth):
    """(name of the first differing step, what was written, what the derivation says)."""
    mine, good = lines_of(reply), lines_of(truth)
    if not any(x.startswith("结论JSON") for x in mine):
        return "no final JSON (cut off or loop)", "", ""
    if not mine or label(mine[0]) != label(good[0]):
        return "another procedure from the first step", mine[0][:160] if mine else "", good[0][:160]
    written = {}
    for a in mine:
        written.setdefault(label(a), body(a))
    for b in good:
        name = label(b)
        if name == "结论JSON":
            continue
        if name not in written:
            return f"step missing: {name}", "", body(b)[:160]
        if written[name] != body(b):
            ca, cb = re.split(r"(?<=[；。])", written[name]), re.split(r"(?<=[；。])", body(b))
            j = next((i for i, (x, y) in enumerate(zip(ca, cb, strict=False)) if x != y), min(len(ca), len(cb)) - 1)  # fmt: skip
            return name, ca[j][:160] if j < len(ca) else "", cb[j][:160] if j < len(cb) else ""
    return "only the final JSON differs", mine[-1][:160], good[-1][:160]


def analyse(split, lab):
    recipe = P6.recipe_of(lab)
    S6.set_steps(P6.manifest(recipe).get("steps", 1) if recipe else 1)  # the model's own version
    try:
        return _analyse(split, lab)
    finally:
        S6.set_steps(1)


def _analyse(split, lab):
    items = {}
    for part in PARTS:
        path = Path("data/phase6") / ("blind" if part == "blind" else split) / (f"{split}.jsonl" if part == "blind" else f"{part}.jsonl")  # fmt: skip
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            items[r["id"]] = r
    scored, first, fields, examples = {}, {}, {}, {}
    for part in PARTS:
        path = ROOT / split / lab / f"{part}_L0.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            meta = items[rec["id"]]["meta"]
            intent = meta.get("intent")
            try:
                derived = S6.derive(intent, meta["pillars"], meta.get("gender")) if intent else None
            except KeyError:
                derived = None
            if derived is None:
                continue
            ok = bool(rec["score"]["correct"])
            wording = "blind" if part == "blind" else "held-out" if rec["score"].get("unseen_wording") else "training"  # fmt: skip
            t = kind(intent)
            cell = scored.setdefault(t, {}).setdefault(wording, [0, 0])
            cell[0] += ok
            cell[1] += 1
            if ok:
                continue
            where, wrote, expected = first_difference(rec["raw_output"], derived[0])
            first.setdefault(t, {}).setdefault(wording, Counter())[where] += 1
            try:
                got = extract_final_json(rec["raw_output"])
            except Exception:
                got = None
            if isinstance(got, dict) and isinstance(meta.get("expected"), dict):
                bad = [k for k, v in meta["expected"].items() if got.get(k) != v] or ["(other)"]
                fields.setdefault(t, Counter()).update(bad)
            ex = examples.setdefault(f"{t} | {wording} | {where}", [])
            if len(ex) < 3:
                ex.append({"id": rec["id"], "wrote": wrote, "derivation": expected})
    return {
        "label": lab,
        "split": split,
        "steps_version": S6.steps_version(),
        "right_of_scored_by_wording": dict(sorted(scored.items())),
        "first_difference_of_wrong_answers": {
            t: {w: dict(c.most_common()) for w, c in v.items()} for t, v in sorted(first.items())
        },
        "wrong_json_fields": {t: dict(c.most_common()) for t, c in sorted(fields.items())},
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev6", choices=("dev6", "valid6", "sealed6"))
    parser.add_argument("--labels", nargs="+", required=True)
    args = parser.parse_args()
    if args.split == "sealed6" and not (ROOT / "selection.json").exists():
        sys.exit("The sealed split stays closed until results/phase6/selection.json exists")
    folder = ROOT / "first_error"
    folder.mkdir(exist_ok=True)
    for lab in args.labels:
        out = analyse(args.split, lab)
        (folder / f"{args.split}_{lab}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1) + "\n"
        )
        print(f"## {lab} on {args.split} (derivation version {out['steps_version']})")
        total = Counter()
        for t, by in out["right_of_scored_by_wording"].items():
            cells = [
                f"{w} {by[w][0]}/{by[w][1]}" for w in ("training", "held-out", "blind") if w in by
            ]
            for w, (a, n) in by.items():
                total[(w, "right")] += a
                total[(w, "n")] += n
            wrong = out["first_difference_of_wrong_answers"].get(t, {}).get("training", {})
            tail = ("; training-wording errors start at: " + ", ".join(f"{k} {v}" for k, v in wrong.items())) if wrong else ""  # fmt: skip
            print(f"- {t}: " + ", ".join(cells) + tail)
        print("- all: " + ", ".join(f"{w} {total[(w, 'right')]}/{total[(w, 'n')]}" for w in ("training", "held-out", "blind")))  # fmt: skip


if __name__ == "__main__":
    main()
