"""Run the Phase 6 evaluation ladder for one or more models.

    .venv-mlx/bin/python scripts/phase6/run6.py --split dev6 --labels p4a1 p5g1 \
        --parts rules indep blind combo cf calendar --levels L0 L1 L2 L3 L3w L4

Writes results/phase6/<split>/<label>/<part>_<level>.jsonl (never overwrites). Replies are
cached by prompt hash in results/phase6/cache/ (not in Git), so a rerun reuses them.

Levels (bazireason.ladder): L0 model alone; L1 raw facts; L2 rule text; L3 the code's
conclusion; L3w a wrong conclusion; L4 self-check of the model's own L0 reply in three
variants (a re-check, b "there is an error", c "this field is wrong"). Calendar rows run at
L0 only. No repair, fallback or routing is applied at any of these levels.

The sealed split stays closed until results/phase6/selection.json exists; every sealed run
is logged in results/phase6/sealed_access.jsonl.
"""

import argparse
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "scripts/phase5")
import run5  # noqa: E402

from bazireason import ladder as L  # noqa: E402
from bazireason import p5_eval as E  # noqa: E402
from bazireason import prompts6 as P6  # noqa: E402
from bazireason import workflow as W  # noqa: E402

ROOT = Path("results/phase6")
DATA = Path("data/phase6")
SAFE = run5.SAFE
MAX_TOKENS = {"calendar": 700}
RULE_TOKENS = 1100
RIGHT_SAMPLE = 60  # right L0 answers per part that also get the L4 checks (break rate)
FOLDERS = {
    "p6u0": "bazireason-qwen3-4b-lora-p6-u0",
    "p6s1": "bazireason-qwen3-4b-lora-p6-s1",
    "p6s2": "bazireason-qwen3-4b-lora-p6-s2",
    "p6s3": "bazireason-qwen3-4b-lora-p6-s3",
    "p6u1": "bazireason-qwen3-4b-lora-p6-u1",
    "p6u1hi": "bazireason-qwen3-4b-lora-p6-u1-hi",
    "p6u2": "bazireason-qwen3-4b-lora-p6-u2",
    **{
        f"p6{r}": f"bazireason-qwen3-4b-lora-p6-{r}"
        for r in ("u3", "u3v", "u4", "u4c", "u5", "u6", "u7", "u8", "u9", "u10")
    },
    "p8s1": "bazireason-qwen3-8b-lora-p6-s1",  # section 11: S1's data on Qwen3-8B
    "p6s2s": "bazireason-qwen3-4b-lora-p6-s2s",  # section 12.4: S2 with the short system prompt
}


class Cache6(run5.Cache):
    """Replies cached by prompt hash. Every prompt gets the system prompt its model was
    trained with (round 2 trains some models with a short one; bazireason.prompts6)."""

    def __init__(self, label):
        self.label = label
        self.path = ROOT / "cache" / f"{label.translate(SAFE)}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                self.store[(r["sha"], r["max_tokens"])] = r["text"]

    def generate(self, bank, conversations, max_tokens, batch):
        return super().generate(bank, P6.apply(self.label, conversations), max_tokens, batch)


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


def load_items(split, part):
    if part == "blind":
        return rows(DATA / "blind" / f"{split}.jsonl")
    data = rows(DATA / split / f"{part}.jsonl")
    if part == "indep":
        return [E.indep_item(r) for r in data]
    if part == "calendar":
        for r in data:
            r["part"] = "calendar"
    return data


def prompt_of(item, level):
    if item.get("part") == "calendar":
        return W.prompt_messages(item)
    return L.messages(item, level, wrong=item.get("_wrong"))


class Bank:
    """One AdapterBank for every adapter of the shared structure; loaded on first use."""

    SHARED = {"num_layers": 16, "rank": 16}

    def __init__(self):
        self.bank = None
        self.single = None

    def get(self, label):
        from bazireason.adapter_bank import ADAPTER_ROOT, DEFAULT_MODEL, AdapterBank
        from bazireason.mlx_backend import MLXChatModel

        name = FOLDERS.get(label, label)
        config = ADAPTER_ROOT / name / "adapter_config.json"
        if config.exists():
            c = json.loads(config.read_text())
            shape = {
                "num_layers": c.get("num_layers"),
                "rank": c.get("lora_parameters", {}).get("rank"),
            }
            base = c.get("model") or DEFAULT_MODEL
            if shape != self.SHARED or base != DEFAULT_MODEL:
                # another adapter structure or base model cannot be swapped in: load it alone
                if self.single is None or self.single.adapter_path != str(ADAPTER_ROOT / name):
                    self.bank = self.single = None
                    self.single = MLXChatModel(base, ADAPTER_ROOT / name, label=label)
                return self.single
        if self.bank is None:
            self.single = None
            self.bank = AdapterBank(first=name if name != "base" else "p4a1")
        return self.bank.use(name)


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for r in records:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")


def tally(label, split, part, level, records, stats):
    """The line printed after a part is scored. valid6 is for selection only: its count stays
    out of the log until selection.json exists (the records are on disk either way)."""
    line = {"label": label, "split": split, "part": part, "level": level, "n": len(records)}
    if split != "valid6" or (ROOT / "selection.json").exists():
        line["correct"] = sum(bool(r["score"]["correct"]) for r in records)
        if level == "L3w":
            line["follows_wrong"] = sum(r["follows_wrong"] for r in records)
    line["stats"] = stats
    return line


def run_level(bank, label, split, part, level, batch):
    out = ROOT / split / label.translate(SAFE) / f"{part}_{level}.jsonl"
    if out.exists():
        print(f"skip {out} (exists)", flush=True)
        return
    items = load_items(split, part)
    cache = Cache6(label)
    max_tokens = MAX_TOKENS.get(part, RULE_TOKENS)
    if level == "L3w":
        kept = []
        for it in items:
            wrong = L.wrong_reference(it, random.Random(f"wrong:{it['id']}"))
            if wrong is not None and wrong != it["meta"]["expected"]:
                kept.append(dict(it, _wrong=wrong))
        items = kept
    if level.startswith("L4"):
        return run_checks(bank, label, split, part, level, batch, cache, items, max_tokens, out)
    convs = [prompt_of(it, level) for it in items]
    texts, stats = cache.generate(bank.get(label), convs, max_tokens, batch)
    records = []
    for it, text in zip(items, texts, strict=True):
        rec = {"id": it["id"], "label": label, "split": split, "part": part, "level": level,
               "intent": it["meta"].get("intent") or it.get("family"), "source": it["meta"].get("source"),
               "raw_output": text, "score": L.score(it, text)}  # fmt: skip
        if level == "L3w":
            rec["wrong_conclusion"] = it["_wrong"]
            rec["follows_wrong"] = bool(L.score(L.with_expected(it, it["_wrong"]), text)["correct"])
        records.append(rec)
    write(out, records)
    print(
        json.dumps(tally(label, split, part, level, records, stats), ensure_ascii=False), flush=True
    )


def run_checks(bank, label, split, part, level, batch, cache, items, max_tokens, out):
    """L4: every wrong L0 answer plus a fixed sample of right ones, three check variants."""
    first_path = ROOT / split / label.translate(SAFE) / f"{part}_L0.jsonl"
    if not first_path.exists():
        raise SystemExit(f"L4 needs {first_path}")
    first = {r["id"]: r for r in rows(first_path)}
    wrong = [it for it in items if not first[it["id"]]["score"]["correct"]]
    right = [it for it in items if first[it["id"]]["score"]["correct"]]
    right = random.Random(f"right:{split}:{part}").sample(right, min(RIGHT_SAMPLE, len(right)))
    chosen = wrong + right
    records = []
    for variant in ("L4a", "L4b", "L4c"):
        convs = [
            L.check_messages(
                it, first[it["id"]]["raw_output"], variant, random.Random(f"field:{it['id']}")
            )
            for it in chosen
        ]
        texts, stats = cache.generate(bank.get(label), convs, max_tokens, batch)
        for it, conv, text in zip(chosen, convs, texts, strict=True):
            was = bool(first[it["id"]]["score"]["correct"])
            now = L.score(it, text)
            records.append({"id": it["id"], "label": label, "split": split, "part": part, "level": variant,
                            "intent": it["meta"].get("intent"), "first_correct": was, "check_request": conv[-1]["content"],
                            "raw_output": text, "score": now,
                            "fixed": (not was) and bool(now["correct"]), "broke": was and not now["correct"]})  # fmt: skip
        print(json.dumps({"label": label, "part": part, "variant": variant, "wrong_first": len(wrong),
                          "fixed": sum(r["fixed"] for r in records if r["level"] == variant),
                          "right_sampled": len(right), "broke": sum(r["broke"] for r in records if r["level"] == variant),
                          "stats": stats}), flush=True)  # fmt: skip
    write(out, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("dev6", "valid6", "sealed6"))
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--parts", nargs="+", default=["rules", "indep", "blind", "combo", "cf"])
    parser.add_argument("--levels", nargs="+", default=["L0"])
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    if args.split == "sealed6":
        if not (ROOT / "selection.json").exists():
            raise SystemExit(
                "The sealed split stays closed until results/phase6/selection.json exists"
            )
        with (ROOT / "sealed_access.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"utc": datetime.now(UTC).isoformat(timespec="seconds"), "labels": args.labels,
                                     "parts": args.parts, "levels": args.levels}) + "\n")  # fmt: skip
    bank = Bank()
    started = time.perf_counter()
    for label in args.labels:
        for part in args.parts:
            levels = ["L0"] if part == "calendar" else args.levels
            for level in levels:
                if level == "L4":
                    run_level(bank, label, args.split, part, "L4", args.batch)
                else:
                    run_level(bank, label, args.split, part, level, args.batch)
    print(f"done in {round(time.perf_counter() - started)} s", flush=True)


if __name__ == "__main__":
    main()
