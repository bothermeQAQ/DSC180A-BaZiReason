"""Run Phase 5 conditions for one or more adapters on one split (real inference, greedy).

    .venv-mlx/bin/python scripts/phase5/run5.py --split dev --labels p3r3 p4a1 base \
        --parts rules indep blind --conditions gf go gr grv
    .venv-mlx/bin/python scripts/phase5/run5.py --split valid --labels mix:p3r3:p4a1:0.25 \
        --parts calendar --conditions w0

One base model is loaded once and adapters are swapped in place (bazireason.adapter_bank).
Outputs: results/phase5/<split>/<label>/<part>_<condition>.jsonl, with a manifest and a
score summary; nothing is overwritten. Replies to byte-identical prompts are reused from
results/phase5/cache/<label>.jsonl (greedy decoding), so a routed run only generates the
items whose prompt differs from the oracle run.

The sealed split needs results/phase5/selection.json, and every access is appended to
results/phase5/sealed_access.jsonl.
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from bazireason import grounded as G
from bazireason import p5_eval as E
from bazireason.p2_eval import score_row

ROOT = Path("results/phase5")
MAX_TOKENS = {"rules": 800, "indep": 600, "blind": 800, "calendar": 700}
SAFE = str.maketrans({":": "-", "/": "-"})


def load_items(split, part):
    if split == "dev" and part == "calendar":
        rows = [json.loads(x) for x in Path("data/phase4/final_test/p2.jsonl").read_text(encoding="utf-8").splitlines()]
        return [r for r in rows if r["task"] == "chart"]
    if split == "dev" and part in ("rules", "indep"):
        rows = [json.loads(x) for x in Path(f"data/phase4/final_test/{'p3' if part == 'rules' else 'indep'}.jsonl").read_text(encoding="utf-8").splitlines()]
        if part == "rules":
            out = []
            for row in rows:
                intent = E.template_intent(row)
                if intent:
                    row["meta"] = dict(row["meta"], intent=intent)
                    out.append(row)
            return out
        return [E.indep_item(r) for r in rows]
    if part == "blind":
        return [json.loads(x) for x in Path(f"data/phase5/blind/{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [json.loads(x) for x in Path(f"data/phase5/{split}/{part}.jsonl").read_text(encoding="utf-8").splitlines()]
    if part == "indep":
        return [E.indep_item(r) for r in rows]
    return rows


def sha(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class Cache:
    def __init__(self, label):
        self.path = ROOT / "cache" / f"{label.translate(SAFE)}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                self.store[(r["sha"], r["max_tokens"])] = r["text"]

    def generate(self, bank, conversations, max_tokens, batch):
        keys = [sha(c) for c in conversations]
        todo = {}
        for key, conv in zip(keys, conversations, strict=True):
            if (key, max_tokens) not in self.store and key not in todo:
                todo[key] = conv
        started = time.perf_counter()
        totals = {}
        if todo:
            texts, totals = bank.generate_batch(list(todo.values()), max_tokens, batch=batch, share_prefix=True)
            with self.path.open("a", encoding="utf-8") as stream:
                for key, text in zip(todo, texts, strict=True):
                    self.store[(key, max_tokens)] = text
                    stream.write(json.dumps({"sha": key, "max_tokens": max_tokens, "text": text}, ensure_ascii=False) + "\n")
        return [self.store[(k, max_tokens)] for k in keys], {
            "generated": len(todo), "reused": len(conversations) - len(todo),
            "seconds": round(time.perf_counter() - started, 1),
            "completion_tokens": totals.get("completion_tokens"),
        }  # fmt: skip


def score_item(part, row, text):
    if part == "calendar":
        return score_row(row, text)
    return E.score5(row, text)


def run_condition(bank, cache, label, split, part, condition, rows, batch):
    meta_of = lambda r: r["meta"]  # noqa: E731
    plans = {}
    if condition in ("go", "gr", "grv"):
        for r in rows:
            plans[r["id"]] = G.oracle_plan(r) if condition == "go" else G.router_plan(r)
    convs = [G.messages_for(r, "w0" if part == "calendar" else condition, plans.get(r["id"])) for r in rows]
    if condition == "grv":
        texts, stats = cache.generate(bank, [G.messages_for(r, "gr", plans[r["id"]]) for r in rows], MAX_TOKENS[part], batch)
    else:
        texts, stats = cache.generate(bank, convs, MAX_TOKENS[part], batch)
    records = []
    repair_jobs = {}
    for r, conv, text in zip(rows, convs, texts, strict=True):
        m = meta_of(r)
        rec = {"id": r["id"], "label": label, "condition": condition, "part": part,
               "intent_oracle": m.get("intent"), "prompt_sha256": sha(conv)}  # fmt: skip
        if part != "calendar":
            plan = plans.get(r["id"])
            if plan:
                rec["intent_used"] = plan["intent"]
                rec["routed_ok"] = plan["intent"] == m.get("intent")
            check_plan = plan or G.oracle_plan(r)
            rec["check"] = G.verify(check_plan, text, m["pillars"], m.get("gender"))
            if condition == "grv" and not rec["check"]["ok"]:
                repair_jobs[r["id"]] = G.repair_messages(conv, text, check_plan, rec["check"])
        rec["raw_output"] = text
        records.append(rec)
    if repair_jobs:
        ids = list(repair_jobs)
        second, rstats = cache.generate(bank, [repair_jobs[i] for i in ids], MAX_TOKENS[part], batch)
        stats["repair"] = rstats
        second = dict(zip(ids, second, strict=True))
        for rec, r in zip(records, rows, strict=True):
            if rec["id"] not in second:
                rec["final_text"], rec["repaired"], rec["fallback"] = rec["raw_output"], False, False
                continue
            m = r["meta"]
            plan = plans[r["id"]]
            check2 = G.verify(plan, second[rec["id"]], m["pillars"], m.get("gender"))
            rec.update(repaired=True, second_output=second[rec["id"]], second_check=check2)
            if check2["ok"]:
                rec["final_text"], rec["fallback"] = second[rec["id"]], False
            else:
                has_reference = plan.get("reference") is not None
                rec["final_text"] = G.engine_answer(plan) if has_reference else second[rec["id"]]
                rec["fallback"] = has_reference
    elif condition == "grv":
        for rec in records:
            rec["final_text"], rec["repaired"], rec["fallback"] = rec["raw_output"], False, False
    for rec, r in zip(records, rows, strict=True):
        rec["score"] = score_item(part, r, rec["raw_output"])
        if condition == "grv":
            rec["final_score"] = score_item(part, r, rec["final_text"])
            m = r["meta"]
            rec["final_contradictions"] = [] if rec["fallback"] else G.verify({"reference": None, "family": "unknown"}, rec["final_text"], m["pillars"], m.get("gender"))["contradictions"]
    return records, stats


def summary(records, condition):
    n = len(records)
    out = {"n": n, "correct": sum(bool(r["score"]["correct"]) for r in records)}
    if records and "check" in records[0]:
        out["with_contradiction"] = sum(bool(r["check"]["contradictions"]) for r in records)
        out["conclusion_disagrees"] = sum(r["check"]["conclusion_agrees"] is False for r in records)
    if records and "routed_ok" in records[0]:
        out["routed_ok"] = sum(bool(r["routed_ok"]) for r in records)
    if condition == "grv":
        out["final_correct"] = sum(bool(r["final_score"]["correct"]) for r in records)
        out["repaired"] = sum(bool(r["repaired"]) for r in records)
        out["fallback"] = sum(bool(r["fallback"]) for r in records)
        out["final_with_contradiction"] = sum(bool(r["final_contradictions"]) for r in records)
    by = Counter()
    tot = Counter()
    for r in records:
        key = (r.get("intent_oracle") or r["score"].get("family") or "").split(":")[0]
        if key == "fact":
            key = "fact:" + r["intent_oracle"].split(":")[1]
        tot[key] += 1
        by[key] += bool(r["score"]["correct"])
    out["by_intent"] = {k: f"{by[k]}/{tot[k]}" for k in sorted(tot)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "valid", "sealed"), required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--parts", nargs="+", default=["rules", "indep", "blind"])
    parser.add_argument("--conditions", nargs="+", default=["w0"], choices=G.CONDITIONS)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--prompt-version", type=int, default=1, choices=(1, 2))
    parser.add_argument("--limit", type=int, default=0, help="first N items per part (prompt tests)")
    args = parser.parse_args()
    G.PROMPT_VERSION["value"] = args.prompt_version
    tag = "" if args.prompt_version == 1 else f"_p{args.prompt_version}"
    tag += f"_n{args.limit}" if args.limit else ""
    if args.split == "sealed":
        if not (ROOT / "selection.json").exists():
            parser.error("The sealed split stays closed until results/phase5/selection.json exists")
        with (ROOT / "sealed_access.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"utc": datetime.now(UTC).isoformat(timespec="seconds"), "labels": args.labels,
                                     "parts": args.parts, "conditions": args.conditions}) + "\n")  # fmt: skip
    from bazireason.adapter_bank import AdapterBank

    bank = None
    for label in args.labels:
        cache = Cache(label)
        for part in args.parts:
            rows = load_items(args.split, part)
            if args.limit:
                rows = rows[:: max(1, len(rows) // args.limit)][: args.limit]
            for condition in (["w0"] if part == "calendar" else args.conditions):
                out_dir = ROOT / args.split / label.translate(SAFE)
                out = out_dir / f"{part}_{condition}{tag}.jsonl"
                if out.exists():
                    print(f"skip {out} (exists)")
                    continue
                if bank is None:
                    bank = AdapterBank()
                bank.use(label)
                out_dir.mkdir(parents=True, exist_ok=True)
                records, stats = run_condition(bank, cache, label, args.split, part, condition, rows, args.batch)
                with out.open("x", encoding="utf-8") as stream:
                    for rec in records:
                        stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
                info = {"split": args.split, "label": label, "part": part, "condition": condition,
                        "n_rows": len(rows), "max_tokens": MAX_TOKENS[part], "temperature": 0,
                        "stats": stats, "summary": summary(records, condition)}  # fmt: skip
                Path(str(out) + ".summary.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n")
                print(json.dumps({k: info[k] for k in ("label", "part", "condition")} | {"summary": {k: v for k, v in info["summary"].items() if k != "by_intent"}, "stats": stats}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
