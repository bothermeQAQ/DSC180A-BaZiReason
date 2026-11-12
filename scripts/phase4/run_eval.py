"""Evaluate one adapter on a Phase 4 split under one workflow (real inference, greedy).

    python scripts/phase4/run_eval.py --label p3r3 --adapter artifacts/adapters/... \
        --split valid --workflow w0
    ... --workflow w1    # verify and repair; needs the w0 run of the same label
    ... --workflow w2    # independent tool check on rows with a supplied tool answer

Outputs go to results/phase4/<split>/<label>_<part>[_w1|_w2].jsonl with a manifest and
a score file; nothing is overwritten. The final test split needs results/phase4/selection.json
(the frozen selection) and every access is appended to results/phase4/final_test_access.jsonl.
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from bazireason import workflow as W
from bazireason.indep_rules import score_indep
from bazireason.p2_eval import score_row, summarize
from bazireason.p3_eval import score3, summarize3

DEFAULT_MODEL = "models/qwen3-4b-instruct-2507-mlx-4bit"
MAX_TOKENS = {"p3": 800, "p2": 700, "indep": 600}
SCORERS = {"p3": score3, "p2": score_row, "indep": score_indep}
ROOT = Path("results/phase4")


def summary(part, scores):
    if part == "p3":
        return summarize3(scores)
    if part == "p2":
        return summarize(scores)
    kinds = Counter(s["kind"] for s in scores)
    return {
        "overall": {"n": len(scores), "k": sum(bool(s["correct"]) for s in scores)},
        "by_kind": {k: sum(bool(s["correct"]) for s in scores if s["kind"] == k) for k in kinds},
        "kind_rows": dict(kinds),
    }


def log_access(args):
    with (ROOT / "final_test_access.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "utc": datetime.now(UTC).isoformat(timespec="seconds"),
                    "label": args.label,
                    "workflow": args.workflow,
                    "parts": args.parts,
                }
            )
            + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter")
    parser.add_argument("--label", required=True)
    parser.add_argument("--split", choices=("valid", "final_test"), required=True)
    parser.add_argument("--parts", nargs="+", default=["p3", "p2", "indep"])
    parser.add_argument("--workflow", choices=("w0", "w1", "w2"), default="w0")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    if args.split == "final_test":
        if not (ROOT / "selection.json").exists():
            parser.error("The final test is sealed until results/phase4/selection.json exists")
        log_access(args)
    from bazireason.mlx_backend import MLXChatModel, file_sha256  # needs MLX; after the seal check

    out_dir = ROOT / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.workflow == "w0" else f"_{args.workflow}"
    chat = None
    for part in args.parts:
        data = Path(f"data/phase4/{args.split}/{part}.jsonl")
        output = out_dir / f"{args.label}_{part}{suffix}.jsonl"
        if output.exists():
            print(f"skip {output} (exists)")
            continue
        rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
        rows = [r for r in rows if "expected" in r["meta"]]
        first = {}
        if args.workflow != "w0":
            base = out_dir / f"{args.label}_{part}.jsonl"
            if not base.exists():
                parser.error(f"{args.workflow} needs the w0 run {base}")
            first = {
                r["id"]: r for r in map(json.loads, base.read_text(encoding="utf-8").splitlines())
            }
        if args.workflow == "w2":
            rows = [r for r in rows if W.is_strict(r) and "supplied" in r["meta"]]
            if not rows:
                continue
        if chat is None:
            chat = MLXChatModel(args.model, args.adapter, label=args.label)
        # Which rows get a (new) generation, and with which conversation.
        jobs = {}
        notes = {}
        for row in rows:
            if args.workflow == "w0":
                jobs[row["id"]] = row["messages"][:-1]
            elif args.workflow == "w1":
                text = first[row["id"]]["raw_output"]
                agrees, fields = W.check_reply(row, text, first[row["id"]]["score"])
                if not agrees:
                    jobs[row["id"]] = W.repair_turn(row, text, fields)
                    notes[row["id"]] = fields
            else:
                meta = row["meta"]
                failed = W.tool_check(meta["inputs"], meta["convention"], meta["supplied"])
                notes[row["id"]] = failed
                if failed:
                    messages = [dict(m) for m in row["messages"][:-1]]
                    messages[-1]["content"] += "\n" + W.tool_note(failed)
                    jobs[row["id"]] = messages
        adapter_file = Path(args.adapter) / "adapters.safetensors" if args.adapter else None
        manifest = {
            "status": "planned",
            "split": args.split,
            "part": part,
            "workflow": args.workflow,
            "dataset": str(data),
            "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            "n_rows": len(rows),
            "n_generated": len(jobs),
            "model": chat.model,
            "adapter_path": args.adapter,
            "adapter_sha256": file_sha256(adapter_file) if adapter_file else None,
            "temperature": 0,
            "max_tokens": MAX_TOKENS[part],
        }
        Path(str(output) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        started = time.perf_counter()
        ids = list(jobs)
        texts, totals = (
            chat.generate_batch(
                [jobs[i] for i in ids], MAX_TOKENS[part], batch=args.batch, share_prefix=True
            )
            if ids
            else ([], {})
        )
        new = dict(zip(ids, texts, strict=True))
        scores = []
        with output.open("x", encoding="utf-8") as stream:
            for row in rows:
                record = {"id": row["id"], "model": chat.model, "workflow": args.workflow}
                if args.workflow == "w0":
                    text = new[row["id"]]
                else:
                    record["first_output"] = first[row["id"]]["raw_output"]
                    record["first_correct"] = bool(first[row["id"]]["score"]["correct"])
                    record["check"] = notes.get(row["id"])
                    record["regenerated"] = row["id"] in new
                    text = new.get(row["id"], first[row["id"]]["raw_output"])
                score = SCORERS[part](row, text)
                scores.append(score)
                record.update({"raw_output": text, "score": score})
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        manifest["status"] = "complete"
        manifest["wall_seconds"] = round(time.perf_counter() - started, 1)
        manifest["generation_totals"] = {
            k: (round(v, 2) if isinstance(v, float) else v) for k, v in totals.items()
        }
        Path(str(output) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        report = {"model": chat.model, "workflow": args.workflow, "summary": summary(part, scores)}
        Path(str(output) + ".scores.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
        right = sum(bool(s["correct"]) for s in scores)
        print(
            json.dumps(
                {
                    "label": args.label,
                    "part": part,
                    "workflow": args.workflow,
                    "correct": f"{right}/{len(scores)}",
                    "generated": len(jobs),
                    "seconds": manifest["wall_seconds"],
                }
            )
        )


if __name__ == "__main__":
    main()
