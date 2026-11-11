"""Evaluate a local model on a Phase 3 held-out split (real inference, greedy decoding).

    python scripts/phase3/run_p3_eval.py --label base --output results/phase3/heldout/p3test_base.jsonl
    python scripts/phase3/run_p3_eval.py --label p3 --adapter artifacts/adapters/<run> --output ...

Every model gets the same prompts and token budget. The output file is never
overwritten. Rows of the training split are refused.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from bazireason.mlx_backend import MLXChatModel, file_sha256
from bazireason.p3_eval import score3, summarize3

DEFAULT_MODEL = "models/qwen3-4b-instruct-2507-mlx-4bit"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter")
    parser.add_argument("--label", required=True)
    parser.add_argument("--data", default="data/phase3/train_v1/test.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--families", nargs="*")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Output exists; choose a new path")
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines()]
    if any(row["split"] == "train" for row in rows):
        parser.error("Refusing to evaluate on training rows")
    rows = [row for row in rows if "expected" in row["meta"]]
    if args.families:
        rows = [row for row in rows if row["family"] in args.families]
    if args.limit:
        rows = rows[:: max(1, len(rows) // args.limit)][: args.limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    chat = MLXChatModel(args.model, args.adapter, label=args.label)
    adapter_file = Path(args.adapter) / "adapters.safetensors" if args.adapter else None
    manifest = {
        "status": "planned",
        "dataset": args.data,
        "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "n_rows": len(rows),
        "model": chat.model,
        "model_path": args.model,
        "model_weights_sha256": file_sha256(Path(args.model) / "model.safetensors"),
        "adapter_path": args.adapter,
        "adapter_sha256": file_sha256(adapter_file) if adapter_file else None,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "enable_thinking": False,
    }
    Path(args.output + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.perf_counter()
    texts, totals = chat.generate_batch(
        [row["messages"][:-1] for row in rows], args.max_tokens, batch=args.batch, share_prefix=True
    )
    scores = []
    with output.open("x", encoding="utf-8") as stream:
        for row, text in zip(rows, texts, strict=True):
            score = score3(row, text)
            scores.append(score)
            stream.write(
                json.dumps(
                    {"id": row["id"], "model": chat.model, "raw_output": text, "score": score},
                    ensure_ascii=False,
                )
                + "\n"
            )
    report = {"model": chat.model, "n_rows": len(rows), "summary": summarize3(scores)}
    manifest["status"] = "complete"
    manifest["wall_seconds"] = round(time.perf_counter() - started, 1)
    manifest["generation_totals"] = {
        k: (round(v, 2) if isinstance(v, float) else v) for k, v in totals.items()
    }
    Path(args.output + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    Path(args.output + ".scores.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    summary = report["summary"]
    print(
        json.dumps(
            {
                "model": chat.model,
                "overall": summary["overall"],
                "reasoning": summary["reasoning_tasks"],
                "seconds": manifest["wall_seconds"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
