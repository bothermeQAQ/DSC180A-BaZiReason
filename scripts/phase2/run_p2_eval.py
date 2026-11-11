"""Run the Phase 2 held-out test split on a local MLX model and score it.

    python scripts/phase2/run_p2_eval.py --label base --output results/phase2/heldout/base.jsonl
    python scripts/phase2/run_p2_eval.py --label tuned --adapter artifacts/adapters/... \
        --output results/phase2/heldout/tuned.jsonl

The prompts are the dataset rows without their last (reference) message, so the
base and the fine-tuned model receive byte-identical inputs. Decoding is greedy.
``--paraphrase-probe`` additionally asks one rule question per held-out chart in
three wordings and reports how often the three answers agree.
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from bazireason import train_tasks as T
from bazireason.mlx_backend import MLXChatModel, file_sha256
from bazireason.p2_eval import score_row, summarize
from bazireason.p2_prompts import (
    OUTPUT_HINTS,
    P2_PROMPT_VERSION,
    chat_messages,
    extract_final_json,
)

DEFAULT_MODEL = "models/qwen3-4b-instruct-2507-mlx-4bit"
PROBE_KINDS = ("ten_god", "hidden", "seasonal", "life_stage", "pattern", "xun_kong", "relations")


def paraphrase_probe(rows, per_kind=12):
    """Three wordings of one rule question for each of ``per_kind`` held-out charts per kind."""
    charts = [r for r in rows if r["family"] == "rule"]
    probes = []
    for k, kind in enumerate(PROBE_KINDS):
        for row in charts[k :: len(PROBE_KINDS)][:per_kind]:
            for variant in range(3):
                rng = random.Random(f"{row['id']}:{kind}")
                question, _text, response = T.rule_task(
                    kind,
                    row["meta"]["pillars"],
                    rng,
                    variant,
                    row["meta"]["gender"],
                    row["meta"]["year"],
                )
                probes.append(
                    {
                        "probe_id": f"{row['id']}:{kind}",
                        "variant": variant,
                        "kind": kind,
                        "messages": chat_messages(question + "\n" + OUTPUT_HINTS["rule"]),
                        "expected": response["answer"],
                    }
                )
    return probes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter")
    parser.add_argument("--label", required=True)
    parser.add_argument("--data", default="data/phase2/train_v1/test.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--paraphrase-probe", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Output exists; choose a new path")
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines()]
    if any(row["split"] == "train" for row in rows):
        parser.error("Refusing to evaluate on training rows")
    # Rows in the benchmark's own prompt formats are scored by the benchmark harness instead.
    rows = [row for row in rows if "expected" in row["meta"]]
    if args.limit:
        rows = rows[: args.limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    chat = MLXChatModel(args.model, args.adapter, label=args.label)
    adapter_file = Path(args.adapter) / "adapters.safetensors" if args.adapter else None
    manifest = {
        "status": "planned",
        "prompt_version": P2_PROMPT_VERSION,
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
            score = score_row(row, text)
            scores.append(score)
            stream.write(
                json.dumps(
                    {"id": row["id"], "model": chat.model, "raw_output": text, "score": score},
                    ensure_ascii=False,
                )
                + "\n"
            )
    report = {"model": chat.model, "n_rows": len(rows), "summary": summarize(scores)}
    if args.paraphrase_probe:
        probes = paraphrase_probe(rows)
        replies, _ = chat.generate_batch([p["messages"] for p in probes], 400, batch=args.batch)
        groups = {}
        for probe, reply in zip(probes, replies, strict=True):
            answer = extract_final_json(reply)
            value = json.dumps(
                answer.get("answer") if isinstance(answer, dict) else None,
                ensure_ascii=False,
                sort_keys=True,
            )
            right = isinstance(answer, dict) and answer.get("answer") == probe["expected"]
            groups.setdefault(probe["probe_id"], []).append((value, right, probe["kind"]))
        agree = sum(len({v for v, _, _ in g}) == 1 and g[0][0] != "null" for g in groups.values())
        all_right = sum(all(r for _, r, _ in g) for g in groups.values())
        report["paraphrase_probe"] = {
            "n_questions": len(groups),
            "wordings_per_question": 3,
            "all_three_answers_identical": agree,
            "identical_rate": agree / len(groups),
            "all_three_correct": all_right,
            "all_correct_rate": all_right / len(groups),
        }
        Path(args.output + ".probe.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        **{k: p[k] for k in ("probe_id", "variant", "kind", "expected")},
                        "raw_output": r,
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for p, r in zip(probes, replies, strict=True)
            )
        )
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
                "exact": summary["calendar"]["exact_four_pillar"],
                "seconds": manifest["wall_seconds"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
