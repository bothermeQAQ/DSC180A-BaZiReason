"""Run one BaziQA condition on a local MLX model.

    python scripts/phase2/run_baziqa.py --who base  --tools none
    python scripts/phase2/run_baziqa.py --who base  --tools verified
    python scripts/phase2/run_baziqa.py --who tuned --tools none     --adapter artifacts/adapters/...
    python scripts/phase2/run_baziqa.py --who tuned --tools verified --adapter artifacts/adapters/...

Each question is asked on its own with the same system prompt, the same option
order as published, and no answer key anywhere in the prompt.

* ``--tools verified``: the prompt carries the verified chart from this
  project's engine and rule tables.
* ``--tools none``: the model first derives the chart itself from the birth
  record (one call per person, no tool output). If it returns a valid chart,
  that chart is shown, labelled as unverified; otherwise only the birth record
  is shown.

Every run samples with the model card's recommended settings and a fixed seed.
A reply without a parseable ``答案：X`` is kept and scored as wrong. The script
appends one line per question and run, and resumes where it stopped.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from bazireason import baziqa as B
from bazireason.mlx_backend import MLXChatModel, file_sha256, messages_sha256
from bazireason.p2_prompts import MCQ_PROMPT_VERSION, extract_choice, extract_final_json

DEFAULT_MODEL = "models/qwen3-4b-instruct-2507-mlx-4bit"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--who", choices=("base", "tuned"), required=True)
    parser.add_argument("--tools", choices=("none", "verified"), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter")
    parser.add_argument("--root", default="external/BaziQA")
    parser.add_argument("--output-dir", default="results/phase2/baziqa")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--persons-per-batch", type=int, default=3)
    parser.add_argument("--limit-persons", type=int)
    parser.add_argument("--label", help="Override the recorded model name")
    args = parser.parse_args()
    if (args.who == "tuned") != bool(args.adapter):
        parser.error("--adapter is required for --who tuned and forbidden for --who base")
    condition = f"{args.who}_{'tools' if args.tools == 'verified' else 'no_tools'}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{condition}.jsonl"
    persons = B.load_contest(args.root)
    if args.limit_persons:
        persons = persons[:: max(1, len(persons) // args.limit_persons)][: args.limit_persons]
    chat = MLXChatModel(
        args.model, args.adapter, label=args.label or f"{args.who}:{Path(args.model).name}"
    )
    adapter_file = Path(args.adapter) / "adapters.safetensors" if args.adapter else None
    files = sorted(Path(args.root, "data").glob("*.json"))
    manifest = {
        "condition": condition,
        "prompt_version": MCQ_PROMPT_VERSION,
        "benchmark": "ChenJiangxi/BaziQA",
        "benchmark_commit": B.PINNED_COMMIT,
        "benchmark_data_sha256": hashlib.sha256(
            b"".join(f.read_bytes() for f in files if f.name != "celebrity50_zh.json")
        ).hexdigest(),
        "n_persons": len(persons),
        "n_questions": sum(len(p["questions"]) for p in persons),
        "runs": args.runs,
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "seed_base": args.seed,
            "enable_thinking": False,
        },
        "chart_source": "verified engine (fixed UTC+08:00, sect 1)"
        if args.tools == "verified"
        else "model's own derivation, unverified",
        "model": chat.model,
        "model_weights_sha256": file_sha256(Path(args.model) / "model.safetensors"),
        "adapter_path": args.adapter,
        "adapter_sha256": file_sha256(adapter_file) if adapter_file else None,
        "contest_data_used_for_training": False,
    }
    manifest_path = out_dir / f"{condition}.manifest.json"
    manifest_path.write_text(
        json.dumps({**manifest, "status": "running"}, indent=2, ensure_ascii=False) + "\n"
    )

    done = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done.add((record["question_id"], record["run"]))
    started = time.perf_counter()

    # Stage one for the no-tool conditions: the model's own chart, greedy, one call per person.
    blocks = {}
    if args.tools == "verified":
        blocks = {p["person_id"]: B.tool_block(p) for p in persons}
    else:
        chart_file = out_dir / f"{condition}.self_charts.jsonl"
        saved = {}
        if chart_file.exists():
            saved = {
                json.loads(line)["person_id"]: json.loads(line)
                for line in chart_file.read_text(encoding="utf-8").splitlines()
            }
        missing = [p for p in persons if p["person_id"] not in saved]
        if missing:
            texts, _ = chat.generate_batch(
                [B.self_chart_messages(p) for p in missing], 700, batch=16
            )
            with chart_file.open("a", encoding="utf-8") as stream:
                for person, text in zip(missing, texts, strict=True):
                    answer = extract_final_json(text)
                    truth, _assumed = B.tool_chart(person)
                    record = {
                        "person_id": person["person_id"],
                        "raw_output": text,
                        "answer": answer,
                        "block": B.self_chart_block(answer),
                        "verified_status": truth["status"],
                        "verified_pillars": truth["pillars"],
                    }
                    saved[person["person_id"]] = record
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        blocks = {pid: record["block"] for pid, record in saved.items()}

    step = args.persons_per_batch
    with output.open("a", encoding="utf-8") as stream:
        for start in range(0, len(persons), step):
            chunk = persons[start : start + step]
            plan = [
                (p, q, B.question_messages(p, q, blocks.get(p["person_id"])))
                for p in chunk
                for q in p["questions"]
            ]
            pending_runs = [
                run
                for run in range(1, args.runs + 1)
                if any((q["question_id"], run) not in done for _, q, _ in plan)
            ]
            if not pending_runs:
                continue
            cached = {}
            if args.tools == "verified":
                for person in chunk:
                    own = [m for p, _, m in plan if p is person]
                    cached[person["person_id"]] = chat.shared_prefix_cache(own)
            for run in pending_runs:
                seed = args.seed + 1000 * run + start
                if args.tools == "verified":
                    texts = []
                    for index, person in enumerate(chunk):
                        cache, length, prompts = cached[person["person_id"]]
                        replies, _ = chat.generate_with_prefix(
                            prompts,
                            cache,
                            length,
                            args.max_tokens,
                            args.temperature,
                            seed + index,
                            args.top_p,
                            args.top_k,
                        )
                        texts.extend(replies)
                else:
                    texts, _ = chat.generate_batch(
                        [m for _, _, m in plan],
                        args.max_tokens,
                        args.temperature,
                        seed,
                        len(plan),
                        args.top_p,
                        args.top_k,
                    )
                for (person, question, messages), text in zip(plan, texts, strict=True):
                    letters = "ABCDE"[: len(question["options"])]
                    choice = extract_choice(text, letters)
                    tokens = len(chat.tokenizer.encode(text, add_special_tokens=False))
                    stream.write(
                        json.dumps(
                            {
                                "question_id": question["question_id"],
                                "person_id": person["person_id"],
                                "condition": condition,
                                "run": run,
                                "answer": choice,
                                "parseable": choice is not None,
                                "truncated": tokens >= args.max_tokens,
                                "completion_tokens": tokens,
                                "raw_output": text,
                                "prompt_sha256": messages_sha256(messages),
                                "seed": seed,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                stream.flush()
            cached.clear()
            elapsed = time.perf_counter() - started
            print(
                f"{condition}: persons {min(start + step, len(persons))}/{len(persons)} elapsed {elapsed:.0f}s",
                flush=True,
            )
    manifest["status"] = "complete"
    manifest["wall_seconds"] = round(time.perf_counter() - started, 1)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"BAZIQA_CONDITION_DONE {condition}")


if __name__ == "__main__":
    main()
