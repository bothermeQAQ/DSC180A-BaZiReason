"""BaziQA with the contest guard (workflow W3): rule facts in the prompt, loop guard, citations.

    python scripts/phase4/run_baziqa_guard.py --adapter <dir> --label p3r3 \
        --output-dir results/phase4/baziqa/p3r3

BaziQA is a previously inspected benchmark; nothing here is tuned on its answer keys.
Same prompt as the `verified` condition of Phases 2 and 3, the same sampling (temperature
0.7, top-p 0.8, top-k 20, three runs, the same seeds) and the same 1,024-token budget.
Differences, all fixed in docs/phase4_preregistration.md:
  * the chart block adds the rule code's deep-structure lines after the verified chart;
  * loop guard: a reply with no parseable option that repeats a span three times is cut
    at the loop and the model is asked for the option letter only (16 tokens, greedy);
  * citation check: book titles not cited by the rule cards are counted (not scored).
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from bazireason import baziqa as B
from bazireason import demo_deep
from bazireason import workflow as W
from bazireason.mlx_backend import MLXChatModel, file_sha256, messages_sha256
from bazireason.p2_prompts import extract_choice

DEFAULT_MODEL = "models/qwen3-4b-instruct-2507-mlx-4bit"
CARD_WORKS = {"渊海子平", "三命通会", "滴天髓", "穷通宝鉴", "子平真诠"}
LETTER_PROMPT = "请只回答一个选项字母。"


def chart_block(person):
    block = B.tool_block(person)
    chart, _ = B.tool_chart(person)
    if chart["status"] == "ok":
        block += "\n【规则代码给出的深入结构（不是预测）】\n" + "\n".join(
            demo_deep.deep_lines(chart["pillars"])
        )
    return block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--root", default="external/BaziQA")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--persons-per-batch", type=int, default=3)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "tuned_tools_w3.jsonl"
    persons = B.load_contest(args.root)
    chat = MLXChatModel(args.model, args.adapter, label=args.label)
    adapter_file = Path(args.adapter) / "adapters.safetensors"
    manifest = {
        "condition": "tuned_tools_w3",
        "workflow": "W3 contest guard (verified chart + rule facts, loop guard, citation check)",
        "benchmark_commit": B.PINNED_COMMIT,
        "n_persons": len(persons),
        "runs": args.runs,
        "decoding": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": args.max_tokens,
                     "seed_base": args.seed, "loop_guard": "greedy, 16 tokens"},
        "model": chat.model,
        "adapter_path": args.adapter,
        "adapter_sha256": file_sha256(adapter_file),
        "contest_data_used_for_training": False,
        "answer_keys_used": False,
    }  # fmt: skip
    done = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done.add((record["question_id"], record["run"]))
    blocks = {p["person_id"]: chart_block(p) for p in persons}
    started = time.perf_counter()
    step = args.persons_per_batch
    with output.open("a", encoding="utf-8") as stream:
        for start in range(0, len(persons), step):
            chunk = persons[start : start + step]
            plan = [
                (p, q, B.question_messages(p, q, blocks[p["person_id"]]))
                for p in chunk
                for q in p["questions"]
            ]
            for run in range(1, args.runs + 1):
                if all((q["question_id"], run) in done for _, q, _ in plan):
                    continue
                seed = args.seed + 1000 * run + start
                texts, _ = chat.generate_batch(
                    [m for _, _, m in plan],
                    args.max_tokens,
                    0.7,
                    seed,
                    len(plan),
                    0.8,
                    20,
                )
                letters_of = ["ABCDE"[: len(q["options"])] for _, q, _ in plan]
                choices = [extract_choice(t, s) for t, s in zip(texts, letters_of, strict=True)]
                guard = [
                    i for i, (t, c) in enumerate(zip(texts, choices, strict=True))
                    if c is None and W.has_loop(t)
                ]  # fmt: skip
                second = {}
                if guard:
                    convos = [
                        plan[i][2]
                        + [
                            {"role": "assistant", "content": W.cut_at_loop(texts[i])},
                            {"role": "user", "content": LETTER_PROMPT},
                        ]
                        for i in guard
                    ]
                    replies, _ = chat.generate_batch(convos, 16, batch=len(convos))
                    second = dict(zip(guard, replies, strict=True))
                for i, ((person, question, messages), text) in enumerate(
                    zip(plan, texts, strict=True)
                ):
                    choice = choices[i]
                    if i in second:
                        choice = extract_choice(second[i], letters_of[i])
                    tokens = len(chat.tokenizer.encode(text, add_special_tokens=False))
                    stream.write(
                        json.dumps(
                            {
                                "question_id": question["question_id"],
                                "person_id": person["person_id"],
                                "condition": "tuned_tools_w3",
                                "run": run,
                                "answer": choice,
                                "parseable": choice is not None,
                                "truncated": tokens >= args.max_tokens,
                                "completion_tokens": tokens,
                                "raw_output": text,
                                "loop_guard_used": i in second,
                                "loop_guard_reply": second.get(i),
                                "titles": W.TITLE.findall(text or ""),
                                "titles_not_in_cards": W.unknown_titles(text, CARD_WORKS),
                                "prompt_sha256": messages_sha256(messages),
                                "seed": seed,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                stream.flush()
            print(f"w3: persons {min(start + step, len(persons))}/{len(persons)} "
                  f"elapsed {time.perf_counter() - started:.0f}s", flush=True)  # fmt: skip
    manifest["status"] = "complete"
    manifest["wall_seconds"] = round(time.perf_counter() - started, 1)
    manifest["prompt_block_sha256"] = hashlib.sha256(
        json.dumps(blocks, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    (out_dir / "tuned_tools_w3.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print("BAZIQA_W3_DONE")


if __name__ == "__main__":
    main()
