"""Interactive latency: one reply at a time, as in the chatbot (needs MLX and the GPU).

    .venv-mlx/bin/python scripts/phase5/latency.py --split valid --labels p4a1 p3r3 --n 40

For a fixed sample of blind questions, measures wall-clock seconds per reply for the model
alone (w0) and for the grounded, verified pipeline (grv, including any repair turn), plus
prompt and completion tokens and peak memory. Writes results/phase5/latency_<split>.json.
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import mlx.core as mx

from bazireason import grounded as G
from bazireason.adapter_bank import AdapterBank


def timed(bank, messages):
    mx.reset_peak_memory()
    start = time.perf_counter()
    text, stats = bank.generate(messages, 800)
    return text, time.perf_counter() - start, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="valid")
    parser.add_argument("--labels", nargs="+", default=["p4a1"])
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args()
    items = [json.loads(x) for x in Path(f"data/phase5/blind/{args.split}.jsonl").read_text(encoding="utf-8").splitlines()]
    sample = random.Random(20261113).sample(items, min(args.n, len(items)))
    bank = AdapterBank()
    out = {"split": args.split, "n": len(sample), "systems": {}}
    for label in args.labels:
        bank.use(label)
        bank.generate(G.messages_for(sample[0], "w0"), 8)  # warm-up
        for condition in ("w0", "grv"):
            seconds, prompt_tokens, completion, peaks, repairs = [], [], [], [], 0
            for item in sample:
                meta = item["meta"]
                start = time.perf_counter()
                if condition == "w0":
                    text, _, stats = timed(bank, G.messages_for(item, "w0"))
                    tokens = [stats["prompt_tokens"], stats["completion_tokens"]]
                else:
                    plan = G.router_plan(item)  # routing and fact sheet are part of the cost
                    messages = G.messages_for(item, "gr", plan)
                    text, _, stats = timed(bank, messages)
                    tokens = [stats["prompt_tokens"], stats["completion_tokens"]]
                    check = G.verify(plan, text, meta["pillars"], meta.get("gender"))
                    if not check["ok"] and plan.get("reference") is not None:
                        repairs += 1
                        _, _, stats2 = timed(bank, G.repair_messages(messages, text, plan, check))
                        tokens[0] += stats2["prompt_tokens"]
                        tokens[1] += stats2["completion_tokens"]
                seconds.append(time.perf_counter() - start)
                prompt_tokens.append(tokens[0])
                completion.append(tokens[1])
                peaks.append(stats["peak_memory_gb"])
            seconds.sort()
            out["systems"][f"{label}/{condition}"] = {
                "median_seconds": round(statistics.median(seconds), 2),
                "p90_seconds": round(seconds[int(0.9 * (len(seconds) - 1))], 2),
                "mean_prompt_tokens": round(statistics.mean(prompt_tokens)),
                "mean_completion_tokens": round(statistics.mean(completion)),
                "peak_memory_gb": round(max(peaks), 2),
                "repair_turns": repairs,
            }
            print(label, condition, out["systems"][f"{label}/{condition}"], flush=True)
    path = Path(f"results/phase5/latency_{args.split}.json")
    path.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
