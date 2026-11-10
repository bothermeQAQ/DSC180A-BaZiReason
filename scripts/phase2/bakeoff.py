"""Measured model-selection bake-off on this machine (one candidate per process).

The pilot items are synthetic and generated here from the verified engine and
the rule tables with a fixed seed. No contest question is used. For each
candidate the script reports rule multiple-choice accuracy, strict JSON copy
validity, reasoned calendar accuracy, generation speed, peak memory and a short
LoRA training throughput probe (real forward and backward passes).

    python scripts/phase2/bakeoff.py --model models/qwen3.5-4b-mlx-4bit --label qwen3.5-4b
"""

import argparse
import json
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

from bazireason import rules
from bazireason.baselines import parse_strict, recover_json
from bazireason.calendar_engine import compute_chart
from bazireason.conventions import PILLARS, STEMS, Convention
from bazireason.mlx_backend import MLXChatModel
from bazireason.prompts import SYSTEM, build_messages
from bazireason.schema import valid_response

SEED = 20261110
REASONED_NOTE = (
    "You may first reason step by step in plain text. Finish with the final JSON object "
    "on its own last line."
)


def random_cases(rng, count):
    cases = []
    while len(cases) < count:
        moment = datetime(rng.randint(1950, 2049), rng.randint(1, 12), rng.randint(1, 28))
        moment += timedelta(seconds=rng.randint(2 * 3600, 21 * 3600))
        inputs = {
            "datetime": moment.isoformat(timespec="seconds"),
            "timezone": "Asia/Shanghai",
        }
        convention = Convention().to_dict()
        result = compute_chart(inputs, convention)
        if result["status"] == "ok":
            cases.append({"input": inputs, "convention": convention, "result": result})
    return cases


def rule_items(rng, count):
    items = []
    kinds = ("ten_god", "hidden", "clash", "hour_stem", "month_stem", "combine")
    for index in range(count):
        kind = kinds[index % len(kinds)]
        day, other = rng.choice(STEMS), rng.choice(STEMS)
        if kind == "ten_god":
            question = f"八字中日主为{day}，见天干{other}，{other}是日主的什么十神？"
            answer, pool = rules.ten_god(day, other), list(rules.TEN_GODS)
        elif kind == "hidden":
            branch = rng.choice("丑寅辰巳未申戌亥")
            question = f"地支{branch}的藏干（本气在前）是哪一组？"
            answer = "".join(rules.HIDDEN_STEMS[branch])
            pool = sorted({"".join(v) for v in rules.HIDDEN_STEMS.values() if len(v) > 1})
        elif kind == "clash":
            branch = rng.choice(rules.BRANCHES)
            question = f"与地支{branch}构成六冲的是哪个地支？"
            answer = next(k for k in rules.BRANCH_CLASHES if branch in k).replace(branch, "")
            pool = list(rules.BRANCHES)
        elif kind == "hour_stem":
            branch = rng.choice(rules.BRANCHES)
            question = f"按五鼠遁，{day}日的{branch}时，时柱是什么？"
            answer = rules.hour_pillar_for(day, branch)
            pool = [s + branch for s in STEMS if rules.valid_pillar(s + branch)]
        elif kind == "month_stem":
            branch = rng.choice(rules.BRANCHES)
            question = f"按五虎遁，{day}年的{branch}月，月柱是什么？"
            answer = rules.month_pillar_for(day, branch)
            pool = [s + branch for s in STEMS if rules.valid_pillar(s + branch)]
        else:
            question = f"与天干{day}构成天干五合的是哪个天干？"
            answer = next(k for k in rules.STEM_COMBINES if day in k).replace(day, "")
            pool = list(STEMS)
        wrong = rng.sample([p for p in pool if p != answer], 3)
        options = wrong + [answer]
        rng.shuffle(options)
        letters = "ABCD"
        text = question + "\n" + "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
        items.append({"question": text, "answer": letters[options.index(answer)]})
    return items


def last_json(text):
    """The last parseable JSON object in a reasoned answer, or None."""
    for match in reversed(list(re.finditer(r"\{", text))):
        depth, end = 0, None
        for position in range(match.start(), len(text)):
            depth += text[position] == "{"
            depth -= text[position] == "}"
            if depth == 0:
                end = position + 1
                break
        if end is None:
            continue
        try:
            value = parse_strict(text[match.start() : end])
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and "status" in value:
            return value
    return None


def training_probe(chat, rows, iters=12, batch_size=4):
    """A few real LoRA optimizer steps; returns tokens per second and peak memory."""
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import TrainingArgs, train
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from bazireason.finetune import PromptCompletionDataset

    chat.net.freeze()
    linear_to_lora_layers(chat.net, 16, {"rank": 8, "dropout": 0.0, "scale": 20.0})
    dataset = CacheDataset(PromptCompletionDataset(rows, chat.tokenizer))
    seen = []

    class Probe:
        def on_train_loss_report(self, info):
            seen.append(info)

        def on_val_loss_report(self, info):
            pass

    mx.reset_peak_memory()
    import tempfile

    mean_tokens = sum(len(dataset[i][0]) for i in range(len(dataset))) / len(dataset)
    with tempfile.TemporaryDirectory() as scratch:
        args = TrainingArgs(
            batch_size=batch_size,
            iters=iters,
            steps_per_report=4,
            steps_per_eval=10**9,
            steps_per_save=10**9,
            max_seq_length=1536,
            adapter_file=str(Path(scratch) / "probe.safetensors"),
        )
        train(
            chat.net,
            optim.Adam(learning_rate=1e-5),
            dataset,
            None,
            args,
            training_callback=Probe(),
        )
    steady = seen[1:] or seen
    rate = sum(s["iterations_per_second"] for s in steady) / len(steady)
    return {
        "batch_size": batch_size,
        "lora_layers": 16,
        "iterations_per_second": rate,
        "sequence_tokens_per_second": rate * batch_size * mean_tokens,
        "loss_tokens_per_second": sum(s["tokens_per_second"] for s in steady) / len(steady),
        "train_peak_memory_gb": max(s["peak_memory"] for s in seen),
        "first_loss": seen[0]["train_loss"],
        "last_loss": seen[-1]["train_loss"],
        "mean_tokens_per_example": mean_tokens,
    }


def probe_rows(cases):
    """Training rows in the benchmark's strict format, used only to time optimizer steps."""
    rows = []
    for c in cases:
        messages = build_messages(
            {"input": c["input"], "convention": c["convention"]}, "tool_intermediate"
        )
        answer = {
            "status": "ok",
            "pillars": c["result"]["pillars"],
            "reason_codes": [],
            "conventions": c["convention"],
        }
        rows.append(
            {
                "messages": messages
                + [{"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}]
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--only-training", action="store_true", help="Update lora_probe only")
    parser.add_argument("--probe-batch-size", type=int, default=4)
    args = parser.parse_args()
    rng = random.Random(SEED)
    cases = random_cases(rng, 24)
    mcq = rule_items(rng, 30)
    chat = MLXChatModel(args.model, label=args.label)
    if args.only_training:
        report = json.loads(Path(args.output).read_text())
        rows = probe_rows(cases)
        try:
            report["lora_probe"] = training_probe(chat, rows * 2, batch_size=args.probe_batch_size)
        except Exception as error:
            report["lora_probe"] = {"error": f"{type(error).__name__}: {error}"[:300]}
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(report["lora_probe"]))
        return
    report = {"label": args.label, "model_path": args.model, "seed": SEED}
    report["load_seconds"] = round(chat.load_seconds, 2)

    # 1. Chinese rule multiple choice: answer with one letter.
    system = "你是八字命理规则助手。只回答正确选项的字母（A、B、C 或 D），不要解释。"
    conversations = [
        [{"role": "system", "content": system}, {"role": "user", "content": item["question"]}]
        for item in mcq
    ]
    start = time.perf_counter()
    texts, stats = chat.generate_batch(conversations, max_tokens=8, batch=15)
    picks = [(re.search(r"[ABCD]", t) or [""])[0] for t in texts]
    report["rule_mcq"] = {
        "n": len(mcq),
        "accuracy": sum(p == item["answer"] for p, item in zip(picks, mcq, strict=True)) / len(mcq),
        "unparsed": sum(not p for p in picks),
        "seconds": round(time.perf_counter() - start, 2),
    }

    # 2. Strict JSON with the engine answer in context (format compliance).
    context_cases = cases[:12]
    conversations = [
        build_messages(
            {"input": c["input"], "convention": c["convention"]}, "tool_context", c["result"]
        )
        for c in context_cases
    ]
    texts, stats = chat.generate_batch(conversations, max_tokens=200, batch=12)
    strict_valid = exact = lenient = 0
    for text, case in zip(texts, context_cases, strict=True):
        try:
            value = parse_strict(text)
        except (ValueError, TypeError):
            value = None
        recovered = value if value is not None else recover_json(text)[0]
        strict_valid += valid_response(value)
        lenient += valid_response(recovered)
        exact += bool(valid_response(value) and value["pillars"] == case["result"]["pillars"])
    report["strict_json_tool_context"] = {
        "n": len(context_cases),
        "strict_valid_rate": strict_valid / len(context_cases),
        "lenient_valid_rate": lenient / len(context_cases),
        "exact_rate": exact / len(context_cases),
        "sample": texts[0][:300],
    }
    report["batch_generation"] = {
        "prompt_tokens": stats["prompt_tokens"],
        "completion_tokens": stats["completion_tokens"],
        "seconds": round(stats["seconds"], 2),
        "peak_memory_gb": round(stats["peak_memory_gb"], 2),
    }

    # 3. Reasoned computation from intermediate tool facts (rule execution).
    reasoned_cases = cases[12:24]
    conversations = []
    for c in reasoned_cases:
        messages = build_messages(
            {"input": c["input"], "convention": c["convention"]}, "tool_intermediate"
        )
        messages[0] = {"role": "system", "content": SYSTEM + REASONED_NOTE}
        conversations.append(messages)
    start = time.perf_counter()
    texts, stats = chat.generate_batch(conversations, max_tokens=700, batch=12)
    pillar_hits = dict.fromkeys(PILLARS, 0)
    exact = parsed = 0
    for text, case in zip(texts, reasoned_cases, strict=True):
        value = last_json(text)
        pillars = value.get("pillars") if isinstance(value, dict) else None
        parsed += isinstance(pillars, dict)
        if isinstance(pillars, dict):
            for name in PILLARS:
                pillar_hits[name] += pillars.get(name) == case["result"]["pillars"][name]
            exact += pillars == case["result"]["pillars"]
    report["reasoned_tool_intermediate"] = {
        "n": len(reasoned_cases),
        "final_json_found": parsed,
        "exact_rate": exact / len(reasoned_cases),
        "per_pillar_accuracy": {k: v / len(reasoned_cases) for k, v in pillar_hits.items()},
        "completion_tokens": stats["completion_tokens"],
        "batch_generation_tps": round(stats["completion_tokens"] / stats["seconds"], 1),
        "seconds": round(time.perf_counter() - start, 2),
        "sample_tail": texts[0][-400:],
    }

    # 4. Single-stream speed, as the chat demo will run it.
    text, single = chat.generate(conversations[0], max_tokens=200)
    report["single_stream"] = {
        "prompt_tokens": single["prompt_tokens"],
        "prompt_tps": round(single["prompt_tps"], 1),
        "generation_tps": round(single["generation_tps"], 1),
        "peak_memory_gb": round(single["peak_memory_gb"], 2),
    }

    # 5. LoRA training throughput probe with real optimizer steps.
    if not args.skip_training:
        rows = probe_rows(cases)
        try:
            report["lora_probe"] = training_probe(chat, rows * 2, batch_size=args.probe_batch_size)
        except Exception as error:  # a candidate that cannot train is a finding, not a crash
            report["lora_probe"] = {"error": f"{type(error).__name__}: {error}"[:300]}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "model_path"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
