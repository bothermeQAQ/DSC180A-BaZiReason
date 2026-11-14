"""Check that swapping adapters in place equals loading them afresh (needs MLX and the GPU).

    .venv-mlx/bin/python scripts/phase5/check_bank.py    # writes results/phase5/bank_check.json

For three prompts and every adapter, the greedy reply of the shared base with the adapter
swapped in must equal the reply of a model loaded with that adapter alone; "base" must equal
the untuned model. Also records swap time and memory.
"""

import json
import time
from pathlib import Path

import mlx.core as mx

from bazireason.adapter_bank import ADAPTERS, AdapterBank, adapter_file
from bazireason.mlx_backend import MLXChatModel
from bazireason.workflow import prompt_messages

OUT = Path("results/phase5/bank_check.json")


def main():
    items = [
        json.loads(x)
        for x in Path("data/phase4/final_test/indep.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[:3]
    ]
    prompts = [prompt_messages(r) for r in items]
    bank = AdapterBank()
    swapped = {}
    for name in ("base", *ADAPTERS, "p3r3"):
        bank.use(name)
        swapped[name] = [bank.generate(p, 120)[0] for p in prompts]
    bank_memory = mx.get_active_memory() / 1e9
    swap_ms = [round(1000 * s, 2) for s in bank.swap_seconds]
    del bank
    mx.clear_cache()
    result = {
        "same_as_fresh_load": {},
        "swap_ms": swap_ms,
        "bank_active_memory_gb": round(bank_memory, 2),
    }
    for name in ("base", *ADAPTERS):
        start = time.perf_counter()
        fresh = MLXChatModel(
            "models/qwen3-4b-instruct-2507-mlx-4bit",
            None if name == "base" else adapter_file(name).parent,
        )
        load = time.perf_counter() - start
        result["same_as_fresh_load"][name] = [
            fresh.generate(p, 120)[0] for p in prompts
        ] == swapped[name]
        result.setdefault("fresh_load_seconds", {})[name] = round(load, 2)
        result["single_model_active_memory_gb"] = round(mx.get_active_memory() / 1e9, 2)
        del fresh
        mx.clear_cache()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
