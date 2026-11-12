"""benchmark_v3 `tool_corrupted` with the independent tool check (workflow W2).

    python scripts/phase4/run_v3_toolcheck.py --label p3r3-w2 --adapter <dir> \
        --output results/phase4/v3/p3r3_w2_tool_corrupted.jsonl

Same cases, prompts, schema and scorer as the strict protocol of Phase 2 and 3. The one
difference: when the supplied (corrupted or clean) tool answer fails a check that does not
use the tool (bazireason.workflow.tool_check), one line naming the failed checks, without
the corrected values, is appended to the user message. Score with `bazireason evaluate`.
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from bazireason import workflow as W
from bazireason.baselines import run_case
from bazireason.calendar_engine import compute_chart
from bazireason.controls import corrupt_context
from bazireason.dataset_generator import read_jsonl
from bazireason.mlx_backend import MLXChatModel, ReplayClient, file_sha256

DATA = "data/synthetic/benchmark_v3.jsonl"
DEFAULT_MODEL = "models/qwen3-4b-instruct-2507-mlx-4bit"


class NotingClient:
    """Appends the W2 note for the current case to the last user message."""

    def __init__(self, inner):
        self.inner = inner
        self.note = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def complete(self, messages, temperature=0, max_tokens=512):
        if self.note:
            messages = [dict(m) for m in messages]
            messages[-1]["content"] += "\n" + self.note
        return self.inner.complete(messages, temperature=temperature, max_tokens=max_tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Output exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = read_jsonl(DATA)
    notes, kinds = {}, Counter()
    for case in cases:
        context = compute_chart(case["input"], case["convention"])
        supplied, corruption = corrupt_context(case, context)
        failed = W.tool_check(case["input"], case["convention"], supplied)
        notes[case["case_id"]] = W.tool_note(failed) if failed else None
        kind = corruption["type"] if corruption else "none"
        kinds[(kind, bool(failed))] += 1
    chat = MLXChatModel(args.model, args.adapter, label=args.label)
    client = NotingClient(ReplayClient(chat.model))
    started = time.perf_counter()
    for case in cases:
        client.note = notes[case["case_id"]]
        run_case(case, client, "tool_corrupted", 0, 0, 512)
    totals = client.inner.run_pending(chat, 512, batch=args.batch)
    records = []
    for case in cases:
        client.note = notes[case["case_id"]]
        record = run_case(case, client, "tool_corrupted", 0, 0, 512)
        record["workflow"] = "w2_tool_check"
        record["w2_note"] = notes[case["case_id"]]
        records.append(record)
    with output.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    adapter_file = Path(args.adapter) / "adapters.safetensors" if args.adapter else None
    manifest = {
        "status": "complete",
        "workflow": "w2_tool_check",
        "dataset_sha256": hashlib.sha256(Path(DATA).read_bytes()).hexdigest(),
        "methods": ["tool_corrupted"],
        "model": chat.model,
        "adapter_path": args.adapter,
        "adapter_sha256": file_sha256(adapter_file) if adapter_file else None,
        "temperature": 0,
        "max_tokens": 512,
        "check_flagged_by_corruption": {f"{k}|{'flagged' if f else 'passed'}": n for (k, f), n in sorted(kinds.items())},
        "wall_seconds": round(time.perf_counter() - started, 1),
        "generation_totals": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in totals.items()},
        "note": "prompt_sha256 in each record is of the prompt before the W2 line was appended",
    }  # fmt: skip
    Path(args.output + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["check_flagged_by_corruption"]))


if __name__ == "__main__":
    main()
