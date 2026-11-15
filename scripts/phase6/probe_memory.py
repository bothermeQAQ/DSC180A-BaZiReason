"""Memory probe for a training set-up: a few real steps on the longest training rows.

    .venv-mlx/bin/python scripts/phase6/probe_memory.py --config configs/phase6/qwen3_4b_lora_p6_u1.yaml \
        --layers 36 [--rank 16] [--grad-checkpoint] [--model models/... --batch 1]

Writes a probe config and data under artifacts/probe/, runs scripts/phase2/train_lora.py for
8 iterations on the 16 longest training rows (batch as in the config), and appends the peak
memory and exit code to results/phase6/memory_probe.jsonl. The set-up fits when the run ends
normally with a peak of at most 25 GB (the machine's recommended working set is 26.8 GB).
Exit code 0 when it fits. The probe adapter is not kept.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

LIMIT_GB = 25.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--model", help="another base model (section 11: the larger base)")
    parser.add_argument(
        "--batch", type=int, choices=(1, 2), help="batch size; updates stay at 4 rows each"
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    tag = f"l{args.layers}-r{args.rank}" + ("-ckpt" if args.grad_checkpoint else "")
    if args.model:
        config["model"] = args.model
        tag = Path(args.model).name + "-" + tag
    if args.batch:
        config["batch_size"] = args.batch
        config["grad_accumulation_steps"] = 4 // args.batch
        tag += f"-b{args.batch}"
    folder = Path("artifacts/probe") / tag
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    rows = Path(config["data"]["train"]).read_text(encoding="utf-8").splitlines()
    longest = sorted(rows, key=len, reverse=True)[:16]
    (folder / "train.jsonl").write_text("\n".join(longest) + "\n", encoding="utf-8")
    (folder / "valid.jsonl").write_text("\n".join(longest[:2]) + "\n", encoding="utf-8")
    config.update(
        run_name=config["run_name"] + "-probe-" + tag,
        data={
            "train": str(folder / "train.jsonl"),
            "valid": str(folder / "valid.jsonl"),
            "valid_rows": 2,
        },
        adapter_dir=str(folder / "adapter"),
        log_dir=str(folder / "log"),
        num_layers=args.layers,
        steps_per_report=2,
        steps_per_eval=1000,
        steps_per_save=1000,
        grad_checkpoint=args.grad_checkpoint,
    )
    config["lora_parameters"]["rank"] = args.rank
    (folder / "probe.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    started = time.perf_counter()
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/phase2/train_lora.py",
            "--config",
            str(folder / "probe.yaml"),
            "--iters",
            "8",
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    metrics = folder / "log" / "metrics.jsonl"
    train = (
        [json.loads(x) for x in metrics.read_text().splitlines() if '"train"' in x]
        if metrics.exists()
        else []
    )
    peak = max((m["peak_memory_gb"] for m in train), default=None)
    fits = proc.returncode == 0 and peak is not None and peak <= LIMIT_GB
    result = {"utc": datetime.now(UTC).isoformat(timespec="seconds"), "config": args.config, "model": config["model"], "layers": args.layers,
              "rank": args.rank, "grad_checkpoint": args.grad_checkpoint, "batch_size": config["batch_size"],
              "longest_row_chars": len(longest[0]), "exit_code": proc.returncode, "peak_memory_gb": peak,
              "seconds_per_iteration": round((train[-1]["elapsed_seconds"] / train[-1]["iteration"]) if train else 0, 2),
              "wall_seconds": round(time.perf_counter() - started, 1), "limit_gb": LIMIT_GB, "fits": fits,
              "stderr_tail": proc.stderr[-400:] if proc.returncode else ""}  # fmt: skip
    with Path("results/phase6/memory_probe.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result) + "\n")
    print(json.dumps(result))
    shutil.rmtree(folder / "adapter", ignore_errors=True)
    sys.exit(0 if fits else 1)


if __name__ == "__main__":
    main()
