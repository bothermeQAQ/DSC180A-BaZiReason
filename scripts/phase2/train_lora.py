"""Local LoRA/QLoRA fine-tuning with MLX-LM, with full logging and checkpoint selection.

    python scripts/phase2/train_lora.py --config configs/phase2/qwen3_4b_lora_v1.yaml
    python scripts/phase2/train_lora.py --config ... --smoke            # 20 real steps
    python scripts/phase2/train_lora.py --config ... --resume           # continue from the last checkpoint

Every reported number comes from real forward and backward passes. The script
writes one JSON line per report to ``metrics.jsonl`` (training loss, validation
loss, learning rate, tokens, speed, peak memory), keeps numbered checkpoints,
and finally copies the checkpoint with the lowest validation loss to
``adapters.safetensors`` so that ``mlx_lm.load(model, adapter_path=...)`` uses it.
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import mlx.core as mx  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.tuner.datasets import CacheDataset  # noqa: E402
from mlx_lm.tuner.trainer import TrainingArgs, train  # noqa: E402
from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: E402

from bazireason.finetune import PromptCompletionDataset  # noqa: E402


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def stratified(rows, count):
    """A fixed subset that keeps every family: round-robin over families in file order."""
    by_family = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
    picked, depth = [], 0
    while len(picked) < min(count, len(rows)):
        for family in sorted(by_family):
            if depth < len(by_family[family]) and len(picked) < count:
                picked.append(by_family[family][depth])
        depth += 1
    return picked


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Recorder:
    def __init__(self, path, offset, started):
        self.path, self.offset, self.started = path, offset, started
        self.val, self.tokens = {}, 0

    def write(self, record):
        record["elapsed_seconds"] = round(time.perf_counter() - self.started, 1)
        with open(self.path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    def on_train_loss_report(self, info):
        self.write(
            {
                "kind": "train",
                "iteration": info["iteration"] + self.offset,
                "train_loss": info["train_loss"],
                "learning_rate": info["learning_rate"],
                "iterations_per_second": info["iterations_per_second"],
                "loss_tokens_per_second": info["tokens_per_second"],
                "loss_tokens_total": info["trained_tokens"],
                "peak_memory_gb": info["peak_memory"],
            }
        )

    def on_val_loss_report(self, info):
        # The trainer evaluates just before step `it` and reports it - 1. Report 0 is the
        # untouched base model; later reports are attributed to the checkpoint saved at `it`
        # (one optimizer step later).
        local = info["iteration"] + 1 if info["iteration"] > 0 else 0
        iteration = local + (self.offset if local else self.offset)
        self.val[iteration] = info["val_loss"]
        self.write(
            {
                "kind": "valid",
                "iteration": iteration,
                "val_loss": info["val_loss"],
                "val_seconds": info["val_time"],
            }
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true", help="20 real steps into a _smoke folder")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--iters", type=int, help="Override the number of iterations")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    suffix = "_smoke" if args.smoke else ""
    adapter_dir = Path(config["adapter_dir"] + suffix)
    log_dir = Path(config["log_dir"] + suffix)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(config["seed"])
    mx.random.seed(config["seed"])

    train_rows = read_rows(config["data"]["train"])
    valid_rows = stratified(read_rows(config["data"]["valid"]), config["data"]["valid_rows"])
    if args.smoke:
        train_rows, valid_rows = stratified(train_rows, 64), valid_rows[:16]
    batch, accumulate = config["batch_size"], config["grad_accumulation_steps"]
    total_iters = args.iters or int(len(train_rows) * config["epochs"] / batch)
    if args.smoke:
        total_iters = 20

    model, tokenizer = load(config["model"])
    model.freeze()
    linear_to_lora_layers(model, config["num_layers"], config["lora_parameters"])
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    total = sum(v.size for _, v in tree_flatten(model.parameters()))

    if config.get("init_adapter"):
        # Warm start: continue from an earlier adapter with a fresh schedule and new data.
        model.load_weights(config["init_adapter"], strict=False)
    start_iteration = 0
    numbered = sorted(adapter_dir.glob("[0-9]*_adapters.safetensors"))
    if args.resume and numbered:
        start_iteration = int(numbered[-1].name.split("_")[0])
        model.load_weights(str(numbered[-1]), strict=False)
        print(f"Resuming from {numbered[-1]} at iteration {start_iteration}")
    elif not args.resume:
        for stale in adapter_dir.glob("*.safetensors"):
            stale.unlink()
        (log_dir / "metrics.jsonl").unlink(missing_ok=True)

    updates = max(1, total_iters // accumulate)
    warmup = min(config["warmup_updates"], max(1, updates // 5))
    schedule = optim.join_schedules(
        [
            optim.linear_schedule(0.0, config["learning_rate"], warmup),
            optim.cosine_decay(
                config["learning_rate"], updates - warmup, config["end_learning_rate"]
            ),
        ],
        [warmup],
    )
    done_updates = start_iteration // accumulate

    def learning_rate(step):
        return schedule(step + done_updates)

    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=config["weight_decay"])
    adapter_config = {
        "fine_tune_type": config["fine_tune_type"],
        "num_layers": config["num_layers"],
        "lora_parameters": config["lora_parameters"],
        "model": config["model"],
        "base_model": config["base_model"],
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2) + "\n")

    train_set = CacheDataset(
        PromptCompletionDataset(train_rows, tokenizer, config["enable_thinking"])
    )
    valid_set = CacheDataset(
        PromptCompletionDataset(valid_rows, tokenizer, config["enable_thinking"])
    )
    sequence_tokens = sum(len(train_set[i][0]) for i in range(len(train_set)))
    plan = {
        "run_name": config["run_name"] + suffix,
        "config_file": args.config,
        "config_sha256": sha256(args.config),
        "init_adapter": config.get("init_adapter"),
        "init_adapter_sha256": sha256(config["init_adapter"])
        if config.get("init_adapter")
        else None,
        "train_file_sha256": sha256(config["data"]["train"]),
        "valid_file_sha256": sha256(config["data"]["valid"]),
        "train_rows": len(train_rows),
        "valid_rows_for_selection": len(valid_rows),
        "train_sequence_tokens_per_epoch": sequence_tokens,
        "iterations": total_iters,
        "start_iteration": start_iteration,
        "batch_size": batch,
        "grad_accumulation_steps": accumulate,
        "optimizer_updates": updates,
        "trainable_parameters": trainable,
        "stored_array_elements": total,
        "note_parameters": "stored_array_elements counts packed 4-bit arrays, not the 4.02e9 "
        "base parameters; trainable_parameters is the exact LoRA size.",
        "software": {"mlx": mx.__version__, "mlx_lm": __import__("mlx_lm").__version__},
        "device": mx.device_info(),
    }
    (log_dir / "training_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({k: v for k, v in plan.items() if k != "device"}, indent=1))

    started = time.perf_counter()
    recorder = Recorder(log_dir / "metrics.jsonl", start_iteration, started)
    # On resume the trainer numbers its checkpoints from 1 again, so it writes into a
    # scratch folder and the files are moved to their global iteration numbers afterwards.
    work_dir = adapter_dir / "resume_work" if start_iteration else adapter_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArgs(
        batch_size=batch,
        iters=total_iters - start_iteration,
        val_batches=-1,
        steps_per_report=config["steps_per_report"],
        steps_per_eval=10 if args.smoke else config["steps_per_eval"],
        steps_per_save=10 if args.smoke else config["steps_per_save"],
        max_seq_length=config["max_seq_length"],
        adapter_file=str(work_dir / "last_adapters.safetensors"),
        grad_accumulation_steps=accumulate,
        # Recomputes layer activations in the backward pass: same gradients, less memory.
        grad_checkpoint=bool(config.get("grad_checkpoint", False)),
    )
    mx.reset_peak_memory()
    train(model, optimizer, train_set, valid_set, training_args, training_callback=recorder)
    duration = time.perf_counter() - started

    if start_iteration:
        for path in sorted(work_dir.glob("[0-9]*_adapters.safetensors")):
            local = int(path.name.split("_")[0])
            path.rename(adapter_dir / f"{local + start_iteration:07d}_adapters.safetensors")
        (work_dir / "last_adapters.safetensors").rename(adapter_dir / "last_adapters.safetensors")
        work_dir.rmdir()
    final = adapter_dir / f"{total_iters:07d}_adapters.safetensors"
    if not final.exists():
        shutil.copyfile(adapter_dir / "last_adapters.safetensors", final)

    history = [json.loads(line) for line in (log_dir / "metrics.jsonl").read_text().splitlines()]
    val = {r["iteration"]: r["val_loss"] for r in history if r["kind"] == "valid"}
    # A validation report at iteration k describes the weights saved at k (k = 0 is the base).
    candidates = {
        k: v for k, v in val.items() if (adapter_dir / f"{k:07d}_adapters.safetensors").exists()
    }
    best = min(candidates, key=candidates.get) if candidates else total_iters
    chosen = adapter_dir / f"{best:07d}_adapters.safetensors"
    shutil.copyfile(chosen, adapter_dir / "adapters.safetensors")
    train_points = [r for r in history if r["kind"] == "train"]
    summary = {
        **{k: v for k, v in plan.items() if k != "device"},
        "duration_seconds": round(duration, 1),
        "final_train_loss": train_points[-1]["train_loss"] if train_points else None,
        "validation_loss_by_iteration": dict(sorted(val.items())),
        "initial_validation_loss": val.get(0),
        "initial_weights": "warm start from init_adapter"
        if config.get("init_adapter")
        else "base model",
        "selected_checkpoint_iteration": best,
        "selected_validation_loss": candidates.get(best),
        "selected_adapter": str(adapter_dir / "adapters.safetensors"),
        "selected_adapter_sha256": sha256(adapter_dir / "adapters.safetensors"),
        "peak_memory_gb": max((r["peak_memory_gb"] for r in train_points), default=None),
        "loss_tokens_total": train_points[-1]["loss_tokens_total"] if train_points else None,
        "sequence_tokens_processed": round(
            sequence_tokens * (total_iters - start_iteration) * batch / len(train_rows)
        ),
        "checkpoints": sorted(p.name for p in adapter_dir.glob("[0-9]*_adapters.safetensors")),
        "resumed_from_iteration": start_iteration or None,
        "resume_note": "optimizer moments restart at a resume; the learning-rate schedule continues"
        if start_iteration
        else None,
    }
    (log_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "TRAINING_COMPLETE",
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "duration_seconds",
                    "final_train_loss",
                    "selected_checkpoint_iteration",
                    "selected_validation_loss",
                    "peak_memory_gb",
                )
            }
        ),
    )


if __name__ == "__main__":
    main()
