"""Training configuration of the larger-base pilot (preregistration section 11).

    .venv/bin/python scripts/phase6/p8_config.py --batch 1 [--grad-checkpoint]
        -> configs/phase6/qwen3_8b_lora_p6_s1.yaml   (never overwritten)

S1's configuration with the base model replaced by Qwen3-8B: the same training file, seed,
LoRA settings, learning rate and schedule. The batch size comes from the memory probe; the
gradient accumulation keeps four rows per update, as for S1.
"""

import argparse
import json
import sys
from pathlib import Path

SOURCE = Path("configs/phase6/qwen3_4b_lora_p6_s1.yaml")
TARGET = Path("configs/phase6/qwen3_8b_lora_p6_s1.yaml")
MODEL = "models/qwen3-8b-mlx-4bit"
REPO = "Qwen/Qwen3-8B-MLX-4bit"
NAME = "bazireason-qwen3-8b-lora-p6-s1"


def revision():
    """Commit of the downloaded weights, from the download metadata (None if unknown)."""
    for path in sorted((Path(MODEL) / ".cache/huggingface/download").glob("*.metadata")):
        lines = path.read_text().splitlines()
        if lines:
            return lines[0].strip()
    return None


def build(batch, grad_checkpoint):
    text = SOURCE.read_text()
    swaps = {
        "# BaZiReason Phase 6, s1: procedures in small steps on fresh charts, template wordings (step size)":
            "# BaZiReason Phase 6, s1 on Qwen3-8B: the base-size pilot (preregistration section 11)",
        "run_name: bazireason-qwen3-4b-lora-p6-s1": f"run_name: {NAME}",
        "model: models/qwen3-4b-instruct-2507-mlx-4bit": f"model: {MODEL}",
        "  repo: Qwen/Qwen3-4B-Instruct-2507": f"  repo: {REPO}",
        "  revision: cdbee75f17c01a7cc42f958dc650907174af0554": f"  revision: {revision() or 'unknown'}",
        '  quantization: "mlx_lm convert -q --q-bits 4 --q-group-size 64 (mlx-lm 0.32.2)"':
            '  quantization: "4-bit MLX weights as published by Qwen"',
        "adapter_dir: artifacts/adapters/bazireason-qwen3-4b-lora-p6-s1": f"adapter_dir: artifacts/adapters/{NAME}",
        "log_dir: results/phase6/training/bazireason-qwen3-4b-lora-p6-s1": f"log_dir: results/phase6/training/{NAME}",
        "batch_size: 2": f"batch_size: {batch}",
        "grad_accumulation_steps: 2": f"grad_accumulation_steps: {4 // batch}",
    }  # fmt: skip
    for old, new in swaps.items():
        if text.count(old) != 1:
            sys.exit(f"cannot rewrite the S1 configuration: {old!r}")
        text = text.replace(old, new)
    if grad_checkpoint:
        text += "grad_checkpoint: true    # same gradients, activations recomputed (memory)\n"
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=(1, 2), required=True)
    parser.add_argument("--grad-checkpoint", action="store_true")
    args = parser.parse_args()
    if TARGET.exists():
        sys.exit(f"{TARGET} exists")
    TARGET.write_text(build(args.batch, args.grad_checkpoint))
    print(
        json.dumps(
            {"config": str(TARGET), "batch": args.batch, "grad_checkpoint": args.grad_checkpoint}
        )
    )


if __name__ == "__main__":
    main()
