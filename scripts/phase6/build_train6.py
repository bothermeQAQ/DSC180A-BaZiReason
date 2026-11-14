"""Build one Phase 6 training mixture and its configuration.

    PYTHONPATH=src .venv-mlx/bin/python scripts/phase6/build_train6.py --recipe u0 --tokens 6000000
    PYTHONPATH=src .venv-mlx/bin/python scripts/phase6/build_train6.py --recipe s3 --tokens 6000000 \
        --replies results/phase6/own_replies_p6s1.jsonl
    ... --check      # rebuild in memory and compare hashes

Writes data/phase6/train/<recipe>/{train,valid}.jsonl and manifest.json, and
configs/phase6/qwen3_4b_lora_p6_<recipe>.yaml (never overwrites). Token counts are exact (the model's tokenizer; run with .venv-mlx). Rows longer than the
sequence limit are dropped, so no target is truncated. Every chart of every evaluation
split (Phase 2-6) is excluded.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from bazireason import splits6 as S
from bazireason import train_data6 as T6

OUT = Path("data/phase6/train")
CONFIG = Path("configs/phase6")
MAX_TOKENS = 2048
SEEDS = {
    "u0": 2026111710,
    "s1": 2026111711,
    "s2": 2026111712,
    "s3": 2026111713,
    "u1": 2026111714,
    "u1hi": 2026111714,
}
TEMPLATE = """# BaZiReason Phase 6, {recipe}: {purpose}
# Trained from the untuned base (no init adapter). Data: scripts/phase6/build_train6.py.
run_name: bazireason-qwen3-4b-lora-p6-{name}
model: models/qwen3-4b-instruct-2507-mlx-4bit
base_model:
  repo: Qwen/Qwen3-4B-Instruct-2507
  revision: cdbee75f17c01a7cc42f958dc650907174af0554
  license: Apache-2.0
  quantization: "mlx_lm convert -q --q-bits 4 --q-group-size 64 (mlx-lm 0.32.2)"
data:
  train: data/phase6/train/{recipe}/train.jsonl
  valid: data/phase6/train/{recipe}/valid.jsonl
  valid_rows: 120
adapter_dir: artifacts/adapters/bazireason-qwen3-4b-lora-p6-{name}
log_dir: results/phase6/training/bazireason-qwen3-4b-lora-p6-{name}
seed: {seed}
fine_tune_type: lora
num_layers: {layers}
lora_parameters:
  rank: {rank}
  scale: 2.0
  dropout: 0.05
optimizer: adamw
weight_decay: 0.01
learning_rate: 2.0e-4      # as the first Phase 2 round: a fresh adapter from the base
warmup_updates: 40
end_learning_rate: 2.0e-5
batch_size: 2
grad_accumulation_steps: 2
epochs: 1.0
max_seq_length: {max_tokens}
steps_per_report: 20
steps_per_eval: 500
steps_per_save: 500
mask_prompt: true
enable_thinking: false
"""
PURPOSE = {
    "u0": "the union of all earlier training data in one joint mixture (joint against sequential)",
    "s1": "procedures in small steps on fresh charts, template wordings (step size)",
    "s2": "s1 plus natural wordings, chart formats and contrast rows (wording coverage)",
    "s3": "s2 plus evidence use, reference checking, self-check and matched twins (evidence and self-check)",
    "u1": "the selected recipe at the full budget",
    "u1hi": "u1 with a larger adapter (all 36 layers): capacity",
}


def exclusions():
    charts = set(S.earlier_eval_charts())
    for path in Path("data/phase6").glob("*6/*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            meta = json.loads(line).get("meta") or {}
            if isinstance(meta.get("pillars"), dict) and len(meta["pillars"]) == 4:
                charts.add(S.chart_key(meta["pillars"]))
    for split in ("dev6", "valid6", "sealed6"):
        for line in (
            (Path("data/phase6/blind") / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ):
            charts.add(S.chart_key(json.loads(line)["meta"]["pillars"]))
    return frozenset(charts)


def load_replies(path):
    if not path:
        return None
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out.append((r["row"], r["reply"], r["correct"]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True, choices=list(SEEDS))
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument(
        "--data-recipe", help="build the data of another recipe (u1: the selected one)"
    )
    parser.add_argument("--replies", help="model replies on training charts, for self-check rows")
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument(
        "--grad-checkpoint", action="store_true", help="recompute activations (memory, not math)"
    )
    parser.add_argument(
        "--calendar-share", type=float, help="screening decision: calendar share of U1"
    )
    parser.add_argument(
        "--drop", nargs="*", default=[], help="screening decision: components left out of U1"
    )
    parser.add_argument("--extra-paraphrases", help="a second file of training-writer paraphrases")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    recipe = args.recipe
    data_recipe = args.data_recipe or recipe
    seed = SEEDS[recipe]
    exclude = exclusions()
    paraphrases = [
        json.loads(x)
        for x in Path("data/phase6/blind/train6.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if args.extra_paraphrases:
        paraphrases += [
            json.loads(x)
            for x in Path(args.extra_paraphrases).read_text(encoding="utf-8").splitlines()
        ]
    replies = load_replies(args.replies)
    options = {"calendar_share": args.calendar_share, "drop": tuple(args.drop)}
    train = T6.build(data_recipe, args.tokens, seed, exclude, paraphrases, replies, **options)
    valid = T6.build(data_recipe, 150_000, seed + 500, exclude, paraphrases, replies, **options)
    limit = MAX_TOKENS - 8 if T6._tokenizer() else int(MAX_TOKENS * 0.95)
    train = [r for r in train if T6.tokens(r) <= limit]
    valid = [r for r in valid if T6.tokens(r) <= limit]
    for r in train + valid:
        pillars = (r.get("meta") or {}).get("pillars")
        assert not (
            isinstance(pillars, dict)
            and len(pillars) == 4
            and S.chart_key(pillars) in exclude
            and r["family"] not in T6.OTHER_FAMILIES
        ), r["id"]
    folder = OUT / recipe
    files = {"train.jsonl": T6.payload(train), "valid.jsonl": T6.payload(valid)}
    if args.check:
        bad = False
        for name, data in files.items():
            same = (folder / name).exists() and hashlib.sha256(
                (folder / name).read_bytes()
            ).hexdigest() == hashlib.sha256(data).hexdigest()
            print(f"{folder / name}: {'ok' if same else 'MISMATCH'}")
            bad |= not same
        sys.exit(int(bad))
    if folder.exists():
        sys.exit(f"{folder} exists; Phase 6 training sets are never rewritten")
    folder.mkdir(parents=True)
    for name, data in files.items():
        (folder / name).write_bytes(data)
    manifest = {"recipe": recipe, "data_recipe": data_recipe, "seed": seed, "token_budget": args.tokens,
                "calendar_share": args.calendar_share, "drop": args.drop, "extra_paraphrases": args.extra_paraphrases,
                "replies": args.replies, "excluded_charts": len(exclude),
                "files": {name: {"sha256": hashlib.sha256(d).hexdigest(), "rows": len(train if name == 'train.jsonl' else valid)} for name, d in files.items()},
                "train_summary": T6.summary(train)}  # fmt: skip
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    name = {"u1hi": "u1-hi"}.get(recipe, recipe)
    config = CONFIG / f"qwen3_4b_lora_p6_{recipe}.yaml"
    if not config.exists():
        text = TEMPLATE.format(recipe=recipe, purpose=PURPOSE[recipe], name=name, seed=seed,
                               layers=args.layers, rank=args.rank, max_tokens=MAX_TOKENS)  # fmt: skip
        if args.grad_checkpoint:
            text += "grad_checkpoint: true    # same gradients, activations recomputed (memory)\n"
        config.write_text(text)
    print(
        json.dumps(
            {"recipe": recipe, "rows": len(train), "valid": len(valid), **T6.summary(train)},
            ensure_ascii=False,
        )[:1500]
    )


if __name__ == "__main__":
    main()
