"""Build one Phase 6 training mixture and its configuration.

    PYTHONPATH=src .venv-mlx/bin/python scripts/phase6/build_train6.py --recipe u0 --tokens 6000000
    PYTHONPATH=src .venv-mlx/bin/python scripts/phase6/build_train6.py --recipe s3 --tokens 6000000 \
        --replies results/phase6/own_replies_p6s1.jsonl
    ... --check      # rebuild in memory and compare hashes

Round 2 (docs/phase6_preregistration.md, section 10):

    ... --recipe u2 --data-recipe s3 --tokens 15000000 --system short --replies ... \
        --extra-paraphrases data/phase6/blind/train6b.jsonl data/phase6/blind/train6c.jsonl
    ... --recipe u3 --parent u2 --tokens 6000000 \
        --weights-from results/phase6/own_replies_p6u2_b.jsonl     # a weighted continuation

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

from bazireason import prompts6 as P6
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
    "s2s": 2026111712,  # section 12.4: S2's seed, recipe and budget with the short system prompt
    "u1": 2026111714,
    "u1hi": 2026111714,
    "u2": 2026111715,
    "u3": 2026111716,
    "u3v": 2026111716,  # the fork of 10.5: U3's seed, derivation version 2
    "u4": 2026111717,
    "u4c": 2026111717,  # the arm of 12.7: U4's seed, with composition rows
    "u5": 2026111718,
    "u6": 2026111719,
    "u7": 2026111720,
    "u8": 2026111721,
    "u9": 2026111722,
    "u10": 2026111723,
}
ROUND2 = ("u2", "u3", "u3v", "u4", "u4c", "u5", "u6", "u7", "u8", "u9", "u10")
FIT_B_SEED = 2026111791  # fit sample B (own_replies.py --sample b); never training input
FIT_B_PER_TYPE = 100
FIT_C_SEED = (
    2026111792  # fit sample C (own_replies.py --sample c): calendar requests, never training input
)
FIT_C_PER_CELL = 40
ERROR_FLOOR = 0.05
FOLDER = "bazireason-qwen3-4b-lora-p6-{name}"
TEMPLATE = """# BaZiReason Phase 6, {recipe}: {purpose}
# {start} Data: scripts/phase6/build_train6.py.
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
learning_rate: {lr}
warmup_updates: 40
end_learning_rate: {end_lr}
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
    "u2": "u1's recipe with the short system prompt at the same token budget (round 2)",
    **dict.fromkeys(
        ("u3", "u4", "u5", "u6", "u7", "u8", "u9", "u10"),
        "a continuation weighted by its parent's measured errors (round 2)",
    ),
    "u3v": "u3 with derivation version 2 (round 2, the fork of preregistration 10.5)",
    "u4c": "u4 with 10% composition rows (round 2, the arm of preregistration 12.7)",
    "s2s": "s2 with the short system prompt at the same token budget (the H7 twin, section 12.4)",
}


def folder_name(recipe):
    return FOLDER.format(name={"u1hi": "u1-hi"}.get(recipe, recipe))


def structure_of(recipe):
    """Adapter structure of a built recipe, read from its training configuration."""
    out = {"layers": 16, "rank": 16, "grad_checkpoint": False}
    for line in (CONFIG / f"qwen3_4b_lora_p6_{recipe}.yaml").read_text().splitlines():
        key, _, value = line.strip().partition(":")
        value = value.split("#")[0].strip()
        if key == "num_layers":
            out["layers"] = int(value)
        elif key == "rank":
            out["rank"] = int(value)
        elif key == "grad_checkpoint":
            out["grad_checkpoint"] = value == "true"
    return out


def fit_b_rows():
    return T6.fit_sample_b(FIT_B_SEED, FIT_B_PER_TYPE, exclusions())


def fit_c_rows():
    return T6.fit_sample_c(FIT_C_SEED, FIT_C_PER_CELL)


def calendar_weights(replies_path):
    """Weights of the calendar cells from the parent's replies to fit sample C (section 12.5):
    error = (wrong + 1) / (n + 2); weight = max(error, 0.05) / the mean of that over the cells."""
    count = {}
    for line in Path(replies_path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        c = count.setdefault(T6.calendar_cell(r["row"]), [0, 0])
        c[0] += 1
        c[1] += not r["correct"]
    raw = {cell: max((w + 1) / (n + 2), ERROR_FLOOR) for cell, (n, w) in sorted(count.items())}
    mean = sum(raw.values()) / len(raw)
    return {
        "source": str(replies_path),
        "scored_wrong": {cell: count[cell] for cell in raw},
        "weights": {cell: round(v / mean, 4) for cell, v in raw.items()},
    }


def training_charts(recipe):
    """Charts of every rule-task row a recipe and its ancestors were trained on."""
    charts = set()
    for name in P6.lineage(recipe):
        path = OUT / name / "train.jsonl"
        if not path.exists():
            sys.exit(f"{path} is missing (rebuild it with build_train6.py)")
        for line in path.read_text(encoding="utf-8").splitlines():
            charts.add(T6.chart_of(json.loads(line)))
    return charts - {None}


def error_weights(replies_path, parent):
    """Weights of the question types from the parent's replies to fit sample B (section 10.2):
    error = (wrong + 1) / (n + 2) over the questions whose chart the parent never trained on;
    weight = max(error, 0.05)."""
    seen = training_charts(parent)
    count = {h: [0, 0] for h in T6.WEIGHTS}
    left_out = 0
    for line in Path(replies_path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if T6.chart_of(r["row"]) in seen:
            left_out += 1
            continue
        c = count[r["row"]["meta"]["head"]]
        c[0] += 1
        c[1] += not r["correct"]
    error = {h: (w + 1) / (n + 2) for h, (n, w) in count.items()}
    return {
        "source": str(replies_path),
        "left_out": left_out,
        "scored_wrong": {h: [n, w] for h, (n, w) in count.items()},
        "error": {h: round(e, 4) for h, e in error.items()},
        "weights": {h: round(max(e, ERROR_FLOOR), 4) for h, e in error.items()},
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
    parser.add_argument(
        "--extra-paraphrases", nargs="*", help="more files of training-writer paraphrases"
    )
    parser.add_argument(
        "--system", choices=("long", "short"), default="long", help="round 2: the system prompt"
    )
    parser.add_argument("--parent", help="round 2: continue this recipe's adapter and options")
    parser.add_argument("--weights-from", help="round 2: the parent's replies to fit sample B")
    parser.add_argument(
        "--steps", type=int, choices=(1, 2), help="round 2: derivation version (steps6)"
    )
    parser.add_argument(
        "--calendar-weights-from", help="round 2: the parent's replies to fit sample C (12.5)"
    )
    parser.add_argument(
        "--compose-share", type=float, help="round 2: share of composition rows (12.7)"
    )
    parser.add_argument(
        "--years",
        choices=("fixed", "life"),
        help="round 2: years of flow and area questions (12.11)",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    recipe = args.recipe
    weighting = None
    if args.parent:  # a continuation keeps its parent's options, system prompt and structure
        if recipe not in ROUND2 or not args.weights_from:
            sys.exit("--parent needs a round-2 recipe and --weights-from")
        inherited = P6.manifest(args.parent)
        args.data_recipe = inherited["data_recipe"]
        args.replies = inherited["replies"]
        args.calendar_share = inherited["calendar_share"]
        args.drop = inherited["drop"]
        extra = inherited["extra_paraphrases"] or []
        args.extra_paraphrases = [extra] if isinstance(extra, str) else list(extra)
        args.system = inherited.get("system", "long")
        args.steps = args.steps or inherited.get("steps", 1)
        if args.compose_share is None:
            args.compose_share = inherited.get("compose_share") or 0.0
        args.years = args.years or inherited.get("years", "fixed")
        shape = structure_of(args.parent)
        args.layers, args.rank = shape["layers"], shape["rank"]
        args.grad_checkpoint = shape["grad_checkpoint"]
        weighting = error_weights(args.weights_from, args.parent)
    data_recipe = args.data_recipe or recipe
    seed = SEEDS[recipe]
    exclude = exclusions()
    block = frozenset()
    if recipe in ROUND2:
        block = frozenset(T6.chart_of(r) for r in fit_b_rows())
    paraphrases = [
        json.loads(x)
        for x in Path("data/phase6/blind/train6.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for extra in args.extra_paraphrases or []:
        paraphrases += [json.loads(x) for x in Path(extra).read_text(encoding="utf-8").splitlines()]
    # A training-writer question on an evaluation chart is left out (section 12.2): the same
    # four pillars can recur on a date of the training years.
    kept = [p for p in paraphrases if S.chart_key(p["meta"]["pillars"]) not in exclude]
    left_out, paraphrases = len(paraphrases) - len(kept), kept
    replies = load_replies(args.replies)
    options = {"calendar_share": args.calendar_share, "drop": tuple(args.drop)}
    round2 = recipe in ROUND2 or args.system == "short"
    calendar_weighting = None
    if args.calendar_weights_from:
        if not args.parent:
            sys.exit("--calendar-weights-from is for continuations (--parent)")
        calendar_weighting = calendar_weights(args.calendar_weights_from)
        options["calendar_weights"] = calendar_weighting["weights"]
        options["calendar_block"] = frozenset(T6.first_user(r) for r in fit_c_rows())
    if round2:
        options["system"] = P6.SHORT_SYSTEM if args.system == "short" else None
        options["weights"] = weighting["weights"] if weighting else None
        options["block"] = block
        options["steps"] = args.steps or 1
        options["compose_share"] = args.compose_share or 0.0
        options["years"] = args.years or "fixed"
    train = T6.build(data_recipe, args.tokens, seed, exclude, paraphrases, replies, **options)
    valid = T6.build(data_recipe, 150_000, seed + 500, exclude, paraphrases, replies, **options)
    limit = MAX_TOKENS - 8 if T6._tokenizer() else int(MAX_TOKENS * 0.95)
    train = [r for r in train if T6.tokens(r) <= limit]
    valid = [r for r in valid if T6.tokens(r) <= limit]

    def held_out(r):  # a row on an evaluation chart or, in round 2, on a chart of fit sample B
        chart = T6.chart_of(r)
        return chart in block or (chart in exclude and r["family"] not in T6.OTHER_FAMILIES)

    # The generators already avoid held-out charts; a row that still lands on one is left out
    # and counted (section 12.2), so that a build cannot stop a queue. None in S1, S2 and S3.
    dropped = [sum(held_out(r) for r in train), sum(held_out(r) for r in valid)]
    train = [r for r in train if not held_out(r)]
    valid = [r for r in valid if not held_out(r)]
    for r in train + valid:
        assert T6.chart_of(r) not in block, r["id"]
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
                "paraphrases_on_evaluation_charts_left_out": left_out,
                "rows_on_held_out_charts_left_out": {"train": dropped[0], "valid": dropped[1]},
                **({"system": args.system, "parent": args.parent, "steps": args.steps or 1, "weighting": weighting,
                    "fit_sample_b_charts_blocked": len(block), "calendar_weighting": calendar_weighting,
                    "compose_share": args.compose_share or 0.0, "years": args.years or "fixed"} if round2 else {}),
                "files": {name: {"sha256": hashlib.sha256(d).hexdigest(), "rows": len(train if name == 'train.jsonl' else valid)} for name, d in files.items()},
                "train_summary": T6.summary(train)}  # fmt: skip
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    name = {"u1hi": "u1-hi"}.get(recipe, recipe)
    config = CONFIG / f"qwen3_4b_lora_p6_{recipe}.yaml"
    if not config.exists():
        start = "Trained from the untuned base (no init adapter)."
        lr = "2.0e-4      # as the first Phase 2 round: a fresh adapter from the base"
        end_lr = "2.0e-5"
        if args.parent:
            start = f"Continues {folder_name(args.parent)} (its files are not modified)."
            lr = "1.0e-4      # a continuation: half the first-round rate"
            end_lr = "1.0e-5"
        text = TEMPLATE.format(recipe=recipe, purpose=PURPOSE[recipe], name=name, seed=seed, start=start, lr=lr,
                               end_lr=end_lr, layers=args.layers, rank=args.rank, max_tokens=MAX_TOKENS)  # fmt: skip
        if args.grad_checkpoint:
            text += "grad_checkpoint: true    # same gradients, activations recomputed (memory)\n"
        if args.parent:
            text += f"init_adapter: artifacts/adapters/{folder_name(args.parent)}/adapters.safetensors\n"
        config.write_text(text)
    print(
        json.dumps(
            {"recipe": recipe, "rows": len(train), "valid": len(valid), **T6.summary(train)},
            ensure_ascii=False,
        )[:1500]
    )


if __name__ == "__main__":
    main()
