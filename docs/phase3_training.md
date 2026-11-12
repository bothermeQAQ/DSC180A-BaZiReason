# Phase 3: rule reasoning — train, evaluate and run locally

Phase 3 continues from the Phase 2 adapter on the same machine and with the same tools (Apple Silicon, MLX, no cloud GPU, no API key). Set up the environment and the base model as in [phase2_training.md](phase2_training.md), sections 1 and 2. Results and their interpretation: [phase3_progress_report.md](phase3_progress_report.md).

Nothing from Phase 1 or Phase 2 is replaced: the frozen checkpoint, both Phase 2 adapters and all earlier result files stay where they were. Phase 3 writes to `data/phase3/`, `configs/phase3/`, `results/phase3/` and to new adapter folders.

## 1. Knowledge base

```bash
.venv/bin/python scripts/phase3/build_knowledge.py --fetch
```

Downloads the cited public-domain texts from zh.wikisource.org into the git-ignored `external/classics/`, checks that every quotation in the rule cards occurs in them, and writes `data/phase3/knowledge/rules_v2.json`. Without `--fetch` it uses the pages already downloaded.

## 2. Datasets

```bash
.venv/bin/python scripts/phase3/build_dataset.py --check               # p3_train_v1
.venv/bin/python scripts/phase3/build_dataset.py --dataset 2 --check   # p3_train_v2
.venv/bin/python scripts/phase3/build_dataset.py --dataset 3 --check   # p3_train_v3
```

All three regenerate byte for byte. Families, label sources and leakage guards: [../data/phase3/README.md](../data/phase3/README.md).

## 3. Training

Each round is a warm start: the configuration's `init_adapter` names the adapter to continue from, and the run writes to its own folder.

```bash
# Round 1: from the Phase 2 adapter, 6,933 rows, about 3 hours, 20 GB peak
nohup caffeinate -i .venv-mlx/bin/python scripts/phase2/train_lora.py \
  --config configs/phase3/qwen3_4b_lora_p3_r1.yaml > artifacts/logs/train_p3_r1.log 2>&1 &

# Round 2: from the round-1 adapter, 4,122 rows in the explicit wording, about 2 hours
nohup caffeinate -i .venv-mlx/bin/python scripts/phase2/train_lora.py \
  --config configs/phase3/qwen3_4b_lora_p3_r2.yaml > artifacts/logs/train_p3_r2.log 2>&1 &

# Round 3 (corrective): from the round-2 adapter, 3,345 rows (half Phase 2 replay),
# learning rate 5e-5, about 1.6 hours
nohup caffeinate -i .venv-mlx/bin/python scripts/phase2/train_lora.py \
  --config configs/phase3/qwen3_4b_lora_p3_r3.yaml > artifacts/logs/train_p3_r3.log 2>&1 &
```

Add `--smoke` for 20 real steps into a `_smoke` folder, or `--resume` to continue an interrupted run from its last checkpoint. Outputs have the same layout as in Phase 2: checkpoints every 500 iterations and `adapters.safetensors` (lowest validation loss) under `artifacts/adapters/<run>/`, logs under `results/phase3/training/<run>/`.

`scripts/phase3/queue_round2.sh` and `queue_round3.sh` chain one round, its adapter selection and every evaluation below, one GPU job at a time.

## 4. Choosing the adapter

```bash
.venv-mlx/bin/python scripts/phase3/run_p3_eval.py --label p3-r1 \
  --adapter artifacts/adapters/bazireason-qwen3-4b-lora-p3-r1 \
  --data data/phase3/train_v1/valid.jsonl --output results/phase3/heldout/valid_p3r1.jsonl
# the same for p3-r2, then:
.venv/bin/python scripts/phase3/select_adapter.py      # writes results/phase3/selection.json
```

The rule is fixed in the script: the round with more correct generated answers on the validation split wins; the test split and the benchmarks are not consulted. The first round's errors were read on `data/phase3/dev_v1/dev.jsonl` (validation years), never on the test split.

Round 3 against round 2 uses a second rule, also written before round 3 was scored:

```bash
.venv/bin/python scripts/phase3/select_final.py   # writes results/phase3/selection_final.json
```

The script adds the correct answers on the Phase 3 validation split (298 rows) and on the Phase 2 validation split `data/phase2/train_v2/valid.jsonl` (203 rows), so calendar skills count as much as rule reasoning. A tie keeps round 2. Round 2 won, 357 to 353, so **the Phase 3 adapter is `artifacts/adapters/bazireason-qwen3-4b-lora-p3-r2`**. Round 3 is still evaluated everywhere to document the tradeoff.

## 5. Evaluation

```bash
PY=.venv-mlx/bin/python
A=$(python3 -c "import json;print(json.load(open('results/phase3/selection_final.json'))['selected_adapter'])")
P2=artifacts/adapters/bazireason-qwen3-4b-lora-v2

# Phase 3 held-out test split: the same prompts for every model
$PY scripts/phase3/run_p3_eval.py --label p3   --adapter $A  --output results/phase3/heldout/p3test_p3.jsonl
$PY scripts/phase3/run_p3_eval.py --label p2   --adapter $P2 --output results/phase3/heldout/p3test_p2.jsonl
$PY scripts/phase3/run_p3_eval.py --label base               --output results/phase3/heldout/p3test_base.jsonl

# benchmark_v3 with the checkpoint's own prompts and scorer (strict, then reasoned)
$PY scripts/phase2/run_v3_local.py --label p3 --adapter $A --protocol strict \
  --methods direct direct_minimal tool_intermediate tool_context tool_corrupted \
  --output results/phase3/v3/p3_strict.jsonl
.venv-mlx/bin/bazireason evaluate --predictions results/phase3/v3/p3_strict.jsonl \
  --output results/phase3/v3/p3_strict.jsonl.scores.json

# Regression on the Phase 2 held-out splits
$PY scripts/phase2/run_p2_eval.py --label p3 --adapter $A --output results/phase3/heldout/p2test_p3.jsonl

# BaziQA with a 1,024-token budget (evaluation only), three runs per condition
$PY scripts/phase2/run_baziqa.py --who tuned --tools none     --adapter $A  --output-dir results/phase3/baziqa1024/p3 --max-tokens 1024
$PY scripts/phase2/run_baziqa.py --who tuned --tools verified --adapter $A  --output-dir results/phase3/baziqa1024/p3 --max-tokens 1024
$PY scripts/phase2/run_baziqa.py --who tuned --tools none     --adapter $P2 --output-dir results/phase3/baziqa1024/p2 --max-tokens 1024
$PY scripts/phase2/run_baziqa.py --who tuned --tools verified --adapter $P2 --output-dir results/phase3/baziqa1024/p2 --max-tokens 1024
.venv/bin/python scripts/phase3/score_baziqa3.py

# One report from everything that exists
$PY scripts/phase3/make_report.py      # writes results/phase3/RESULTS.md and summary.json
```

The calendar check split `data/phase3/p2check_v1/test.jsonl` is scored with `run_p2_eval.py --data` for the Phase 2 adapter and rounds 2 and 3. The file names used by the queue scripts carry the round (`p3test_p3r2.jsonl` and so on); `make_report.py` reads those. No runner overwrites an existing output. `scripts/phase3/rescore.py` recomputes a score file from saved raw outputs without running the model.

## 6. Chat demo

```bash
.venv-mlx/bin/python -m bazireason.demo_app            # then open http://127.0.0.1:8765
```

- **Three models.** The menu offers the Phase 3 adapter (round 2 by default), the Phase 2 adapter and the untuned base model. All three receive the same prompt. `--adapter` and `--adapter-p2` point at other adapter folders; `--preload all` loads the three models at start (about 9 GB).
- **Verified chart.** The four pillars, the Ten Gods and the "深入结构" panel (strength count, pattern by 透干, 用神 by school, relations, stars) are computed by the engine and the rule code, never by the model.
- **Question routing.** A free question is mapped to the closest trained request (`bazireason.demo_deep.plan`): whole-chart overview, strength, pattern, 用神, relations, a flow year, the traditional view of an area of life, a star, a list of statements to check. "查看实际发给模型的提问" under each reply shows the exact text the model received.
- **Checks after each reply.** The reply's structured conclusion is compared with the rule code. On a mismatch the page shows the rule code's value and its full derivation.
- **No predictions.** A question that asks for a definite personal outcome ("我什么时候会结婚？") is routed to the declining format. "用神是什么？" without a school, or "这个八字好不好？", is routed to the format that says there is no unique answer and why.
- **Privacy.** Unchanged from Phase 2: local by default, nothing stored, feedback only with the consent box, conversation only with the second box, `--share-lan` with an access code for friends on the same network.

## 7. Tests

```bash
.venv/bin/python -m pytest -q          # 176 Phase 1 + 28 Phase 2 + 24 Phase 3 tests
.venv-mlx/bin/ruff check src scripts tests
```

The Phase 3 tests need no model: rule tables against the twelve-stage table and the element cycle, the declared conventions, quotations against the downloaded pages, dataset hashes and leakage guards, and the scorers.
