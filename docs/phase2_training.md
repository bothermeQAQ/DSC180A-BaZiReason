# Phase 2: train, evaluate and run BaZiReason locally

Everything here runs on an Apple Silicon Mac with MLX. It was executed on a MacBook Pro M5 with 32 GB. No cloud GPU and no API key is needed. Weights are never committed; they live under `models/` and `artifacts/adapters/`, which Git ignores.

## 1. Environment

The Phase 1 environment (`.venv`, Python 3.14) is unchanged and still runs the calendar tests. Training uses a second environment, because one diagnostic dependency of Phase 1 (`sxtwl`) does not build on Python 3.12:

```bash
uv venv .venv-mlx --python 3.12
uv pip install --python .venv-mlx/bin/python -e '.[mlx]' pytest==9.2.0 ruff==0.17.3
```

Pinned versions: `mlx==0.32.4`, `mlx-lm==0.32.2`.

## 2. Base model (frozen revision)

```bash
.venv-mlx/bin/python scripts/phase2/download_models.py --only qwen3-4b-2507
SRC=$(python3 -c "import json;print(json.load(open('artifacts/models/download_manifest.json'))['qwen3-4b-2507']['local_path'])")
HF_HUB_OFFLINE=1 .venv-mlx/bin/python -m mlx_lm convert --hf-path "$SRC" \
  --mlx-path models/qwen3-4b-instruct-2507-mlx-4bit -q --q-bits 4 --q-group-size 64
shasum -a 256 models/qwen3-4b-instruct-2507-mlx-4bit/model.safetensors
# expected 2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f
```

Why this model: [phase2_model_selection.md](phase2_model_selection.md).

## 3. Dataset

```bash
.venv-mlx/bin/python scripts/phase2/build_dataset.py --check   # regenerate in memory, compare hashes
.venv-mlx/bin/python scripts/phase2/build_dataset.py --force   # rewrite data/phase2/train_v1
```

Description and leakage guards: [../data/phase2/README.md](../data/phase2/README.md).

## 4. Training

```bash
# 20 real optimizer steps into *_smoke folders (about two minutes)
.venv-mlx/bin/python scripts/phase2/train_lora.py --config configs/phase2/qwen3_4b_lora_v1.yaml --smoke

# the full first epoch (2,171 iterations, about 2 h 11 min, 22 GB peak)
nohup .venv-mlx/bin/python scripts/phase2/train_lora.py \
  --config configs/phase2/qwen3_4b_lora_v1.yaml > artifacts/logs/train_v1.log 2>&1 &
```

Outputs:

| Path | Content |
|---|---|
| `artifacts/adapters/<run>/NNNNNNN_adapters.safetensors` | a checkpoint every 250 iterations and at the end |
| `artifacts/adapters/<run>/adapters.safetensors` | the checkpoint with the lowest validation loss (what `mlx_lm.load` reads) |
| `artifacts/adapters/<run>/adapter_config.json` | LoRA shape and the base model revision |
| `results/phase2/training/<run>/metrics.jsonl` | one line per report: training loss, validation loss, learning rate, speed, tokens, peak memory |
| `results/phase2/training/<run>/training_summary.json` | counts, duration, selected checkpoint and its SHA-256 |

Keep the Mac awake and plugged in. Close other heavy applications: other work on the machine roughly halves the training speed, and the run needs about 22 GB.

### Resume an interrupted run

```bash
.venv-mlx/bin/python scripts/phase2/train_lora.py --config configs/phase2/qwen3_4b_lora_v1.yaml --resume
```

The script loads the highest-numbered checkpoint in the adapter folder and continues to the configured number of iterations. The learning-rate schedule continues from the same point; the optimizer's moment estimates restart, and the data order is reshuffled.

### Train further from a finished run (second epoch)

```bash
mkdir -p artifacts/adapters/bazireason-qwen3-4b-lora-v1-e2
cp artifacts/adapters/bazireason-qwen3-4b-lora-v1/0002171_adapters.safetensors \
   artifacts/adapters/bazireason-qwen3-4b-lora-v1-e2/
.venv-mlx/bin/python scripts/phase2/train_lora.py --config configs/phase2/qwen3_4b_lora_v1_e2.yaml --resume
```

`qwen3_4b_lora_v1_e2.yaml` is the same configuration with `epochs: 2.0` and its own output folders, so iterations 2,172-4,343 are a second pass over the data.

### If memory runs out

Lower `max_seq_length`, set `num_layers` to 8, or switch the configuration's `model` to the 2 B fallback (`models/minicpm5-2b-mlx-4bit`, about 12 GB at batch 4). Qwen3.5 and Gemma 4 small models do not train on this machine with MLX-LM 0.32.2 at these sequence lengths.

## 5. Evaluation

```bash
A=artifacts/adapters/bazireason-qwen3-4b-lora-v1
PY=.venv-mlx/bin/python

# held-out split of the training distribution (unseen years, some unseen wordings)
$PY scripts/phase2/run_p2_eval.py --label base  --output results/phase2/heldout/base.jsonl --paraphrase-probe
$PY scripts/phase2/run_p2_eval.py --label tuned --adapter $A --output results/phase2/heldout/tuned.jsonl --paraphrase-probe

# benchmark_v3 with the checkpoint's own prompts and scorer
$PY scripts/phase2/run_v3_local.py --label base --protocol strict \
  --methods direct direct_minimal tool_intermediate tool_context tool_corrupted \
  --output results/phase2/v3/base_strict.jsonl
.venv-mlx/bin/bazireason evaluate --predictions results/phase2/v3/base_strict.jsonl \
  --output results/phase2/v3/base_strict.jsonl.scores.json
# add --adapter $A --label tuned for the fine-tuned model; --protocol reasoned for visible reasoning

# BaziQA contest benchmark (evaluation only)
$PY scripts/phase2/fetch_benchmarks.py
$PY scripts/phase2/run_baziqa.py --who base  --tools none
$PY scripts/phase2/run_baziqa.py --who base  --tools verified
$PY scripts/phase2/run_baziqa.py --who tuned --tools none     --adapter $A
$PY scripts/phase2/run_baziqa.py --who tuned --tools verified --adapter $A
$PY scripts/phase2/score_baziqa.py

# one report from everything that exists
$PY scripts/phase2/make_report.py        # writes results/phase2/RESULTS.md and summary.json
```

Output files are never overwritten: every runner refuses an existing path, and the BaziQA runner resumes by appending the missing questions and runs. Measured wall times on this machine: held-out split 25 min (tuned) to 50 min (base, which writes longer replies); benchmark_v3 strict 16-19 min per model; one BaziQA condition 40-60 min for three runs.

## 6. Chat demo

```bash
.venv-mlx/bin/python -m bazireason.demo_app            # then open http://127.0.0.1:8765
```

- The chart on the left always comes from the verified engine. The model on the right explains and answers; a menu switches between the fine-tuned adapter and the untuned base model, and both get the same prompt.
- After each reply the app checks the statements it can verify (named pillars, day master, year pillars, Ten Gods of named stems, luck direction) against the engine and the rule tables, and flags any that disagree.
- "对比另一个出生时间" opens a second birth form; the comparison highlights which pillars change.
- Nothing is stored. Feedback is saved to `demo_feedback/feedback.jsonl` only when the tester ticks the consent box, and the conversation only with the second box.

Letting friends try it:

- **Same Wi-Fi, your Mac does the work:** `.venv-mlx/bin/python -m bazireason.demo_app --share-lan` prints an access code. Friends open `http://<your-Mac's-address>:8765` and enter the code. Their questions are processed on your Mac. Use this only on a network you trust; the connection is not encrypted.
- **Their own Apple Silicon Mac:** they clone the repository, follow sections 1 and 2, and copy your `artifacts/adapters/bazireason-qwen3-4b-lora-v1/` folder (`adapters.safetensors` and `adapter_config.json`, about 59 MB) into the same place. The adapter is not in Git; share it directly or publish it yourself when you decide to.

## 7. Tests

```bash
.venv/bin/python -m pytest -q          # 200 tests: 176 from Phase 1 plus 24 for Phase 2
.venv-mlx/bin/ruff check src scripts tests
```

The Phase 2 tests need no model: they check the rule tables against lunar-python, the worked answers against the engine and an independent oracle, the dataset hashes, the split and exclusion guards, and the scorers.
