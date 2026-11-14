#!/bin/sh
# Phase 6 U1-hi (preregistration section 5): trained only if U1 answers fewer than 95% of fresh
# training-distribution questions right at L0, and only if it starts by 27 November. Same data
# as U1 (same recipe, seed and options); only the adapter differs. Memory fallbacks are those of
# section 9: all 36 layers; else the same with gradient checkpointing; else rank 64 on 16 layers.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
export PYTHONPATH=src
run() { echo "=== $(date "+%H:%M:%S") $*"; "$@" || { echo "STEP_FAILED $*"; return 1; }; }
ALL="rules indep blind combo cf"
until grep -q "QUEUE_U1_DONE" artifacts/logs/p6_queue_u1.log; do sleep 60; done
D=".venv/bin/python scripts/phase6/decide6.py"
FLAGS=$($D flags)
if ! .venv/bin/python -c 'import json,sys; sys.exit(0 if json.load(open("results/phase6/fit_p6u1.json"))["share"] < 0.95 else 1)'; then
  echo "U1-hi not trained: U1 fit is at least 95%"; echo "QUEUE_U1HI_DONE $(date '+%H:%M:%S')"; exit 0
fi
if [ "$(date '+%Y%m%d')" -gt 20261127 ]; then
  echo "U1-hi not trained: after 27 November"; echo "QUEUE_U1HI_DONE $(date '+%H:%M:%S')"; exit 0
fi
P="$PY scripts/phase6/probe_memory.py --config configs/phase6/qwen3_4b_lora_p6_u1.yaml"
if run $P --layers 36; then OPT="--layers 36"
elif run $P --layers 36 --grad-checkpoint; then OPT="--layers 36 --grad-checkpoint"
else OPT="--layers 16 --rank 64"; fi
echo "U1-hi adapter: $OPT"
# shellcheck disable=SC2086
run $PY scripts/phase6/build_train6.py --recipe u1hi --data-recipe s3 --tokens 15000000 --replies results/phase6/own_replies_p6s1.jsonl --extra-paraphrases data/phase6/blind/train6b.jsonl $FLAGS $OPT
if cmp -s data/phase6/train/u1/train.jsonl data/phase6/train/u1hi/train.jsonl; then echo "U1-hi data identical to U1: yes"
else echo "STEP_FAILED U1-hi data differs from U1"; exit 1; fi
run $PY scripts/phase2/train_lora.py --config configs/phase6/qwen3_4b_lora_p6_u1hi.yaml
run $PY scripts/phase6/own_replies.py --label p6u1hi --n 2400
run $PY scripts/phase6/run6.py --split dev6 --labels p6u1hi --parts $ALL calendar --levels L0
run $PY scripts/phase6/run6.py --split valid6 --labels p6u1hi --parts $ALL calendar --levels L0
echo "QUEUE_U1HI_DONE $(date '+%H:%M:%S')"
