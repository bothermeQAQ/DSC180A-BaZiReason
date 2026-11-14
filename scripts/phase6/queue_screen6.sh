#!/bin/sh
# Phase 6 screening: S1, S2, S3 (6.0 M tokens each, from the base), each scored on dev6.
# S3's self-check rows come from S1's own replies on fresh training-distribution questions.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
export PYTHONPATH=src
run() { echo "=== $(date '+%H:%M:%S') $*"; "$@" || echo "STEP_FAILED $*"; }
ALL="rules indep blind combo cf"
until grep -q "QUEUE_U0_DONE" artifacts/logs/p6_queue_u0.log; do sleep 60; done
run $PY scripts/phase2/train_lora.py --config configs/phase6/qwen3_4b_lora_p6_s1.yaml
run $PY scripts/phase6/run6.py --split dev6 --labels p6s1 --parts $ALL calendar --levels L0
run $PY scripts/phase6/own_replies.py --label p6s1 --n 2400
run $PY scripts/phase2/train_lora.py --config configs/phase6/qwen3_4b_lora_p6_s2.yaml
run $PY scripts/phase6/run6.py --split dev6 --labels p6s2 --parts $ALL calendar --levels L0 L1 L4
run $PY scripts/phase6/build_train6.py --recipe s3 --tokens 6000000 --replies results/phase6/own_replies_p6s1.jsonl
run $PY scripts/phase2/train_lora.py --config configs/phase6/qwen3_4b_lora_p6_s3.yaml
run $PY scripts/phase6/run6.py --split dev6 --labels p6s3 --parts $ALL calendar --levels L0 L1 L4
run $PY scripts/phase6/run6.py --split dev6 --labels p6u0 p4a1 --parts $ALL --levels L1 L4
echo "QUEUE_SCREEN6_DONE $(date '+%H:%M:%S')"
