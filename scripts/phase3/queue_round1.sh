#!/bin/zsh
# Runs after the round-1 training process exits: one GPU job at a time.
# Usage: scripts/phase3/queue_round1.sh <training pid>
# Only the development split (validation years) is scored here; the test split and the
# benchmarks stay unopened until an adapter has been selected.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
P3=artifacts/adapters/bazireason-qwen3-4b-lora-p3-r1
P2=artifacts/adapters/bazireason-qwen3-4b-lora-v2
H=results/phase3/heldout
while kill -0 "$1" 2>/dev/null; do sleep 30; done
[ -f $P3/adapters.safetensors ] || { echo "QUEUE_ABORT no adapter"; exit 1; }
mkdir -p $H
run() { echo "=== $(date '+%H:%M:%S') $*"; "$@" || echo "STEP_FAILED $*"; }
run $PY scripts/phase3/run_p3_eval.py --label p3-r1 --adapter $P3 --data data/phase3/dev_v1/dev.jsonl --output $H/dev_p3r1.jsonl
run $PY scripts/phase3/run_p3_eval.py --label p3-r1 --adapter $P3 --data data/phase3/train_v1/valid.jsonl --output $H/valid_p3r1.jsonl
run $PY scripts/phase3/run_p3_eval.py --label p2-v2 --adapter $P2 --data data/phase3/train_v1/valid.jsonl --output $H/valid_p2.jsonl
echo "QUEUE_ROUND1_DONE $(date '+%H:%M:%S')"
