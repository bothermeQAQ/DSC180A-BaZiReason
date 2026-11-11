#!/bin/zsh
# Runs after the round-1 training process exits: one GPU job at a time.
# Usage: scripts/phase3/queue_round1.sh <training pid>
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
P3=artifacts/adapters/bazireason-qwen3-4b-lora-p3-r1
P2=artifacts/adapters/bazireason-qwen3-4b-lora-v2
H=results/phase3/heldout
V=results/phase3/v3
METHODS=(direct direct_minimal tool_intermediate tool_context tool_corrupted)
while kill -0 "$1" 2>/dev/null; do sleep 30; done
[ -f $P3/adapters.safetensors ] || { echo "QUEUE_ABORT no adapter"; exit 1; }
mkdir -p $H $V
run() { echo "=== $(date '+%H:%M:%S') $*"; "$@" || echo "STEP_FAILED $*"; }
run $PY scripts/phase3/run_p3_eval.py --label p3-r1 --adapter $P3 --output $H/p3test_p3r1.jsonl
run $PY scripts/phase3/run_p3_eval.py --label p2-v2 --adapter $P2 --output $H/p3test_p2.jsonl
run $PY scripts/phase2/run_v3_local.py --label p3-r1 --adapter $P3 --protocol strict --methods $METHODS --output $V/p3r1_strict.jsonl
run .venv-mlx/bin/bazireason evaluate --predictions $V/p3r1_strict.jsonl --output $V/p3r1_strict.jsonl.scores.json
run $PY scripts/phase2/run_p2_eval.py --label p3-r1 --adapter $P3 --output $H/p2test_p3r1.jsonl --paraphrase-probe
run $PY scripts/phase2/run_p2_eval.py --label p3-r1 --adapter $P3 --data data/phase2/train_v2/test.jsonl --output $H/p2fresh_p3r1.jsonl
echo "QUEUE_ROUND1_DONE $(date '+%H:%M:%S')"
