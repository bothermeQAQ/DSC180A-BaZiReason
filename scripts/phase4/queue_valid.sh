#!/bin/zsh
# Phase 4 part 1: wait for the A1 training run, score every adapter on the validation split
# (workflow W0 only), then apply the frozen selection rule. One GPU job at a time.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
A=artifacts/adapters
run() { echo "=== $(date '+%H:%M:%S') $*"; "$@" || echo "STEP_FAILED $*"; }
while pgrep -f "train_lora.py --config configs/phase4" > /dev/null; do sleep 60; done
[ -f $A/bazireason-qwen3-4b-lora-p4-a1/adapters.safetensors ] || { echo "QUEUE_ABORT no A1 adapter"; exit 1; }
for pair in "p2 bazireason-qwen3-4b-lora-v2" "p3r2 bazireason-qwen3-4b-lora-p3-r2" \
            "p3r3 bazireason-qwen3-4b-lora-p3-r3" "p4a1 bazireason-qwen3-4b-lora-p4-a1"; do
  label=${pair%% *}; dir=${pair#* }
  run $PY scripts/phase4/run_eval.py --label $label --adapter $A/$dir --split valid --workflow w0
done
run .venv-mlx/bin/python scripts/phase4/select.py
echo "QUEUE_VALID_DONE $(date '+%H:%M:%S')"
