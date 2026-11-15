#!/bin/sh
# Phase 6, the larger-base pilot (preregistration section 11): S1's training file and settings
# on Qwen3-8B, after the screening queue and before U1. It always ends with QUEUE_P8_DONE, so
# U1 goes on if the pilot stops for a technical reason.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
export PYTHONPATH=src
run() { echo "=== $(date "+%H:%M:%S") $*"; "$@" || { echo "STEP_FAILED $*"; return 1; }; }
ALL="rules indep blind combo cf"
NAME=bazireason-qwen3-8b-lora-p6-s1
until grep -q "QUEUE_SCREEN6_DONE" artifacts/logs/p6_queue_screen6.log; do sleep 60; done
pilot() {
  [ -f models/qwen3-8b-mlx-4bit/model.safetensors ] || { echo "no Qwen3-8B weights"; return 1; }
  if [ ! -f configs/phase6/qwen3_8b_lora_p6_s1.yaml ]; then
    # Memory probe on the 16 longest S1 rows; a failed probe prints STEP_FAILED by design.
    P="$PY scripts/phase6/probe_memory.py --config configs/phase6/qwen3_4b_lora_p6_s1.yaml --model models/qwen3-8b-mlx-4bit --layers 16"
    if run $P --batch 2; then OPT="--batch 2"
    elif run $P --batch 1; then OPT="--batch 1"
    elif run $P --batch 1 --grad-checkpoint; then OPT="--batch 1 --grad-checkpoint"
    else echo "Qwen3-8B does not fit in memory with any probed set-up"; return 1; fi
    echo "Qwen3-8B set-up: $OPT"
    # shellcheck disable=SC2086
    run .venv/bin/python scripts/phase6/p8_config.py $OPT || return 1
  fi
  [ -f "results/phase6/training/$NAME/training_summary.json" ] || run $PY scripts/phase2/train_lora.py --config configs/phase6/qwen3_8b_lora_p6_s1.yaml || return 1
  run $PY scripts/phase6/run6.py --split dev6 --labels p8s1 --parts $ALL calendar --levels L0 || return 1
}
pilot || echo "PILOT_STOPPED $(date '+%H:%M:%S')"
echo "QUEUE_P8_DONE $(date '+%H:%M:%S')"
