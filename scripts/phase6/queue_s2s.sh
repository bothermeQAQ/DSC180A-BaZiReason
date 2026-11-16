#!/bin/sh
# Phase 6, the H7 twin (preregistration 12.4): S2's recipe, seed and token budget with the short
# system prompt. Stops at the first failed step and says so; queue_main.sh waits for its marker.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
export PYTHONPATH=src
run() { echo "=== $(date "+%H:%M:%S") $*"; "$@" || { echo "STEP_FAILED $*"; return 1; }; }
fail() { echo "QUEUE_S2S_FAILED $(date '+%m-%d %H:%M:%S')"; exit 1; }
ALL="rules indep blind combo cf"
[ -f results/phase6/training/bazireason-qwen3-4b-lora-p6-s2s/training_summary.json ] || run $PY scripts/phase2/train_lora.py --config configs/phase6/qwen3_4b_lora_p6_s2s.yaml || fail
run $PY scripts/phase6/run6.py --split dev6 --labels p6s2s --parts $ALL calendar --levels L0 || fail
echo "QUEUE_S2S_DONE $(date '+%m-%d %H:%M:%S')"
