#!/bin/zsh
# After the development queue: train G1 (one GPU job at a time), then G1's development runs.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
until grep -q "QUEUE_DEV3_DONE" artifacts/logs/p5_queue_dev3.log; do sleep 30; done
echo "=== $(date '+%H:%M:%S') train G1"
$PY scripts/phase2/train_lora.py --config configs/phase5/qwen3_4b_lora_p5_g1.yaml > artifacts/logs/train_p5_g1.log 2>&1 || echo "STEP_FAILED train"
echo "=== $(date '+%H:%M:%S') G1 development runs"
$PY scripts/phase5/run5.py --split dev --labels p5g1 --parts rules indep blind calendar --conditions w0 go gr grv --prompt-version 2 || echo "STEP_FAILED dev g1"
echo "QUEUE_TRAIN_DONE $(date '+%H:%M:%S')"
