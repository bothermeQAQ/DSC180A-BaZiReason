#!/bin/sh
# Wait for the development L0 baseline, stop the rest of queue_base6 (its later levels run on
# valid6 and sealed6 instead), then train U0 and score it on dev6 at L0.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
export PYTHONPATH=src
run() { echo "=== $(date '+%H:%M:%S') $*"; "$@" || echo "STEP_FAILED $*"; }
until grep -q -- "--levels L1" artifacts/logs/p6_queue_base6.log; do sleep 30; done
pkill -f queue_base6.sh; pkill -f "run6.py --split dev6 --labels p4a1 p5g1 base"; sleep 20
echo "stopped queue_base6 after the dev6 L0 baseline $(date '+%H:%M:%S')"
run $PY scripts/phase2/train_lora.py --config configs/phase6/qwen3_4b_lora_p6_u0.yaml
run $PY scripts/phase6/run6.py --split dev6 --labels p6u0 --parts rules indep blind combo cf calendar --levels L0
echo "QUEUE_U0_DONE $(date '+%H:%M:%S')"
