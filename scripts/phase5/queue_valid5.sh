#!/bin/zsh
# Phase 5 validation (selection only), then the frozen selection. One GPU job at a time.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
run() { echo "=== $(date '+%H:%M:%S') $*"; "$@" || echo "STEP_FAILED $*"; }
until grep -q "QUEUE_TRAIN_DONE" artifacts/logs/p5_queue_train.log; do sleep 30; done
run .venv/bin/python scripts/phase5/select.py alpha
M=$(.venv/bin/python -c "import json;print(json.load(open('results/phase5/dev_alpha.json'))['label'])")
run $PY scripts/phase5/run5.py --split valid --batch 32 --labels p2 p3r3 p4a1 $M p5g1 --parts calendar --conditions w0
run $PY scripts/phase5/run5.py --split valid --batch 32 --labels p4a1 $M p5g1 --parts rules indep blind --conditions w0
run $PY scripts/phase5/run5.py --split valid --batch 32 --labels p4a1 p5g1 p3r3 $M base --parts rules indep blind --conditions gr grv
run $PY scripts/phase5/convo.py run --split valid --batch 32 --labels p4a1 --conditions w0 grv
run $PY scripts/phase5/convo.py run --split valid --batch 32 --labels p5g1 p3r3 $M base --conditions grv
run .venv/bin/python scripts/phase5/select.py final
echo "QUEUE_VALID5_DONE $(date '+%H:%M:%S')"
