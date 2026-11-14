#!/bin/zsh
# Secondary measures on validation (docs/phase5_preregistration.md section 6), after the sealed runs:
# the 2 x 3 table {base, R3, A1} x {w0, gr} needs base and R3 alone; gf and go for G* separate the
# contribution of the fact sheet, of routing and of the router's errors. One GPU job at a time.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
run() { echo "=== $(date '+%H:%M:%S') $*"; "$@" || echo "STEP_FAILED $*"; }
until grep -q "QUEUE_SEALED5_DONE" artifacts/logs/p5_queue_sealed5.log; do sleep 30; done
G=$(.venv/bin/python -c "import json;print(json.load(open('results/phase5/selection.json'))['g_star'])")
run $PY scripts/phase5/run5.py --split valid --batch 32 --labels p3r3 base --parts rules indep blind --conditions w0
run $PY scripts/phase5/run5.py --split valid --batch 32 --labels $G --parts rules indep blind --conditions gf go
echo "QUEUE_VALID_EXTRA_DONE $(date '+%H:%M:%S')"
