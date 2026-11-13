#!/bin/zsh
# Phase 5 development runs, part 3: prompt version 2 for A1, then the interpolation grid (model alone).
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
run() { echo "=== $(date '+%H:%M:%S') $*"; "$@" || echo "STEP_FAILED $*"; }
run $PY scripts/phase5/run5.py --split dev --labels p4a1 --parts rules indep blind --conditions go gr grv --prompt-version 2
run $PY scripts/phase5/run5.py --split dev --labels mix:p3r3:p4a1:0.15 mix:p3r3:p4a1:0.25 mix:p3r3:p4a1:0.35 --parts calendar rules indep blind --conditions w0
run $PY scripts/phase5/run5.py --split dev --labels p2 p3r3 p4a1 --parts calendar --conditions w0
echo "QUEUE_DEV3_DONE $(date '+%H:%M:%S')"
