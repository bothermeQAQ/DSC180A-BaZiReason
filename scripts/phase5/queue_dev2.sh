#!/bin/zsh
# Phase 5 development runs, part 2 (after A1): the untuned base first, then round 3 (no full-sheet run).
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
run() { echo "=== $(date '+%H:%M:%S') $*"; "$@" || echo "STEP_FAILED $*"; }
run $PY scripts/phase5/run5.py --split dev --labels base --parts rules indep blind --conditions w0 gf go gr grv
run $PY scripts/phase5/run5.py --split dev --labels p3r3 --parts rules indep blind --conditions w0 go gr grv
echo "QUEUE_DEV2_DONE $(date '+%H:%M:%S')"
