#!/bin/zsh
# Phase 5 development runs (Phase 4 final test + blind dev questions). One GPU job at a time.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
run() { echo "=== $(date '+%H:%M:%S') $*"; "$@" || echo "STEP_FAILED $*"; }
run $PY scripts/phase5/run5.py --split dev --labels p4a1 p3r3 base --parts rules indep blind --conditions w0 gf go gr grv
run $PY scripts/phase5/run5.py --split dev --labels mix:p3r3:p4a1:0.15 mix:p3r3:p4a1:0.25 mix:p3r3:p4a1:0.35 p2 p3r3 p4a1 --parts calendar rules indep blind --conditions w0
echo "QUEUE_DEV_DONE $(date '+%H:%M:%S')"
