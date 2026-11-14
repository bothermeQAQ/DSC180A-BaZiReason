#!/bin/sh
# Phase 6 baseline: the fit diagnostic, then the development ladder for the existing models.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
export PYTHONPATH=src
run() { echo "=== $(date '+%H:%M:%S') $*"; "$@" || echo "STEP_FAILED $*"; }
run $PY scripts/phase6/diag_fit.py
run $PY scripts/phase6/run6.py --split dev6 --labels p4a1 p5g1 p2 base --parts rules indep blind combo cf calendar --levels L0
run $PY scripts/phase6/run6.py --split dev6 --labels p4a1 p5g1 base --parts rules indep blind combo cf --levels L1 L2 L3 L3w
run $PY scripts/phase6/run6.py --split dev6 --labels p4a1 p5g1 --parts rules indep blind combo cf --levels L4
echo "QUEUE_BASE6_DONE $(date '+%H:%M:%S')"
