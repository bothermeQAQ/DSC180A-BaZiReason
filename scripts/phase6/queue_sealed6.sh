#!/bin/sh
# Phase 6 sealed runs (preregistration sections 8 and 12.4), started by hand once results/phase6/selection.json
# exists. Every run6.py / run_l5.py call on sealed6 appends to results/phase6/sealed_access.jsonl.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
export PYTHONPATH=src
run() { echo "=== $(date "+%H:%M:%S") $*"; "$@" || { echo "STEP_FAILED $*"; return 1; }; }
fail() { echo "QUEUE_SEALED6_FAILED $(date '+%m-%d %H:%M:%S')"; exit 1; }
ALL="rules indep blind combo cf"
[ -f results/phase6/selection.json ] || { echo "no selection.json: the sealed split stays closed"; exit 1; }
M=$(.venv/bin/python -c 'import json; print(json.load(open("results/phase6/selection.json"))["selected"])')
echo "selected model: $M"
# Section 12.4: L0 also for S2 and S2s (H7), U3 and U3v (H9), S1 and S1-8B (H10), U1, U2, M's ancestors
EXTRA=$(.venv/bin/python scripts/phase6/decide6.py sealed-extra) || exit 1
echo "round-2 L0 runs: ${EXTRA:-none}"
# L0 (model alone) and calendar: M and A1 first (the primary comparison), then the references
# shellcheck disable=SC2086
run $PY scripts/phase6/run6.py --split sealed6 --labels "$M" p4a1 base p2 p5g1 p6u0 $EXTRA --parts $ALL calendar --levels L0 || fail
# Levels of help and self-correction
run $PY scripts/phase6/run6.py --split sealed6 --labels "$M" p4a1 p5g1 --parts $ALL --levels L1 L2 L3 L3w L4 || fail
run $PY scripts/phase6/run6.py --split sealed6 --labels base --parts $ALL --levels L1 L2 L3 || fail
# L5: the Phase 5 hybrid system, reported apart
run $PY scripts/phase6/run_l5.py --split sealed6 --labels p5g1 "$M" || fail
echo "QUEUE_SEALED6_DONE $(date '+%H:%M:%S')"
