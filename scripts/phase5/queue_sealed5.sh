#!/bin/zsh
# Phase 5 sealed evaluation: the runs listed in docs/phase5_preregistration.md section 5, once.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
run() { echo "=== $(date '+%H:%M:%S') $*"; "$@" || echo "STEP_FAILED $*"; }
until grep -q "QUEUE_VALID5B_DONE" artifacts/logs/p5_queue_valid5b.log; do sleep 30; done
[ -f results/phase5/selection.json ] || { echo "QUEUE_ABORT no selection"; exit 1; }
G=$(.venv/bin/python -c "import json;print(json.load(open('results/phase5/selection.json'))['g_star'])")
M=$(.venv/bin/python -c "import json;s=json.load(open('results/phase5/selection.json'));print(s['m_star'] if s['m_star_is_candidate'] else '')")
GR_LABELS=($G base); [ "$G" != "p4a1" ] && GR_LABELS+=(p4a1)
run $PY scripts/phase5/run5.py --split sealed --batch 32 --labels p4a1 --parts rules indep blind --conditions w0
run $PY scripts/phase5/run5.py --split sealed --batch 32 --labels $GR_LABELS --parts rules indep blind --conditions gr grv
run $PY scripts/phase5/run5.py --split sealed --batch 32 --labels $G --parts rules indep blind --conditions go
CAL=(p2 p3r3 p4a1 p5g1); [ -n "$M" ] && CAL+=($M)
run $PY scripts/phase5/run5.py --split sealed --batch 32 --labels $CAL --parts calendar --conditions w0
run $PY scripts/phase5/convo.py run --split sealed --batch 32 --labels $G --conditions w0 grv
[ "$G" != "p4a1" ] && run $PY scripts/phase5/convo.py run --split sealed --batch 32 --labels p4a1 --conditions grv
LAT=(p4a1); [ "$G" != "p4a1" ] && LAT+=($G)
run $PY scripts/phase5/latency.py --split valid --labels $LAT --n 40
run $PY scripts/phase5/check_bank.py
echo "QUEUE_SEALED5_DONE $(date '+%H:%M:%S')"
