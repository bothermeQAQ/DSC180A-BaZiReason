#!/bin/sh
# Phase 6, preregistration 12.14: UD, the chain's best run continued with derivation version 3
# (each decision after its check). One GPU job at a time, so it waits for the main queue
# (QUEUE_MAIN_DONE: the chain, U1 and the diagnostics), then:
#   1. decide6.py ud: the parent (the chain's best run on dev6) and the twin (the chain level built
#      from it), recorded in results/phase6/round2.json;
#   2. the parent's fit samples B and C if missing, the build (the twin's seed, --steps 3), training;
#   3. dev6 and valid6 at L0 (valid6 stays unread until the selection).
# It stops at the first failed step and prints QUEUE_UD_FAILED; a restart skips what is there.
# The sealed split is not touched here.
#   nohup sh scripts/phase6/queue_ud.sh >> artifacts/logs/p6_queue_ud.log 2>&1 &
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
export PYTHONPATH=src
run() { echo "=== $(date "+%m-%d %H:%M:%S") $*"; "$@" || { echo "STEP_FAILED $*"; return 1; }; }
fail() { echo "QUEUE_UD_FAILED $(date '+%m-%d %H:%M:%S')"; exit 1; }
ALL="rules indep blind combo cf"
D=".venv/bin/python scripts/phase6/decide6.py"
echo "QUEUE_UD_WAITING $(date '+%m-%d %H:%M:%S') for QUEUE_MAIN_DONE"
until grep -q "QUEUE_MAIN_DONE" artifacts/logs/p6_queue_main.log; do sleep 300; done
echo "QUEUE_UD_STARTED $(date '+%m-%d %H:%M:%S')"
if [ "$(date '+%Y%m%d')" -gt 20261127 ]; then echo "UD not started: after 27 November"; fail; fi
LINE=$($D ud) || fail
echo "UD: $LINE"
# shellcheck disable=SC2086
set -- $LINE
R=$1
P=$2
shift 2
[ "$R" = "ud" ] || fail
B="results/phase6/own_replies_p6${P}_b.jsonl"
C="results/phase6/own_replies_p6${P}_c.jsonl"
[ -f "$B" ] || run $PY scripts/phase6/own_replies.py --label "p6$P" --sample b || fail
[ -f "$C" ] || run $PY scripts/phase6/own_replies.py --label "p6$P" --sample c || fail
# shellcheck disable=SC2068
[ -d "data/phase6/train/$R" ] || run $PY scripts/phase6/build_train6.py --recipe "$R" --parent "$P" --tokens 6000000 --weights-from "$B" --calendar-weights-from "$C" $@ || fail
[ -f "results/phase6/training/bazireason-qwen3-4b-lora-p6-$R/training_summary.json" ] || run $PY scripts/phase2/train_lora.py --config "configs/phase6/qwen3_4b_lora_p6_$R.yaml" || fail
run $PY scripts/phase6/run6.py --split dev6 --labels "p6$R" --parts $ALL calendar --levels L0 || fail
run $PY scripts/phase6/run6.py --split valid6 --labels "p6$R" --parts $ALL calendar --levels L0 || fail
echo "QUEUE_UD_DONE $(date '+%m-%d %H:%M:%S')"
