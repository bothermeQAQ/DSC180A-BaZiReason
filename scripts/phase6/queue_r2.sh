#!/bin/sh
# Phase 6 round 2 (preregistration section 10): U2 (U1's recipe with the short system prompt at
# the same token budget), then up to three continuations weighted by the parent's measured
# errors on fit sample B. Every decision is made by decide6.py and logged in
# results/phase6/round2.json. The sealed split is not touched here.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
export PYTHONPATH=src
run() { echo "=== $(date "+%H:%M:%S") $*"; "$@" || { echo "STEP_FAILED $*"; return 1; }; }
cutoff() { [ "$(date '+%Y%m%d')" -gt 20261127 ]; }
ALL="rules indep blind combo cf"
# (A failed memory probe prints STEP_FAILED in the U1-hi log by design, so only its last line counts.)
until grep -q "QUEUE_U1HI_DONE" artifacts/logs/p6_queue_u1hi.log; do sleep 60; done
D=".venv/bin/python scripts/phase6/decide6.py"
FLAGS=$($D flags) || exit 1
STRUCT=$($D structure) || exit 1
echo "round-2 adapter options: ${STRUCT:-rank 16 on the last 16 layers}"
if cutoff; then echo "STOP: after 27 November"; exit 0; fi
[ -f data/phase6/blind/train6c.jsonl ] || run $PY scripts/phase6/gen_blind6.py --splits train6c || exit 1
# shellcheck disable=SC2086
run $PY scripts/phase6/build_train6.py --recipe u2 --data-recipe s3 --tokens 15000000 --system short --replies results/phase6/own_replies_p6s1.jsonl --extra-paraphrases data/phase6/blind/train6b.jsonl data/phase6/blind/train6c.jsonl $FLAGS $STRUCT || exit 1
run $PY scripts/phase2/train_lora.py --config configs/phase6/qwen3_4b_lora_p6_u2.yaml || exit 1
run $PY scripts/phase6/run6.py --split dev6 --labels p6u2 --parts $ALL calendar --levels L0 || exit 1
run $PY scripts/phase6/run6.py --split valid6 --labels p6u2 --parts $ALL calendar --levels L0 || exit 1
while :; do
  NEXT=$($D next) || exit 1
  [ "$NEXT" = "stop" ] && break
  if cutoff; then echo "STOP: after 27 November"; break; fi
  # shellcheck disable=SC2086
  set -- $NEXT
  R=$1; P=$2
  echo "continuation $R from $P"
  [ -f "results/phase6/own_replies_p6${P}_b.jsonl" ] || run $PY scripts/phase6/own_replies.py --label "p6$P" --sample b || exit 1
  run $PY scripts/phase6/build_train6.py --recipe "$R" --parent "$P" --tokens 6000000 --weights-from "results/phase6/own_replies_p6${P}_b.jsonl" || exit 1
  run $PY scripts/phase2/train_lora.py --config "configs/phase6/qwen3_4b_lora_p6_$R.yaml" || exit 1
  run $PY scripts/phase6/run6.py --split dev6 --labels "p6$R" --parts $ALL calendar --levels L0 || exit 1
  run $PY scripts/phase6/run6.py --split valid6 --labels "p6$R" --parts $ALL calendar --levels L0 || exit 1
done
echo "QUEUE_R2_TRAINING_DONE $(date '+%H:%M:%S')"
# Diagnostic only: fit sample B for every U run that has no such file yet (what each run still fails).
for L in p6u1 p6u1hi p6u2 p6u3 p6u4 p6u5; do
  [ -f "results/phase6/dev6/$L/rules_L0.jsonl" ] || continue
  [ -f "results/phase6/own_replies_${L}_b.jsonl" ] || run $PY scripts/phase6/own_replies.py --label "$L" --sample b
done
echo "QUEUE_R2_DONE $(date '+%H:%M:%S')"
