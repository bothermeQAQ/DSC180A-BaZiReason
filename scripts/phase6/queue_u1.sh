#!/bin/sh
# Phase 6 U1 (15 M tokens, from the base): the frozen screening decision, the second set of
# training-writer paraphrases, U1's data and training, its fit check, dev6, and the valid6 runs
# that the frozen selection rule reads (candidates, and Phase 2 calendar).
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
export PYTHONPATH=src
run() { echo "=== $(date "+%H:%M:%S") $*"; "$@" || { echo "STEP_FAILED $*"; return 1; }; }
ALL="rules indep blind combo cf"
until grep -q "QUEUE_SCREEN6_DONE" artifacts/logs/p6_queue_screen6.log; do sleep 60; done
run .venv/bin/python scripts/phase6/decide6.py screen || exit 1
D=".venv/bin/python scripts/phase6/decide6.py"
if $D small-steps-dropped; then
  echo "STOP: small steps were worse than U0 on dev6; U1's recipe needs a recorded decision"; exit 1
fi
FLAGS=$($D flags)
BEST=$($D best)
echo "screening flags:$FLAGS; best S run: $BEST"
[ -f data/phase6/blind/train6b.jsonl ] || run $PY scripts/phase6/gen_blind6.py --splits train6b
# shellcheck disable=SC2086
run $PY scripts/phase6/build_train6.py --recipe u1 --data-recipe s3 --tokens 15000000 --replies results/phase6/own_replies_p6s1.jsonl --extra-paraphrases data/phase6/blind/train6b.jsonl $FLAGS
run $PY scripts/phase2/train_lora.py --config configs/phase6/qwen3_4b_lora_p6_u1.yaml
run $PY scripts/phase6/own_replies.py --label p6u1 --n 2400
run $PY scripts/phase6/run6.py --split dev6 --labels p6u1 --parts $ALL calendar --levels L0
run $PY scripts/phase6/run6.py --split valid6 --labels p6u1 "$BEST" --parts $ALL calendar --levels L0
run $PY scripts/phase6/run6.py --split valid6 --labels p2 --parts calendar --levels L0
echo "QUEUE_U1_DONE $(date '+%H:%M:%S')"
