#!/bin/sh
# Phase 6 main queue after the recovery (preregistration section 12.4), one GPU job at a time:
#   1. U2: S2's recipe at 15 M tokens, with the system prompt chosen from the H7 twin (S2s vs S2)
#   2. the chain: the fork U3 / U3v, then weighted continuations up to U10 (sections 10.2, 10.5)
#   3. valid6 for the other candidates (S2, S2s)
#   4. U1 as registered (S3's recipe), then diagnostics
# It stops at the first failed step of parts 1-3 and prints QUEUE_MAIN_FAILED; a restart skips
# what is already there. Every decision is made by decide6.py (results/phase6/round2.json).
# The sealed split is not touched here.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
export PYTHONPATH=src
run() { echo "=== $(date "+%m-%d %H:%M:%S") $*"; "$@" || { echo "STEP_FAILED $*"; return 1; }; }
fail() { echo "QUEUE_MAIN_FAILED $(date '+%m-%d %H:%M:%S')"; exit 1; }
cutoff() { [ "$(date '+%Y%m%d')" -gt 20261127 ]; }
done_training() { [ -f "results/phase6/training/bazireason-qwen3-4b-lora-p6-$1/training_summary.json" ]; }
ALL="rules indep blind combo cf"
D=".venv/bin/python scripts/phase6/decide6.py"
until grep -q "QUEUE_S2S_DONE\|QUEUE_S2S_FAILED" artifacts/logs/p6_queue_s2s.log; do sleep 60; done
grep -q "QUEUE_S2S_DONE" artifacts/logs/p6_queue_s2s.log || { echo "the H7 twin did not finish"; fail; }

# ---- 1. U2 --------------------------------------------------------------------------------------
if [ ! -d data/phase6/train/u2 ]; then
  PROMPT=$($D prompt) || fail
  echo "U2 system prompt: $PROMPT"
  [ -f data/phase6/blind/train6c.jsonl ] || run $PY scripts/phase6/gen_blind6.py --splits train6c || fail
  # shellcheck disable=SC2086
  run $PY scripts/phase6/build_train6.py --recipe u2 --data-recipe s2 --tokens 15000000 $PROMPT --extra-paraphrases data/phase6/blind/train6b.jsonl data/phase6/blind/train6c.jsonl || fail
fi
done_training u2 || run $PY scripts/phase2/train_lora.py --config configs/phase6/qwen3_4b_lora_p6_u2.yaml || fail
run $PY scripts/phase6/run6.py --split dev6 --labels p6u2 --parts $ALL calendar --levels L0 || fail
run $PY scripts/phase6/run6.py --split valid6 --labels p6u2 --parts $ALL calendar --levels L0 || fail

# ---- 2. the chain -----------------------------------------------------------------------------------
# One arm of a level: fit sample B for the parent, build, train, dev6 and valid6.
arm() {
  R=$1; P=$2; shift 2
  echo "continuation $R from $P $*"
  [ -f "results/phase6/own_replies_p6${P}_b.jsonl" ] || run $PY scripts/phase6/own_replies.py --label "p6$P" --sample b || return 1
  # shellcheck disable=SC2068
  [ -d "data/phase6/train/$R" ] || run $PY scripts/phase6/build_train6.py --recipe "$R" --parent "$P" --tokens 6000000 --weights-from "results/phase6/own_replies_p6${P}_b.jsonl" $@ || return 1
  done_training "$R" || run $PY scripts/phase2/train_lora.py --config "configs/phase6/qwen3_4b_lora_p6_$R.yaml" || return 1
  run $PY scripts/phase6/run6.py --split dev6 --labels "p6$R" --parts $ALL calendar --levels L0 || return 1
  run $PY scripts/phase6/run6.py --split valid6 --labels "p6$R" --parts $ALL calendar --levels L0 || return 1
}
while :; do
  NEXT=$($D next) || fail
  [ "$NEXT" = "stop" ] && break
  if cutoff; then echo "the chain ends: after 27 November"; break; fi
  echo "$NEXT" > artifacts/logs/p6_main_level.txt
  DONE_ARMS=0
  while read -r LINE <&3; do
    [ -n "$LINE" ] || continue
    # shellcheck disable=SC2086
    if arm $LINE; then DONE_ARMS=$((DONE_ARMS + 1)); else echo "ARM_SKIPPED $LINE"; fi
  done 3< artifacts/logs/p6_main_level.txt
  if [ "$DONE_ARMS" -eq 0 ]; then echo "no arm of this level finished"; fail; fi
done

# ---- 3. valid6 for the other candidates -----------------------------------------------------------
run $PY scripts/phase6/run6.py --split valid6 --labels p6s2 p6s2s --parts $ALL calendar --levels L0 || fail
echo "QUEUE_MAIN_CHAIN_DONE $(date '+%m-%d %H:%M:%S')"

# ---- 4. U1 as registered (S3's recipe), then diagnostics; a failure here undoes nothing above --------
u1() {
  [ -f data/phase6/blind/train6b.jsonl ] || return 1
  [ -d data/phase6/train/u1 ] || run $PY scripts/phase6/build_train6.py --recipe u1 --data-recipe s3 --tokens 15000000 --replies results/phase6/own_replies_p6s1.jsonl --extra-paraphrases data/phase6/blind/train6b.jsonl || return 1
  done_training u1 || run $PY scripts/phase2/train_lora.py --config configs/phase6/qwen3_4b_lora_p6_u1.yaml || return 1
  run $PY scripts/phase6/run6.py --split dev6 --labels p6u1 --parts $ALL calendar --levels L0 || return 1
  run $PY scripts/phase6/run6.py --split valid6 --labels p6u1 --parts $ALL calendar --levels L0 || return 1
  [ -f results/phase6/own_replies_p6u1.jsonl ] || run $PY scripts/phase6/own_replies.py --label p6u1 --n 2400 || return 1
  run .venv/bin/python scripts/phase6/fit_clean.py --label p6u1 || return 1
}
if cutoff; then echo "U1 not trained: after 27 November"; else u1 || echo "U1_STOPPED $(date '+%m-%d %H:%M:%S')"; fi
for L in p6s2 p6s2s p6u1 p6u2 p6u3 p6u3v p6u4 p6u5 p6u6 p6u7 p6u8 p6u9 p6u10; do
  [ -f "results/phase6/dev6/$L/rules_L0.jsonl" ] || continue
  [ -f "results/phase6/own_replies_${L}_b.jsonl" ] || run $PY scripts/phase6/own_replies.py --label "$L" --sample b
done
echo "QUEUE_MAIN_DONE $(date '+%m-%d %H:%M:%S')"
