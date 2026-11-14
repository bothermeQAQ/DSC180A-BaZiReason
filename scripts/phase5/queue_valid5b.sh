#!/bin/zsh
# Validation, second part: the untuned base's gr/grv runs, which ran out of GPU memory at batch 32
# (long repair prompts), rerun with the padded-token budget in run5.TOKEN_BUDGET; then the frozen
# selection. Waits for queue_valid5.sh. Replies already generated are reused from the cache.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
run() { echo "=== $(date '+%H:%M:%S') $*"; "$@" || echo "STEP_FAILED $*"; }
until grep -q "QUEUE_VALID5_DONE" artifacts/logs/p5_queue_valid5.log; do sleep 30; done
run $PY scripts/phase5/run5.py --split valid --batch 32 --labels base --parts rules indep blind --conditions gr grv
run $PY scripts/phase5/convo.py run --split valid --batch 32 --labels base --conditions grv
run .venv/bin/python scripts/phase5/select.py final
echo "QUEUE_VALID5B_DONE $(date '+%H:%M:%S')"
