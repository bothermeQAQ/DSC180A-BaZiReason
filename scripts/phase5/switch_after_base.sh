#!/bin/zsh
# Wait for the base oracle run on the dev rule rows, stop queue_dev2.sh, start queue_dev3.sh.
cd "$(dirname "$0")/../.." || exit 1
until [ -f results/phase5/dev/base/rules_go.jsonl.summary.json ]; do sleep 20; done
pkill -f queue_dev2.sh; sleep 2; pkill -f "run5.py --split dev --labels base"; sleep 5
nohup caffeinate -i scripts/phase5/queue_dev3.sh > artifacts/logs/p5_queue_dev3.log 2>&1 &
echo "SWITCHED $(date '+%H:%M:%S')"
