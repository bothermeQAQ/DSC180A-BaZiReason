#!/bin/zsh
# Wait until A1's last development run (blind, grv) is written, stop queue_dev.sh, start queue_dev2.sh.
cd "$(dirname "$0")/../.." || exit 1
until [ -f results/phase5/dev/p4a1/blind_grv.jsonl.summary.json ]; do sleep 20; done
pkill -f queue_dev.sh; sleep 2; pkill -f "run5.py --split dev --labels p4a1 p3r3 base"; sleep 5
nohup caffeinate -i scripts/phase5/queue_dev2.sh > artifacts/logs/p5_queue_dev2.log 2>&1 &
echo "SWITCHED $(date '+%H:%M:%S')"
