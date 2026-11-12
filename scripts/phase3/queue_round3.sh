#!/bin/zsh
# Corrective round 3, final selection between rounds 2 and 3, then the comparisons.
# One GPU job at a time. Usage: scripts/phase3/queue_round3.sh
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
R2=artifacts/adapters/bazireason-qwen3-4b-lora-p3-r2
R3=artifacts/adapters/bazireason-qwen3-4b-lora-p3-r3
P2=artifacts/adapters/bazireason-qwen3-4b-lora-v2
H=results/phase3/heldout
V=results/phase3/v3
Q=results/phase3/baziqa1024
CHECK=data/phase3/p2check_v1/test.jsonl
METHODS=(direct direct_minimal tool_intermediate tool_context tool_corrupted)
run() { echo "=== $(date '+%H:%M:%S') $*"; "$@" || echo "STEP_FAILED $*"; }
run $PY scripts/phase2/train_lora.py --config configs/phase3/qwen3_4b_lora_p3_r3.yaml
[ -f $R3/adapters.safetensors ] || { echo "QUEUE_ABORT no round-3 adapter"; exit 1; }
# Validation only, for the pre-specified choice between rounds 2 and 3.
run $PY scripts/phase3/run_p3_eval.py --label p3-r3 --adapter $R3 --data data/phase3/train_v1/valid.jsonl --output $H/valid_p3r3.jsonl
run $PY scripts/phase2/run_p2_eval.py --label p3-r2 --adapter $R2 --data data/phase2/train_v2/valid.jsonl --output $H/p2valid_p3r2.jsonl
run $PY scripts/phase2/run_p2_eval.py --label p3-r3 --adapter $R3 --data data/phase2/train_v2/valid.jsonl --output $H/p2valid_p3r3.jsonl
FINAL=$(.venv/bin/python scripts/phase3/select_final.py)
echo "FINAL_SELECTED $FINAL"
# Never-inspected calendar check split: Phase 2, round 2, round 3.
run $PY scripts/phase2/run_p2_eval.py --label p3-r3 --adapter $R3 --data $CHECK --output $H/p2check_p3r3.jsonl
run $PY scripts/phase2/run_p2_eval.py --label p3-r2 --adapter $R2 --data $CHECK --output $H/p2check_p3r2.jsonl
run $PY scripts/phase2/run_p2_eval.py --label p2-v2 --adapter $P2 --data $CHECK --output $H/p2check_p2.jsonl
# Round 3 on the Phase 3 test split, the Phase 2 splits and benchmark_v3.
run $PY scripts/phase3/run_p3_eval.py --label p3r3 --adapter $R3 --output $H/p3test_p3r3.jsonl
run $PY scripts/phase2/run_p2_eval.py --label p3r3 --adapter $R3 --output $H/p2test_p3r3.jsonl
run $PY scripts/phase2/run_p2_eval.py --label p3r3 --adapter $R3 --data data/phase2/train_v2/test.jsonl --output $H/p2fresh_p3r3.jsonl
run $PY scripts/phase2/run_v3_local.py --label p3r3 --adapter $R3 --protocol strict --methods $METHODS --output $V/p3r3_strict.jsonl
run .venv-mlx/bin/bazireason evaluate --predictions $V/p3r3_strict.jsonl --output $V/p3r3_strict.jsonl.scores.json
run $PY scripts/phase2/run_v3_local.py --label p3r3 --adapter $R3 --protocol reasoned --methods direct tool_intermediate --output $V/p3r3_reasoned.jsonl
run .venv-mlx/bin/bazireason evaluate --predictions $V/p3r3_reasoned.jsonl --output $V/p3r3_reasoned.jsonl.scores.json
# BaziQA for round 3 (same settings as the other models).
mkdir -p $Q/p3r3
for tools in none verified; do
  run $PY scripts/phase2/run_baziqa.py --who tuned --tools $tools --adapter $R3 --output-dir $Q/p3r3 --max-tokens 1024 --label "p3:p3r3"
done
echo "QUEUE_ROUND3_DONE $(date '+%H:%M:%S')"
