#!/bin/zsh
# Phase 4 part 2 (after results/phase4/selection.json exists): the one-time final test,
# then the inspected benchmarks. One GPU job at a time. Order follows the pre-registration.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
A=artifacts/adapters
V=results/phase4/v3
Q=results/phase4/baziqa
run() { echo "=== $(date '+%H:%M:%S') $*"; "$@" || echo "STEP_FAILED $*"; }
[ -f results/phase4/selection.json ] || { echo "QUEUE_ABORT no selection"; exit 1; }
typeset -A DIR
DIR=(p2 bazireason-qwen3-4b-lora-v2 p3r2 bazireason-qwen3-4b-lora-p3-r2 \
     p3r3 bazireason-qwen3-4b-lora-p3-r3 p4a1 bazireason-qwen3-4b-lora-p4-a1)
# 1. Final test, W0 for every adapter.
for label in p4a1 p3r3 p2 p3r2; do
  run $PY scripts/phase4/run_eval.py --label $label --adapter $A/$DIR[$label] --split final_test --workflow w0
done
# 2. W1 (verify and repair) for R3 and the selected adapter (P2), plus A1 (exploratory);
#    W2 (tool check) on the strict rows for P2, R3, A1.
for label in p2 p3r3 p4a1; do
  run $PY scripts/phase4/run_eval.py --label $label --adapter $A/$DIR[$label] --split final_test --workflow w1
done
for label in p4a1 p3r3 p2; do
  run $PY scripts/phase4/run_eval.py --label $label --adapter $A/$DIR[$label] --split final_test --workflow w2 --parts p3
done
# 3. benchmark_v3 (inspected): A1 under both Phase 3 protocols; W2 on tool_corrupted.
mkdir -p $V
run $PY scripts/phase2/run_v3_local.py --label p4a1 --adapter $A/$DIR[p4a1] --protocol strict \
  --methods direct direct_minimal tool_intermediate tool_context tool_corrupted --output $V/p4a1_strict.jsonl
run .venv-mlx/bin/bazireason evaluate --predictions $V/p4a1_strict.jsonl --output $V/p4a1_strict.jsonl.scores.json
run $PY scripts/phase2/run_v3_local.py --label p4a1 --adapter $A/$DIR[p4a1] --protocol reasoned \
  --methods direct tool_intermediate --output $V/p4a1_reasoned.jsonl
run .venv-mlx/bin/bazireason evaluate --predictions $V/p4a1_reasoned.jsonl --output $V/p4a1_reasoned.jsonl.scores.json
for label in p4a1 p3r3 p2; do
  run $PY scripts/phase4/run_v3_toolcheck.py --label $label --adapter $A/$DIR[$label] --output $V/${label}_w2_tool_corrupted.jsonl
  run .venv-mlx/bin/bazireason evaluate --predictions $V/${label}_w2_tool_corrupted.jsonl --output $V/${label}_w2_tool_corrupted.jsonl.scores.json
done
# 4. BaziQA (inspected): A1 under the Phase 3 conditions, then W3 for the selected adapter
#    (P2), R3 and A1.
mkdir -p $Q/p4a1 $Q/p3r3 $Q/p2
run $PY scripts/phase4/run_baziqa_guard.py --adapter $A/$DIR[p2] --label p2 --output-dir $Q/p2
for tools in none verified; do
  run $PY scripts/phase2/run_baziqa.py --who tuned --tools $tools --adapter $A/$DIR[p4a1] \
    --output-dir $Q/p4a1 --max-tokens 1024 --label "p4:p4a1"
done
run $PY scripts/phase4/run_baziqa_guard.py --adapter $A/$DIR[p4a1] --label p4a1 --output-dir $Q/p4a1
run $PY scripts/phase4/run_baziqa_guard.py --adapter $A/$DIR[p3r3] --label p3r3 --output-dir $Q/p3r3
run .venv/bin/python scripts/phase4/score_baziqa4.py
echo "QUEUE_FINAL_DONE $(date '+%H:%M:%S')"
