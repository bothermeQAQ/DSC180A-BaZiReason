#!/bin/zsh
# Round 2 training, adapter selection on the validation split, then every final evaluation.
# One GPU job at a time. Usage: scripts/phase3/queue_round2.sh
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-mlx/bin/python
R1=artifacts/adapters/bazireason-qwen3-4b-lora-p3-r1
R2=artifacts/adapters/bazireason-qwen3-4b-lora-p3-r2
P2=artifacts/adapters/bazireason-qwen3-4b-lora-v2
H=results/phase3/heldout
V=results/phase3/v3
Q=results/phase3/baziqa1024
METHODS=(direct direct_minimal tool_intermediate tool_context tool_corrupted)
mkdir -p $H $V $Q/p2 $Q/p3
run() { echo "=== $(date '+%H:%M:%S') $*"; "$@" || echo "STEP_FAILED $*"; }
run $PY scripts/phase2/train_lora.py --config configs/phase3/qwen3_4b_lora_p3_r2.yaml
[ -f $R2/adapters.safetensors ] || { echo "QUEUE_ABORT no round-2 adapter"; exit 1; }
run $PY scripts/phase3/run_p3_eval.py --label p3-r2 --adapter $R2 --data data/phase3/train_v1/valid.jsonl --output $H/valid_p3r2.jsonl
run $PY scripts/phase3/run_p3_eval.py --label p3-r2 --adapter $R2 --data data/phase3/dev_v1/dev.jsonl --output $H/dev_p3r2.jsonl
BEST=$(.venv/bin/python scripts/phase3/select_adapter.py)
echo "SELECTED $BEST"
[ "$BEST" = "$R2" ] && OTHER=$R1 || OTHER=$R2
[ "$BEST" = "$R2" ] && { BN=p3r2; ON=p3r1; } || { BN=p3r1; ON=p3r2; }
# Held-out Phase 3 test split: selected adapter, the other round, Phase 2.
run $PY scripts/phase3/run_p3_eval.py --label $BN --adapter $BEST --output $H/p3test_$BN.jsonl
run $PY scripts/phase3/run_p3_eval.py --label p2-v2 --adapter $P2 --output $H/p3test_p2.jsonl
# benchmark_v3, strict protocol, with the checkpoint's own evaluator.
run $PY scripts/phase2/run_v3_local.py --label $BN --adapter $BEST --protocol strict --methods $METHODS --output $V/${BN}_strict.jsonl
run .venv-mlx/bin/bazireason evaluate --predictions $V/${BN}_strict.jsonl --output $V/${BN}_strict.jsonl.scores.json
# BaziQA with a 1,024-token budget, three runs each.
for tools in none verified; do
  run $PY scripts/phase2/run_baziqa.py --who tuned --tools $tools --adapter $BEST --output-dir $Q/p3 --max-tokens 1024 --label "p3:$BN"
done
for tools in none verified; do
  run $PY scripts/phase2/run_baziqa.py --who tuned --tools $tools --adapter $P2 --output-dir $Q/p2 --max-tokens 1024 --label "p2:lora-v2"
done
run .venv/bin/python scripts/phase3/score_baziqa3.py
# Regression on the Phase 2 held-out splits.
run $PY scripts/phase2/run_p2_eval.py --label $BN --adapter $BEST --output $H/p2test_$BN.jsonl --paraphrase-probe
run $PY scripts/phase2/run_p2_eval.py --label $BN --adapter $BEST --data data/phase2/train_v2/test.jsonl --output $H/p2fresh_$BN.jsonl
# benchmark_v3, reasoned protocol.
run $PY scripts/phase2/run_v3_local.py --label $BN --adapter $BEST --protocol reasoned --methods direct tool_intermediate --output $V/${BN}_reasoned.jsonl
run .venv-mlx/bin/bazireason evaluate --predictions $V/${BN}_reasoned.jsonl --output $V/${BN}_reasoned.jsonl.scores.json
# The remaining comparisons.
run $PY scripts/phase3/run_p3_eval.py --label $ON --adapter $OTHER --output $H/p3test_$ON.jsonl
run $PY scripts/phase3/run_p3_eval.py --label base --output $H/p3test_base.jsonl
echo "QUEUE_ROUND2_DONE $(date '+%H:%M:%S')"
