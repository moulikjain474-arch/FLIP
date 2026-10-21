#!/usr/bin/env bash
source "$(dirname "$0")/../env.sh"
# Causal + position tests at stage checkpoints. Run from the repo folder, in the background:
#   nohup bash scripts/run_causal.sh > $PERSIST/log_causal.txt 2>&1 &
# Checkpoints come from the dense 1.4B grid: 5000 and 7000 are the coin-flip stage
# (correct ~= distractor ~= 0.4), 143000 is the trained model. 410M steps are from its
# main grid; update them once its dense grid is in (see EXPERIMENTS.md, Changes).
set -u
OUT=${PERSIST:-.}
SPECS=(
  "EleutherAI/pythia-1.4b 5000"
  "EleutherAI/pythia-1.4b 7000"
  "EleutherAI/pythia-1.4b 143000"
  "EleutherAI/pythia-410m 16000"
  "EleutherAI/pythia-410m 32000"
  "EleutherAI/pythia-410m 143000"
)
for spec in "${SPECS[@]}"; do
  set -- $spec
  echo "=== position $1 step $2 ==="
  python experiments/position_test.py --model "$1" --step "$2" --out "$OUT/position" || echo "FAILED position $1 $2"
  for task in vartrack_d1 vartrack_d2 vartrack_d3; do
    echo "=== causal $1 step $2 $task ==="
    python core/steer_stage.py --model "$1" --step "$2" --task "$task" --layout blocked --n 600 \
      --out "$OUT/steer_stage" || echo "FAILED steer_stage $1 $2 $task"
    python core/steer.py --model "$1" --step "$2" --task "$task" --layout blocked --n 600 \
      --out "$OUT/steer" || echo "FAILED steer $1 $2 $task"
  done
done
echo "ALL DONE"
