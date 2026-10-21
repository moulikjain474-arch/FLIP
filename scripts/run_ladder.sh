#!/usr/bin/env bash
source "$(dirname "$0")/../env.sh"
# Checkpoint ladder (claim C30, EXPERIMENTS.md section 4m): Pythia-6.9B, layer 22, depths 1 and 3, six checkpoints.
# Per checkpoint: the plant/erase test with 5 random-subspace controls (core/steer_stage.py) and the chain-identity probe
# (experiments/chain_probe.py). Run AFTER scripts/run_final12h.sh (it needs the patched core/steer_stage.py that scripts/run_final12h.sh creates).
#   export PERSIST=/scratch/flip_out HF_HOME=/scratch/hf_cache
#   bash scripts/run_ladder.sh
# Score:  python3 analysis/ladder_report.py --steer $PERSIST/fullday/ladder --probe $PERSIST/fullday/ladder_probe
# Order is chosen so a truncated run still shows the early/middle/late shape; each checkpoint is independent.
cd "$(dirname "$0")"
OUT=${PERSIST:?set PERSIST first}/fullday
: ${HF_HOME:?set HF_HOME first}
mkdir -p $OUT
python3 -c "import sklearn" 2>/dev/null || pip3 install --break-system-packages scikit-learn >/dev/null 2>&1
python3 -c "import sklearn" || { echo "FAILED ladder: scikit-learn missing"; exit 1; }
python3 experiments/patch_steer_stage.py >/dev/null; python3 experiments/patch_plant_rand.py >/dev/null
M=EleutherAI/pythia-6.9b
echo "=== LADDER $(date)"
for s in 4000 16000 64000 8000 32000 143000; do
  echo "=== LADDER step $s $(date)"
  for t in vartrack_d1 vartrack_d3; do
    python3 core/steer_stage.py --model $M --step $s --task $t --layer 22 --scope single \
      --n-rand 5 --plant-rand --out $OUT/ladder || echo "FAILED ladder steer $s $t"
  done
  python3 experiments/chain_probe.py --model $M --step $s --tasks vartrack_d1 vartrack_d3 --every 4 --batch-size 16 \
    --out $OUT/ladder_probe || echo "FAILED ladder probe $s"
done
rm -rf $HF_HOME/hub/models--EleutherAI--pythia-6.9b
python3 analysis/ladder_report.py --steer $OUT/ladder --probe $OUT/ladder_probe || true
echo "=== LADDER DONE $(date)"
