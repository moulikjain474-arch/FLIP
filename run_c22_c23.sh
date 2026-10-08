#!/usr/bin/env bash
# Registered claims C22 (uncertainty gating) and C23 (fresh question set). EXPERIMENTS.md section 4f.
# Commit EXPERIMENTS.md BEFORE running this, so the dated commit shows the claims came first.
#   export PERSIST=/shared-docker/moulik_earlymind/s9 HF_HOME=/shared-docker/moulik_earlymind/hf_cache_c
#   setsid nohup bash run_c22_c23.sh > $PERSIST/log_c22_c23.txt 2>&1 < /dev/null &
# 6.9B first (smaller), then 12B. Per model: C22 split, then C23 fresh seed. Results land in
# $PERSIST/fullday/uncertainty and $PERSIST/fullday/perturb_seed1.
cd "$(dirname "$0")"
OUT=${PERSIST:?set PERSIST first}/fullday
: ${HF_HOME:?set HF_HOME first}
mkdir -p $OUT
cleanrepo() { rm -rf $HF_HOME/hub/models--${1//\//--}; }
echo "=== C22/C23 start $(date)"
python check_seed.py || { echo "FAILED seed check: C23 would not use fresh questions"; exit 1; }
for m in pythia-6.9b pythia-12b; do
  echo "=== $m $(date)"
  python uncertainty_gate.py --model EleutherAI/$m --step 143000 --out $OUT/uncertainty || echo "FAILED c22 $m"
  python perturb_test.py --model EleutherAI/$m --step 143000 --seed 1 --out $OUT/perturb_seed1 || echo "FAILED c23 $m"
  cleanrepo EleutherAI/$m
done
echo "=== C22/C23 DONE $(date)"
