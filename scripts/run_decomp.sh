#!/usr/bin/env bash
source "$(dirname "$0")/../env.sh"
# Exploratory (not registered): which chain does the model pick in each order, and does it depend on confidence?
# Motivated by the 12B result for C22. Results: $PERSIST/fullday/decomp/
#   export PERSIST=/shared-docker/moulik_earlymind/s9 HF_HOME=/shared-docker/moulik_earlymind/hf_cache_c
#   setsid nohup bash scripts/run_decomp.sh > $PERSIST/log_decomp.txt 2>&1 < /dev/null &
cd "$(dirname "$0")"
OUT=${PERSIST:?set PERSIST first}/fullday/decomp
: ${HF_HOME:?set HF_HOME first}
mkdir -p $OUT
cleanrepo() { rm -rf $HF_HOME/hub/models--${1//\//--}; }
echo "=== decomposition start $(date)"
for m in pythia-12b pythia-6.9b; do
  python experiments/position_decomposition.py --model EleutherAI/$m --step 143000 --out $OUT || echo "FAILED decomp $m"
  cleanrepo EleutherAI/$m
done
for m in allenai/OLMo-2-1124-7B allenai/OLMo-2-0425-1B Qwen/Qwen2.5-7B; do
  python experiments/position_decomposition.py --model $m --revision main --out $OUT || echo "FAILED decomp $m"
  cleanrepo $m
done
echo "=== decomposition DONE $(date)"
