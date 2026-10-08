#!/usr/bin/env bash
# Extra scale points (exploratory): reorder-vs-rename test on mid-size models between 7B and 72B.
# Waits until the other queue has finished (so two 30-65 GB models never share the card).
#   PERSIST=/shared-docker/moulik_earlymind/s6 HF_HOME=/shared-docker/moulik_earlymind/hf_cache_b bash run_scale.sh
cd "$(dirname "$0")"
OUT=${PERSIST:?set PERSIST first}/fullday
: ${HF_HOME:?set HF_HOME first}
mkdir -p $OUT
cleanrepo() { rm -rf $HF_HOME/hub/models--${1//\//--}; }
while pgrep -f "queue_b.sh|run_fullday.sh|run_extra.sh" > /dev/null; do sleep 60; done
for m in allenai/OLMo-2-1124-13B Qwen/Qwen2.5-14B Qwen/Qwen2.5-32B; do
  echo "=== SCALE $m $(date)"
  python perturb_test.py --model $m --revision main --batch-size 8 --instruct --out $OUT/perturb || echo "FAILED scale $m"
  cleanrepo $m
done
echo "=== SCALE ALL DONE $(date)"
