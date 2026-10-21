#!/usr/bin/env bash
source "$(dirname "$0")/../env.sh"
# Waits for scripts/run_final12h.sh to finish, then runs the additions in order of cost (cheapest first):
#   1 position padding (audit A3), 2 digit breakdown (audit A2), 3 sublayer patching (A4), 4 the checkpoint ladder (C30).
# Start it once, at any time:
#   export PERSIST=/scratch/flip_out HF_HOME=/scratch/hf_cache
#   (setsid nohup bash scripts/run_after.sh > $PERSIST/log_after.txt 2>&1 < /dev/null &)
# Progress:  grep -E "^=== |FAILED" $PERSIST/log_after.txt | tail -20
cd "$(dirname "$0")"
OUT=${PERSIST:?set PERSIST first}/fullday
: ${HF_HOME:?set HF_HOME first}
mkdir -p $OUT
while pgrep -f "[r]un_final12h.sh" >/dev/null; do sleep 30; done
echo "=== AFTER: main run finished $(date)"

echo "=== A3 position padding $(date)"
python3 experiments/head_pad_test.py --model EleutherAI/pythia-12b --step 143000 --out $OUT/headpad || echo "FAILED headpad pythia-12b"
rm -rf $HF_HOME/hub/models--EleutherAI--pythia-12b

echo "=== A2 digit breakdown $(date)"
for m in Qwen/Qwen2.5-7B-Instruct allenai/OLMo-2-1124-7B-Instruct; do
  for th in med coins; do
    python3 experiments/digit_breakdown.py --model $m --revision main --theme $th --depths 3 --out $OUT/digits || echo "FAILED digits $m $th"
  done
  rm -rf $HF_HOME/hub/models--${m//\//--}
done
python3 experiments/digit_breakdown.py --model EleutherAI/pythia-6.9b --step 143000 --theme coins --depths 1 3 --out $OUT/digits || echo "FAILED digits pythia-6.9b"
rm -rf $HF_HOME/hub/models--EleutherAI--pythia-6.9b

echo "=== A4 sublayer patching $(date)"
python3 experiments/component_patch.py --selftest || echo "FAILED component_patch selftest"
python3 experiments/component_patch.py --model EleutherAI/pythia-6.9b --step 143000 --every 4 --out $OUT/component_patch || echo "FAILED component_patch"

echo "=== C30 ladder $(date)"
bash scripts/run_ladder.sh
echo "=== AFTER ALL DONE $(date)"
