#!/usr/bin/env bash
# Five-minute check of the new tests on tiny runs, BEFORE the long day. Any line that fails
# here is cheaper to fix now than at hour 6.  Run from the repo folder:
#   export PERSIST=/shared-docker/moulik_earlymind HF_HOME=$PERSIST/hf_cache; bash smoke.sh
set -e
cd "$(dirname "$0")"
S=${PERSIST:?set PERSIST first}/smoke
rm -rf $S; mkdir -p $S
python -c "import datasets" 2>/dev/null || pip install -q datasets
python test_flip.py
python perturb_test.py --model EleutherAI/pythia-410m --step 143000 --n 40 --tasks vartrack_d1 --out $S/perturb
python perturb_test.py --model allenai/OLMo-2-0425-1B --revision main --n 40 --tasks vartrack_d1 --out $S/perturb
python position_test.py --model EleutherAI/pythia-410m --step 143000 --layout interleaved --n 40 --tasks vartrack_d1 --out $S/position_interleaved/pythia-410m
python benchmark_flip.py --model Qwen/Qwen2.5-0.5B --n 40 --out $S/benchmark
python distance_test.py --model EleutherAI/pythia-410m --step 143000 --n 20 --tasks vartrack_d1 --out $S/distance
python patch_test.py --model EleutherAI/pythia-410m --step 143000 --n 20 --tasks vartrack_d1 --every 8 --out $S/patch
python story_test.py --model EleutherAI/pythia-410m --step 143000 --n 20 --depths 1 --out $S/story
python export_csv.py --dir $S --out $S/smoke.csv
rm -rf $HF_HOME/hub/models--EleutherAI--pythia-410m $HF_HOME/hub/models--allenai--OLMo-2-0425-1B $HF_HOME/hub/models--Qwen--Qwen2.5-0.5B
echo "SMOKE OK"
