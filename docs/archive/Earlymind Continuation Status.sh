#!/usr/bin/env bash
# 2-minute check of ONLY what the 3h session runs (Parts 1 and 2), on a tiny model. Run before run_fullday.sh:
#   export PERSIST=/shared-docker/moulik_earlymind HF_HOME=$PERSIST/hf_cache; bash smoke_core.sh
set -e
cd "$(dirname "$0")"
S=${PERSIST:?set PERSIST first}/smoke_core
rm -rf $S; mkdir -p $S
python -c "import datasets, accelerate" 2>/dev/null || pip install -q datasets accelerate
python test_flip.py
python swap_test.py --model EleutherAI/pythia-410m --step 143000 --n 40 --tasks vartrack_d1 --out $S/swap
python perturb_test.py --model EleutherAI/pythia-410m --step 143000 --n 40 --tasks vartrack_d1 --instruct --out $S/perturb
ls -R $S | head -20
echo "SMOKE CORE OK"
