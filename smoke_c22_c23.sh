#!/usr/bin/env bash
# Quick check (a few minutes, 410M) of the C22 / C23 code before the real runs.
#   export PERSIST=/shared-docker/moulik_earlymind/s9 HF_HOME=/shared-docker/moulik_earlymind/hf_cache_c
#   bash smoke_c22_c23.sh
set -eo pipefail
cd "$(dirname "$0")"
S=${PERSIST:?set PERSIST first}/smoke_c22
rm -rf $S; mkdir -p $S
python test_flip.py | tail -3
python check_seed.py
python uncertainty_gate.py --model EleutherAI/pythia-410m --step 143000 --n 90 --tasks vartrack_d2 --out $S/uncertainty
test -f $S/uncertainty/pythia-410m__step143000.json
python perturb_test.py --model EleutherAI/pythia-410m --step 143000 --n 60 --tasks vartrack_d2 --seed 1 --out $S/perturb_seed1
test -f $S/perturb_seed1/pythia-410m__step143000__seed1.json
python export_csv.py --dir $S --out $S/smoke.csv
grep -q ",gate," $S/smoke.csv && grep -q "perturb_seed1" $S/smoke.csv
python score_new.py --dir $S 2>&1 | grep -E "C22|C23" | cut -c1-160
echo "SMOKE C22/C23 OK"
