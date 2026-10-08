#!/usr/bin/env bash
# Quick check (few minutes, 410M) of the new exploratory tests. Run before run_extra.sh / Part 9.
set -eo pipefail
cd "$(dirname "$0")"
S=${PERSIST:?set PERSIST first}/smoke_extra
rm -rf $S; mkdir -p $S
python test_flip.py
python patch_test.py --model EleutherAI/pythia-410m --step 143000 --n 40 --every 8 --tasks vartrack_d1 --n-control 20 --out $S/patch_choice
python attention_test.py --model EleutherAI/pythia-410m --step 143000 --n 10 --tasks vartrack_d2 --out $S/attention
python noise_check.py --model EleutherAI/pythia-410m --step 143000 --n 40 --tasks vartrack_d1 --out $S/noise
python chain_probe.py --model EleutherAI/pythia-410m --step 143000 --n 80 --every 8 --tasks vartrack_d2 --out $S/chain_probe
python chain_probe_report.py --dir $S/chain_probe | tail -4
if grep -q -- "--rank-sweep" steer_stage.py; then
  python steer_stage.py --model EleutherAI/pythia-410m --step 143000 --task vartrack_d1 --layer 12 --scope single --n-rand 1 --n 120 --save-items --rank-sweep --out $S/steer_items
  python plant_margin.py --dir $S/steer_items | tee $S/plant_margin.txt
  grep -q "P3" $S/plant_margin.txt || { echo "plant_margin printed nothing"; exit 1; }
else
  echo "(steer_stage.py not patched yet: run python3 patch_steer_stage.py to test P3/P4)"
fi
python commit_depth.py --dir $S | tee $S/commit.txt
grep -q "choice layer" $S/commit.txt || { echo "commit_depth printed nothing"; exit 1; }
if ls $PERSIST/fullday/swap/*.json >/dev/null 2>&1; then
  python extra_stats.py --dir $PERSIST/fullday | tee $S/xstat.txt | tail -3
  grep -q "strict" $S/xstat.txt || { echo "extra_stats printed nothing"; exit 1; }
else
  echo "(no swap results in $PERSIST/fullday yet: extra_stats not checked)"
fi
python attn_vs_pref.py --dir $S 2>&1 | tail -3
echo "SMOKE EXTRA OK"
