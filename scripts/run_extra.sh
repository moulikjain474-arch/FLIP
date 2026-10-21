#!/usr/bin/env bash
source "$(dirname "$0")/../env.sh"
# Extra exploratory runs (mentor feedback + changes doc). Run AFTER scripts/run_fullday.sh has finished.
#   export PERSIST=/shared-docker/moulik_earlymind/s2 HF_HOME=/shared-docker/moulik_earlymind/hf_cache
#   ONLY="X Y Z" setsid nohup bash scripts/run_extra.sh > $PERSIST/log_extra.txt 2>&1 < /dev/null &
# X  attention at every 12B trajectory checkpoint (does the last token's attention move with the preference? J2)
# Y  attention at the 410M seed checkpoints
# Z  numerical-noise check (two batch sizes)
# P  chain-identity probe across 1.4B checkpoints (R2b): does the representation move with the preference?
# M  plant specificity (P3) and rank sweep (P4) on 6.9B. NEEDS core/steer_stage.py patched first: python3 experiments/patch_steer_stage.py
# Results go to $PERSIST/fullday/ next to the position results they are compared with; nothing is moved aside.
cd "$(dirname "$0")"
OUT=${PERSIST:?set PERSIST first}/fullday
: ${HF_HOME:?set HF_HOME first}
mkdir -p $OUT
cleanrepo() { rm -rf $HF_HOME/hub/models--${1//\//--}; }
clean() { cleanrepo EleutherAI/$1; }
want() { [[ -z "${ONLY:-}" || " $ONLY " == *" $1 "* ]]; }
ATT="--n ${ATT_N:-100} --tasks vartrack_d2 vartrack_d3"

if want X; then
echo "=== PART X: attention at the 12B trajectory checkpoints $(date)"
for s in ${X_STEPS:-1000 2000 4000 6000 8000 12000 16000 24000 32000 48000 64000 100000 120000 143000}; do
  python experiments/attention_test.py --model EleutherAI/pythia-12b --step $s $ATT --out $OUT/attention || echo "FAILED attention 12b $s"
  clean pythia-12b
done
echo "=== PART X DONE"
fi

if want Y; then
echo "=== PART Y: attention at the 410M seed checkpoints $(date)"
for m in pythia-410m pythia-410m-seed1 pythia-410m-seed2 pythia-410m-seed3 pythia-410m-seed5 pythia-410m-seed6 pythia-410m-seed7 pythia-410m-seed8 pythia-410m-seed9; do
  for s in 16000 64000; do
    python experiments/attention_test.py --model EleutherAI/$m --step $s $ATT --out $OUT/attention || echo "FAILED attention $m $s"
  done
  clean $m
done
echo "=== PART Y DONE"
fi

if want Z; then
echo "=== PART Z: numerical-noise check $(date)"
for m in pythia-1.4b pythia-12b; do
  python experiments/noise_check.py --model EleutherAI/$m --step 143000 --out $OUT/noise || echo "FAILED noise $m"
  clean $m
done
echo "=== PART Z DONE"
fi

if want P; then
echo "=== PART P: chain-identity probe across Pythia-1.4B checkpoints $(date)"
for s in ${P_STEPS:-1000 4000 16000 32000 64000 143000}; do
  python experiments/chain_probe.py --model EleutherAI/pythia-1.4b --step $s --out $OUT/chain_probe || echo "FAILED chain_probe 1.4b $s"
  clean pythia-1.4b
done
echo "=== PART P DONE"
fi

if want M; then
echo "=== PART M: plant specificity and rank sweep on 6.9B $(date)"
if ! grep -q -- "--rank-sweep" core/steer_stage.py; then echo "FAILED part M: run python3 experiments/patch_steer_stage.py first"; else
for t in vartrack_d1 vartrack_d3; do
  for L in ${M_LAYERS:-14 18 22}; do
    python core/steer_stage.py --model EleutherAI/pythia-6.9b --step 143000 --task $t --layer $L --scope single --n-rand 1 \
      --save-items --rank-sweep --out $OUT/steer_items || echo "FAILED steer_items 6.9b $t $L"
  done
done
clean pythia-6.9b
fi
echo "=== PART M DONE"
fi

echo "=== EXTRA ALL DONE $(date)"
