#!/usr/bin/env bash
# The 12-hour slot (9 Oct). Ordered by value, so if time runs short only the cheap tail is lost.
#   export PERSIST=/shared-docker/moulik_earlymind/s11 HF_HOME=/shared-docker/moulik_earlymind/hf_cache_f
#   setsid nohup bash run_final12h.sh > $PERSIST/log_final12h.txt 2>&1 < /dev/null &
# Check progress:  grep -E "^=== |FAILED" $PERSIST/log_final12h.txt | tail -20
#
# S  smoke: unit tests, both patches, one tiny steering run with the new control, one tiny ARC run
# R  claim C24: plant into RANDOM subspaces of the same rank (6.9B layers 14-26, depths 1 and 3; 12B layer sweep)
# M  models: perturb / decomposition / MMLU (n=1000) / ARC option order for 10 models, one download each
# D  C9 follow-up (exploratory): 410M seeds at steps 4000, 8000, 32000, 143000
# E  attention (410M seeds, trimmed 12B) and the numerical-noise check
# ONLY="R M" runs just those parts. Results go to $PERSIST/fullday (same layout as before) and benchmark_arc/.
cd "$(dirname "$0")"
OUT=${PERSIST:?set PERSIST first}/fullday
: ${HF_HOME:?set HF_HOME first}
mkdir -p $OUT
cleanrepo() { rm -rf $HF_HOME/hub/models--${1//\//--}; }
want() { [[ -z "${ONLY:-}" || " $ONLY " == *" $1 "* ]]; }
SMOKE_OK=1

# ---------------------------------------------------------------- S
if want S; then
echo "=== PART S: smoke $(date)"
python3 test_flip.py > $PERSIST/smoke_tests.txt 2>&1 && echo "unit tests ok" || { echo "FAILED unit tests (see $PERSIST/smoke_tests.txt)"; SMOKE_OK=0; }
python3 benchmark_flip.py --selftest || SMOKE_OK=0
python3 patch_steer_stage.py && python3 patch_plant_rand.py || { echo "FAILED patches"; SMOKE_OK=0; }
python steer_stage.py --model EleutherAI/pythia-410m --step 143000 --task vartrack_d1 --n 300 --layer 12 \
  --scope single --n-rand 2 --plant-rand --out $OUT/smoke_plant > $PERSIST/smoke_plant.txt 2>&1 \
  && python3 plant_control_report.py --dir $OUT/smoke_plant | tail -3 || { echo "FAILED steer smoke (see $PERSIST/smoke_plant.txt)"; SMOKE_OK=0; }
cleanrepo EleutherAI/pythia-410m
python benchmark_flip.py --model Qwen/Qwen2.5-0.5B --dataset arc --n 40 --out $OUT/smoke_arc > $PERSIST/smoke_arc.txt 2>&1 \
  && echo "arc smoke ok" || { echo "FAILED arc smoke (see $PERSIST/smoke_arc.txt)"; }
cleanrepo Qwen/Qwen2.5-0.5B
echo "=== PART S DONE smoke_ok=$SMOKE_OK"
fi

# ---------------------------------------------------------------- R
if want R; then
echo "=== PART R: random-subspace plant control (C24) $(date)"
if [[ "$SMOKE_OK" != 1 ]] && want S; then echo "SKIPPED part R: smoke failed"; else
python3 patch_steer_stage.py >/dev/null; python3 patch_plant_rand.py >/dev/null
for t in vartrack_d1 vartrack_d3; do
  for L in 14 18 22 26; do
    python steer_stage.py --model EleutherAI/pythia-6.9b --step 143000 --task $t --layer $L --scope single \
      --n-rand 5 --plant-rand --save-items --out $OUT/plant_rand || echo "FAILED plant_rand 6.9b $t $L"
  done
done
cleanrepo EleutherAI/pythia-6.9b
for t in vartrack_d1 vartrack_d3; do
  for L in 8 12 16 20 24 28 32; do
    python steer_stage.py --model EleutherAI/pythia-12b --step 143000 --task $t --layer $L --scope single \
      --n-rand 3 --plant-rand --out $OUT/plant_rand || echo "FAILED plant_rand 12b $t $L"
  done
done
cleanrepo EleutherAI/pythia-12b
python3 plant_control_report.py --dir $OUT/plant_rand || true
fi
echo "=== PART R DONE"
fi

# ---------------------------------------------------------------- M
# run_model MODEL BATCH PERTURB DECOMP MMLU [extra benchmark flags]; ARC option order always runs
run_model() {
  local m=$1 b=$2 dp=$3 dd=$4 dm=$5; shift 5
  echo "=== MODEL $m $(date)"
  if [[ $dp == 1 ]]; then python perturb_test.py --model $m --revision main --batch-size $b --instruct --out $OUT/perturb || echo "FAILED perturb $m"; fi
  if [[ $dd == 1 ]]; then python position_decomposition.py --model $m --revision main --batch-size $b --out $OUT/decomp || echo "FAILED decomp $m"; fi
  if [[ $dm == 1 ]]; then python benchmark_flip.py --model $m --batch-size $b "$@" --out $OUT/benchmark || echo "FAILED mmlu $m"; fi
  python benchmark_flip.py --model $m --batch-size $b "$@" --dataset arc --out $OUT/benchmark_arc || echo "FAILED arc $m"
  cleanrepo $m
}
if want M; then
echo "=== PART M: models $(date)"
run_model allenai/OLMo-2-1124-7B 16 0 0 0
run_model Qwen/Qwen2.5-7B 16 0 0 0
run_model mistralai/Mistral-7B-v0.1 16 0 0 0 --bos
run_model allenai/OLMo-2-1124-7B-Instruct 16 0 1 0
run_model Qwen/Qwen2.5-7B-Instruct 16 0 0 0
run_model Qwen/Qwen2.5-14B 8 1 1 1
run_model allenai/OLMo-2-1124-13B 8 1 1 1
run_model Qwen/Qwen2.5-32B 8 1 1 1
run_model allenai/OLMo-2-0325-32B 8 0 1 0
run_model Qwen/Qwen2.5-72B 4 0 0 0
echo "=== PART M DONE"
fi

# ---------------------------------------------------------------- D
if want D; then
echo "=== PART D: 410M seeds at more steps (exploratory follow-up to C9) $(date)"
for m in pythia-410m pythia-410m-seed1 pythia-410m-seed2 pythia-410m-seed3 pythia-410m-seed5 pythia-410m-seed6 pythia-410m-seed7 pythia-410m-seed8 pythia-410m-seed9; do
  for s in 4000 8000 32000 143000; do
    python position_test.py --model EleutherAI/$m --step $s --out $OUT/position/$m || echo "FAILED pos $m $s"
  done
  cleanrepo EleutherAI/$m
done
echo "=== PART D DONE"
fi

# ---------------------------------------------------------------- E
if want E; then
echo "=== PART E: attention and noise $(date)"
ONLY="Y Z" bash run_extra.sh
ONLY="X" X_STEPS="4000 16000 64000 143000" bash run_extra.sh
echo "=== PART E DONE"
fi
echo "=== FINAL12H ALL DONE $(date)"
