#!/usr/bin/env bash
# Full Exea day. Runs the tests registered as C8-C11 in EXPERIMENTS.md, plus the 6.9B / 12B
# main-grid sweeps needed to score P1-P4. Run from the repo folder:
#   export PERSIST=/shared-docker/moulik_earlymind HF_HOME=$PERSIST/hf_cache
#   setsid nohup bash run_fullday.sh > $PERSIST/log_fullday.txt 2>&1 < /dev/null &
# Every model checkpoint is deleted from the cache right after use (the disk is shared).
# Parts run in priority order; each writes "=== PART n DONE" so progress is easy to follow.
set -u
cd "$(dirname "$0")"
OUT=${PERSIST:?set PERSIST first}/fullday
mkdir -p $OUT
clean() { rm -rf $HF_HOME/hub/models--EleutherAI--$1; }
pos()  { python position_test.py --model EleutherAI/$1 --step $2 --out $OUT/position/$1 || echo "FAILED pos $1 $2"; }
swp()  { python swap_test.py --model EleutherAI/$1 --step $2 --out $OUT/swap || echo "FAILED swap $1 $2"; }

echo "=== PART 0: checkpoint integrity (file checksums; no downloads) $(date)"
python - <<'EOF'
from huggingface_hub import HfApi, list_repo_files
api = HfApi()
for m, steps in [("pythia-6.9b", [4000, 16000, 64000, 143000]), ("pythia-12b", [4000, 16000, 64000, 143000]),
                 ("pythia-410m-seed1", [16000, 64000, 143000])]:
    seen = {}
    for s in steps:
        r = f"step{s}"; M = f"EleutherAI/{m}"
        files = [f for f in list_repo_files(M, revision=r) if f.endswith((".bin", ".safetensors"))]
        sig = tuple(sorted((i.path, i.lfs.sha256[:12]) for i in api.get_paths_info(M, files, revision=r) if i.lfs))
        flag = "DUPLICATE of " + seen[sig] if sig in seen else "ok"
        seen.setdefault(sig, r); print(m, r, flag)
EOF
echo "=== PART 0 DONE"

echo "=== PART 1: C10 order-swap across training (6.9B, 12B) $(date)"
for m in pythia-6.9b pythia-12b; do
  for s in 16000 32000 64000; do swp $m $s; clean $m; done
done
echo "=== PART 1 DONE"

echo "=== PART 2: C8 12B position trajectory (+ unseen 6.9B steps) $(date)"
for s in 1000 2000 4000 6000 8000 12000 16000 24000 32000 48000 64000 100000 120000 143000; do pos pythia-12b $s; clean pythia-12b; done
for s in 1000 2000 6000 12000 24000 48000 100000 120000; do pos pythia-6.9b $s; clean pythia-6.9b; done
echo "=== PART 2 DONE"

echo "=== PART 3: C9 410M seeds at unseen steps $(date)"
for m in pythia-410m pythia-410m-seed1 pythia-410m-seed2 pythia-410m-seed3 pythia-410m-seed5 pythia-410m-seed6 pythia-410m-seed7 pythia-410m-seed8 pythia-410m-seed9; do
  for s in 16000 64000; do pos $m $s; done
  clean $m
done
echo "=== PART 3 DONE"

echo "=== PART 4: C11 single-layer plant on 6.9B $(date)"
for s in 16000 143000; do
  for t in vartrack_d1 vartrack_d3; do
    for L in 4 8 10 12 14 16 18 20 22 26; do
      python steer_stage.py --model EleutherAI/pythia-6.9b --step $s --task $t --layer $L --scope single --n-rand 1 \
        --out $OUT/steer_layers || echo "FAILED layers 6.9b $s $t $L"
    done
  done
  clean pythia-6.9b
done
echo "=== PART 4 DONE"

echo "=== PART 5: 6.9B and 12B main-grid sweeps for P1-P4 $(date)"
for m in pythia-6.9b pythia-12b; do
  for s in 0 32 128 512 1000 2000 4000 8000 16000 32000 64000 100000 143000; do
    python sweep.py --model EleutherAI/$m --steps $s --layout blocked --out $OUT/runs || echo "FAILED sweep $m $s"
    clean $m
  done
done
echo "=== PART 5 DONE"
echo "=== ALL DONE $(date)"
