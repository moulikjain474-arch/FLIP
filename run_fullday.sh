#!/usr/bin/env bash
# Full Exea day. Runs the tests registered as C8-C17 in EXPERIMENTS.md, plus the 6.9B / 12B
# main-grid sweeps needed to score P1-P4. Run from the repo folder:
#   export PERSIST=/shared-docker/moulik_earlymind HF_HOME=$PERSIST/hf_cache
#   setsid nohup bash run_fullday.sh > $PERSIST/log_fullday.txt 2>&1 < /dev/null &
# Every model is deleted from the cache right after use (the disk is shared).
# Parts run in priority order; each writes "=== PART n DONE" so progress is easy to follow.
# Short slot? Run only some parts, e.g.  ONLY="2 3 4" setsid nohup bash run_fullday.sh ...
#   (2 = reorder vs rename, 3 = OLMo-2, 4 = interleaved layout, 5 = MMLU; 0-1 and 6-9 are the rest)
set -u
cd "$(dirname "$0")"
OUT=${PERSIST:?set PERSIST first}/fullday
: ${HF_HOME:?set HF_HOME first}
mkdir -p $OUT
python -c "import datasets" 2>/dev/null || pip install -q datasets
cleanrepo() { rm -rf $HF_HOME/hub/models--${1//\//--}; }
clean() { cleanrepo EleutherAI/$1; }
want() { [[ -z "${ONLY:-}" || " $ONLY " == *" $1 "* ]]; }
pos()  { python position_test.py --model EleutherAI/$1 --step $2 --out $OUT/position/$1 || echo "FAILED pos $1 $2"; }
swp()  { python swap_test.py --model EleutherAI/$1 --step $2 --out $OUT/swap || echo "FAILED swap $1 $2"; }
prt()  { python perturb_test.py --model EleutherAI/$1 --step $2 --out $OUT/perturb || echo "FAILED perturb $1 $2"; }

if want 0; then
echo "=== PART 0: checkpoint integrity (file checksums; no downloads) $(date)"
python - <<'PY'
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
PY
fi

if want 0; then
echo "=== PART 0 DONE"
fi

if want 1; then
echo "=== PART 1: C10 order-swap across training (6.9B, 12B) $(date)"
for m in pythia-6.9b pythia-12b; do
  for s in 16000 32000 64000; do swp $m $s; clean $m; done
done
fi

if want 1; then
echo "=== PART 1 DONE"
fi

if want 2; then
echo "=== PART 2: C12 / C13 / C17 reorder vs rename, with per-question results $(date)"
for s in 143000 64000 32000; do prt pythia-12b $s; clean pythia-12b; done
for s in 143000 64000 32000; do prt pythia-6.9b $s; clean pythia-6.9b; done
for m in pythia-1.4b pythia-410m; do prt $m 143000; clean $m; done
fi

if want 2; then
echo "=== PART 2 DONE"
fi

if want 3; then
echo "=== PART 3: C16 second model family (OLMo-2), same task $(date)"
for m in allenai/OLMo-2-0425-1B allenai/OLMo-2-1124-7B; do
  python perturb_test.py --model $m --revision main --out $OUT/perturb || echo "FAILED olmo $m"
  cleanrepo $m
done
fi

if want 3; then
echo "=== PART 3 DONE"
fi

if want 4; then
echo "=== PART 4: C14 interleaved layout $(date)"
for m in pythia-410m pythia-1.4b pythia-6.9b; do
  python position_test.py --model EleutherAI/$m --step 143000 --layout interleaved \
    --out $OUT/position_interleaved/$m || echo "FAILED interleaved $m"
  clean $m
done
fi

if want 4; then
echo "=== PART 4 DONE"
fi

if want 5; then
echo "=== PART 5: C15 real benchmark (MMLU option order) $(date)"
python benchmark_flip.py --selftest
python benchmark_flip.py --model allenai/OLMo-2-1124-7B --out $OUT/benchmark || echo "FAILED bench olmo"; cleanrepo allenai/OLMo-2-1124-7B
python benchmark_flip.py --model Qwen/Qwen2.5-7B --out $OUT/benchmark || echo "FAILED bench qwen"; cleanrepo Qwen/Qwen2.5-7B
python benchmark_flip.py --model mistralai/Mistral-7B-v0.1 --bos --out $OUT/benchmark || echo "FAILED bench mistral"; cleanrepo mistralai/Mistral-7B-v0.1
fi

if want 5; then
echo "=== PART 5 DONE"
fi

if want 6; then
echo "=== PART 6: C8 12B position trajectory (+ unseen 6.9B steps) $(date)"
for s in 1000 2000 4000 6000 8000 12000 16000 24000 32000 48000 64000 100000 120000 143000; do pos pythia-12b $s; clean pythia-12b; done
for s in 1000 2000 6000 12000 24000 48000 100000 120000; do pos pythia-6.9b $s; clean pythia-6.9b; done
fi

if want 6; then
echo "=== PART 6 DONE"
fi

if want 7; then
echo "=== PART 7: C9 410M seeds at unseen steps $(date)"
for m in pythia-410m pythia-410m-seed1 pythia-410m-seed2 pythia-410m-seed3 pythia-410m-seed5 pythia-410m-seed6 pythia-410m-seed7 pythia-410m-seed8 pythia-410m-seed9; do
  for s in 16000 64000; do pos $m $s; done
  clean $m
done
fi

if want 7; then
echo "=== PART 7 DONE"
fi

if want 8; then
echo "=== PART 8: C11 single-layer plant on 6.9B $(date)"
for s in 16000 143000; do
  for t in vartrack_d1 vartrack_d3; do
    for L in 4 8 10 12 14 16 18 20 22 26; do
      python steer_stage.py --model EleutherAI/pythia-6.9b --step $s --task $t --layer $L --scope single --n-rand 1 \
        --out $OUT/steer_layers || echo "FAILED layers 6.9b $s $t $L"
    done
  done
  clean pythia-6.9b
done
fi

if want 8; then
echo "=== PART 8 DONE"
fi

if want 9; then
echo "=== PART 9: 6.9B and 12B main-grid sweeps for P1-P4 $(date)"
for m in pythia-6.9b pythia-12b; do
  for s in 0 32 128 512 1000 2000 4000 8000 16000 32000 64000 100000 143000; do
    python sweep.py --model EleutherAI/$m --steps $s --layout blocked --out $OUT/runs || echo "FAILED sweep $m $s"
    clean $m
  done
done
fi

if want 9; then
echo "=== PART 9 DONE"
fi

echo "=== ALL DONE $(date)"
