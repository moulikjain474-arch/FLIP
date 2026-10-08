#!/usr/bin/env bash
# Full Exea day. Runs the tests registered as C8-C20 in EXPERIMENTS.md (C21 runs on a laptop:
# frontier_flip.py). The old P1-P4 main-grid sweeps are no longer run (see EXPERIMENTS.md Changes). Run from the repo folder:
#   export PERSIST=/shared-docker/moulik_earlymind HF_HOME=$PERSIST/hf_cache
#   setsid nohup bash run_fullday.sh > $PERSIST/log_fullday.txt 2>&1 < /dev/null &
# Every model is deleted from the cache right after use (the disk is shared).
# Parts run in priority order; each writes "=== PART n DONE" so progress is easy to follow.
# Short slot? Run only some parts, e.g.  ONLY="2 3 4" setsid nohup bash run_fullday.sh ...
#   (2 = reorder vs rename, 3 = OLMo-2, 4 = interleaved layout, 5 = MMLU, 5e = OLMo-2 trajectory,
#    5f = 32B/72B scale check (+ code test), 5g = code test, 5b-5d, 5h-5j = exploratory
#    distance / patching / story task; 0-1 and 6-8 are the older replications)
set -u
cd "$(dirname "$0")"
OUT=${PERSIST:?set PERSIST first}/fullday
: ${HF_HOME:?set HF_HOME first}
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then   # never score leftovers from an earlier attempt
  mv "$OUT" "$OUT.old.$(date +%m%d-%H%M)"; echo "moved earlier results aside: $OUT.old.*"
fi
mkdir -p $OUT
python -c "import datasets, accelerate" 2>/dev/null || pip install -q datasets accelerate
cleanrepo() { rm -rf $HF_HOME/hub/models--${1//\//--}; }
clean() { cleanrepo EleutherAI/$1; }
want() { [[ -z "${ONLY:-}" || " $ONLY " == *" $1 "* ]]; }
pos()  { python position_test.py --model EleutherAI/$1 --step $2 --out $OUT/position/$1 || echo "FAILED pos $1 $2"; }
swp()  { python swap_test.py --model EleutherAI/$1 --step $2 --out $OUT/swap || echo "FAILED swap $1 $2"; }
prt()  { python perturb_test.py --model EleutherAI/$1 --step $2 --instruct --out $OUT/perturb || echo "FAILED perturb $1 $2"; }

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
for s in ${P2_STEPS:-143000 64000 32000}; do prt pythia-12b $s; clean pythia-12b; done
for s in ${P2_STEPS:-143000 64000 32000}; do prt pythia-6.9b $s; clean pythia-6.9b; done
for m in pythia-1.4b pythia-410m; do prt $m 143000; clean $m; done
fi

if want 2; then
echo "=== PART 2 DONE"
fi

if want 3; then
echo "=== PART 3: C16 second model family (OLMo-2), same task $(date)"
for m in allenai/OLMo-2-0425-1B allenai/OLMo-2-1124-7B; do
  python perturb_test.py --model $m --revision main --instruct --out $OUT/perturb || echo "FAILED olmo $m"
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
if ! python benchmark_flip.py --model mistralai/Mistral-7B-v0.1 --bos --out $OUT/benchmark; then
  echo "FAILED bench mistral (gated or unavailable): registered fallback falcon-7b"
  python benchmark_flip.py --model tiiuae/falcon-7b --out $OUT/benchmark || echo "FAILED bench falcon"; cleanrepo tiiuae/falcon-7b
fi
cleanrepo mistralai/Mistral-7B-v0.1
fi

if want 5; then
echo "=== PART 5 DONE"
fi

if want 5e; then
echo "=== PART 5e: C18 OLMo-2-7B position trajectory across its training checkpoints $(date)"
echo "    (each checkpoint is downloaded ONCE; the exploratory OLMo runs that need it happen in the same pass)"
OM=allenai/OLMo-2-1124-7B
REVS=$(python -c "import model_utils as mu; print(' '.join(r for _, _, r in mu.nearest_revisions('$OM', [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])))")
EARLY=$(python -c "import model_utils as mu; print(mu.nearest_revisions('$OM', [0.05])[0][2])")
LAST=$(python -c "import model_utils as mu; print(mu.nearest_revisions('$OM', [1.0])[0][2])")
echo "OLMo-2-7B checkpoints: $REVS (early: $EARLY, last: $LAST)"
[ -z "$REVS" ] && echo "FAILED olmo revisions: could not list OLMo-2-7B checkpoints (C18 will show MISSING)"
for r in $REVS; do
  python position_test.py --model $OM --revision $r --out $OUT/position_olmo/OLMo-2-1124-7B || echo "FAILED olmo traj $r"
  python perturb_test.py --model $OM --revision $r --out $OUT/perturb_olmo_traj || echo "FAILED olmo perturb $r"
  if [ "$r" = "$EARLY" ] || [ "$r" = "$LAST" ]; then
    python attention_test.py --model $OM --revision $r --out $OUT/attention || echo "FAILED olmo attention $r"
  fi
  if [ "$r" = "$EARLY" ]; then
    python patch_test.py --model $OM --revision $r --out $OUT/patch || echo "FAILED olmo patch $r"
  fi
  cleanrepo $OM
done
echo "=== PART 5e DONE"
fi

if want 5f; then
echo "=== PART 5f: C19 scale check: OLMo-2-32B and Qwen2.5-72B $(date)"
for m in allenai/OLMo-2-0325-32B Qwen/Qwen2.5-72B; do
  python perturb_test.py --model $m --revision main --batch-size 8 --instruct --out $OUT/perturb || echo "FAILED scale perturb $m"
  python benchmark_flip.py --model $m --batch-size 4 --out $OUT/benchmark || echo "FAILED scale bench $m"
  python code_test.py --model $m --revision main --batch-size 4 --out $OUT/code || echo "FAILED scale code $m"
  cleanrepo $m
done
echo "=== PART 5f DONE"
fi

if want 5g; then
echo "=== PART 5g: C20 Order-Flip Test on short Python programs $(date)"
for m in pythia-1.4b pythia-6.9b pythia-12b; do
  python code_test.py --model EleutherAI/$m --step 143000 --out $OUT/code || echo "FAILED code $m"
  clean $m
done
for m in allenai/OLMo-2-1124-7B Qwen/Qwen2.5-7B; do
  python code_test.py --model $m --revision main --out $OUT/code || echo "FAILED code $m"
  cleanrepo $m
done
echo "=== PART 5g DONE"
fi

if want 5b; then
echo "=== PART 5b: exploratory: distance to the question (filler lines) $(date)"
for m in pythia-12b pythia-1.4b pythia-410m; do
  python distance_test.py --model EleutherAI/$m --step 143000 --out $OUT/distance || echo "FAILED distance $m"
  clean $m
done
echo "=== PART 5b DONE"
fi

if want 5c; then
echo "=== PART 5c: exploratory: activation patching swapped -> original, per layer $(date)"
for m in pythia-1.4b pythia-12b; do
  python patch_test.py --model EleutherAI/$m --step 143000 --out $OUT/patch || echo "FAILED patch $m"
  clean $m
done
echo "=== PART 5c DONE"
fi

if want 5d; then
echo "=== PART 5d: exploratory: second binding task (story) $(date)"
for m in pythia-410m pythia-1.4b pythia-6.9b pythia-12b; do
  python story_test.py --model EleutherAI/$m --step 143000 --out $OUT/story || echo "FAILED story $m"
  clean $m
done
python story_test.py --model EleutherAI/pythia-12b --step 32000 --out $OUT/story || echo "FAILED story 12b 32000"; clean pythia-12b
echo "=== PART 5d DONE"
fi

if want 5h; then
echo "=== PART 5h: exploratory: attention from the last token to each chain $(date)"
for s in 7000 143000; do python attention_test.py --model EleutherAI/pythia-1.4b --step $s --out $OUT/attention || echo "FAILED attention 1.4b $s"; done
clean pythia-1.4b
python attention_test.py --model EleutherAI/pythia-12b --step 143000 --out $OUT/attention || echo "FAILED attention 12b"
clean pythia-12b
echo "=== PART 5h DONE"
fi

if want 5i; then
echo "=== PART 5i: exploratory: the same tests on OLMo-2-7B (and Qwen2.5-7B) $(date)"
OM=allenai/OLMo-2-1124-7B
python distance_test.py --model $OM --revision main --out $OUT/distance || echo "FAILED olmo distance"
python patch_test.py --model $OM --revision main --out $OUT/patch || echo "FAILED olmo patch main"
python story_test.py --model $OM --revision main --out $OUT/story || echo "FAILED olmo story"
python position_test.py --model $OM --revision main --layout interleaved --out $OUT/position_interleaved/OLMo-2-1124-7B || echo "FAILED olmo interleaved"
cleanrepo $OM
python perturb_test.py --model Qwen/Qwen2.5-7B --revision main --instruct --out $OUT/perturb || echo "FAILED qwen7b perturb"
python story_test.py --model Qwen/Qwen2.5-7B --revision main --out $OUT/story || echo "FAILED qwen7b story"
cleanrepo Qwen/Qwen2.5-7B
echo "=== PART 5i DONE"
fi

if want 5j; then
echo "=== PART 5j: exploratory: base vs chat-tuned (instruct) versions of the same model $(date)"
for m in Qwen/Qwen2.5-7B-Instruct allenai/OLMo-2-1124-7B-Instruct; do
  python perturb_test.py --model $m --revision main --instruct --out $OUT/perturb || echo "FAILED instruct perturb $m"
  python benchmark_flip.py --model $m --out $OUT/benchmark || echo "FAILED instruct bench $m"
  cleanrepo $m
done
echo "=== PART 5j DONE"
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
echo "=== PART 8 DONE"
fi

if want 9; then
echo "=== PART 9: exploratory (mentor R1): where is the CHOICE made? order patching + single-chain control, 1.4B $(date)"
for s in ${P9_STEPS:-4000 16000 32000 64000 143000}; do
  python patch_test.py --model EleutherAI/pythia-1.4b --step $s --every 1 --tasks vartrack_d1 vartrack_d2 vartrack_d3 \
    --out $OUT/patch_choice || echo "FAILED patch_choice 1.4b $s"
done
clean pythia-1.4b
echo "=== PART 9 DONE"
fi

echo "=== ALL DONE $(date)"
