#!/usr/bin/env bash
source "$(dirname "$0")/../env.sh"
# Five-minute check of the new tests on tiny runs, BEFORE the long day. Any line that fails
# here is cheaper to fix now than at hour 6.  Run from the repo folder:
#   export PERSIST=/shared-docker/moulik_earlymind HF_HOME=$PERSIST/hf_cache; bash scripts/smoke.sh
set -e
cd "$(dirname "$0")"
S=${PERSIST:?set PERSIST first}/smoke
rm -rf $S; mkdir -p $S
python -c "import datasets, accelerate" 2>/dev/null || pip install -q datasets accelerate
python experiments/test_flip.py
python experiments/frontier_flip.py --selftest
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
from huggingface_hub import hf_hub_download
for m in ["EleutherAI/pythia-12b", "allenai/OLMo-2-0425-1B", "allenai/OLMo-2-1124-7B", "allenai/OLMo-2-0325-32B",
          "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-72B", "Qwen/Qwen2.5-7B-Instruct", "allenai/OLMo-2-1124-7B-Instruct", "mistralai/Mistral-7B-v0.1", "tiiuae/falcon-7b"]:
    try:
        hf_hub_download(m, "config.json"); print("downloadable", m)
    except Exception as e:   # gated or missing: only Mistral has a registered fallback
        print("NOT DOWNLOADABLE", m, type(e).__name__)
        if m not in ("mistralai/Mistral-7B-v0.1", "tiiuae/falcon-7b"):
            raise
from transformers import AutoTokenizer
import benchmark_flip as bf
for m in ("tiiuae/falcon-7b", "mistralai/Mistral-7B-v0.1"):
    try:
        print("letter ids ok", m, bf.letter_ids(AutoTokenizer.from_pretrained(m)))
    except Exception as e:
        print("letter ids FAILED", m, type(e).__name__, e)
import model_utils as mu
print("OLMo-2-7B checkpoints:", [r for _, _, r in mu.nearest_revisions("allenai/OLMo-2-1124-7B", [0.01, 0.5, 1.0])])
PY
python experiments/perturb_test.py --model EleutherAI/pythia-410m --step 143000 --n 40 --tasks vartrack_d1 --instruct --out $S/perturb
python experiments/perturb_test.py --model allenai/OLMo-2-0425-1B --revision main --n 40 --tasks vartrack_d1 --out $S/perturb
python experiments/position_test.py --model EleutherAI/pythia-410m --step 143000 --layout interleaved --n 40 --tasks vartrack_d1 --out $S/position_interleaved/pythia-410m
python experiments/benchmark_flip.py --model Qwen/Qwen2.5-0.5B --n 40 --out $S/benchmark
python experiments/distance_test.py --model EleutherAI/pythia-410m --step 143000 --n 20 --tasks vartrack_d1 --out $S/distance
python experiments/patch_test.py --model EleutherAI/pythia-410m --step 143000 --n 20 --tasks vartrack_d1 --every 8 --out $S/patch
python experiments/story_test.py --model EleutherAI/pythia-410m --step 143000 --n 20 --depths 1 --out $S/story
python experiments/position_test.py --model allenai/OLMo-2-0425-1B --revision main --n 20 --tasks vartrack_d1 --out $S/position_olmo/OLMo-2-0425-1B
python experiments/code_test.py --model EleutherAI/pythia-410m --step 143000 --n 20 --levels 2 --out $S/code
python experiments/code_test.py --model Qwen/Qwen2.5-0.5B --revision main --n 20 --levels 2 --out $S/code
python experiments/attention_test.py --model EleutherAI/pythia-410m --step 143000 --n 10 --tasks vartrack_d1 --out $S/attention
python experiments/story_test.py --model Qwen/Qwen2.5-0.5B --revision main --n 20 --depths 1 --out $S/story
R1=$(python -c "import model_utils as mu; print(mu.nearest_revisions('allenai/OLMo-2-0425-1B', [0.5])[0][2])")
python experiments/perturb_test.py --model allenai/OLMo-2-0425-1B --revision $R1 --n 20 --tasks vartrack_d1 --out $S/perturb
python experiments/patch_test.py --model allenai/OLMo-2-0425-1B --revision main --n 10 --tasks vartrack_d1 --every 8 --out $S/patch
python experiments/attention_test.py --model allenai/OLMo-2-0425-1B --revision main --n 5 --tasks vartrack_d1 --out $S/attention
python analysis/export_csv.py --dir $S --out $S/smoke.csv
rm -rf $HF_HOME/hub/models--EleutherAI--pythia-410m $HF_HOME/hub/models--allenai--OLMo-2-1124-7B $HF_HOME/hub/models--allenai--OLMo-2-0425-1B $HF_HOME/hub/models--Qwen--Qwen2.5-0.5B
echo "SMOKE OK"
