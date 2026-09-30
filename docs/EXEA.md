# Running EarlyMind on Exea (AMD MI300X, SSH) - final version

Replaces every earlier exea_commands file. Order = priority: the causal and position tests
decide the project, so they run first. You commit everything yourself; nothing here pushes.

Known setup: SSH, ROCm PyTorch preinstalled, internet OK, ~720 GB persistent disk, nohup for
background jobs, plenty of GPU-hours.

## 0. Before the first long run
- The latest EXPERIMENTS.md must be committed and pushed. Its commit time is the record that
  the plan and predictions P1-P4 came before the 6.9B / 12B runs.
- The distractor field is confirmed (`item["meta"]["distractor"]`), nothing to check.

## 1. Setup
```bash
df -h                                    # find the ~720 GB disk
export PERSIST=/path/to/that/disk        # edit
export HF_HOME=$PERSIST/hf_cache
export HF_TOKEN=<HF_TOKEN>
echo "export PERSIST=$PERSIST" >> ~/.bashrc
echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
echo "export HF_TOKEN=$HF_TOKEN" >> ~/.bashrc
cd $PERSIST
git clone https://<GH_TOKEN>@github.com/moulikjain474-arch/EarlyMind.git
cd EarlyMind
```

## 2. Check GPU and torch (never pip install torch)
```bash
rocm-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Need True, an MI300X, torch >= 2.6 (6.9B / 12B ship .bin weights). Otherwise stop and ask Exea.

## 3. Install the rest and smoke test
```bash
grep -v -e '^torch' -e '^gradio' requirements.txt > /tmp/req_exea.txt
pip install -r /tmp/req_exea.txt          # add --break-system-packages if pip refuses
python smoke_test.py
```

## 4. Quick check of the new scripts (about 10 min)
```bash
python position_test.py --model EleutherAI/pythia-1.4b --step 5000 --n 200 --out $PERSIST/check
python steer_stage.py --model EleutherAI/pythia-1.4b --step 5000 --task vartrack_d1 --n 200 --out $PERSIST/check
```
Look at: the example prompt printed by position_test (is the order detection right?), and that
steer_stage prints every condition (natural, erase_*, pca_top, plant_dis, plant_cor,
random0-4) and ends with `primary_erase` and `primary_plant`. Sanity checks: erase_cor should
LOWER correct; plant_cor should raise "planted" well above natural. If either fails, the
subspaces are broken, so stop and paste the output to Claude before step 5.

## 5. Causal + position tests (the deciding experiment)
```bash
cd $PERSIST/EarlyMind
nohup bash run_causal.sh > $PERSIST/log_causal.txt 2>&1 &
tail -n 30 $PERSIST/log_causal.txt
```

## 6. Big models (can run at the same time; 192 GB is enough for both)
```bash
nohup python sweep.py --model EleutherAI/pythia-6.9b --layout blocked --out $PERSIST/runs > $PERSIST/log_69b.txt 2>&1 &
```
When 6.9B is done:
```bash
nohup python sweep.py --model EleutherAI/pythia-12b --layout blocked --out $PERSIST/runs > $PERSIST/log_12b.txt 2>&1 &
```
Then position tests on the final big models:
```bash
python position_test.py --model EleutherAI/pythia-6.9b --step 143000 --out $PERSIST/position
python position_test.py --model EleutherAI/pythia-12b --step 143000 --out $PERSIST/position
```

## 7. Dense grid for 410M (checks the 1.4B stage pattern at a second size)
```bash
nohup python sweep.py --model EleutherAI/pythia-410m --steps 5000 6000 7000 9000 10000 12000 14000 18000 20000 24000 28000 --layout blocked --out $PERSIST/runs_dense > $PERSIST/log_410m_dense.txt 2>&1 &
```

## 8. Optional, lowest priority: one modern model
Already known that modern models bind better (Ravulapalli & Chadha 2026); this only checks the
position shortcut. Stops with a clear error if the answer numbers are not single tokens.
```bash
python position_test.py --model allenai/OLMo-2-1124-7B --revision main --out $PERSIST/position
```

## 9. Copy results to your Mac (run ON YOUR MAC)
```bash
scp -r <user>@<host>:<PERSIST>/{runs,runs_dense,steer,steer_stage,position} ~/Downloads/earlymind_exea/
```
Upload `runs` and `runs_dense` to Drive for errors.py / probes.py on Colab.

## Checking on jobs (safe to close SSH)
```bash
tail -n 20 $PERSIST/log_*.txt
ls $PERSIST/runs | wc -l
```

## Stop rules
- torch can't see the GPU or is < 2.6: stop, ask Exea.
- smoke test or step 4 fails: stop, paste the full error to Claude.
- Never start step 6 before EXPERIMENTS.md is pushed.
- No git commit/push from Exea.
- "Compute provided by Exea Labs (AMD MI300X)" only for results that ran there.
