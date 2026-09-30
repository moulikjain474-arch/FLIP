# Exea full day (AMD MI300X, JupyterLab terminal)

Tested on the 1 Oct session: Docker container, you are root, one shared 697 GB disk at
/shared-docker, torch 2.12 (ROCm), python at /opt/venv. Other people use this machine:
never touch files that are not yours, and never use File > Shut Down in Jupyter.

## 0. Before the day (on your Mac)
- EXPERIMENTS.md with section 4b (C8-C11) must be committed and pushed. The commit time is
  what makes C8-C11 pre-registered.
- Package the repo for upload (the GitHub token cannot be typed into the Jupyter terminal):
  ```bash
  cd ~/Downloads/EarlyMind-repo && git archive -o ~/Downloads/EarlyMind.zip HEAD
  ```

## 1. Log in and upload
Open the URL, log in with the token. Upload EarlyMind.zip with the up-arrow in the file
browser. Open File > New > Terminal.

## 2. Set up (paste as one block)
```bash
export PERSIST=/shared-docker/moulik_earlymind HF_HOME=/shared-docker/moulik_earlymind/hf_cache
mkdir -p $PERSIST/EarlyMind $HF_HOME
cd $PERSIST/EarlyMind && python -m zipfile -e /shared-docker/EarlyMind.zip . && rm /shared-docker/EarlyMind.zip
grep -v -e '^torch' -e '^gradio' requirements.txt > /tmp/r.txt && pip install -q -r /tmp/r.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
grep -c "C8" EXPERIMENTS.md; ls run_fullday.sh score_new.py swap_test.py
df -h /shared-docker | tail -1
```
Need: torch 2.12... True, a non-zero count, the three files listed, and disk below 60%.

## 3. Start the day
```bash
cd $PERSIST/EarlyMind
setsid nohup bash run_fullday.sh > $PERSIST/log_fullday.txt 2>&1 < /dev/null &
sleep 60; head -20 $PERSIST/log_fullday.txt
```
Part 0 prints a checksum line per checkpoint. Any "DUPLICATE" means that model's
checkpoints are unreliable: tell Claude before trusting its results.

## 4. Check progress (any time; closing the tab is safe)
```bash
grep "=== PART" $PERSIST/log_fullday.txt; grep FAILED $PERSIST/log_fullday.txt | tail; df -h /shared-docker | tail -1
```
Parts run in priority order: 0 integrity, 1 order-swap (C10), 2 12B trajectory (C8),
3 seeds (C9), 4 6.9B layers (C11), 5 main-grid sweeps (P1-P4). If time runs out, the
most important parts are already done. Disk above 88%: stop and ask.

## 5. Score and export (after "ALL DONE", or whenever you have to stop)
```bash
cd $PERSIST/EarlyMind && python score_new.py --dir $PERSIST/fullday | tee $PERSIST/fullday/score.txt
cd $PERSIST && python - <<'EOF'
import json, glob, csv
w = csv.writer(open("/shared-docker/earlymind_fullday.csv", "w"))
w.writerow(["kind","model","step","task","layer","a","b","c","d","e","n"])
for f in glob.glob("fullday/position/**/*.json", recursive=True):
    s = json.load(open(f))
    for t, v in s["tasks"].items():
        pl = v["picks_later"]
        w.writerow(["pos", s["model"].split("/")[1], s["revision"], t, "", pl["rate"], pl["k"], v["query_chain_later"]["correct"], v["query_chain_earlier"]["correct"], "", pl["n"]])
for f in glob.glob("fullday/swap/*.json"):
    s = json.load(open(f))
    for t, r in s["tasks"].items():
        w.writerow(["swap", s["model"].split("/")[1], f'step{s["step"]}', t, "", r["acc_original"], r["acc_swapped"], r["acc_combined"], r["same_answer_both_orders"], r["correct_in_exactly_one_order"], r["n"]])
for f in glob.glob("fullday/steer_layers/*.json"):
    s = json.load(open(f)); r = s["rates"]
    w.writerow(["layer", s["model"].split("/")[1], s["revision"], s["task"], s["layer"], r["plant_cor"]["planted"], r["plant_dis"]["planted"], r["natural"]["correct"], r["erase_cor"]["correct"], "", s["n_test"]])
print("done")
EOF
cp $PERSIST/fullday/score.txt /shared-docker/earlymind_score.txt
```
Download earlymind_fullday.csv and earlymind_score.txt from the file browser (a small
text file downloads fine; if not, open it and copy-paste). The main-grid sweep outputs
(fullday/runs) are larger: run errors.py on Exea and export its CSV instead:
```bash
cd $PERSIST/EarlyMind && python errors.py --runs $PERSIST/fullday/runs --out /shared-docker/earlymind_errors_big.csv
```

## 6. Clean up (after the files are on your Mac)
```bash
cd / && rm -rf /shared-docker/moulik_earlymind /shared-docker/earlymind_*.csv /shared-docker/earlymind_score.txt
sed -i '/export PERSIST=/d; /export HF_HOME=/d' ~/.bashrc; history -c
df -h /shared-docker | tail -1
```
Close the terminal tab and the browser tab. Do not use File > Shut Down.

## Rules
- No git commit or push from Exea; commit on your Mac.
- "Compute provided by Exea Labs (AMD MI300X)" in the acknowledgements.
- Pythia-2.8B is excluded (duplicated public checkpoints).
