# Exea full day (AMD MI300X, JupyterLab terminal)

Tested on the 1 Oct session: Docker container, you are root, one shared 697 GB disk at
/shared-docker, torch 2.12 (ROCm), python at /opt/venv. Other people use this machine:
never touch files that are not yours, and never use File > Shut Down in Jupyter.

## 0. Before the day (on your Mac)
- EXPERIMENTS.md with sections 4b (C8-C11) AND 4c (C12-C17) must be committed and pushed.
  The commit time is what makes the claims pre-registered. Check: `git log -1 --format=%cd`.
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
grep -v -e '^torch' -e '^gradio' requirements.txt > /tmp/r.txt && pip install -q -r /tmp/r.txt datasets
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
grep -c "C17" EXPERIMENTS.md; ls run_fullday.sh smoke.sh score_new.py perturb_test.py benchmark_flip.py
df -h /shared-docker | tail -1
```
Need: torch 2.12... True, a non-zero count, the five files listed, and disk below 60%.

## 2b. Smoke test (about 5 minutes; do not skip)
```bash
cd $PERSIST/EarlyMind && bash smoke.sh 2>&1 | tail -25
```
It must end with `SMOKE OK`. It runs the new tests on tiny runs (40 questions) so a broken
line shows up now, not at hour 6. If it fails, send the last 25 lines to Claude before
starting the day. (The smoke test stops at its first failure. If the failing line is the
OLMo-2 or Qwen one, the rest of the day can still run; Parts 3 and 5 will just report FAILED.)

## 3. Start the day
```bash
cd $PERSIST/EarlyMind
setsid nohup bash run_fullday.sh > $PERSIST/log_fullday.txt 2>&1 < /dev/null &
sleep 60; head -20 $PERSIST/log_fullday.txt
```
Part 0 prints a checksum line per checkpoint. Any "DUPLICATE" means that model's
checkpoints are unreliable: tell Claude before trusting its results.

Short slot (say 2-3 hours instead of a full day)? Run only the parts that matter most:
```bash
ONLY="2 3 4 5" setsid nohup bash run_fullday.sh > $PERSIST/log_fullday.txt 2>&1 < /dev/null &
```
Parts 2-5 hold the new claims (C12-C17); 0-1 and 6-9 are the older replications and sweeps.

## 4. Check progress (any time; closing the tab is safe)
```bash
grep "=== PART" $PERSIST/log_fullday.txt; grep FAILED $PERSIST/log_fullday.txt | tail; df -h /shared-docker | tail -1
```
Parts run in priority order: 0 integrity, 1 order-swap (C10), 2 reorder vs rename (C12,
C13, C17), 3 OLMo-2 (C16), 4 interleaved layout (C14), 5 MMLU benchmark (C15), 6 12B
trajectory (C8), 7 seeds (C9), 8 6.9B layers (C11), 9 main-grid sweeps (P1-P4). If time runs
out, the most important parts are already done. Disk above 88%: stop and ask.

## 5. Score and export (after "ALL DONE", or whenever you have to stop)
```bash
cd $PERSIST/EarlyMind
python score_new.py --dir $PERSIST/fullday | tee $PERSIST/fullday/score.txt
python export_csv.py --dir $PERSIST/fullday --out /shared-docker/earlymind_fullday.csv
cp $PERSIST/fullday/score.txt /shared-docker/earlymind_score.txt
python errors.py --runs $PERSIST/fullday/runs --out /shared-docker/earlymind_errors_big.csv
```
Download earlymind_fullday.csv and earlymind_score.txt from the file browser (a small
text file downloads fine; if not, open it and copy-paste). The CSV is one number per row
(kind, model, revision, task, layer, metric, value, n).

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
