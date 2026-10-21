# FLIP

Do language models track which value belongs to which variable, or do they lean on where it was written?

FLIP asks every question twice, with the order of two variable chains swapped, and once more with only the variable names changed. If a model really tracks variables, the order shouldn't matter. It does.

## What I found

- Swapping the chains changes **41–49%** of Pythia-12B's answers. Just renaming the variables changes **17–23%**.
- The side a model prefers flips during training: five times in Pythia-1.4B, up to six in Pythia-6.9B.
- Asking in both orders and combining the answers lifts Pythia-12B from **77.3% to 90.3%**. A one-pass instruction gets 75.2%, an equal-cost rename ensemble 79.3%.
- Overwriting a digit subspace steers the answer only from the middle layers onward. Matched random subspaces never did.
- I wrote down 23 claims before running them: 16 held, 5 missed, 2 weren't run. The misses are in the paper too.

It's one synthetic task on open-weight models, with one run per size above 410M. Think of it as a probe, not a verdict.

## What's here

```
core/         the task, model loading, stats and probes
experiments/  one script per study
analysis/     reports, stats and figures
claims/       the claims, registered before each run
scripts/      the shell scripts that ran everything
results/      final result files
docs/         notes and the compute runbook
EXPERIMENTS.md  the claims log
```

## Running it

```bash
pip install -r requirements.txt
source env.sh
python experiments/swap_test.py --help
```

The full runs live in `scripts/`. `scripts/run_fullday.sh` is the big one, and `docs/EXEA.md` explains the GPU setup I used.

## A note on the claims

`EXPERIMENTS.md` is the log I wrote before running each study. The commits are dated, so you can check that every claim came before its result.
