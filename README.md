# EarlyMind

**How a language model learns which variable you asked about, and why it never fully gets it right.**

EarlyMind follows Pythia language models through pretraining (13 main checkpoints plus a dense
grid, sizes 160M to 12B, four extra training seeds at 160M) on a variable-tracking task: two
chains of assignments in a Python REPL, one of them queried, answered as the next token.

Every answer falls into one of four types: **correct**, **distractor** (the other chain's
value), **shot_copy** (the answer of a worked example) or **other**. Tracking these types
across training shows the stages the model goes through, and activation edits test what
drives its wrong-chain answers.

## Questions
1. **Stages.** What does the model do before it answers correctly? (`stages.py`)
2. **Early warning.** Does anything, inside the model (probes) or in its mistakes, signal
   competence before plain accuracy does? (`probes.py`, `analyze.py`, `monitor.py`)
3. **Cause.** At the stage where the model picks between chains almost at random, does
   erasing the wrong chain's value from its activations make it pick the right one?
   (`steer_stage.py`, with `steer.py` as the general erasure test)
4. **Position.** Does the model choose a chain by where it sits in the prompt?
   (`position_test.py`)

The analysis plan, what counts as support for each claim, and every change to the plan
(with dates and reasons) are in [EXPERIMENTS.md](EXPERIMENTS.md).

## Pipeline

| Step | Script | Output |
|---|---|---|
| Run every task at every checkpoint | `sweep.py` | `runs/*.npz` (candidate logits + activations) |
| Probes and onsets (main grid only) | `probes.py`, `analyze.py` | `results.csv`, figures |
| Answer types per checkpoint | `errors.py` | `error_types.csv` |
| Stage table and figure | `stages.py` | `figs/stages_table.csv`, `figs/stages.png` |
| Mistakes vs accuracy as a warning | `monitor.py` | `figs/monitor.csv` |
| General erasure test | `steer.py` | `steer/*.json` |
| Wrong-chain erasure test | `steer_stage.py` | `steer_stage/*.json` |
| Position test | `position_test.py` | `position/*.json` |
| All causal + position tests | `run_causal.sh` | the three folders above |

Always use `--layout blocked` (the default in the new scripts, not in `sweep.py` / `steer.py`).

```bash
# main grid for one model
python sweep.py --model EleutherAI/pythia-1.4b --layout blocked --out runs
# dense grid, kept separate so the pre-registered main-grid analysis is untouched
python sweep.py --model EleutherAI/pythia-1.4b --steps 3000 5000 6000 7000 9000 10000 11000 12000 --layout blocked --out runs_dense
python errors.py --runs runs --out figs/error_types.csv
python errors.py --runs runs_dense --out figs/error_types_dense.csv
python stages.py --errors figs/error_types.csv figs/error_types_dense.csv --out figs/stages
python monitor.py --errors figs/error_types.csv figs/error_types_dense.csv
nohup bash run_causal.sh > log_causal.txt 2>&1 &
```

## Scope and caveats
- One synthetic task, Pythia models, a blocked prompt layout. The blocked layout may allow a
  partial shortcut, so results are stated as "follows the correct chain", not as reasoning.
- Interleaved layout was tried first and failed at depth 2-3 even at 1.4B; it is reported.
- Closest prior work: Ravulapalli & Chadha 2026 (binding capacity in finished models),
  Feng & Steinhardt 2024 (binding IDs), Gur-Arieh et al. 2026 (retrieval mechanisms),
  Wu et al. 2025 (training phases in from-scratch models). See EXPERIMENTS.md.

## License
MIT
