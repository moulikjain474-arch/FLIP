# FLIP

**FLIP (First-or-Last Interference Profile): language models choose by position, not meaning, and the habit keeps flipping during training.**

*(Formerly EarlyMind. The repository name is unchanged.)*

Give a language model two chains of assignments and ask about one of them:

```
>>> t = 0        >>> n = 1
>>> v = t        >>> c = n
>>> v            ->  0   (the other chain's value, 1, is the typical mistake)
```

FLIP follows Pythia models (410M to 12B parameters, up to 24 checkpoints per model,
9 independent training runs at 410M) through pretraining on this task. It asks what the
model does before it can reliably pick the right chain, where inside the network the
answer is decided, and how much the answer depends on something that should not matter:
the order in which the two chains are written.

## Main results (1 Oct 2026; details in [EXPERIMENTS.md](EXPERIMENTS.md))

1. **Position shortcuts that flip.** When unsure, models take the chain that comes first
   or the one that comes last, and which one keeps changing during training (5
   statistically significant reversals over 24 checkpoints of Pythia-1.4B, up to 4 in
   6.9B; 1.4B-deduped instead keeps a first-chain habit at depths 2-3). The flips are not
   measurement noise
   (the same checkpoint on new questions differs by about 0.02; checkpoints 1000 steps
   apart differ by up to 0.31) and are not caused by the worked examples in the prompt.
2. **The final habit depends on size and random seed.** Fully trained models end with
   different habits; across 9 training runs of the same 410M model, only the random
   seed differs, yet the depth-3 preference ranges from 0.31 to 0.69.
3. **The answer is decided halfway through the network.** Writing a number into the
   model's activations does nothing before about layer 12 of 24, and controls the answer
   after about layer 14, at every stage of training. Errors come from putting the wrong
   chain's value into that slot, not from failing to read the values.
4. **Reordering changes the answer, even at 12B.** Pythia-12B is right in one order and
   wrong in the other on 41-49% of questions. Asking in both orders and combining the
   answers raises its accuracy from 79% to 91%, with no training.

Reported as nulls: erasing the wrong chain's representation does not help (pre-registered
test C4), and the model's mistakes give no earlier warning of competence than accuracy.
Public Pythia-2.8B intermediate checkpoints load identical weights and are excluded.

## Pre-registration
The analysis plan, thresholds and every change (dated, with reasons) are in
[EXPERIMENTS.md](EXPERIMENTS.md), committed before the runs they govern. Results found
exploratorily on 1 Oct are re-tested on unseen checkpoints as claims C8-C11, and the new
controls and generalisation tests C12-C17, all registered before those runs and scored by
`score_new.py`.

## Pipeline

| Step | Script | Output |
|---|---|---|
| Run the task at checkpoints | `sweep.py` | `runs/*.npz` |
| Answer types, stages | `errors.py`, `stages.py` | `error_types.csv`, stage table and figure |
| Probes and onsets | `probes.py`, `analyze.py` | `results.csv` |
| Position test (first vs last chain) | `position_test.py` | `position/*.json` |
| Order-swap test | `swap_test.py` | `swap/*.json` |
| Causal erase / plant, per layer | `steer_stage.py` | `steer_stage/*.json` |
| Reorder vs rename, with per-question results | `perturb_test.py` | `perturb/*.json` |
| Order-Flip Test on a real benchmark (MMLU options) | `benchmark_flip.py` | `benchmark/*.json` |
| Distance, patching, story task (exploratory) | `distance_test.py`, `patch_test.py`, `story_test.py` | `distance/`, `patch/`, `story/` |
| Order-Flip Test on closed frontier models (API, laptop) | `frontier_flip.py` | `frontier_results/*.json` |
| Order-Flip Test on short Python programs | `code_test.py`, `code_task.py` | `code/*.json` |
| Score the registered claims C8-C20 | `score_new.py` | printed hit / miss |
| Flatten results to one CSV | `export_csv.py` | `earlymind_fullday.csv` |
| Quick check of the new tests | `smoke.sh`, `test_flip.py` | `SMOKE OK` |
| Full compute day | `run_fullday.sh` | everything above |
| Figures from the 1 Oct session | `results/exea_2026-10-01/make_figs.py` | `plots/*.png` |

Always use `--layout blocked` (the default everywhere except `sweep.py`).

## Scope and caveats
- One synthetic task in one layout, one model family (Pythia).
- The layer and plant results show where the output is fixed, not the full circuit that
  chooses the chain.
- Combining both orders is shown to help only on this task.
- Related work: Wu, Geiger & Milliere 2025 (heuristics in toy models trained from
  scratch), Gur-Arieh et al. 2026 (retrieval mechanisms in trained models), Ravulapalli &
  Chadha 2026 (binding capacity in finished models), Guo & Vosoughi 2025 (serial-position
  effects in LLMs).

Compute: Google Colab; Exea Labs (AMD MI300X).

## License
MIT
