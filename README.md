# FLIP

**FLIP, short for the Order-Flip Test: language models choose by position, not meaning, and the habit keeps flipping during training.**

FLIP is deliberately narrow but testable: one task, one layout, the models tested, with the claims and thresholds
written down before the runs. Pre-registered nulls are reported as results.

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
2. **The final habit differs across sizes; a seed effect is not confirmed.** Fully trained
   models end with different habits. Across 9 training runs of the same 410M model the
   depth-3 preference ranged from 0.31 to 0.69 in the 1 Oct exploration, but the registered
   test of opposite directions (C9) was a MISS: at steps 16,000 and 64,000 most runs were
   first-chain biased (5 of 9 and 4 of 9) and only 1 of 9 was later-chain biased.
3. **The answer is decided halfway through the network.** Writing a number into the
   model's activations does nothing before about layer 12 of 24, and controls the answer
   after about layer 14, at every stage of training. Errors come from putting the wrong
   chain's value into that slot, not from failing to read the values.
4. **Reordering changes the answer, even at 12B.** Pythia-12B is right in one order and
   wrong in the other on 41-49% of questions. Asking in both orders and combining the
   answers raises its accuracy from 79% to 91%, with no training. Combining orders is a
   known cheap fix (permutation self-consistency, Tang et al. 2023); what FLIP adds is
   measuring when it helps, and that reordering moves answers more than renaming does
   (claim C13), so the gain is not just "asking twice".

Reported as nulls: erasing the wrong chain's representation does not help (pre-registered
test C4), and the model's mistakes give no earlier warning of competence than accuracy.
Public Pythia-2.8B intermediate checkpoints load identical weights and are excluded.

## Final results (8 Oct 2026; numbers in [results_08Oct/](results_08Oct/))
Where these differ from the 1 Oct figures above, cite these. Scripts regenerate every number
(`paper_numbers.py`, `order_specific_edge.py`, `exploratory_summary.py`, `make_figures.py`).

- **Claims:** 15 registered claims scored, 13 hit, 2 missed (C9, C22). Both misses are reported.
- **C9 (seed decides) missed.** Nine 410M runs, depth 3: first-chain bias in 5 of 9 (step 16,000)
  and 4 of 9 (step 64,000), later-chain bias in 1 of 9 each. The seeds mostly agree in direction.
- **C22 (confidence) missed.** Pythia-6.9B: the least-confident third flips 2.6-2.8 times as often as the
  most-confident third (depths 1-2). Pythia-12B: ratio 1.0-1.2, p >= 0.13. The claim needed both.
- **C23 (fresh question set) held** for both models.
- **Order-specific gain (exploratory, run after the data were in).** Against an equal-cost control
  (original + renamed), asking in both orders wins clearly in models of about 7B and up. Depth 3:
  OLMo-2-7B-Instruct 83.5% vs 60.2%, OLMo-2-32B 94.7% vs 80.0%, Qwen2.5-7B 99.8% vs 94.8%; Pythia-12B
  depth 1: 90.3% vs 79.3% (exact McNemar p < 1e-7 each, 600 questions). In Pythia-410M, 1.4B and
  6.9B at depth 3 the gain is mostly generic ensembling. Qwen2.5-7B at depth 2: reordering and
  renaming change answers equally often (6.5% vs 6.5%).
- **Scale.** Qwen2.5-72B is at the ceiling on the toy task (99.8-100%) but changes 10.1% of MMLU answers.
- **Recency (exploratory).** Wrong answers are mostly the later-written chain in both orders:
  Pythia-12B depth 1, 39% later vs 3% earlier; OLMo-2-1B depth 3, 69% vs 0% (`position_decomposition.py`).
- **MMLU option order, 9 models.** 10-30% of answers change. Asking in both orders gains 1.5-2.8 points in
  the seven main-run models (95% intervals above zero); Qwen2.5-14B +2.3 and Qwen2.5-32B +0.3 at 300
  questions (reported separately, `results_08Oct/merged/bench_n300`).
- **Steering, Pythia-6.9B.** A planted answer wins about 0% at layer 14 and 55-76% (depth 3) or 19-35%
  (depth 1) at layer 22. There is no random-direction control of the same size yet.

## Confounds ruled out
| Possible explanation | Control | Where |
|---|---|---|
| The particular variable names | Rename every variable; flips under renaming are the baseline (C12) | `perturb_test.py` |
| Distance from the question | Insert no-op lines between the chains (exploratory) | `distance_test.py` |
| The layout of the chains | Interleaved layout (C14) | `position_test.py --layout interleaved` |
| Copying the worked examples | No worked examples (shots 0) at 1.4B | `position_test.py --shots 0` |
| The wording of the prompt | Instruction version of the prompt (exploratory) | `perturb_test.py --instruct` |
| The task format | Story task, short Python programs, MMLU option order (C15 replicates Zong et al. 2024) | `story_test.py`, `code_test.py`, `benchmark_flip.py` |
| Numerical noise in batched scoring | Same prompts at two batch sizes (exploratory) | `noise_check.py` |
| Chance | Flip rate expected if answers ignored order, plus an exact Fisher test (exploratory) | `extra_stats.py` |

Swapping the chains keeps the correct answer unchanged by construction (the chains are
independent), and `test_flip.py` checks this on 500 generated prompts.

## Pre-registration
The analysis plan, thresholds and every change (dated, with reasons) are in
[EXPERIMENTS.md](EXPERIMENTS.md), committed before the runs they govern. Results found
exploratorily on 1 Oct are re-tested on unseen checkpoints as claims C8-C11, and the new
controls and generalisation tests C12-C20, all registered before those runs and scored by
`score_new.py`.

## Pipeline

| Step | Script | Output |
|---|---|---|
| Run the task at checkpoints (earlier work, not in the full-day run) | `sweep.py` | `runs/*.npz` |
| Answer types, stages (earlier work) | `errors.py`, `stages.py` | `error_types.csv`, stage table and figure |
| Probes and onsets | `probes.py`, `analyze.py` | `results.csv` |
| Position test (first vs last chain) | `position_test.py` | `position/*.json` |
| Order-swap test | `swap_test.py` | `swap/*.json` |
| Causal erase / plant, per layer | `steer_stage.py` | `steer_stage/*.json` |
| Reorder vs rename, with per-question results | `perturb_test.py` | `perturb/*.json` |
| Order-Flip Test on a real benchmark (MMLU options) | `benchmark_flip.py` | `benchmark/*.json` |
| Distance, patching, story task (exploratory). Patching includes a single-chain control (where a VALUE is read out vs where the CHOICE of chain is made), per-question log-prob margins and a digit-token check | `distance_test.py`, `patch_test.py`, `story_test.py` | `distance/`, `patch/`, `patch_choice/`, `story/` |
| Order-Flip Test on closed frontier models, incl. hard mode, a same-order repeat (noise floor), optional `--vote K` and `--effort` (C21; API, laptop) | `frontier_flip.py` | `frontier_results/*.json` |
| Attention from the last token to each chain (exploratory) | `attention_test.py` | `attention/*.json` |
| Order-Flip Test on short Python programs | `code_test.py`, `code_task.py` | `code/*.json` |
| Flip rate against model size (descriptive trend, after the run) | `size_fit.py` | printed fit |
| Strict / lenient accuracy, chance flip rate, Fisher and McNemar tests (exploratory) | `extra_stats.py` (also in `score_new.py`, `export_csv.py`) | printed, `xstat_*` rows |
| Does attention track the preference across training? (exploratory) | `attn_vs_pref.py` | printed |
| Commit layer as a fraction of depth (exploratory) | `commit_depth.py` | printed |
| Numerical-noise check, two batch sizes (exploratory) | `noise_check.py` | `noise/*.json` |
| Extra GPU runs: attention across training, noise check | `run_extra.sh` | `attention/`, `noise/` |
| Order effect concentrates where the model is unsure (registered C22) | `uncertainty_gate.py` (`run_c22_c23.sh`) | `uncertainty/*.json` |
| Order effect on a freshly drawn question set (registered C23) | `perturb_test.py --seed 1` (`run_c22_c23.sh`) | `perturb_seed1/*.json` |
| Which chain is picked in each order, by confidence (exploratory, post hoc) | `position_decomposition.py` (`run_decomp.sh`) | `decomp/*.json` |
| Check any model's order sensitivity with one command | `flip_check.py` | printed report |
| Figures straight from the result files | `make_figures.py` | `figures/*.png` |
| Every number for the paper, printed from the results | `paper_numbers.py` | printed |
| Score the registered claims C8-C20, C22, C23 (C21: `frontier_flip.py --score`) | `score_new.py` | printed hit / miss |
| Flatten results to one CSV | `export_csv.py` | `earlymind_fullday.csv` |
| Both orders vs the equal-cost control (exploratory) | `order_specific_edge.py` | `results_08Oct/order_specific_edge.txt` |
| Decomposition and C22 numbers from saved JSONs | `exploratory_summary.py` | `results_08Oct/exploratory_summary.txt` |
| Quick check of the new tests | `smoke.sh`, `test_flip.py` | `SMOKE OK` |
| Full compute day | `run_fullday.sh` | every test above except the frontier API test |
| Figures from the 1 Oct session | `results/exea_2026-10-01/make_figs.py` | `plots/*.png` |

Always use `--layout blocked` (the default everywhere except the earlier `sweep.py`).

## Scope and caveats
- Training-dynamics results are on Pythia (and OLMo-2-7B checkpoints); the order tests also
  cover OLMo-2, Qwen2.5 and Mistral/Falcon. Main task is synthetic; code, story and MMLU
  tests extend it.
- The layer and plant results show where the output is fixed, not the full circuit that
  chooses the chain.
- Combining both orders helps a lot where the order effect is order-specific on this task (about 7B and
  up) and only a little on MMLU (1.5-2.8 points). It doubles inference cost, and
  with two orders it needs log-probabilities to break ties (a plain vote ties on every
  flip); black-box APIs without log-probabilities need several shuffles and a vote
  (`frontier_flip.py --vote`, exploratory).
- Model-size trends are descriptive: five sizes, one run each above 410M, and seed-to-seed
  variation at 410M is large, so no scaling law is claimed.
- Results above 410M come from one training run per size; a reversal counts as seed-robust
  only if it is larger than the 410M seed spread (0.31 to 0.69).
- Related work, and how FLIP differs:
  - Ravulapalli & Chadha 2026: how many bindings finished models hold (capacity). FLIP:
    which binding a model picks when unsure, and how that changes during training.
  - Gur-Arieh et al. 2026: finished models retrieve bindings mainly by position. FLIP: how
    that positional preference forms and flips during pretraining.
  - Wu, Geiger & Milliere 2025: in small models trained from scratch, a position heuristic
    appears once and is replaced. FLIP: in real pretrained models it reverses repeatedly
    and ends differently across sizes (a seed effect was not confirmed, C9).
  - Tang et al. 2023 (permutation self-consistency, arXiv 2310.07712): shuffle a list several
    times and aggregate the rankings. FLIP does not claim the fix; it measures when it helps
    (reorder vs rename control) and uses a log-probability sum because two orders tie a vote.
  - Mistry et al. 2025 (how primacy- and recency-like attention structure changes during training
    of GPT-2) and Salvatore et al. 2025 (a U-shaped position bias appears in models trained from
    scratch on simple memory tasks). FLIP: in real pretraining runs the preferred
    direction reverses repeatedly and is committed mid-network.
  - Guo & Vosoughi 2025: primacy and recency effects in LLMs; prompt fixes inconsistent.
    FLIP: a measured fix (asking in both orders) against a plain prompt instruction.
  - Exploiting Primacy Effect to Improve LLMs [check authors]: fine-tuning amplifies primacy in
    multiple-choice QA. FLIP compares base and chat-tuned versions of the same models.
  - A study of vision-language training [check title and authors]: text-only training
    encourages positional shortcuts in binding. FLIP follows such shortcuts through pretraining.
  - Order effects in finished models: Chen et al. 2024 (premise order), Zheng et al. 2024
    (multiple-choice option position), Hsieh et al. 2024 (RULER), Liu et al. 2024 (lost in
    the middle).

Compute: Google Colab; Exea Labs (AMD MI300X).

## License
MIT
