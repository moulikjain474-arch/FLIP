# FLIP: final scored results (9 Oct 2026)

Every registered claim C8-C30 is reported here, hit or miss. Thresholds are the ones written in EXPERIMENTS.md before
each run; none was changed after a result was seen. Full scorer output, with every per-model line, is kept verbatim in
`results/final_9oct/score_union_final.txt` and `results/final_9oct/all_reports.txt`; every raw result file is in
`results/final_9oct/flip_results_final.zip`.

## How the numbers were produced

- Two node days were combined: the earlier day (hand-assembled in `results_08Oct/merged`) and the 9 Oct run
  (`fullday`). They were copied into one folder with no overwriting (`cp -rn`: on a name collision the earlier-day file
  is kept), and scored once: `python3 score_new.py --dir <union folder>`.
- Scoring only the 9 Oct folder gives many false MISSING lines (it lacks the earlier day's files), and scoring only
  the earlier day leaves C24-C29 NOT RUN or MISS (it lacks the 9 Oct results). The union is the only complete scoring.
- C30 is not scored by `score_new.py`; it is scored by `ladder_report.py` and `ladder_report_all.py` (see C30 below).
- `results/final_9oct/SHA256SUMS` holds the hashes of the zip and the CSV.
  Zip: 4,973,011 bytes, 1,185 entries. CSV (`earlymind_union.csv`): 12,718 lines.

## Scoreboard

23 claims registered as C8-C30: **16 HIT, 5 MISS, 2 NOT RUN.**

| Claim | What it says | Verdict |
|---|---|---|
| C8 | 12B flips direction at least twice (depth 2 or 3) | HIT |
| C9 | Independent training runs of the same model disagree on the preferred chain (at least 2 later and 2 first) at depth 3, step 16000 or 64000 | **MISS** |
| C10 | Correct in exactly one order at least 0.20, depth 2 (6.9B, 12B; steps 16k/32k/64k) | HIT |
| C11 | 6.9B: planting does nothing at layers 4, 8, 10 and works at some layer up to 22 | HIT |
| C12 | Reordering flips more answers than renaming (12B, depth 2, three checkpoints) | HIT (3 of 3 cells, difference at least 0.05) |
| C13 | The both-orders ensemble beats the equal-cost rename ensemble (12B, depth 1 and 2) | HIT |
| C14 | The position shortcut also appears with interleaved chains (depth 3, final checkpoints) | HIT (2 of 3 models biased) |
| C15 | Order dependence on a real benchmark (MMLU option order) | HIT (3 models, all with flip at least 0.10, gain in all) |
| C16 | The order effect appears in a second family (OLMo-2) | HIT |
| C17 | Both-orders gain over the single-order average has a 95% interval above 0 (12B final) | HIT |
| C18 | OLMo-2-7B's position habit changes direction during training | HIT |
| C19 | Order dependence at 32B-72B scale | HIT |
| C20 | Order dependence on short Python programs | HIT (3 of 3 eligible models, flip at least 0.10) |
| C21 | Shuffled facts hurt closed frontier models | **NOT RUN** (needs the author's own API key) |
| C22 | The order effect concentrates in the low-confidence third of questions (6.9B and 12B) | **MISS** (1 of 2 models) |
| C23 | The order effect replicates on a freshly drawn question set | HIT (2 of 2 model runs) |
| C24 | Planting in the learned subspace beats random subspaces of the same rank by at least 0.15 | HIT |
| C25 | ARC-Challenge option order: flip at least 0.03 and a both-orders gain above 0, in at least 7 of 10 models | **MISS** (6 of 10) |
| C26 | Medication-record wording, depth 3, in at least 4 of 7 models | **MISS** (3 of 7) |
| C27 | Harder medication wording (depth 6 or 9) at 72B | HIT |
| C28 | Generated answers with step-by-step reasoning still flip (at least 2 of 4 models) | HIT (3 of 4) |
| C29 | A frontier model via API, medication wording | **NOT RUN** (needs the author's own API key) |
| C30 | A small learned subspace controls the answer later than the chain is readable (Pythia-6.9B ladder) | **MISS** as registered (see below) |

## The misses, with the numbers behind them

**C9, seeds do not disagree as registered.** Per-seed preference at depth 3 (L = picks the later chain, F = the earlier
one, - = neither, one character per run, nine 410M runs):
step 16000 `F-LF--FFF` (L=1, F=5); step 64000 `--FFL--FF` (L=1, F=4). The rule needed at least two L and two F.
The runs agree more than they disagree, so a seed-dependent direction is not supported as registered.

**C22, the order effect does not concentrate where the model is unsure, in both models.** Flip rate by confidence third,
least to most confident:

| Model | Low | Mid | High | Ratio (low/high) | Low minus high [95% interval] | Fisher p | Supported |
|---|---|---|---|---|---|---|---|
| Pythia-6.9B | 0.400 | 0.250 | 0.145 | 2.76 | +0.255 [+0.170, +0.335] | 1.24e-08 | yes |
| Pythia-12B | 0.510 | 0.420 | 0.440 | 1.16 | +0.070 [-0.035, +0.160] | 0.193 | no |

Accuracy by third: 6.9B 0.64 / 0.88 / 0.96; 12B 0.59 / 0.75 / 0.88. The rule needs both models, so C22 is a MISS and
the wording "when unsure" is removed from every text (as EXPERIMENTS.md section 4f says).

**C25, ARC-Challenge option order (6 of 10).** Registered models, flip original vs reversed and both-orders gain
[95% interval] (n = 1000 each):

| Model | Accuracy | Flip | Gain | Supported |
|---|---|---|---|---|
| Mistral-7B-v0.1 | 0.771 | 0.190 | +0.032 [+0.018, +0.046] | yes |
| OLMo-2-0325-32B | 0.901 | 0.079 | +0.018 [+0.010, +0.027] | yes |
| OLMo-2-1124-13B | 0.822 | 0.145 | +0.025 [+0.013, +0.036] | yes |
| OLMo-2-1124-7B-Instruct | 0.801 | 0.170 | +0.020 [+0.007, +0.033] | yes |
| OLMo-2-1124-7B | 0.798 | 0.193 | +0.028 [+0.014, +0.042] | yes |
| Qwen2.5-14B | 0.932 | 0.049 | +0.007 [+0.001, +0.015] | yes |
| Qwen2.5-32B | 0.954 | 0.031 | +0.004 [-0.001, +0.010] | no |
| Qwen2.5-72B | 0.957 | 0.030 | +0.005 [-0.001, +0.011] | no |
| Qwen2.5-7B-Instruct | 0.911 | 0.065 | +0.005 [-0.003, +0.013] | no |
| Qwen2.5-7B | 0.900 | 0.060 | +0.005 [-0.003, +0.013] | no |

(Qwen2.5-72B-Instruct was also run, accuracy 0.951, flip 0.035, gain +0.004 [-0.001, +0.011]; it is not one of the ten
registered models.) All four non-supporting models are Qwen2.5: their gain interval includes 0. Reading, after the fact:
the Qwen2.5 models are accurate enough on ARC that order leaves little room to help.

**C26, medication-record wording at depth 3 (3 of 7).** Registered models (both orders, accuracy; flip; gain):

| Model | Accuracy | Flip | Gain | Supported |
|---|---|---|---|---|
| OLMo-2-1124-7B-Instruct | 0.863 / 0.857 | 0.270 | +0.107 [+0.087, +0.125] | yes |
| OLMo-2-1124-13B | 0.973 / 0.977 | 0.050 | +0.025 [+0.017, +0.034] | yes |
| Qwen2.5-7B-Instruct | 0.917 / 0.938 | 0.145 | +0.064 [+0.050, +0.079] | yes |
| Qwen2.5-14B | 1.000 / 1.000 | 0.000 | +0.000 | no |
| Qwen2.5-32B | 1.000 / 1.000 | 0.000 | +0.000 | no |
| OLMo-2-0325-32B | 1.000 / 1.000 | 0.000 | +0.000 | no |
| Qwen2.5-72B | 0.998 / 1.000 | 0.002 | +0.001 [+0.000, +0.003] | no |

Four of the seven models are at the ceiling on this wording at depth 3, so the order effect has nothing to act on.
That is why C27 was registered (harder depths). The registered rule stands: C26 is a MISS. Models outside the rule, shown
for completeness: Pythia-12B 0.473 / 0.560, flip 0.940, gain +0.158 [+0.122, +0.195]; Pythia-6.9B 0.535 / 0.585,
flip 0.830, gain +0.147 [+0.112, +0.181]; DeepSeek-R1-Distill-Llama-8B 0.688 / 0.677, flip 0.562, gain +0.168
[+0.140, +0.195]; OLMo-2-1124-7B 0.877 / 0.872, flip 0.252, gain +0.121 [+0.103, +0.138]; Qwen2.5-7B 0.992 / 0.993,
flip 0.015, gain +0.007 [+0.003, +0.013]; Llama-3.1-70B-Instruct 1.000 / 1.000, flip 0.000.

**C30, readable versus used (Pythia-6.9B, layer 22).** The five lines the report scripts printed:

| Report | Result |
|---|---|
| `ladder_report.py`, registered steps (4000 to 143000) | MISS: the chain-identity probe is already at 1.000 at the first registered step, so "readable before used" cannot be shown there |
| `ladder_report.py`, early steps only | MISS (registered checkpoints not run in that folder) |
| `ladder_report_all.py`, registered plus early steps | HIT (exploratory: early steps 0 to 2000 were added after the miss) |
| `ladder_report_all.py`, fresh question sample (seed 1) | HIT (exploratory) |
| `ladder_report.py`, Pythia-1.4B | MISS |

The registered verdict is MISS. The extended ladders are exploratory and are not a replacement verdict. The 1.4B result
disagrees with the 6.9B extension and is reported, not set aside. The 6.9B ladder is one training run, so "seed 1" here
means a fresh question sample, not an independent training run.

## NOT RUN

- C21 and C29 (frontier models via API) need the author's own API key; no frontier-model result is claimed.
- SAE pilot (`sae_pilot.py`): Gemma-2-2B access was still "awaiting a review" at the end of the node session, so the
  pilot was never run. Its decision rule is fixed in the script docstring.
- Mistral-7B-v0.1 failed the medication-wording and deep-medication runs and the seed-1 perturbation run; it is not
  one of the seven C26 models. It did complete ARC (C25).

## Supporting numbers for the hits

- **C24.** Depth 3, Pythia-6.9B, step 143000, layer 22: learned-subspace plant 0.677, random-subspace plant 0.000,
  difference +0.677, p = 2.8e-75 (n = 300). The depth-1 line is in `score_union_final.txt`.
- **C27.** Qwen2.5-72B-Instruct, depth 9: flip 0.080, both-orders gain +0.037 [+0.026, +0.048]; depth 6: flip 0.002,
  gain -0.001 [-0.003, +0.000]. Qwen2.5-72B, depth 9: flip 0.023, gain +0.012 [+0.006, +0.018]; depth 6: flip 0.002,
  gain +0.001 [+0.000, +0.003]. The rule needs flip at least 0.05 and a gain above 0 at depth 6 or 9 in one of the two,
  met by Qwen2.5-72B-Instruct at depth 9. For comparison, Qwen2.5-7B-Instruct at depth 9: flip 0.650, gain +0.223
  [+0.196, +0.250].

## Exploratory results (all added after results were seen; none registered)

**Onset of chain readability, nine 410M runs** (`seed_onset.py`; best-layer probe accuracy, chance about 0.5;
shuffled-label control in the last column):

| Step | d1 range | d3 range | Shuffled control |
|---|---|---|---|
| 1000 | 0.547 to 0.593 | 0.582 to 0.627 | 0.46 to 0.56 |
| 2000 | 0.818 to 0.985 | 0.890 to 1.000 | 0.46 to 0.51 |
| 4000 | 0.973 to 1.000 | 0.993 to 1.000 | 0.46 to 0.53 |

In all nine runs the probe is only weakly above its shuffled control at step 1000 (0.55 to 0.63 against 0.46 to 0.56),
reaches at least 0.82 at step 2000 and at least 0.97 at step 4000. Seed 4 is not in these files.

**6.9B ladder on a fresh question sample** (`ladder_report_all.py`, seed 1; plant minus random-subspace plant, and the
erase effect on the correct answer's log-probability, vs random subspaces; steps 0 to 32000 shown, later steps are in
`all_reports.txt`):

| Step | Depth | Natural accuracy | Probe (chance) | Plant minus random (p) | Erase minus random (p) |
|---|---|---|---|---|---|
| 1000 | d1 | 0.123 | 0.573 (0.517) | -0.008 (0.45) | +0.113 (1.8e-10) |
| 1000 | d3 | 0.097 | 0.587 (0.518) | +0.005 (0.7) | +0.030 (0.0023) |
| 2000 | d1 | 0.113 | 0.973 (0.517) | +0.030 (0.02) | +0.163 (4.2e-14) |
| 2000 | d3 | 0.127 | 0.998 (0.518) | +0.025 (0.074) | +0.144 (1.2e-13) |
| 4000 | d1 | 0.243 | 0.998 (0.517) | +0.267 (8.6e-21) | -0.023 (0.2) |
| 4000 | d3 | 0.187 | 1.000 (0.518) | +0.090 (1.1e-07) | +0.009 (0.64) |
| 8000 | d1 | 0.350 | 1.000 (0.517) | +0.480 (2.3e-44) | -0.031 (0.027) |
| 8000 | d3 | 0.300 | 1.000 (0.518) | +0.191 (5.8e-15) | -0.026 (0.073) |
| 16000 | d1 | 0.510 | 1.000 (0.517) | +0.783 (2.6e-101) | -0.015 (0.048) |
| 16000 | d3 | 0.543 | 1.000 (0.518) | +0.630 (1.6e-66) | -0.026 (0.0077) |
| 32000 | d1 | 0.660 | 1.000 (0.517) | +0.813 (5.4e-111) | -0.029 (3.4e-07) |
| 32000 | d3 | 0.633 | 1.000 (0.518) | +0.550 (8.9e-54) | -0.054 (1.7e-10) |

Reading: the chain is readable (probe at least 0.97) at step 2000, while planting a value in the learned subspace
moves the answer by at most +0.03 at steps 1000 and 2000, then at step 4000 by +0.27 (d1) and +0.09 (d3), and keeps
growing.
Caveats: this is one training run; the model's own accuracy at steps 1000 and 2000 is at chance (0.10 to 0.13), and the
erase column is significantly positive at steps 1000 and 2000, which this write-up does not interpret. A probe finds
information; it does not show the model uses it.

**Norm-matched and out-of-distribution checks** (`plant_normmatch.py`, `ood_check.py`; Pythia-6.9B, step 143000, layer
22). Controls that push the state by the same per-item norm as the learned plant. In depth 1, the learned plant stays
closest to natural states (share of items beyond the natural 95th percentile of the off-manifold residual: 0.29, natural
0.05), while random whole-space pushes are far off (0.99 to 1.00) and random pushes inside the subspace are 0.42 to
0.64. Depth 3 shows the same pattern (0.14 for the plant). The median planted edit is 0.12 to 0.13 of the state's length.

**Black-box baseline for the two-order flag** (`error_catch_baseline.py`, exploratory, post hoc). Flagging a question
when the two orders disagree catches most order-driven errors in the stronger models, and beats a ridge-regression
no-model baseline at the same review budget there; for example Qwen2.5-7B-Instruct depth 3: catches 100.0% of errors vs
15.9% for the baseline (difference +0.841 [+0.738, +0.926]); Pythia-6.9B depth 1: 94.7% vs 13.2% (+0.816
[+0.683, +0.938]). It does not beat the baseline in the weakest models: Pythia-410M (depth 1 difference -0.009
[-0.111, +0.092]), Pythia-1.4B depths 2 and 3 (+0.025 [-0.070, +0.117], -0.047 [-0.133, +0.041]) and OLMo-2-0425-1B
depth 3 (-0.024 [-0.096, +0.049]). Some Qwen2.5 cells have 0 or 1 errors, so intervals there are uninformative.
The flag says "check this", not which answer is right.

**Value probe: uninformative.** About 0.35 at random initialization and unchanged at step 512; stopped early; not used as
evidence.

## What the results support, and what they do not

Supported: reordering the same facts changes answers well beyond renaming them (C12, C23); asking in both orders
beats an equal-cost control and has a positive gain in the models where there is headroom (C13, C17, C15, C16, C19,
C20, C27, C28); the effect scales with how hard the question is for the model (headroom), so strong models need harder
depths or wording to show it (C26, C27); a small learned subspace is a causal handle on the answer in 6.9B (C11, C24).

Not supported, and to be said in the paper: a seed-dependent direction (C9); the order effect concentrating where the
model is unsure (C22); a registered readable-before-used ordering (C30, MISS as registered; exploratory extensions
agree on 6.9B only); order effects on ARC in every Qwen2.5 model (C25); order effects at depth 3 in every medication
model (C26).

Wording rules: do not say "when unsure"; do not say models "pick the later chain" without a size, family and depth;
describe the ladder result as exploratory; say that steering shows control, not where the computation is.

## To verify before submission

- Numbers in README.md and docs/PAPER_DRAFT_v4.md (for example "41-49%", "79% to 91%", "five reversals (registered interval rule)",
  "0.31 to 0.69") come from the 1 Oct session and were not re-derived here. Re-check each against
  `results/final_9oct/results_08Oct/paper_numbers.txt` and `all_reports.txt`.
- The reason seed 4 is absent from the 410M seed files, and a with/without table, are still owed.
- A1 to A6 (random-subspace layer search, digits, padding, whole-state and per-head patching) are in
  `all_reports.txt`; their numbers are not copied into this file.

## Errata (4q). Corrections and additions to 4p, 9 Oct 2026 (second commit)

Source for all numbers: results/final_9oct/ (score_union_final.txt, all_reports.txt) and results/exea_2026-10-01/results.csv.
Nothing below changes a registered verdict. The first commit (f147d16) is not rewritten; this section corrects it.

### Corrections
1. **Pythia-1.4B ladder (C30).** Commit f147d16 and parts of 4p, docs/RESULTS_FINAL.md and docs/STORY.md say the ladder
   "disagrees on 1.4B" or is "a MISS" in a way that reads as evidence against the 6.9B result. That is overstated.
   The 1.4B ladder only ran steps 4000 and later. At step 4000 the chain probe is already 0.998-1.000 and planting
   already works (+0.467 depth 1, +0.170 depth 3), so "first readable" and "first used" coincide at the first step by
   construction. The registered rule scores this as a MISS, and it stays a MISS. It cannot separate readable from
   used, because no early 1.4B step was run. It is not evidence against the 6.9B extension.
2. **12B accuracy gain.** Earlier text said "79% to 91%". Final numbers (600 questions, single = mean of the two orders):
   depth 1 77.3% -> 90.3% (gain +13.0, 95% +10.6 to +15.3); depth 2 73.7% -> 85.8%; depth 3 65.1% -> 72.7%.
   A plain instruction gives 75.2% / 71.7% / 66.7%. Swap flip 41.3 / 45.7 / 48.5%; rename 18.3 / 16.7 / 22.5%.
   Both orders vs equal-cost rename ensemble (exact McNemar): depth 1 94:28 p=1.6e-09; depth 2 95:39 p=1.4e-06; depth 3 100:65 p=0.0079.
3. **1.4B reversals.** Earlier text said "over 24 checkpoints". 24 is correct: the 1 Oct position data hold 24 checkpoints per depth for
   Pythia-1.4B (a 33 in an earlier draft of this note was wrong). By the 95% Wilson-interval rule there are 2 direction changes at depth 1 and 5 each at depths 2 and 3; say "up to five".
4. **Layer statement.** "Layer 12-14 of 24" is the Pythia-1.4B figure (single-layer plant, 1 Oct). In the final files
   planting with a learned subspace on 6.9B works from about layer 18-22 of 32 (depth 3: L18 +0.113, L22 +0.677, L26 +0.817;
   depth 1: L18 +0.007, L22 +0.280, L26 +0.460) and on 12B from about layer 20 of 36 (L20 +0.243/+0.330, L24 +0.803/+0.880).
   Equal-size random subspaces give 0.000 everywhere.
5. **410M seeds.** "Nine runs" = the main 410M run plus seeds 1, 2, 3, 5, 6, 7, 8, 9. Seed 4 has three checkpoints only
   (8000, 32000, 143000) and is excluded as a suspected bad checkpoint (see 1 Oct entry). Depth-3 later-share at step 143000
   over these nine runs: 0.312-0.685 (seed 4 would be 0.523, so including it does not change the range).
   The registered C9 test (disagreement at steps 16000 or 64000) is a MISS: L=1, F=5 and L=1, F=4.
6. **"Three of four model variants"** (old paper draft) could not be reproduced from the final files and is dropped.

### Additions (exploratory, Pythia-6.9B step 143000 unless stated)
- **A1, selection-matched random-subspace control: PASS.** Best random draw is 0.000 in all 5 draws; learned best 0.460 (depth 1) and
  0.817 (depth 3), both at layer 26. 12B (3 draws): learned 0.990 / 0.967 at layer 32, random 0.000.
- **A2, digit breakdown:** no result is in the final files. Do not cite A2.
- **A3, padding / distance (headpad).** Padding before the question or at the start moves the later-share by under 0.1;
  inserting 8 lines between the chains raises the later-share (12B +0.17 / +0.24 / +0.30; 6.9B +0.17 / +0.20 / +0.25; 1.4B +0.43 / +0.56 / +0.60)
  and lowers accuracy. Distance between the chains matters as well as order.
- **A4, attention vs MLP (6.9B):** no single sublayer carries the order effect; best attention transfer 0.147 (depth 1) and
  0.135 (depth 3), both at layer 16; MLP at most 0.067 / 0.065.
- **A5, whole-state patching:** transfer from the other order rises over roughly 50-70% of depth (12B depth 1: 0.419 at L18, 0.915 at L20 of 36;
  6.9B depth 3: 0.350 at L12, 0.603 at L16, 0.780 at L20). A single-chain control saturates in the same window, so this shows when
  the information has been moved, not which component does it.
- **A6, per-head patching (6.9B):** layer 16 head 27 reaches 0.160 (depth 1) and 0.121 (depth 3); all other layers are at or below 0.047.
  No head is a bottleneck.
- **No-examples control (1.4B, later-share at step 143000, depths 1-3):** 0.43 / 0.26 / 0.20 without worked examples vs 0.52 / 0.42 / 0.34 with.
  Same direction, so the preference is not caused by copying the examples.

### Wording rules after this section
- Say "Pythia-1.4B cannot test readable-versus-used because it was run from step 4000", never "disagrees".
- Say 77.3% -> 90.3% (depth 1), never 79% -> 91%.
- Say "24 checkpoints" and "up to five reversals" for the 1.4B count.
