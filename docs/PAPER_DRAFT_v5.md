# IRIS synopsis, v5 (every number below was checked against the final scored files of 9 Oct 2026)
Items in [BRACKETS] must be filled before submission. No school, city, state or repo link anywhere.
Limits: abstract 250, introduction 100-150, innovation 50-100, methodology 150-250, results 100-150, acknowledgements 50-100.

## Abstract
When a program mentions several variables, a language model must work out which value belongs to the one being asked about. I studied how this develops during pretraining, using Pythia models from 410 million to 12 billion parameters at many training checkpoints, plus nine training runs of the 410M model that differ only in random seed. Each prompt held two chains of variable assignments, and the model was asked for the value at the end of one chain. Before models answered reliably, they often favoured the chain written first or last, and in the 1.4-billion-parameter model this preference changed direction up to five times across 24 checkpoints. In the nine 410M runs it ended anywhere from 0.31 to 0.69 at depth 3, but my pre-registered prediction that runs would disagree in direction at two chosen checkpoints missed. Writing a number into internal activations took control of the answer only from the middle layers on, while equal-size random subspaces did nothing. The 12-billion-parameter model changed its answer when the two chains were swapped on 41-49% of questions, and combining both orders raised accuracy from 77% to 90% at depth 1, more than an equal-cost rename control. Several other pre-registered predictions missed and are reported. Single-order benchmarks can misjudge a model's reliability.

## Introduction
Code assistants and long-document readers must track which value belongs to which variable. Language models often fail at this binding, and recent work measured the limit in finished models, but how the ability forms during training, and what models do before they master it, is less clear. Small models trained from scratch have shown shortcuts based on line position; whether real pretrained models use similar shortcuts, and whether those persist, was not known to me. Because the Pythia suite publishes checkpoints from throughout training, this can be observed directly. My objectives were to (1) track how models choose between competing variables across training, (2) find at which layers the answer becomes controllable, and (3) measure how much answers depend on the irrelevant order in which information is presented.

## Innovation
Earlier studies measured binding in finished models or in small models trained from scratch. This project follows real pretrained models across checkpoints, sizes and random seeds, and measures an order-swap audit (FLIP) with rename controls. Asking in both orders is a known fix; I measure how much of the order sensitivity it removes compared with a plain instruction and an equal-cost rename ensemble. Claims and thresholds were registered before the runs, and registered misses are reported as misses.

## Methodology
Task: Python-style prompts with two chains of variable assignments (1-3 steps deep) and two worked examples; the model chooses among ten digits; 600 questions per checkpoint and depth.
Models: Pythia 410M, 1.4B, 1.4B-deduplicated, 6.9B and 12B at many checkpoints, plus eight extra 410M seeds. Pythia-2.8B was excluded because its public checkpoints are duplicated.
Position test: share of answers taken from the later-written chain, with 95% intervals; controls used new question sets and no worked examples.
Order-swap test: every question was also asked with the chains reversed; I measured how often correctness changed and whether combining both orders helped, against a rename ensemble of equal cost (exact McNemar test).
Activation interventions: at the final token, low-dimensional subspaces for candidate values were fitted on half the questions and overwritten at single layers; held-out questions measured how often the model output the planted number. Random subspaces of equal size, given the same layer search, were the control.
Pre-registration: hypotheses, thresholds and a dated change log were committed before each set of runs; later analyses are labelled exploratory.

## Results
In 1.4B the later-chain preference changed direction up to five times over 24 checkpoints, with swings of 0.31 between neighbouring checkpoints against a noise level of 0.02; it persisted without worked examples. At depth 3 the 410M seeds ended between 0.31 and 0.69. Planting a number controlled the answer only from about layers 18-22 of 32 (6.9B) and layer 20 of 36 (12B), with random subspaces at 0.000. Pythia-12B changed its answer under a swap on 41-49% of questions, versus 17-23% under renaming; both orders raised accuracy from 77.3% to 90.3% (depth 1), versus 75.2% with a plain instruction. Misses: seed disagreement (C9), readable-before-used timing (C30), and confidence predicting flips in 12B (C22).
Limitations: one synthetic task, mostly one model family, one training run per size above 410M.

## Acknowledgements and references
Guide: [name, role]. Compute: Google Colab; I thank AMD and Exea for access to AMD Instinct MI300X GPUs. [Describe AI-assistant help truthfully: what it was used for, e.g. coding and drafting support, and that all experiments were designed, run and checked by me.]
References: Biderman et al. 2023 (Pythia); van der Wal et al. 2025 (PolyPythias); OLMo Team 2025 (OLMo 2); Ravulapalli & Chadha 2026; Gur-Arieh et al. 2026; Wu, Geiger & Milliere 2025; Guo & Vosoughi 2025; Feng & Steinhardt 2024; Chen et al. 2024; Zheng et al. 2024; Tang et al. 2023 (permutation self-consistency).

## What changed from v4 (all checked)
1. 78/79% -> 91% was wrong. Final: 77.3% -> 90.3% (depth 1), 73.7% -> 85.8% (depth 2), 65.1% -> 72.7% (depth 3).
2. "three of four model variants" removed (could not be reproduced from the final files).
3. "up to 24 checkpoints" kept: 24 is the count in the 1 Oct position data (an earlier note saying 33 was wrong).
4. "layer 14 of 24" was the 1.4B figure; replaced by the 6.9B and 12B onsets from the final files (layers 18-22 of 32, layer 20 of 36).
5. "ended with opposite preferences" -> range 0.31-0.69, and the registered C9 test is reported as a miss.
6. "when unsure" removed everywhere (C22 missed in 12B).
7. "Erasing the wrong chain's representation did not help" dropped from the synopsis (limit; kept in EXPERIMENTS.md).
8. Added: C9, C22, C30 misses; equal-cost rename control; plain-instruction comparison.
9. Still to verify by you before submission: the noise level 0.02 and swing 0.31 come from the 1 Oct runs, not re-run in the final union.
