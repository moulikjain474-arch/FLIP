# IRIS synopsis, v4 (word counts checked against IRIS limits)
Items in [BRACKETS] must be filled/verified before submission. No school/city/state anywhere.

## Abstract (limit 250; now ~243)
When a program or document mentions several variables, a language model must work out which value belongs to the one being asked about. I studied how this ability develops during pretraining, using Pythia models from 410 million to 12 billion parameters, up to 24 training checkpoints per model, and nine independent training runs of one model. Each prompt contained two chains of variable assignments, and the model was asked for the value of one chain. Before models could reliably choose the right chain, they did not guess at random: they took the chain written first or the chain written last, and this preference reversed direction repeatedly during training (five statistically significant reversals across 24 checkpoints of the 1.4-billion-parameter model). The reversals were far larger than measurement noise, did not come from the worked examples in the prompt, and appeared in three of four model variants. Fully trained runs differing only in random seed ended with opposite preferences. By writing numbers directly into the model's internal activations, I found the answer is fixed about halfway through the network at every training stage, so errors come from choosing the wrong chain, not from failing to read values. The 12-billion-parameter model was correct in one order and wrong in the other on 41-49% of questions, and asking in both orders raised its accuracy from [VERIFY 78 or 79]% to 91% without retraining. Single-order benchmarks can therefore misjudge a model's reliability. [TBD: one sentence on the C8-C11 replication.]

## Introduction (100-150; ~133)
Code assistants and long-document readers must track which value belongs to which variable. Language models often fail at this "binding", and recent work measured the limit in finished models, but it is not known how the ability forms during training or what models do before they master it. In small models trained from scratch, researchers found shortcuts based on line position; whether real pretrained models use similar shortcuts, and whether those shortcuts persist, has not been tested. Because the Pythia suite publishes checkpoints from throughout training, this can be observed directly. My objectives were to (1) track how models choose between competing variables across training, (2) locate where inside the network the answer is fixed, and (3) measure how much a model's answers depend on the irrelevant order in which information is presented.

## Innovation (50-100; ~86)  [wording fixed: asking in both orders is a KNOWN fix]
Earlier studies measured binding in finished models or in small models trained from scratch on one task. This project follows real pretrained models across their training checkpoints, sizes and random seeds, and finds that their primacy/recency shortcut does not settle once, as in toy models, but reverses repeatedly and ends differently by seed. It introduces FLIP, an order-swap audit with rename and layout controls, and measures how much of the order sensitivity a known fix (asking in both orders, as in permutation self-consistency) removes compared with a plain prompt instruction. Claims were registered in advance of the replication runs.

## Methodology (150-250; ~190)
Task: Python-style prompts with two chains of variable assignments (1-3 steps deep) and two worked examples. The model chooses among ten single-digit answers; 600 questions per checkpoint and depth.
Models: Pythia 410M, 1.4B, 1.4B-deduplicated, 6.9B and 12B at up to 24 checkpoints, plus eight additional training seeds of 410M. All checkpoint files were verified by checksum and weight fingerprint; Pythia-2.8B was excluded because its public checkpoints are duplicated.
Position test: the share of answers taken from the chain written later, with 95% intervals; controls re-tested the same checkpoints on new question sets and without worked examples.
Order-swap test: every question was also asked with the chains' order reversed; I measured how often correctness changed and whether combining both orders helped.
Activation interventions: at the final token, I computed low-dimensional subspaces that encode each candidate value (from a training half of the questions) and overwrote them at single layers with the representation of a chosen number, measuring on held-out questions how often the model then output that number. Random subspaces of equal size served as controls.
Pre-registration: hypotheses, thresholds and a dated change log were committed before each set of runs.

## Results (100-150; ~133)
Across training, the preference for the first or last chain reversed repeatedly (five significant reversals in 1.4B; up to 0.31 between checkpoints 1000 steps apart, versus 0.02 measurement noise) and ended differently across sizes and seeds (0.31-0.69 across nine 410M runs). Writing a number at layer 14 or later of 24 controlled the answer; earlier layers had no effect, at every training stage. The 12B model changed correctness on 41-49% of questions when the order was swapped; combining both orders raised accuracy from [VERIFY]% to 91%. Erasing the wrong chain's representation did not improve answers (pre-registered null). [TBD: C8-C11.]
Conclusion: pretrained models choose between variables by position long after they can read the values, and that shortcut is unstable, seed-dependent and not removed by scale. Limitations: one synthetic task and one model family.

## Acknowledgements and references (50-100; ~67)
Guide: [name, role]. Compute: Google Colab; we thank AMD and Exea for access to AMD Instinct MI300X GPUs. [Describe any AI-assistant help truthfully, e.g. coding and drafting support.]
References: Biderman et al. 2023 (Pythia); van der Wal et al. 2025 (PolyPythias); OLMo Team 2025 (OLMo 2); Ravulapalli & Chadha 2026; Gur-Arieh et al. 2026; Wu, Geiger & Milliere 2025; Guo & Vosoughi 2025; Feng & Steinhardt 2024; Chen et al. 2024; Zheng et al. 2024.

## Flags
1. "78 vs 79%": my notes say d1 78->91; your draft says 79. Check final scored output.
2. Every number must be re-checked after the final merge/score; "five reversals", "0.31", "41-49%" are from earlier runs.
3. "nine independent training runs" = 410M default + 8 seeds; confirm count.
4. "three of four model variants": confirm vs final position results.
5. Fix the ack line to truthfully disclose AI help.
