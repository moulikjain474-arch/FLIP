# IRIS synopsis draft (v3, after the 1 Oct Exea session)

Word limits from IRIS. [TBD] = numbers from the full-day runs (C8-C11); if a claim misses,
rewrite that sentence to say so. No school, city or state anywhere, and no repo link.

---

## Title
FLIP: Language Models Choose by Position, Not Meaning, and the Habit Keeps Flipping
During Training

## 1. Abstract (max 250 words)
When a program or document mentions several variables, a language model must work out
which value belongs to the one being asked about. I studied how this ability develops
during pretraining, using Pythia models from 410 million to 12 billion parameters, up to
24 training checkpoints per model, and nine independent training runs of one model. Each
prompt contained two chains of variable assignments, and the model was asked for the
value of one chain. Before models could reliably choose the right chain, they did not
guess at random: they took the chain written first or the chain written last, and this
preference reversed direction repeatedly during training (five statistically significant
reversals across 24 checkpoints of the 1.4-billion-parameter model). These reversals
were far larger than measurement noise, did not come from the worked examples in the
prompt, appeared in three of four model variants, and fully trained runs that differed
only in their random seed ended with opposite preferences. By writing numbers directly into the model's internal
activations, I found that the answer is fixed about halfway through the network at every
stage of training, so errors come from choosing the wrong chain, not from failing to read
values. The consequence is large: the 12-billion-parameter model was correct in one order
and wrong in the other on 41-49% of questions, and simply asking in both orders raised
its accuracy from 79% to 91% without any retraining. Single-order benchmarks can
therefore misjudge a model's reliability. [TBD: one sentence on C8-C11 replication.]

## 2. Introduction and objective (100-150 words)
Code assistants and long-document readers must track which value belongs to which
variable. Language models often fail at this "binding", and recent work measured the
limit in finished models, but it is not known how the ability forms during training or
what models do before they master it. In small models trained from scratch, researchers
found shortcuts based on line position; whether real pretrained models use similar
shortcuts, and whether those shortcuts persist, has not been tested. Because the Pythia
suite publishes checkpoints from throughout training, this can be observed directly. My
objectives were to (1) track how models choose between competing variables across
training, (2) locate where inside the network the answer is fixed, and (3) measure how
much a model's answers depend on the irrelevant order in which information is presented.

## 3. Innovation (50-100 words)
Previous studies examined binding either in finished models or in small models trained
on one task. This project follows real pretrained models across their whole training, at
five sizes and nine seeds, and shows that their position shortcut can reverse repeatedly and
settles differently depending on the random seed. It introduces an order-swap test that
exposes hidden inconsistency in otherwise accurate models, and it pre-registered its
analyses, including a second round of claims fixed before replication runs.

## 4. Methodology (150-250 words)
Task: Python-style prompts with two chains of variable assignments (1-3 steps deep) and
two worked examples. The model chooses among ten single-digit answers; 600 questions per
checkpoint and depth.
Models: Pythia 410M, 1.4B, 1.4B-deduplicated, 6.9B and 12B at up to 24 checkpoints, plus
eight additional training seeds of 410M. All checkpoint files were verified by checksum
and weight fingerprint; Pythia-2.8B was excluded because its public checkpoints are
duplicated.
Position test: the share of answers taken from the chain written later, with 95%
intervals; controls re-tested the same checkpoints on new question sets and without
worked examples.
Order-swap test: every question was also asked with the chains' order reversed; I
measured how often correctness changed and whether combining both orders helped.
Activation interventions: at the final token, I computed low-dimensional subspaces that
encode each candidate value (from a training half of the questions) and overwrote them at
single layers with the representation of a chosen number, measuring on held-out questions
how often the model then output that number. Random subspaces of equal size served as
controls.
Pre-registration: hypotheses, thresholds and a dated change log were committed before
each set of runs.

## 5. Results and conclusions (100-150 words)
Across training, the preference for the first or last chain reversed repeatedly (five
significant reversals in 1.4B; up to 0.31 between checkpoints 1000 steps apart, versus
0.02 measurement noise) and ended
differently across sizes and seeds (0.31-0.69 across nine 410M runs). Writing a number at
layer 14 or later of 24 controlled the answer; earlier layers had no effect, at every
training stage. The 12B model changed correctness on 41-49% of questions when the order
was swapped; combining both orders raised accuracy from 79% to 91%. Erasing the wrong
chain's representation did not improve answers (pre-registered null). [TBD: C8-C11.]
Conclusion: pretrained models choose between variables by position long after they can
read the values, and that shortcut is unstable, seed-dependent and not removed by scale.
Limitations: one synthetic task and one model family.

## 6. Acknowledgement and references (50-100 words)
Guide: [name, role]. Compute: Google Colab; Exea Labs (AMD MI300X).
References: Biderman et al. 2023 (Pythia); van der Wal et al. 2025 (PolyPythias); Wu,
Geiger & Milliere 2025 (variable binding in transformers); Gur-Arieh et al. 2026 (Mixing
mechanisms); Ravulapalli & Chadha 2026 (in-context binding capacity); Guo & Vosoughi 2025
(serial position effects of LLMs); Feng & Steinhardt 2024 (binding IDs).

## 7. Video (90 s) outline
1. (10 s) Hook: "Ask a 12-billion-parameter AI the same question twice, just reordered.
   Nearly half the time, it changes whether it's right."
2. (25 s) The task and the flip figure: first-chain vs last-chain habit swinging during
   training.
3. (20 s) Why it's real: new questions, no examples, nine seeds, five sizes.
4. (20 s) Inside the model: the answer is fixed halfway through; errors are choosing the
   wrong chain.
5. (15 s) So what: ask in both orders (+12 points); single-order tests mislead.
