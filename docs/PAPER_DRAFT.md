# IRIS synopsis draft (EarlyMind, reframed)

Word limits from IRIS. [TBD] marks numbers that come from runs not done yet; fill them from
the outputs, and rewrite any sentence the results do not support. No school, city or state
anywhere (anonymity rule), and do not link the repo.

---

## Title
Knowing the Value but Not the Variable: How Language Models Learn, and Fail to Finish
Learning, Which Variable Was Asked About

## 1. Abstract (max 250 words)
Language models often mix up which value belongs to which variable, a failure that matters
for code assistants and long documents. Recent work measured this limit in finished models
but left open how it forms during training and what causes wrong choices. I tracked
Pythia language models (160 million to 12 billion parameters, up to 21 checkpoints each,
four independent training runs at the smallest size) on a variable-tracking task with two
competing chains of assignments, classifying every answer as correct, the other chain's
value, a copied example answer, or other. Models passed through the same stages: first
copying the worked example, then answering with a chain's value while choosing between the
two chains almost at random (1.4B: 0.41 correct vs 0.41 wrong chain at step 6000), then
slowly preferring the right chain. Even fully trained, the wrong chain was still chosen on
22-40% of questions. Neither internal probes nor the model's mistakes signalled competence
earlier than plain accuracy. To test the cause, I erased the directions encoding the wrong
chain's value from the model's activations at the random-choice stage: correct answers
[TBD: rose from X to Y / did not change], compared with [TBD] for random erasures of the
same size. [TBD: one sentence on the position test.] The ability to find a value develops
well before the ability to pick the right variable, and [TBD: the causal result] shows
[TBD]. These results give a training-time account of a binding failure that scale did not
remove in this model family.

## 2. Introduction and objective (100-150 words)
When a model reads code or a long document, it must track which value belongs to which
entity. Language models are known to fail at this "binding", and a 2026 study showed the
failure persists across model sizes in some model families, but it measured only finished
models and noted that behaviour alone cannot reveal whether a model has the answer but
retrieves the wrong one. Because the Pythia suite releases checkpoints from throughout
training, the development of this failure can be observed directly. My objective was to
(1) describe the stages a model passes through while learning to answer questions about
chained variables, (2) test whether any signal predicts competence before accuracy rises,
and (3) test causally whether wrong-chain answers depend on the wrong chain's value being
represented at the moment of answering.

## 3. Innovation (50-100 words)
Prior work studied binding in finished models (Feng & Steinhardt 2024; Ravulapalli &
Chadha 2026) or in small models trained from scratch (Wu et al. 2025). This project follows
binding errors across the whole pretraining of real models at five sizes and four seeds,
on multi-step chains, and runs an activation-erasure test at each learning stage, an
experiment the 2026 study proposed but did not perform. The analysis plan and predictions
for the larger models were fixed and timestamped before those runs.

## 4. Methodology (150-250 words)
Task: Python-style prompts with two chains of variable assignments (depth 1, 2 or 3 steps)
and two worked examples; the model must output the starting value of the queried chain,
chosen among 10 single-token numbers. 600 questions per task and checkpoint.
Models: Pythia 160M, 410M, 1.4B, 6.9B, 12B at 13 checkpoints from initialization to the end
of training; PolyPythia 160M seeds 1-4; a dense grid of 8 extra checkpoints for 1.4B around
the transition.
Answer types: each prediction was labelled correct, wrong-chain, copied-example or other;
stage and event definitions were written before the dense and large-model runs.
Probes: logistic probes on the last-token residual stream at the middle layer, with
shuffled-label and untrained-model controls; onset = first checkpoint whose lower 95%
confidence bound clears a fixed threshold twice in a row.
Causal test: at checkpoints in each stage, the subspace encoding the wrong chain's value
(class means on a training half, with the correct-value directions projected out) was set
to its mean at the middle and all later layers; effects on held-out questions were compared
with five random subspaces of equal rank and with principal-component erasure. Paired
t-tests, Holm-corrected.
Position test: questions split by whether the queried chain appears first or second.

## 5. Results and conclusions (100-150 words)
[Write after the runs. Structure:]
- Stages: copy -> random choice between chains -> slow preference for the right chain, in
  [TBD]/[TBD] model x depth runs; the final wrong-chain rate at 1.4B is 0.22 / 0.33 / 0.40
  (depth 1/2/3) and at 12B [TBD].
- No early warning: probes and error-type signals fired no earlier than accuracy.
- Causal: [TBD].
- Position: [TBD].
Conclusion: models learn to retrieve a value long before they learn to select the correct
variable, and this selection remains incomplete at the end of training.
Limitations: one synthetic task, one model family, a blocked layout that may allow partial
shortcuts.

## 6. Acknowledgement and references (50-100 words)
Guide: [name, role]. Compute: Google Colab; Exea Labs (AMD MI300X) [only if used].
References: Biderman et al. 2023 (Pythia); Feng & Steinhardt 2024 (binding in context);
Gur-Arieh, Geva & Geiger 2026 (Mixing mechanisms); Ravulapalli & Chadha 2026 (in-context
binding capacity); Wu, Geiger & Milliere 2025 (variable binding in transformers); Kim et al.
2024 (code pretraining and entity tracking).

## 7. Video (90 s) outline
1. (10 s) Hook: "Ask an AI which value x holds when y is also nearby. Watch it guess."
2. (25 s) The stage figure: copy -> coin flip -> slow fix, across sizes.
3. (25 s) The causal test, one before/after bar chart.
4. (15 s) What it means for code assistants and long documents.
5. (15 s) Limits and next step.
