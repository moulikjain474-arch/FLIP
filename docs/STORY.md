# FLIP: the story (9 Oct 2026)

Working notes for the synopsis, video and interview. Numbers marked [after run] are filled in from the 9 Oct results.

## Thesis
Language models often answer by where a fact sits, not what it says.

## Three findings
1. **Position, not noise.** Swapping the order of two facts changes the answer far more often than renaming them
   (OLMo-2-7B-Instruct, depth 3: 70% vs 20% of answers change). Most order errors pick the same slot in both orders.
2. **It forms during training.** Across Pythia and OLMo checkpoints the shortcut appears early, reverses direction,
   and is still present in the final models (figure: position_over_training.png). C30 asks when the model first
   reads which chain the question points at, versus when a small learned subspace first controls the answer
   (Pythia-6.9B, six checkpoints).
   Result: the registered C30 is a MISS (the chain probe was already at 1.000 at the first registered checkpoint, step 4000).
   Early checkpoints added afterwards (exploratory) show readable before used on 6.9B: the probe is at 0.97 or more by
   step 2000 while planting in the learned subspace moves the answer by at most +0.03, and by +0.27 (d1) at step 4000.
   The same ladder on Pythia-1.4B is a registered MISS, but it only ran from step 4000 (probe already ~1.0), so it cannot separate readable from used. One training run per size, so this is not a seed result.
3. **A fix that beats an equal-cost control.** Asking in both orders: 60% -> 84% correct. Asking twice with renamed
   variables (same cost): 60%. Exact McNemar p = 2e-21.

Everything else: "N predictions registered before running; X hit, Y missed" (one table, appendix).

## What is new
Prior work shows that premise order (Chen et al., ICML 2024) and answer-option order (Pezeshkpour & Hruschka, 2023)
change answers of finished models. We show when the shortcut forms during training, that it is driven by position
rather than random instability, that a small learned subspace is a causal handle on the answer (erase or overwrite it
and the answer changes; random subspaces of the same size do not do this, C24; A1 repeats the check with the random
subspaces given the same layer search: PASS on Pythia-6.9B step 143000, learned margin over the largest of 5 random draws +0.46 (depth 1) and +0.82 (depth 3) at layer 26), and a fix that beats an equal-cost control. We do not claim to have
located where the computation happens.

## Answers to the obvious attacks
- **Toy task / 72B is perfect.** A controlled task is what makes exact measurement possible. Harder depths (C27),
  medication-record wording (C26), and real benchmarks: Qwen2.5-72B still changes 10% of MMLU answers when the
  options are reordered. C26 (medication wording, depth 3) is a MISS: 3 of 7 models, because four are at the ceiling
  (accuracy 1.000 or 0.998). C27 (harder depths) is a HIT: Qwen2.5-72B-Instruct at depth 9 flips 8.0% of answers and
  both orders gain +0.037 [+0.026, +0.048].
- **Just asking twice.** The equal-cost control (rename and ask twice) does not help; reordering does.
- **Order errors vs knowledge errors (exploratory, post hoc).** On the controlled task, order errors are nearly all
  of a strong model's errors (the two-order check catches 100% of Qwen2.5-7B-Instruct's depth-3 errors while
  flagging 18% of questions). On MMLU they are a minority (at least 21-34% of errors). This is why the fix adds
  about +23 points on the controlled task but about +2 on MMLU. The flag says "check this", not which answer is right.
- **Chain-of-thought.** C28 tests generated answers with step-by-step reasoning, including
  DeepSeek-R1-Distill-Llama-8B. Result: C28 HIT, 3 of 4 models supported (per-model rows in
  results/final_9oct/score_union_final.txt).
- **Frontier models.** C21 and C29 were NOT RUN (they need the author's own API key). No frontier-model result is claimed.

## How far each claim goes (evidence ladder)
| Rung | Question | Evidence | Where it stops |
|---|---|---|---|
| 1. The behaviour | Does the answer change when only the order of facts changes? | Flip rate with exact McNemar test; rename control at equal cost | Synthetic and benchmark tasks, not real documents |
| 2. Readable | Does the network represent which chain the question points at? | Cross-validated linear probe by layer and checkpoint (C30) | A probe finds information; it does not show the model uses it |
| 3. Causal handle | Does a small learned subspace control the answer? | Erase it, or overwrite it with another value (C24); random subspaces of the same rank do nothing (A1 gives them the same layer search) | Steering shows control, not location; whole-state and attention-vs-MLP patching narrow where, per-head patching not done |
| 4. Specific | Is it order, not digits or padding? | Flip rate by digit (A2); padding at start, before the first chain, between chains (A3) | Pythia and two instruct models; one task family |

## Pre-registration
Predictions C24-C29 were committed to the repository before the 12-hour run (the latest registration commit is 8fbc5c6);
C30 and audits A1-A3 were registered at about 13:40 IST on 9 Oct while the run was in progress (EXPERIMENTS.md,
section 4m). A1 was defined after Part R had written its files and before any of them was scored. No threshold was
changed after a result was seen. Every claim is reported, hit or miss.

## What did not work (say it before they ask)
- EarlyMind's original idea, a sudden early phase of variable binding, was not supported by the data. The seed test (C9) also missed.
- The confidence gate (C22) was supported in 1 of 2 models, so it stays a miss.
- The "error check" on MMLU is a lower bound: only 21-34% of MMLU errors are order errors, and the flag says "check",
  not which answer is right. It is exploratory and lives in the appendix.
- Mistral-7B-v0.1 did not complete the medication-wording run on 9 Oct; it is not one of the C26 models.
- C9: nine 410M runs mostly agree (step 16000: 1 later, 5 first; step 64000: 1 later, 4 first), so a seed-dependent
  direction is not supported. The direction is not universal across size, family and depth, so it is reported per
  condition, never as "models pick the later chain".
- C25 (ARC, 6 of 10): the four Qwen2.5 models that miss are accurate on ARC and the gain interval includes 0.
- C26 (3 of 7): four models are at the ceiling at depth 3; C27 repeats the test at depth 9, where it is a HIT.
- C30 (registered ladder): the probe was already at 1.000 at the first registered step. The early-step extension is
  exploratory, agrees on 6.9B and disagrees on 1.4B.
- Not run: C21, C29 (API key), the SAE pilot (Gemma access), a usable value probe (random-init baseline).

## Limits we state
- Synthetic tasks make exact measurement possible; they do not show the same rate in real documents.
- Five random subspaces per test: the smallest exact p is 1/6, so effect size, not p, is the evidence there (A1).
- Planting works in a subspace we found by fitting; it shows a causal handle, not where the mechanism sits.
- The checkpoint ladder has six points, so "earlier" means earlier among those six.

## Iteration
The project began as EarlyMind: testing whether models learn variable binding in a sudden early phase. The data did
not support that; the anomaly that kept appearing, answers changing with the order of facts, became FLIP.

## Interview prep
1. What did you do vs tools/AI? Answer specifically and truthfully (own decisions: question, models, controls,
   registered thresholds, interpretation; tool help: code writing/debugging, drafting).
2. Why Pythia/OLMo? Public training checkpoints and seeds, so the shortcut can be watched as it forms.
3. Why care? Same facts, different order, different answer: a silent error in records, contracts, code. The two-pass
   check is free to deploy.
4. What failed? C9 (seeds), C22 (confidence gate), and any 9 Oct misses; what each taught us.
5. More compute? Frontier models, real documents (with ethics approval), training with order-shuffled data.
6. Is it position or something else? Digits (A2) and padding (A3) are the two checks; say what each found.
7. How do you know the subspace is special? Random subspaces of the same size do much less (C24); A1 gives them the
   same layer search: PASS on Pythia-6.9B step 143000, learned margin over the largest of 5 random draws +0.46 (depth 1) and +0.82 (depth 3) at layer 26. Say plainly that this shows a causal handle, not where the computation is.
8. Why Pythia checkpoints? You can watch readable-versus-used form during training (C30).
