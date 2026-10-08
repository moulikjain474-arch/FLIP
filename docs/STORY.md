# FLIP: the story (9 Oct 2026)

Working notes for the synopsis, video and interview. Numbers marked [after run] are filled in from the 9 Oct results.

## Thesis
Language models often answer by where a fact sits, not what it says.

## Three findings
1. **Position, not noise.** Swapping the order of two facts changes the answer far more often than renaming them
   (OLMo-2-7B-Instruct, depth 3: 70% vs 20% of answers change). Most order errors pick the same slot in both orders.
2. **It forms during training.** Across Pythia and OLMo checkpoints the shortcut appears early, reverses direction,
   and is still present in the final models (figure: position_over_training.png).
3. **A fix that beats an equal-cost control.** Asking in both orders: 60% -> 84% correct. Asking twice with renamed
   variables (same cost): 60%. Exact McNemar p = 2e-21.

Everything else: "N predictions registered before running; X hit, Y missed" (one table, appendix).

## What is new
Prior work shows that premise order (Chen et al., ICML 2024) and answer-option order (Pezeshkpour & Hruschka, 2023)
change answers of finished models. We show when the shortcut forms during training, that it is driven by position
rather than random instability, where it is carried inside the network (causal intervention with a random-direction
control, C24), and a fix that beats an equal-cost control.

## Answers to the obvious attacks
- **Toy task / 72B is perfect.** A controlled task is what makes exact measurement possible. Harder depths (C27),
  medication-record wording (C26), and real benchmarks: Qwen2.5-72B still changes 10% of MMLU answers when the
  options are reordered. [after run: C26, C27]
- **Just asking twice.** The equal-cost control (rename and ask twice) does not help; reordering does.
- **Order errors vs knowledge errors (exploratory, post hoc).** On the controlled task, order errors are nearly all
  of a strong model's errors (the two-order check catches 100% of Qwen2.5-7B-Instruct's depth-3 errors while
  flagging 18% of questions). On MMLU they are a minority (at least 21-34% of errors). This is why the fix adds
  about +23 points on the controlled task but about +2 on MMLU. The flag says "check this", not which answer is right.
- **Chain-of-thought.** C28 tests generated answers with step-by-step reasoning, including
  DeepSeek-R1-Distill-Llama-8B. [after run]
- **Frontier models.** C29 (frontier_med.py) if API access is available. [after run]

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
