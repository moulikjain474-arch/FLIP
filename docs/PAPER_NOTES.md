# Paper notes (8 Oct 2026). Text to drop into the paper/synopsis once results are in.
Placeholders in [BRACKETS] must be filled from the scored results. Nothing here states a result.

## One-sentence question
How do language models choose between competing variable chains, how does that choice change during
pretraining, where in the network is it decided, and what does it cost?

## Three findings (J3). Fill each from the scored results; if a registered claim missed, say so and reword.
1. The position preference [reverses / does not reverse] repeatedly during pretraining and [depends / does not depend] on the seed. (C8, C9)
2. It is decided [mid-network: layers ~[X]-[Y] of [L]] (plant C11 + order patching, exploratory). Attention [does / does not] track it (attn_vs_pref.py, exploratory).
3. It costs accuracy [beyond / not beyond] the chance flip rate and the rename control (C12), and combining both orders [recovers [Z]] (C13, C17); beyond the toy task: [story, code, MMLU, frontier].

## Seed results on Trott's two axes (F2)
- Developmental: does the trajectory replicate across the 9 runs of 410M? (C9)  [fill]
- Positional: is the commit layer the same across checkpoints / sizes? (commit_depth.py, exploratory) [fill]

## Discussion points (F5), each only if the data supports it
- Compare the commit layer with the layer-conductance finding that recency grows with depth [title needed].
- Contrast late reversals with early stabilisation of behaviour in Kishino et al.
- Alur et al.: why decoder-only models struggle to be order-invariant.

## Video hook (F4, describes our own data)
"Same facts, different order, different answer."

## Intro stakes line (F3) - see docs/CITATIONS.md wording rule. Do not say "same note, different diagnosis".

## Plant validity paragraph (N2)
Regime structure across layers; a matched random-subspace control; negative-control layers; replication gap (plant run on 6.9B only).

## Limitations to state
One training run per size above 410M; the fix doubles inference cost and needs log-probabilities with two orders;
synthetic task; size trend is descriptive, not a scaling law; layer results show where the output is fixed, not the circuit.
