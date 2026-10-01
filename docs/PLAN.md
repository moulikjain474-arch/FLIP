# Plan to the IRIS submission (target 12 Oct, deadline 13 Oct 2026)

Goal: qualify for ISEF through IRIS. What judges score, and where the project stands:

| Judges look for | Status | Still to do |
|---|---|---|
| A clear, answerable question | Done: how do models choose which variable was asked, across training? | Say it in one sentence everywhere |
| Sound design with controls | Strong: pre-registration, random-subspace controls, noise floor, no-example control, 9 seeds, 5 sizes, checkpoint checksums | Replicate on unseen checkpoints (C8-C11); controls and generalisation (C12-C17) |
| Careful execution and analysis | Good | Add confidence intervals to every figure |
| A surprising, memorable result | "12B changes correctness on nearly half the questions when reordered" | Lead with it |
| A consequence outside the lab | Asking in both orders: 79% to 91% at 12B; single-order tests mislead | One slide, one sentence |
| Honesty about limits and nulls | C4 null, early-warning null, 2.8B excluded | Keep them in the paper |
| The student understands it all | - | Interview practice (below) |

The pattern shared by strong interpretability entries: one clear question, one headline
number, controls that pre-empt the obvious objection, and a practical takeaway. The
remaining work is replication (one compute day) and communication, not new ideas.

## Timeline
| Dates | Work |
|---|---|
| 1-3 Oct | Commit EXPERIMENTS.md with C8-C17 registered BEFORE any new run. Register the project on the IRIS portal, pay, add the guide. Ask Exea for the full day. IRIS confirmed 13 Oct (1 Oct). |
| Full Exea day (as early as possible) | `smoke.sh` first, then `run_fullday.sh` (runbook: docs/EXEA.md). Then `score_new.py` and `export_csv.py`. |
| Same or next day | Export the CSV, make final figures, (P1-P4 are registered but no longer tested; see EXPERIMENTS.md Changes). |
| 4-7 Oct | Write the synopsis (docs/PAPER_DRAFT.md), fill every [TBD], rewrite anything a result contradicts. |
| 8-9 Oct | Figures final (3 in the main text: position flips, noise floor / seeds, order-swap; 1 mechanism figure). Video script and recording. |
| 10-11 Oct | Full read-through against IRIS rules: word limits, anonymity (no school, city, state, no repo link), references. Someone else reads it cold. |
| 12 Oct | Submit. 13 Oct is the buffer, not the plan. |

If the full compute day does not happen: submit with the 1 Oct results, and state that
C8-C11 are registered and pending. The story does not depend on them; they make it
stronger.

## If a registered claim misses
- C8 miss (12B does not flip): the flips are a feature of smaller models; the 12B result
  is then "one stable habit", and the order-swap result still stands.
- C9 miss (seeds agree): the habit is set by the data order, not by chance. Say so.
- C10 miss (order dependence small earlier in training): report where it appears.
- C11 miss: the halfway-commitment result is specific to 1.4B.
A miss changes a sentence, not the project.

## Interview questions to be ready for
0. Isn't this already known? (About 20 seconds:) "Four close papers. Ravulapalli & Chadha
   measured how MANY bindings finished models hold; I measure WHICH one a model picks when
   unsure and how that changes. Gur-Arieh et al. showed finished models retrieve mainly by
   position; I show how that positional preference forms and flips during pretraining.
   Wu, Geiger & Milliere found a position heuristic that appears once and is replaced in
   small models trained from scratch; in real pretrained models it reverses repeatedly and
   ends differently by seed. Guo & Vosoughi found primacy and recency effects whose prompt
   fixes were inconsistent; I measure a fix, asking in both orders, against a plain prompt
   instruction. The gap is the position preference across real pretraining checkpoints,
   sizes and seeds, where it is decided inside the model, and what it costs."
1. Isn't the flipping just noise? (Same checkpoint on new questions: SD 0.02; swings up
   to 0.31; five significant reversals.)
2. Maybe the model copies the worked examples? (With no examples the first-chain habit at
   the end of 1.4B training gets stronger: 90% vs 31%.)
3. Why Pythia? (Only suite with public checkpoints across training at many sizes and
   seeds, and the same data order for every size.)
4. Is writing into activations just steering the output token? (Near the output, yes;
   that is why the informative part is where it starts working: nothing before about
   layer 12, and the same at every training stage.)
5. What does this mean for real use? (Test models with the order of facts varied; a
   single accuracy number can hide answers that flip.)
6. What failed? (Erasing the wrong chain did not help; mistakes did not warn early; one
   model size had corrupted checkpoints.)
7. What would you do next? (Find the attention heads that choose the chain; test other
   binding tasks and model families; training interventions that remove the shortcut.)
