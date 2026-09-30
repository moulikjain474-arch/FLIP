# EarlyMind: pre-registered analysis plan (v2, reframed)

Written on 30 Sept 2026, before the dense-grid, 6.9B and 12B runs and before any causal
test. Commit this file before those runs; the commit time is the record. Anything changed
later is listed under "Changes" with a date and a reason.

## 1. Question
When a pretrained language model learns to follow a variable chain, what does it do on the
way? And can we see competence coming before it arrives, either inside the model (probes)
or in its mistakes?

## 2. Data
- Task: vartrack depth 1/2/3, blocked layout, 2 worked examples, K = 10 candidates.
- Main grid (fixed earlier, used for H1): 13 checkpoints, 0 ... 143000.
- Dense grid (this plan): extra checkpoints around transitions, stored in `runs_dense`,
  used only for the stage and warning analyses in sections 3 to 5. The H1 analysis stays on
  the main grid.
- Models: Pythia 160M (+ PolyPythias seeds 1-4), 410M, 1.4B, 6.9B, 12B.

## 3. Answer types and events (stages.py)
Each prediction is one of: correct, distractor (the other chain's value), shot_copy (the
answer of a worked example), other.
- **Stage label** of a checkpoint = its most common answer type.
- **T_cor (competence)**: first checkpoint where correct is the most common answer AND
  >= 0.3, and the same holds at the next checkpoint.
- **B_acc (baseline alarm)**: first checkpoint where correct >= 0.2 (2x chance).
- **S_err (error signal)**: first checkpoint where distractor >= 0.2 AND distractor >= correct.
- **S_copy (copy-decline signal)**: first checkpoint where shot_copy has peaked at >= 0.4
  and has since fallen by >= 0.15.

## 4. Claims and what counts as support
- **C1, stages.** In models that reach T_cor, the stage order is copy, then distractor,
  then correct. Supported if this order holds in most runs that reach T_cor. Runs that skip
  a stage at checkpoint resolution are reported, not dropped.
- **C2, mistakes warn early.** Lead = doublings of tokens from signal to T_cor. A signal is
  useful only if it fires before B_acc. Supported if S_err (or S_copy) fires strictly
  earlier than B_acc in a majority of runs that reach T_cor on the dense grid. If it fires at
  the same checkpoint, C2 is not supported and the paper says so.
  - Known risk: on the main grid, S_err and B_acc fire at the same checkpoint for 1.4B and
    410M. The dense grid is the real test.
  - Known risk: the main grid doubles (1000, 2000, 4000 ...), so any "2x" gap between
    events could be produced by the spacing. Only dense-grid gaps are interpreted.
- **C3, prediction (revised 30 Sept, see Changes).** Written before 6.9B / 12B are run.
  For each of 6.9B and 12B, at every depth, on the 13-checkpoint grid:
  - P1: the dominant-answer sequence is O > S > D > C (a single D/C flip-back is allowed,
    as seen in 1.4B d1). A skipped stage counts as a miss.
  - P2: correct and distractor rise together: S_err and B_acc fire at the same checkpoint
    or one checkpoint apart.
  - P3: T_cor is no later than 1.4B's on the main grid (8000 steps).
  - P4: the distractor rate at step 143000 stays >= 0.15 (the model never fully resolves
    which chain; 1.4B ends at 0.22 / 0.33 / 0.40).
  Report every P for every depth and size as hit or miss.
- **C4, causal (steer_stage.py).** At an error-stage checkpoint, erasing the
  distractor-minus-correct subspace increases log p(correct) compared with the mean of 5
  random same-rank erasures (primary: paired t, held-out items). Secondary: accuracy,
  distractor rate. Controls reported: random, pca_top, erase_cor. The same test at the
  final checkpoint is the comparison.
- **H1 (kept from v1, reported as is).** Probe-behavior lead vs depth, main grid,
  thresholds unchanged. Current result: no lead at 410M and 1.4B; d2/d3 censored.
- **C5, position shortcut (position_test.py).** At the coin-flip stage (1.4B steps 5000,
  7000), wrong-chain answers are explained by position if "picks later chain" is >= 0.65
  or <= 0.35 AND correct rates differ by >= 0.2 between the two position groups. Same test
  at step 143000 and on 6.9B/12B/final modern models shows whether the shortcut persists.
- **C6, does scale fix it?** Distractor rate at step 143000 against model size
  (160M, 410M, 1.4B, 6.9B, 12B), per depth. Reported as a curve; either a shrinking or a
  flat floor is a finding. P4 above is the locked version.
- **C7, modern models.** Final checkpoints of current open models (revision main, no
  training history) on the same task and prompts. Reported as correct / distractor rates
  and the position test. Only models whose answer options are single tokens are usable
  (model_utils.candidate_ids stops otherwise); every model tried is listed, including ones
  that could not be run.

## 5. Statistics
- Holm correction across all C4 primary tests (every model x checkpoint x depth run).
- Every run is reported, including failures and nulls.
- 160M runs rarely reach T_cor; they count toward "signal fired without competence".

## 6. Claims we will NOT make
- "First" anything; "early-warning system" in general; "reasoning" (say "follows the
  correct chain", because the blocked layout may allow a partial shortcut).
- Anything beyond this task, this layout and Pythia.

## Changes
- 30 Sept 2026, after the dense 1.4B grid (steps 3000-12000) and before any 6.9B/12B run:
  - C2 result so far: on the dense 1.4B grid, S_err fires at the SAME checkpoint as the
    plain-accuracy baseline for d1 and d2 (step 3000) and one dense checkpoint earlier for
    d3. The wrong-chain answers rise together with the correct ones, not before them.
    C2 is not supported for 1.4B. 410M (main grid only) still to be checked on its dense grid.
  - C3 rewritten. The numeric rules in `monitor.py --predict` were dropped as the locked
    prediction: rule A extrapolates from only two model sizes and puts 6.9B/12B competence
    at step 0-1000, before the untrained model could plausibly do anything, and rule B's
    range (x2 to x16) is too wide to be falsified. The qualitative predictions P1-P4 replace
    them. `monitor.py --predict` output is exploratory only.
  - Added C5 (position shortcut), C6 (scale) and C7 (modern models), before any of them
    was run. Causal checkpoints for 1.4B moved to 5000 / 7000 / 143000, the coin-flip stage
    found on the dense grid.
  - Novelty check done on "In-Context Binding Capacity in Language Models" (Ravulapalli &
    Chadha, arXiv 2609.30634, 24 Sept 2026), read in full:
    - Already shown there, on frozen final models: binding capacity in the Pythia ladder is
      flat from 410M to 12B (k* = 3-6); modern recipes (Qwen2.5, OLMo-2) hold far more
      bindings; serial position predicts which trials fail. So C6 ("scale does not fix it"
      in Pythia) and C7 (modern models do better) extend known results and are reported as
      consistency checks, not headline findings. C5 is new only in its across-training form.
    - Not done there: behavior of binding errors across pretraining checkpoints (they only
      track declaration-site geometry across Pythia-1.4B checkpoints and formation times in
      small from-scratch models), multi-hop chains, and any activation intervention. They
      state that recall alone cannot tell retention from access and propose, but do not run,
      intervention tests (their Sections 7.1 and 9.5). C1 (stages across training) and C4
      (causal erasure across training) are the contributions.
    - Their Appendix A: read the query site, not the entity token. Our probes and edits use
      the last token (the query site), which is consistent with that advice.
    Earlier note kept for the record: Feng & Steinhardt (ICLR 2024) and "Mixing Mechanisms"
    (arXiv 2510.06182) study binding in trained models, not across training.
