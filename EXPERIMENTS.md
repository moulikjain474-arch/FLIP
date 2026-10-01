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
- **C4b, causal plant (steer_stage.py, added 30 Sept, see Changes).** At the same
  checkpoints, overwriting the distractor-value subspace with the class mean of a third
  number p (not the answer, not the distractor; chosen per item with a fixed seed) makes
  the model answer p more often than with no edit (primary: paired t on "answer = p",
  plant_dis vs natural, held-out items). Supported only if the rate under plant_dis is
  also above 1/K = 0.1 absolute. plant_cor (the same inside the answer-value subspace) is
  reported as secondary. A hit means the value the model outputs is READ from that
  subspace, not only correlated with it: the edit predicts which exact wrong answer appears.
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

## 4b. New claims, registered 1 Oct 2026 BEFORE the full-day runs (C8-C11)
Tonight's exploratory findings (see Changes, 1 Oct) become confirmatory only if they hold
on data not yet seen. Definitions: a checkpoint is LATER-biased if the lower end of the 95%
Wilson interval of picks_later is above 0.5, FIRST-biased if the upper end is below 0.5.
- **C8, the flip replicates at 12B.** On the 12B position trajectory (steps 1000, 2000,
  4000, 6000, 8000, 12000, 16000, 24000, 32000, 48000, 64000, 100000, 120000, 143000; only
  32000 and 143000 have been seen), at depth 2 or depth 3, the sequence of biased
  checkpoints changes direction at least twice (for example LATER, FIRST, LATER). Also
  reported for the unseen 6.9B steps.
- **C9, the seed decides.** For 410M seeds 1-9 (seed 4 excluded) plus the main run, at
  steps 16000 and 64000 (not yet seen), depth 3: at one of the two steps or both, at least
  2 runs are LATER-biased and at least 2 are FIRST-biased.
- **C10, order dependence persists in training.** Order-swap test on 6.9B and 12B at steps
  16000, 32000 and 64000 (not yet seen): the share of questions correct in exactly one
  order is >= 0.20 at depth 2 in every one of these runs. Secondary: for 12B at 32000 and
  64000, combining both orders raises depth-1 and depth-2 accuracy by >= 0.03.
- **C11, mid-network commitment generalises.** Single-layer plant (plant_cor, scope
  single) on 6.9B (32 layers) at steps 16000 and 143000, depths 1 and 3: planted answers
  stay below 0.10 at every tested layer up to 10, and reach at least 0.50 at some layer up
  to 22.
Every result is reported, hit or miss. A miss is written up as a miss.

## 4c. New claims, registered 1 Oct 2026 BEFORE the full-day runs (C12-C17)
Two objections to the 1 Oct findings: (a) "any small edit to a prompt changes some answers,
so reordering is not special", and (b) "one synthetic task, one model family". The claims
below test both. None of the runs behind them has been done: renaming has never been run,
the OLMo-2, interleaved-layout and benchmark runs are new. The 600 questions per depth are
the same item set (seed 0) as before; only the new versions of them are new.

Definitions. A question is asked in four versions with the same correct answer: orig,
swap (the two chains reversed), rename (every variable renamed to a new random name of the
same length, order unchanged), swap_rename. Flip rate F(A, B) = share of questions answered
correctly in exactly one of versions A and B. An ensemble sums the log-probabilities of its
versions. "Single-order accuracy" = the mean of the orig and swap accuracies (so neither
order is privileged). Every version is checked to keep the same answer before any model is
run (perturb_test.py).
- **C12, the order effect is not generic fragility.** Pythia-12B, depth 2, steps 32000,
  64000 and 143000: F(orig, swap) exceeds F(orig, rename) by at least 0.05 in at least 2 of
  the 3 steps. (6.9B is reported, not part of the claim.)
  A miss means renaming flips about as many answers as reordering: the finding is then
  "answers are fragile to small prompt edits", and the paper says that instead.
- **C13, order diversity is what helps.** Pythia-12B, step 143000, depths 1 and 2: the
  accuracy of the ensemble orig + swap exceeds that of the equal-cost ensemble orig + rename
  by at least 0.03 at both depths. (Steps 32000 and 64000 are reported.)
  A miss means the gain from asking twice is generic ensembling, not specific to order.
- **C14, not a layout artifact.** Interleaved layout, final checkpoints (step 143000) of
  Pythia-410M, 1.4B and 6.9B, depth 3: picks_later is biased (95% Wilson interval excludes
  0.5) in at least 2 of the 3 models.
- **C15, the test works on a real benchmark.** MMLU (1,000 random test questions, seed 0,
  0-shot), options shown in the original and the reversed order, base models OLMo-2-1124-7B,
  Qwen2.5-7B and Mistral-7B-v0.1 (or tiiuae/falcon-7b if Mistral cannot be downloaded; see
  Changes, 1 Oct ~20:15): F(orig, reversed) >= 0.10 for at least 2 of the 3 models,
  AND the 2-order ensemble beats the single-order average for all 3 (4 cyclic orders are
  reported, not part of the claim). This tests the TOOL on a real task; it is not the same
  mechanism as the variable-chain result, and option-order sensitivity is known from earlier work.
- **C16, a second model family.** OLMo-2-0425-1B and OLMo-2-1124-7B (final), depth 3: for at
  least one of them, F(orig, swap) >= 0.10 and the orig + swap ensemble beats the
  single-order accuracy by at least 0.02.
- **C17, the headline gain is not noise.** Pythia-12B, step 143000, depths 1 and 2: the
  paired-bootstrap 95% interval (10,000 resamples over questions) of [ensemble orig + swap
  correct] minus [mean of orig and swap correct] lies above 0 at both depths. The point
  estimates were seen on 1 Oct (+0.13 and +0.12); the interval needs per-question results
  that were not saved then.
Notes. Renamed variables can tokenize to a different number of tokens than the originals
(swap does not change the length much); the mean length change is reported with C12. All
thresholds are compared with a tolerance of 1e-9 so that a rate sitting exactly on a
threshold counts as meeting it.
Exploratory only (not claims): distance filler lines between the chains, the 4-version
ensemble, how rename and swap flips change across training, and which questions flip.
Every result is reported, hit or miss. A miss is written up as a miss.

## 4d. Claims registered 1 Oct 2026 (~14:30), still BEFORE any full-day run (C18-C20)
- **C18, the flip in a second, modern family during training.** OLMo-2-7B (allenai/OLMo-2-1124-7B)
  at the 10 stage-1 checkpoints nearest to 0.5%, 1%, 2%, 5%, 10%, 20%, 35%, 50%, 75% and 100% of
  pretraining (chosen by model_utils.nearest_revisions before any is run): at depth 2 or
  depth 3, the sequence of biased checkpoints (definition as in 4b) changes direction at
  least once. A miss means the reversals are specific to Pythia.
- **C19, scale does not remove order dependence.** OLMo-2-32B (allenai/OLMo-2-0325-32B) and
  Qwen2.5-72B, final models, depth 3: F(orig, swap) >= 0.05 for at least one of the two. Their
  MMLU option-order results are reported, not part of the claim. A miss means the largest
  open models tested no longer show the effect on this task, and the paper says so.
- **C20, the order effect on realistic code.** code_task.py builds short Python programs
  (2-4 small functions, 2-4 values, one printed; no negative numbers; answer on the line
  after "Output:", candidates "0"-"9") whose definitions can be reordered without
  changing the output (checked by running every program in both orders; every value prints a
  different digit, so the others act as distractors). Final models Pythia-1.4B, 6.9B, 12B,
  OLMo-2-7B, Qwen2.5-7B, OLMo-2-32B, Qwen2.5-72B; level 2 (3 functions, 3 values); 600
  programs, 2 worked examples. Among models whose single-order accuracy is >= 0.30 (3x
  chance), at least half have F(orig, reversed) >= 0.10, and at least 2 models are eligible.
  A miss means the order effect does not carry over to realistic code, and the paper says so.
- Exploratory, not claims: frontier_flip.py asks closed frontier models through their APIs
  (temperature 0) with 2, 4 or 6 chains in both orders and reports the flip rate.
Every result is reported, hit or miss.

## 4e. Claim registered 1 Oct 2026 (~20:40), before any frontier run (C21)
- **C21, shuffled facts hurt frontier models.** frontier_flip.py hard mode (levels hard1-hard4:
  4 to 10 chains of 4 to 10 hops, 0 to ~6,000 words of irrelevant filler; facts given in
  dependency order vs randomly shuffled; answer only, temperature 0; OpenAI reasoning models,
  which reject those settings, run at their default with a 4,000-token budget and are reported
  separately; replies that contain no digit count as wrong and their share is reported). For at least one closed
  frontier model tested, at some level, accuracy with shuffled facts is at least 0.05 lower than
  with facts in dependency order, with a paired-bootstrap 95% interval of the difference above 0.
  Scored by `python3 frontier_flip.py --score <results folder>`. Every model and level is
  reported. A miss means current frontier models are robust to fact order on this task.
  Design follows published failure modes: premise order (Chen et al., ICML 2024), more hops
  and distractors (same; RULER, COLM 2024), long irrelevant context (RULER).

## 5. Statistics
- Holm correction across all C4 primary tests (every model x checkpoint x depth run), and
  separately across all C4b primary tests.
- C5 also reports a two-sided binomial test of "picks later chain" vs 0.5 and a 95% Wilson
  interval; the C5 decision rule above is unchanged.
- C12-C17 are fixed thresholds, not p-value tests; the only interval is the C17 paired
  bootstrap (10,000 resamples over questions). Headline gains use the single-order average
  as baseline, not the "original" order.
- Every run is reported, including failures and nulls.
- 160M runs rarely reach T_cor; they count toward "signal fired without competence".

## 6. Claims we will NOT make
- "First" anything; "early-warning system" in general; "reasoning" (say "follows the
  correct chain", because the blocked layout may allow a partial shortcut).
- That FLIP is the first to study variable binding across training: Wu, Geiger & Milliere
  (2025) follow it in small models trained from scratch, and Ravulapalli & Chadha (2026)
  report formation times. The claim is narrower: the position preference, followed across
  real pretraining checkpoints, model sizes and random seeds.
- "Primacy" and "recency" are used as descriptive names for the first-chain and last-chain
  preference only, not as a claim about human-like memory.
- Anything beyond this task, this layout and the models tested.
- That position habits are "like human memory": the serial-position comparison is an analogy
  only.
- That combining both orders is a general fix: it is shown only on this task.

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
- 30 Sept 2026 (later the same day), code review, before any causal run:
  - Added C4b (plant test) to steer_stage.py. It is the intervention Ravulapalli & Chadha
    propose but do not run, and it predicts which exact answer changes, which a plain
    erasure cannot.
  - stages.py now scores P1-P4 automatically (columns P1_order ... P4_final_dis_ge_015) and
    reports a coin-flip window (correct and wrong-chain both >= 0.3, within 0.05). P1-P4
    thresholds are unchanged.
  - Bug fixes that change no past result: monitor.py treated a signal at step 0 (0 tokens)
    as "never fired"; steer.py and steer_stage.py ignored --layer 0 (never used); steer.py's
    default layout changed to blocked (every run so far passed --layout blocked).
- 1 Oct 2026, after a code check on CPU (Pythia-410M step 143000, depth 1, n=200, i.e. 100
  test items; a code check, not one of the registered runs, reported here because it was
  seen before them):
  - Sanity checks passed: erase_cor lowered accuracy (0.67 -> 0.43); plant_cor made the
    model answer the planted value on 63% of items (0% unedited).
  - Seen: erase_dis_orth LOWERED log p(correct) vs random (-0.11, p < 0.001), the opposite
    of C4's direction; plant_dis gave 21% planted answers (0% unedited); picks_later = 0.68
    (95% CI 0.61-0.74) with a 0.33 correct-rate gap, i.e. C5's shortcut criterion is met.
  - Added plant_dis_orth as a SECONDARY condition: the raw distractor subspace overlaps the
    answer subspace (erase_dis hurt accuracy as much as erase_cor), so plant_dis could work
    through that overlap. plant_dis_orth plants only in the part that excludes the answer
    directions. C4 and C4b primary tests and thresholds are unchanged.
- 1 Oct 2026, Exea test session (AMD MI300X, about one hour). All numbers are in
  results/exea_2026-10-01/results.csv; figures from make_figs.py in the same folder.
  - REGISTERED TESTS (decision rules as written above, unchanged):
    - C4 (erase) at the 1.4B coin-flip checkpoints (steps 5000, 7000; depths 1-3; 300 test
      items each): log p(correct) under erase_dis_orth vs the random mean went +0.048,
      +0.006, -0.040, +0.038, +0.016, -0.006 (smallest p = 0.011). None survives Holm
      across the six. C4 NOT SUPPORTED. At step 143000 the effect is negative (-0.02, -0.06,
      -0.01).
    - C4b (plant) at the same six checkpoints: planted-value answers under plant_dis were
      0.48-0.79 vs 0.01-0.04 unedited, every p far below 0.001 and every rate above 0.1.
      C4b SUPPORTED. Secondary: plant_dis_orth gave only 0.02-0.08 and plant_cor about as
      much as plant_dis. Read together, the edited subspace holds the value the model is
      about to output, shared across both chains, not a separate wrong-chain slot. The
      paper states C4b in that weaker form.
    - C5 (position) at the coin-flip checkpoints: picks_later 0.42-0.61 with correct-rate
      gaps below 0.2, so NOT MET there. At step 143000: met for 1.4B depth 3 (0.34, gap
      0.31) and 410M (0.66-0.69, gaps 0.31-0.33); not met for 6.9B (0.56-0.59, gaps 0.15).
    - P1-P4 are not scored yet: they need full 6.9B / 12B sweeps.
  - EXPLORATORY ANALYSES, all designed after seeing the registered results above and
    reported as exploratory:
    - Position preference (picks_later) at every 1000-4000 steps of 1.4B, and across
      training for 410M, 1.4B-deduped, 6.9B and two 12B checkpoints. Main observation: the
      preference swings between later-chain and first-chain repeatedly during training.
    - Noise floor: 1.4B steps 5000/6000/7000 re-run on two new item sets (item seeds 1, 2).
      Within-checkpoint SD 0.023; adjacent checkpoints differ by up to 0.31, and 61% of
      adjacent pairs at depths 2-3 differ by more than 2 standard errors.
    - No worked examples (shots 0) at 1.4B steps 7000, 32000, 143000.
    - PolyPythia 410M seeds 1-9 at steps 32000 and 143000 (plus 8000 for seeds 1-4). The
      depth-3 final preference ranges 0.31-0.69 across runs. Seed 4 at step 143000 is at
      chance (0.10-0.17 correct) and is excluded as a suspected bad checkpoint.
    - Single-layer plant (scope single) at layers 4-20, 1.4B steps 3000, 5000, 10000,
      32000, 143000: planting is ignored at layers up to 10 and controls the answer from
      layer 12-14 on, at every step.
    - Order-swap test (swap_test.py, new): each question asked with the two chains in both
      orders. The share of questions correct in exactly one order is 0.12-0.56 across
      models and stages (12B: 0.41-0.49). Combining both orders raises 12B accuracy from
      0.79 to 0.91 (depth 1).
  - DATA INTEGRITY: the public EleutherAI/pythia-2.8b intermediate checkpoints load
    identical weights (same fingerprint at steps 4000 and 143000; the same safetensors file
    at steps 4000 and 32000; a .bin shared with pythia-2.8b-deduped). All 2.8B results are
    excluded. The other models give different results at different steps.
  - CODE: model_utils.load falls back to "<|endoftext|>" as the pad token when a tokenizer
    has neither a pad nor an eos token (pythia-1.4b-deduped and pythia-2.8b checkpoints).
    This changes nothing for the models that already loaded.
    Earlier note kept for the record: Feng & Steinhardt (ICLR 2024) and "Mixing Mechanisms"
    (arXiv 2510.06182) study binding in trained models, not across training.
- 1 Oct 2026, afternoon, before any run below:
  - Registered C12-C17 (section 4c). Added perturb.py (swap and rename edits, shared with
    swap_test.py: checked identical to the committed swap function on 2,000 prompts),
    perturb_test.py, benchmark_flip.py, flipstats.py, export_csv.py, smoke.sh, test_flip.py.
  - swap_test.py gains --revision (for models without step checkpoints). The registered C10
    runs are unaffected.
  - The headline order-swap gain is now stated against the average of the two orders: 12B
    depth 1 goes 0.776 to 0.905 and depth 2 0.740 to 0.858 (the earlier 0.79 to 0.91 and
    0.75 to 0.86 used the "original" order as baseline).
  - A fixed-position rule (always put the asked-about chain in the better position) was
    checked on the 1 Oct data and gains almost nothing: the two orders have nearly equal
    accuracy; the instability is per question. Not pursued.
  - Later the same afternoon, still before any run: score_new.py compares thresholds with a
    1e-9 tolerance (an independent review found that float rounding could score an exact-
    threshold result as a miss); perturb.py fails with clearer errors; perturb_test.py also
    reports token-length changes and warns if the computed answer differs from the item
    label; C15 is scored only on its three registered models.
- 1 Oct 2026, ~14:00, after the C8-C17 push and before any run: added three EXPLORATORY
  analyses (no claims, no thresholds; reported as exploratory whatever they show):
  - distance_test.py: no-op lines ('>>> pass') between the chains or before the question,
    to see whether the first-or-last habit follows distance to the question.
  - patch_test.py: activation patching of the last-token state from the swapped-order run
    into the original run, one layer at a time, to see at which layer the order-dependent
    choice is present.
  - story_test.py + story_task.py: the same two-chain structure told as a short English
    story ("Mia has as many coins as Ravi"), both orders, as a second binding task. Its
    items come from their own seed; the vartrack items in tasks.py are unchanged.
  - perturb.py renames variables using the same letters as tasks.py (only the names change).
    Checked on all 1,800 real blocked items: every edit keeps the label.

- 1 Oct 2026, ~14:30: registered C18-C19 (section 4d) and added frontier_flip.py
  (exploratory; it also asks the code programs). Registered C20 (code programs, code_task.py,
  code_test.py). position_test.py now runs non-Pythia models in bf16 (Pythia unchanged, fp16).
- 1 Oct 2026, same afternoon, before any run (independent review): code programs end the
  prompt with "Output:" plus a newline and use bare-digit candidates, so the answer is one
  token for Qwen and OLMo tokenizers too; "twice" adds instead of subtracts (no negative
  numbers); frontier_flip.py saves after every setting, keeps partial results on API
  errors, uses 6x3 (not 6x4, which needed more names than exist), and sends the
  reasoning-model request format when needed. smoke.sh also runs the code test on Qwen.
- 1 Oct 2026, ~19:45, before any run: added attention_test.py (EXPLORATORY, no claim):
  attention from the last token to each chain's tokens, per layer and head, and how it
  differs between questions answered with the later vs the earlier chain's value. Run on
  Pythia-1.4B at steps 7000 and 143000, Pythia-12B at 143000, and OLMo-2-7B at the
  stage-1 checkpoints nearest 5% and 100% of training (part 5h).
- 1 Oct 2026, ~19:50, before any run: part 5i (EXPLORATORY, no claim) runs the other
  tests on OLMo-2-7B as well: order-swap and rename at the same 10 training checkpoints as
  C18 (written to perturb_olmo_traj/, kept apart from the C16 scoring), activation patching at
  the 5% checkpoint and the final model, the distance control, the story task and the
  interleaved layout on the final model; plus rename and story on Qwen2.5-7B. The story task
  now asks "Q: How many coins does X have? A:" with the digit on the next line, so the answer
  is one token for OLMo-2 and Qwen too (found in an audit: the earlier format would have
  crashed on them). Nothing had been run with the old format.
- 1 Oct 2026, ~20:15, still before any run (two independent full reviews):
  - C15 fallback, fixed now before any data: if mistralai/Mistral-7B-v0.1 cannot be
    downloaded on the day (it may require accepting terms with a login), tiiuae/falcon-7b
    takes its place as the third model. The smoke test reports which applies.
  - score_new.py: each claim is scored separately so one missing or broken result cannot
    hide the others; C11 now requires every tested layer up to 10 (4, 8, 10) to be present
    and below 0.10, as the claim says; C20 counts only its 7 registered models; missing
    checkpoints, seeds and models are printed as MISSING.
  - attention_test.py runs in fp32 and refuses non-finite values (fp16 eager attention can
    overflow). code_task.py: "bump" multiplies by 3 instead of 2 so answers are not skewed
    towards even digits. Each OLMo-2-7B checkpoint is downloaded once (part 5e runs the
    exploratory OLMo checks in the same pass). model_utils.load loads straight onto the GPU
    (patch_model_utils.py) so a 72B model does not need ~145 GB of host RAM first.
- 1 Oct 2026, ~20:20: frontier_flip.py (exploratory) gets two harder levels, 8 chains of
  depth 4 and 10 chains of depth 5 (10 is the maximum: each chain needs its own digit), as a
  difficulty ladder. Every level is reported.
- 1 Oct 2026, ~20:40, before any run: registered C21 (section 4e). frontier_flip.py: the easy
  2x3/4x3 chains and code programs stay; the 6x3/8x4/10x5 levels are replaced by hard mode.
  SCOPE: the 6.9B/12B main-grid sweeps that would score P1-P4 (section 4, C3) are no longer
  run: they test the earlier stage framing, not the position-shortcut question, and cost about
  26 large model loads. P1-P4 are reported as "registered, not tested".
- 1 Oct 2026, ~20:45, before any run (exploratory, no claim): perturb_test.py --instruct also
  asks every question with a plain instruction to attend to the asked variable, in both
  orders, as a prompt-based baseline against asking in both orders (the kind of prompt fix
  Guo & Vosoughi 2025 found inconsistent). Used in parts 2, 3 and 5f. size_fit.py fits
  flip rate against parameter count on the final models after the run (descriptive; printed
  pooled and per family, since families differ in data and recipe as well as size).
