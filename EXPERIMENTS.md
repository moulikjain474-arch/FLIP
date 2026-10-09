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

- **Amendment to C21, 8 Oct 2026, before any frontier run.** The threshold and the test are unchanged.
  frontier_flip.py now also re-asks the first 20 questions of each setting a second time in the
  identical order (`--repeat 20`, default) and reports the same-order disagreement rate next to every
  C21 line, because API models are not deterministic even at temperature 0 and OpenAI reasoning models
  take no temperature setting. A shuffle drop is read against this floor. Open-model runs need no repeat
  (log-probability scoring is deterministic).

## 4f. Claims registered 8 Oct 2026 (~22:30 IST), AFTER the main results were seen, BEFORE these two tests were run (C22, C23)

Disclosure: the main results (C8-C18) had been scored when these were written, and the C22 questions are
the same seed-0 questions as the swap tests. What did NOT exist when the thresholds below were fixed:
any per-question confidence split of the order effect (C22), and any run on a fresh question set (C23).
Thresholds are fixed here and are not changed after the run. Each claim is scored by `score_new.py`
(`score_c22`, `score_c23`). Commit this file before running the tests, so the dated commit shows the order.

- **C22, the order effect concentrates where the model is unsure.** `uncertainty_gate.py`; Pythia-6.9B and 12B,
  step 143000, depth 2, 600 questions (seed 0), asked in the original and the swapped order. Confidence of a
  question is the log-probability gap between the model's best and second-best answer in the ORIGINAL order.
  Questions are cut into three equal groups by confidence. A question flips if it is answered correctly in
  exactly one of the two orders. Support for a model: the flip rate of the least-confident third is at least
  twice that of the most-confident third AND an exact two-sided Fisher test on those two thirds gives p < 0.05.
  C22 holds if both models are supported. A miss means the order effect is not concentrated in unsure
  questions, and the wording "when unsure" is removed from every text. Descriptive only (not part of the
  rule): the same split by swapped-order margin and by the smaller of the two margins, the rank correlation
  of margin with flipping, and accuracy per third (the confident third should be more accurate).
- **C23, the order effect is not a quirk of one question set.** `perturb_test.py --seed 1`, a freshly drawn
  question set (and fresh renamings), same two models, step 143000, depth 2, 600 questions. Support for a
  model: flip rate between original and swapped order >= 0.20, flip(original, swapped) - flip(original,
  renamed) >= 0.05, and the 95% paired-bootstrap interval of the both-orders accuracy gain over the
  single-order average lies above 0. C23 holds if both models are supported (every fresh seed that was run
  must be supported). A miss means the seed-0 numbers do not generalise to new questions.
- If a model cannot be run in time, that claim is reported as NOT RUN for that model, never as a hit.
  Nothing here changes C1-C21. Both are labelled exploratory-to-confirmatory: they were written after the main
  run, so they are prospective tests of new predictions, not part of the original pre-registration.

## 4g. Outcomes recorded 8 Oct 2026 (~23:40 IST)
Scored by `score_new.py` on the merged results (`results_08Oct/merged`); full tables in `results_08Oct/*.txt`.
- C8, C10-C20 and C23: HIT as scored (C12: 3 of 3 primary cells; C14: 2 of 3 biased; C15: 3 models; C19, C20 as listed).
  C9: MISS. C22: MISS (1 of 2 models).
- **C9 detail.** Depth 3, nine 410M runs: step 16,000 = F-LF--FFF (L=1, F=5); step 64,000 = --FFL--FF (L=1, F=4).
  The rule needed at least 2 L and 2 F at one step. The runs mostly agree in direction, so the habit is not
  shown to depend on the seed. Every text that said otherwise is reworded.
- **C22 detail.** Pythia-6.9B supported at every depth; Pythia-12B not supported (flip rates by confidence third
  at depth 2: 0.51 / 0.42 / 0.44, ratio 1.16, Fisher p = 0.19). The claim needed both models.
- Exploratory, run after the results were seen, and not registered: `position_decomposition.py` (recency),
  `order_specific_edge.py` (both orders vs original + renamed, exact McNemar, 33 cells; only cells with
  p < 0.0015 survive a Bonferroni correction for 33 tests), the MMLU runs at n=300 for Qwen2.5-14B and 32B, and
  the M steering readout by layer. None of them changes a registered claim.

## 4h. New claims, registered 9 Oct 2026 BEFORE the 12-hour run (C24, C25)

Thresholds are fixed now and will not change after seeing data. Anything else the run produces is labelled exploratory.

- **C24 (random-subspace plant control).** Planting the class mean of the exact third number into the learned
  (distractor) subspace makes the model answer that number. Control: plant the same values inside random subspaces of
  the same rank (5 draws, same layer, same items). HIT if, for Pythia-6.9B step 143000 at layer 22, at depth 1 AND
  depth 3, (plant_dis - plant_rand) >= 0.15 and paired p < 0.05. Otherwise MISS. Other layers and 12B are descriptive.
  Script: plant_control_report.py (patch_plant_rand.py adds --plant-rand to steer_stage.py).
- **C25 (ARC-Challenge option order).** Same test as MMLU: answers change when options are reordered, and the
  two-order ensemble helps. HIT if in at least 7 of the 10 models the answer-change rate (orig vs reversed) is >= 0.03
  AND the two-order ensemble gain has a 95% CI entirely above 0. Otherwise MISS.
- **Exploratory only (no threshold):** 410M seeds at steps 4000, 8000, 32000, 143000 (follow-up to the C9 miss),
  attention and numerical-noise checks, position decomposition and perturbation for the larger models.

## 4i. Claim registered 9 Oct 2026 (~02:00 IST) BEFORE the 12-hour run (C26)

- **C26 (medication-record wording).** The same two-chain task, worded as a medication record
  ("Patient A takes 4 mg of warfarin. Patient B takes the same warfarin dose as patient A."; both chains use the same
  drug; doses 1-9 mg). Swapping the order of the two records must not change the answer. HIT if, at depth 3, in at
  least 4 of these 7 models the flip rate is >= 0.05 AND the both-orders ensemble gain has a 95% CI entirely above 0:
  Qwen2.5-7B-Instruct, OLMo-2-1124-7B-Instruct, Qwen2.5-14B, OLMo-2-1124-13B, Qwen2.5-32B, OLMo-2-0325-32B,
  Qwen2.5-72B. A model not run counts as not supported. Otherwise MISS. Pythia 6.9B/12B and Qwen2.5-72B-Instruct
  are descriptive. Script: story_test.py --theme med; scored by score_new.py (score_c26).
- This is synthetic text in a clinical format, not real patient data, and says nothing about any deployed clinical
  system. Prior work already shows answer-OPTION shuffling changes medical QA answers (e.g. MedGemma on MedMCQA);
  C26 is about the order of FACTS in the context, which those papers do not test.

## 4j. Claim registered 9 Oct 2026 (~02:15 IST) BEFORE the 12-hour run (C27)

- **C27 (is the shortcut only a small-model problem?).** Qwen2.5-72B is ~100% correct on depths 1-3, so those depths
  cannot show whether the largest models still rely on position. The medication wording is also run at depths 6 and 9
  for every model in part M. HIT if Qwen2.5-72B or Qwen2.5-72B-Instruct has flip >= 0.05 AND both-orders gain 95% CI
  entirely above 0 at depth 6 or depth 9. Otherwise MISS. The size trend across the other models is descriptive.
  Scored by score_new.py (score_c27).

## 4k. Claim registered 9 Oct 2026 (~02:30 IST) BEFORE the 12-hour run (C28)

- **C28 (does writing out reasoning remove the order effect?).** All other tests score the log-probability of the
  answer token. Here the model GENERATES its answer (greedy, chat template when available) to the medication-record
  task, zero-shot, depths 3 and 6, 150 questions each, both record orders; mode "direct" (answer only) and mode "cot"
  ("think step by step, then 'Answer: N mg'"). The final dose is parsed from the text; raw outputs are saved.
  HIT if, in the cot mode, the flip rate is >= 0.05 at depth 3 or depth 6 with fewer than 20% unparsed answers,
  in at least 2 of these 4 models: Qwen2.5-7B-Instruct, OLMo-2-1124-7B-Instruct, Qwen2.5-72B-Instruct,
  DeepSeek-R1-Distill-Llama-8B (cot only, up to 1024 tokens). A model not run counts as not supported.
  Otherwise MISS. Llama-3.1-70B-Instruct (if access is granted) and the direct mode are descriptive.
  Script: gen_test.py; scored by score_new.py (score_c28). The parser is fixed now; if hand-checking shows it
  misreads answers, any re-parse is reported as a correction alongside the registered result, not instead of it.

## 4l. Claim registered 9 Oct 2026 (~02:00 IST) BEFORE any frontier run (C29)

- **C29 (frontier model, medication wording).** frontier_med.py asks one closed frontier model through its API the
  same 100 medication-record questions per depth (3 and 6) as gen_test.py, in both record orders, in the direct and
  the step-by-step mode, temperature 0 where the API allows it. HIT if, for at least one frontier model, the flip
  rate is >= 0.03 at depth 3 or 6 in either mode with fewer than 20% unparsed answers. Otherwise MISS. Which model
  is run depends on API access; the model name is recorded in the result file. Every reply is saved.

## 4m. Registered 9 Oct 2026 (~13:40 IST), added to the queue while the 12-hour run was in progress (C30, audits A1-A3)

Prompted by reviewer-style questions from a mechanistic-interpretability mentor (absolute position vs fact order;
random-direction controls; where the mechanism sits) and by practices in recent interpretability-for-discovery papers
(selection-matched nulls; separating readable from causally used). The checks below were fixed before the runs they
describe; where the data already existed, that is stated.

- **C30 (checkpoint ladder).** Pythia-6.9B, layer 22 (the C24 layer), depths 1 and 3, checkpoints 4000, 8000, 16000,
  32000, 64000, 143000. A checkpoint is READABLE if the cross-validated chain-identity probe (best layer;
  chain_probe.py, every 4th layer) beats the majority class by >= 0.05. It is USED if planting the distractor class means in the learned
  subspace beats the mean over five random subspaces of the same rank by >= 0.15 with paired p < 0.05 (the C24
  criterion; steer_stage.py --plant-rand). HIT if at BOTH depths a first readable checkpoint exists, a first used
  checkpoint exists, and the readable one is strictly earlier. Otherwise MISS, and the table reports which of: used
  first, same checkpoint, readable but never used, never readable, never used. Six checkpoints give a coarse
  timeline: "earlier" means earlier among these six. Rank-0 subspaces count as not used. Scripts: run_ladder.sh,
  ladder_report.py (score_new.py score_c30). The probe reads WHICH CHAIN the question points at, not the numeric value.
- **A1 (selection-matched null; audit, not a claim).** The plant results report the best of several layers (6.9B:
  14, 18, 22, 26; 12B: 8-32). Each random draw gets the same search: take its best layer, and require the learned
  subspace's best layer to beat the best layer of EVERY draw by >= 0.15 at both depths (6.9B step 143000). Defined after
  Part R had produced its files and before any of them was scored (only the progress log had been read). With five
  draws the smallest exact permutation p is 1/6, so the margin is the evidence. Script: selection_null.py. C24 itself
  is unchanged: it fixes layer 22 in advance, so it involves no layer search.
- **A2 (digit breakdown; exploratory, no threshold).** The flip rate (correct in exactly one order) by correct-answer
  digit and by distractor digit, with a 2000-shuffle permutation test of the spread between the highest and lowest
  digit, for Qwen2.5-7B-Instruct and OLMo-2-1124-7B-Instruct (medication and coins wording, depth 3) and Pythia-6.9B
  (coins, depths 1 and 3). Pairs cancel digit difficulty by construction; this checks it. Script: digit_breakdown.py.
- **A3 (absolute position vs distance; exploratory).** Pythia-12B step 143000, depths 1-3, no-op lines ('>>> pass') put
  at the very start of the prompt (64 lines), directly before the first chain (8), or between the chains (8); the
  answer never changes. Rule fixed in advance: a placement CHANGES the first-or-last habit if the later-chain rate
  moves by >= 0.05 against no padding. start or head changes it, between does not: absolute position. Between changes
  it, start and head do not: distance to the question or order. Both or neither are reported as such. Scripts:
  head_pad_test.py, headpad_report.py.
- **A4 (which sublayer; exploratory).** Pythia-6.9B step 143000, depths 1 and 3: the last-token output of one attention
  or MLP sublayer from the order-swapped run is written into the original run; reported as the share of
  order-dependent questions whose answer follows the swapped order. Whole-residual patching already exists (patch_test.py,
  C-series); this separates attention from MLP. Script: component_patch.py. No threshold; reported as found.
- Not done: a rank sweep of the planted subspace (how many dimensions the effect needs), and per-head patching.
  Steering shows the learned subspace is a causal handle on the answer; sublayer patching narrows where, not how;
  both are stated as limitations.

## 4n. Registered 9 Oct 2026 (~14:40 IST), added to the queue before it started (A5, ladder extensions)

- **A5 (whole-state patching; exploratory).** patch_test.py on Pythia-6.9B step 143000, depths 1 and 3, every 4th layer,
  with the single-chain control (where a value is read out, versus where the choice of chain is made). Reported as found.
- **Ladder extension 1 (how many dimensions).** At every ladder checkpoint the plant test also overwrites only the top-r
  directions of the subspace, r = 1..rank (steer_stage.py --rank-sweep), scored by plant_margin.py. Exploratory, no threshold.
- **Ladder extension 2 (second model size).** The same ladder on Pythia-1.4B, layer 14 of 24, as an exploratory
  replication. C30 itself (the registered claim, thresholds and verdict rule in section 4m) is scored on 6.9B only;
  a 1.4B result that disagrees is reported, not used to change the verdict.
- **A6 (per-head patching; exploratory).** head_patch.py on Pythia-6.9B step 143000, depths 1 and 3, layers 16, 20, 24, 28, 32:
  one head's last-token contribution from the order-swapped run is written into the original run; reported per head
  and for the best 5 heads together. A flat map (no head matters alone) is reported as such. No threshold.
- Not done: the frontier-model test C29 (needs an API key).

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
- That combining the two orders is a new method: it is permutation self-consistency (Tang et
  al. 2023; also Sadanandan & Behzadan 2026). We measure when it helps and whether it beats an
  equally costly rename ensemble (C13).
- That position effects changing during training is new: Mistry et al. 2025 (primacy/recency-like
  attention structure during GPT-2 training) and Salvatore et al. 2025 (position bias in models trained
  from scratch) study it in small models. Ours is narrower: in
  real pretraining runs the preferred direction reverses repeatedly, depends on the seed, and is
  committed mid-network.
- A scaling law for the flip rate: five sizes, one run each above 410M, and seed variation at
  410M is large (Choshen et al., ICML). The size plot is a descriptive trend only.
- That the MMLU option test (C15) is a contribution: it replicates Zong et al. (ICML 2024) and is
  a sanity check on a real benchmark.
- That the fix is free: it doubles inference cost, and with two orders it needs log-probabilities
  (a plain vote ties on every flip).
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
  stage-1 checkpoints nearest 5% and 100% of training (part 5h; the OLMo runs moved to 5e,
  see ~20:15).
- 1 Oct 2026, ~19:50, before any run (OLMo checkpoint runs later moved to 5e, see ~20:15):
  part 5i (EXPLORATORY, no claim) runs the other
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
- 1 Oct 2026, ~20:55, before any run (exploratory, no claim): part 5j runs the same
  order-swap and rename tests, and MMLU option order, on the chat-tuned versions
  Qwen2.5-7B-Instruct and OLMo-2-1124-7B-Instruct, to compare with their base versions (a
  primacy-bias study on multiple-choice QA reports that fine-tuning amplifies the bias;
  chat-tuned versions are what products deploy). Prompts are the same raw text as for base
  models (no chat template), which is stated as a limitation. The base Qwen2.5-7B run now
  also includes the instruction variants so both sides have the same versions.
  attention_test.py states two hypotheses to check: first-chain preference from attention
  to the start of the input ("attention sink"), last-chain preference from causal masking
  and position decay.
- 1 Oct 2026, ~21:00, before any run (final review): run_fullday.sh moves any earlier
  results aside before starting (so nothing stale is scored) and reports FAILED if the
  OLMo-2-7B checkpoint list cannot be fetched. The last OLMo trajectory point is the final
  stage-1 checkpoint, not the released model after stage 2. Base vs instruct comparisons
  (5j) use raw prompts without chat templates, so prompt format is a confound.

- 8 Oct 2026, exploratory and descriptive, added AFTER pre-registration (they re-express or
  decompose pre-registered numbers and change no claim): flipstats.py `order_stats` and
  `pair_metrics` now also report, per question set asked in two orders: strict accuracy (right in
  both orders) and lenient accuracy (right in at least one); the both-right / flip / both-wrong
  split; the chance flip rate p1(1-p2)+p2(1-p1) built from each order's own accuracy; a two-sided exact
  Fisher test on the 2x2 table (odds ratio below 1 means errors are anti-correlated across orders);
  and an exact McNemar test of the two-order ensemble against the original-order answer (questions
  fixed vs broken). They appear in the swap results and in the CSV export automatically.
- 8 Oct 2026, exploratory additions, all labelled exploratory, none changes a registered claim:
  (1) `extra_stats.py`: strict / lenient accuracy, both-right / flip / both-wrong, chance flip rate,
  Fisher and McNemar tests, computed from saved results (swap results keep only rates, so the 2x2 table is
  reconstructed from them exactly). (2) `patch_test.py`: single-chain control, per-question log-prob
  margin, digit-token check; run as Part 9 on Pythia-1.4B at 5 checkpoints (mentor feedback via the
  program director, 8 Oct). (3) `attn_vs_pref.py`, `commit_depth.py`, `noise_check.py`, `run_extra.sh`
  (attention at every 12B trajectory checkpoint and 410M seed checkpoint, depth fraction of the commit layer,
  batch-size noise check). (4) `frontier_flip.py`: same-order repeat (amendment to C21 above),
  `--vote K` (black-box fix, hard levels, capped questions), `--effort` (reasoning-effort confound; each effort
  writes its own result file, and `--score` lists them as separate rows). (5) Wording: C13 is the headline of
  the fix result; combining orders is a known fix; position-bias-in-training is not claimed as new; the size
  fit is a descriptive trend; C15 is a replication. Seed framing: 1.4B, 6.9B and 12B have one run each, so C8
  describes what happened in one training run.
- 8 Oct 2026, wording only, no claim changes: (a) C15 is a replication of Zong et al. (ICML 2024) and a sanity
  check on a real benchmark, not a contribution. (b) C11 intervention-validity wording (in the terms of recent
  intervention papers): regime structure across layers (no effect at layers 4, 8, 10; effect at some layer
  <= 22), a matched random-subspace control (`--n-rand`), negative-control layers where planting must fail,
  and the replication gap (the plant is run on 6.9B only). (c) The project is "narrow but testable": one task,
  one layout, the models tested; pre-registered nulls (C4 erase test, early-warning) are results.
- 8 Oct 2026 (~22:30 IST): registered C22 (uncertainty gating of the order effect) and C23 (fresh-seed
  replication), section 4f, before either was run. New code: `uncertainty_gate.py`, `flipstats.confidence_gate`
  and `top2_margin`, `score_new.py` `score_c22` / `score_c23`, `check_seed.py`; `perturb_test.py` now writes
  `__seed<N>` file names for non-zero seeds so a fresh-seed run can never overwrite the seed-0 result.
- 8 Oct 2026 (~22:45 IST), exploratory, written AFTER the first C22 result (6.9B supported, 12B not):
  `position_decomposition.py` sorts each question's pair of answers (original order, swapped order) into
  right chain twice, wrong chain twice, first-written chain twice, second-written chain twice, or neither,
  overall and inside confidence thirds, for Pythia-12B and 6.9B (step 143000) and, as a check on the
  pattern, OLMo-2-7B, OLMo-2-1B and Qwen2.5-7B. It is motivated by the C22 result at 12B, so it is
  post hoc, registers no claim, and changes none. C22 stays a MISS (1 of 2 models).

- 8 Oct 2026 (~23:40 IST), after all runs: outcomes recorded in section 4g. C9 and C22 are misses, reported as
  misses; the seed wording in README.md and docs/PAPER_DRAFT_v4.md was changed to match C9. Results, scripts
  and figures are committed under `results_08Oct/`. No threshold was changed after the fact.
- 9 Oct 2026: registered C24 and C25 (section 4h) before the 12-hour run. Thresholds not changed afterwards.
- 9 Oct 2026 ~02:00 IST: registered C26 (section 4i) before the 12-hour run. Thresholds not changed afterwards.
- 9 Oct 2026 ~02:15 IST: registered C27 (section 4j) before the 12-hour run. Thresholds not changed afterwards.
- 9 Oct 2026 ~02:30 IST: registered C28 (section 4k) before the 12-hour run. Thresholds not changed afterwards.
- 9 Oct 2026 ~02:00 IST: registered C29 (section 4l) before any frontier run. Thresholds not changed afterwards.
- 9 Oct 2026 ~13:40 IST: registered C30 and audits A1-A3 (section 4m) while the 12-hour run was in progress. Thresholds not changed afterwards.
- 9 Oct 2026 ~14:40 IST: registered A5 and ladder extensions (section 4n) before the queue started. Thresholds not changed afterwards.
