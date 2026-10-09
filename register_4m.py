"""Run in the repo on the Mac BEFORE starting run_after.sh: registers C30 and audits A1-A3 in EXPERIMENTS.md (idempotent). Then commit it."""
SRC = "EXPERIMENTS.md"
MARK = "## 4m. Registered 9 Oct 2026"
SECTION = '''## 4m. Registered 9 Oct 2026 (~13:40 IST), added to the queue while the 12-hour run was in progress (C30, audits A1-A3)

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

'''
OLD = '''- Not done: a rank sweep of the planted subspace (how many dimensions the effect needs), and activation patching to
  locate where the value is read out. Steering shows the learned subspace is a causal handle on the answer, not where
  the mechanism lives; this is stated as a limitation.'''
NEW = '''- **A4 (which sublayer; exploratory).** Pythia-6.9B step 143000, depths 1 and 3: the last-token output of one attention
  or MLP sublayer from the order-swapped run is written into the original run; reported as the share of
  order-dependent questions whose answer follows the swapped order. Whole-residual patching already exists (patch_test.py,
  C-series); this separates attention from MLP. Script: component_patch.py. No threshold; reported as found.
- Not done: a rank sweep of the planted subspace (how many dimensions the effect needs), and per-head patching.
  Steering shows the learned subspace is a causal handle on the answer; sublayer patching narrows where, not how;
  both are stated as limitations.'''
src = open(SRC).read()
if MARK in src:
    if OLD in src:   # first version of 4m already applied: bring it up to date (A4 added, patching claim corrected)
        open(SRC, "w").write(src.replace(OLD, NEW))
        print("updated 4m (A4 added)")
    else:
        print("already registered")
else:
    anchor = "\n## 5. Statistics"
    assert src.count(anchor) == 1, "anchor '## 5. Statistics' not found exactly once; nothing changed"
    src = src.replace(anchor, "\n" + SECTION.rstrip("\n") + "\n" + anchor)
    src = src.rstrip("\n") + "\n- 9 Oct 2026 ~13:40 IST: registered C30 and audits A1-A3 (section 4m) while the 12-hour run was in progress. Thresholds not changed afterwards.\n"
    open(SRC, "w").write(src)
    print("registered 4m")
