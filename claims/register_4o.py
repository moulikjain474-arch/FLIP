"""Run in the repo on the Mac: records section 4o in EXPERIMENTS.md (idempotent). Then commit it yourself.
Unlike 4m/4n, everything here was added AFTER results were seen, and the section says so."""
SRC = "EXPERIMENTS.md"
MARK = "## 4o. Added 9 Oct 2026, AFTER results were seen"
SECTION = '''## 4o. Added 9 Oct 2026, AFTER results were seen (all exploratory, none registered in advance)

These analyses were added in the evening of 9 Oct 2026 (about 17:00 to 19:30 IST), after the main run, A1 to A6 and the C30
ladder had finished and been read. None has a pre-set threshold except where stated. They are reported as post hoc.

- **C30 stays MISS as registered.** The registered ladder (steps 4000 to 143000) found the chain-identity probe already at
  1.000 at the first step. Early steps (0, 256, 512, 1000, 2000) were then added to find the onset (ladder_report_all.py).
  That extended report is exploratory and is not a replacement verdict.
- **Random-init and early baselines.** Chain probe and value probe at step 0 and steps 256 to 2000. The value probe was found
  to score about 0.35 at random init and unchanged at step 512, so it is uninformative; its queue was stopped early and its
  results are not used as evidence.
- **Norm-matched plant control (plant_normmatch.py).** Pythia-6.9B step 143000, layer 22, d1 (also d3 and layer 16 where run).
  Controls push the state by the same per-item norm as the learned plant, in a random whole-space direction and in a random
  direction inside the learned subspace. Added because the same-rank random-subspace control may be trivially small.
- **Early-vs-final prediction (seed_predict.py)** on the nine 410M seeds (seed 4 absent from these files).
- **Stability tables (stability_table.py).** The first-or-last share across models, seeds, checkpoints and layouts. The
  layouts differ between result folders; a layout-matched matrix (run_extra4.sh) was added to fix that.
- **Per-seed steering on the 410M seeds (run_extra3.sh), a Mamba behavioural test (mamba_position.py), fresh-question
  replication (perturb_test --seed 1) and a seed-1 replication of the early onset (run_extra5.sh).**
- **Error-catching and ensemble comparisons (error_catch.py, extra_stats.py).** The swap-ensemble versus rename-ensemble
  McNemar tests and the flag-on-disagreement analysis are post hoc. Many tests were run; only very small p-values are quoted.
- **SAE pilot (sae_pilot.py), NOT RUN.** Gemma-2-2B access was pending at the time of writing. Decision rule fixed in the
  script docstring before any run: GO needs AUROC >= 0.60 for the SAE swap change, >= 0.05 above the rename control with a
  bootstrap lower bound above 0, and no worse than the raw-residual change by more than 0.02, on >= 40 flipped questions.
- Not done: the frontier-model test C29 (needs the author's own API key).

'''
src = open(SRC).read()
if MARK in src:
    print("already recorded")
else:
    anchor = "\n## 5. Statistics"
    assert src.count(anchor) == 1, "anchor '## 5. Statistics' not found exactly once; nothing changed"
    src = src.replace(anchor, "\n" + SECTION.rstrip("\n") + "\n" + anchor)
    src = src.rstrip("\n") + "\n- 9 Oct 2026 evening IST: recorded section 4o. These analyses were added after results were seen and are exploratory.\n"
    open(SRC, "w").write(src)
    print("recorded 4o")
