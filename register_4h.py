"""Run in the repo on the Mac BEFORE the slot: registers C24 and C25 in EXPERIMENTS.md (idempotent). Then commit it."""
SRC = "EXPERIMENTS.md"
MARK = "## 4h. New claims, registered 9 Oct 2026"
SECTION = '''## 4h. New claims, registered 9 Oct 2026 BEFORE the 12-hour run (C24, C25)

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

'''
src = open(SRC).read()
if MARK in src:
    print("already registered")
else:
    anchor = "\n## 5. Statistics"
    assert src.count(anchor) == 1, "anchor '## 5. Statistics' not found exactly once; nothing changed"
    src = src.replace(anchor, "\n" + SECTION.rstrip("\n") + "\n" + anchor)
    src = src.rstrip("\n") + "\n- 9 Oct 2026: registered C24 and C25 (section 4h) before the 12-hour run. Thresholds not changed afterwards.\n"
    open(SRC, "w").write(src)
    print("registered 4h")
