"""Run in the repo on the Mac: registers A5 and the ladder extensions (section 4n) in EXPERIMENTS.md (idempotent). Then commit it."""
SRC = "EXPERIMENTS.md"
MARK = "## 4n. Registered 9 Oct 2026"
SECTION = '''## 4n. Registered 9 Oct 2026 (~14:40 IST), added to the queue before it started (A5, ladder extensions)

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

'''
src = open(SRC).read()
if MARK in src:
    print("already registered")
else:
    anchor = "\n## 5. Statistics"
    assert src.count(anchor) == 1, "anchor '## 5. Statistics' not found exactly once; nothing changed"
    src = src.replace(anchor, "\n" + SECTION.rstrip("\n") + "\n" + anchor)
    src = src.rstrip("\n") + "\n- 9 Oct 2026 ~14:40 IST: registered A5 and ladder extensions (section 4n) before the queue started. Thresholds not changed afterwards.\n"
    open(SRC, "w").write(src)
    print("registered 4n")
