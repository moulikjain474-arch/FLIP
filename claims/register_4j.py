"""Run in the repo on the Mac BEFORE the slot, after register_4i.py: registers C27 (idempotent). Then commit it."""
SRC = "EXPERIMENTS.md"
MARK = "## 4j. Claim registered 9 Oct 2026"
SECTION = '''## 4j. Claim registered 9 Oct 2026 (~02:15 IST) BEFORE the 12-hour run (C27)

- **C27 (is the shortcut only a small-model problem?).** Qwen2.5-72B is ~100% correct on depths 1-3, so those depths
  cannot show whether the largest models still rely on position. The medication wording is also run at depths 6 and 9
  for every model in part M. HIT if Qwen2.5-72B or Qwen2.5-72B-Instruct has flip >= 0.05 AND both-orders gain 95% CI
  entirely above 0 at depth 6 or depth 9. Otherwise MISS. The size trend across the other models is descriptive.
  Scored by score_new.py (score_c27).

'''
src = open(SRC).read()
if MARK in src:
    print("already registered")
else:
    anchor = "\n## 5. Statistics"
    assert src.count(anchor) == 1, "anchor '## 5. Statistics' not found exactly once; nothing changed"
    src = src.replace(anchor, "\n" + SECTION.rstrip("\n") + "\n" + anchor)
    src = src.rstrip("\n") + "\n- 9 Oct 2026 ~02:15 IST: registered C27 (section 4j) before the 12-hour run. Thresholds not changed afterwards.\n"
    open(SRC, "w").write(src)
    print("registered 4j")
