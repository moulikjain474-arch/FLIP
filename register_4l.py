"""Run in the repo on the Mac BEFORE any frontier run: registers C29 (idempotent). Then commit it."""
SRC = "EXPERIMENTS.md"
MARK = "## 4l. Claim registered 9 Oct 2026"
SECTION = '''## 4l. Claim registered 9 Oct 2026 (~02:00 IST) BEFORE any frontier run (C29)

- **C29 (frontier model, medication wording).** frontier_med.py asks one closed frontier model through its API the
  same 100 medication-record questions per depth (3 and 6) as gen_test.py, in both record orders, in the direct and
  the step-by-step mode, temperature 0 where the API allows it. HIT if, for at least one frontier model, the flip
  rate is >= 0.03 at depth 3 or 6 in either mode with fewer than 20% unparsed answers. Otherwise MISS. Which model
  is run depends on API access; the model name is recorded in the result file. Every reply is saved.

'''
src = open(SRC).read()
if MARK in src:
    print("already registered")
else:
    anchor = "\n## 5. Statistics"
    assert src.count(anchor) == 1, "anchor '## 5. Statistics' not found exactly once; nothing changed"
    src = src.replace(anchor, "\n" + SECTION.rstrip("\n") + "\n" + anchor)
    src = src.rstrip("\n") + "\n- 9 Oct 2026 ~02:00 IST: registered C29 (section 4l) before any frontier run. Thresholds not changed afterwards.\n"
    open(SRC, "w").write(src)
    print("registered 4l")
