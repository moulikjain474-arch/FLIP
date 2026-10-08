"""Run in the repo on the Mac BEFORE the slot: registers C26 in EXPERIMENTS.md (idempotent). Then commit it."""
SRC = "EXPERIMENTS.md"
MARK = "## 4i. Claim registered 9 Oct 2026"
SECTION = '''## 4i. Claim registered 9 Oct 2026 (~02:00 IST) BEFORE the 12-hour run (C26)

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

'''
src = open(SRC).read()
if MARK in src:
    print("already registered")
else:
    anchor = "\n## 5. Statistics"
    assert src.count(anchor) == 1, "anchor '## 5. Statistics' not found exactly once; nothing changed"
    src = src.replace(anchor, "\n" + SECTION.rstrip("\n") + "\n" + anchor)
    src = src.rstrip("\n") + "\n- 9 Oct 2026 ~02:00 IST: registered C26 (section 4i) before the 12-hour run. Thresholds not changed afterwards.\n"
    open(SRC, "w").write(src)
    print("registered 4i")
