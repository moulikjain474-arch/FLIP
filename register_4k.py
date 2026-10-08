"""Run in the repo on the Mac BEFORE the slot, after register_4j.py: registers C28 (idempotent). Then commit it."""
SRC = "EXPERIMENTS.md"
MARK = "## 4k. Claim registered 9 Oct 2026"
SECTION = '''## 4k. Claim registered 9 Oct 2026 (~02:30 IST) BEFORE the 12-hour run (C28)

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

'''
src = open(SRC).read()
if MARK in src:
    print("already registered")
else:
    anchor = "\n## 5. Statistics"
    assert src.count(anchor) == 1, "anchor '## 5. Statistics' not found exactly once; nothing changed"
    src = src.replace(anchor, "\n" + SECTION.rstrip("\n") + "\n" + anchor)
    src = src.rstrip("\n") + "\n- 9 Oct 2026 ~02:30 IST: registered C28 (section 4k) before the 12-hour run. Thresholds not changed afterwards.\n"
    open(SRC, "w").write(src)
    print("registered 4k")
