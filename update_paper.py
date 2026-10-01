"""One-time update of docs/PAPER_DRAFT.md: innovation wording (no "first" claim), a related-work
paragraph for the full paper, and the reference list. Run once in the repo:
  python3 update_paper.py
Every replaced block is checked first; if anything differs, nothing is written.
"""
p = "docs/PAPER_DRAFT.md"
s = open(p).read()


def block(text, start, end):
    i = text.index(start)
    return i, text.index(end, i)


EXTRA = (" Fine-tuning has been reported to amplify primacy bias in multiple-choice QA (Exploiting\n"
         "Primacy Effect to Improve LLMs [check authors]), so FLIP compares base and chat-tuned versions;\n"
         "and text-only training has been reported to encourage positional shortcuts in binding [check\n"
         "vision-language paper title and authors].")
ANCHOR = "and long contexts (Liu et al., 2024; Hsieh et al., 2024)."
if "## 2b. Related work" in s:
    if EXTRA in s:
        print("PAPER_DRAFT.md already updated"); raise SystemExit(0)
    assert ANCHOR in s, "related-work paragraph not found: send PAPER_DRAFT.md to Claude, nothing changed"
    open(p, "w").write(s.replace(ANCHOR, ANCHOR + EXTRA))
    print("PAPER_DRAFT.md: two related-work lines added"); raise SystemExit(0)

i, j = block(s, "## 3. Innovation (50-100 words)", "## 4. Methodology")
s = s[:i] + """## 3. Innovation (50-100 words)
Earlier studies measured binding in finished models or in small models trained from scratch on
one task. This project follows real pretrained models across their training checkpoints, sizes
and random seeds, and finds that their primacy/recency shortcut does not settle once, as in toy
models, but reverses repeatedly and ends differently by seed. It introduces FLIP, an order-swap
audit with rename and layout controls, tests a fix (asking in both orders) against a plain prompt
instruction, and registered its claims in advance of the replication runs.

""" + s[j:]

i, _ = block(s, "## 3. Innovation (50-100 words)", "\n")
s = s[:i] + """## 2b. Related work (full paper; not part of the IRIS word limits)
Ravulapalli & Chadha (2026) measured how many bindings finished models can hold; FLIP asks which
binding a model picks when unsure, and how that changes during training. Gur-Arieh et al. (2026)
showed that finished models retrieve bindings mainly by position; FLIP shows how that positional
preference forms and reverses during pretraining. Wu, Geiger & Milliere (2025) found that small
models trained from scratch on this task pass through a position heuristic that appears once and
is then replaced; in real pretrained models the heuristic reverses repeatedly and ends differently
by seed. Guo & Vosoughi (2025) documented primacy and recency effects in LLMs and found prompt
fixes inconsistent; FLIP measures a fix, asking in both orders, against a plain prompt instruction.
Order effects in finished models are known for premises (Chen et al., 2024), multiple-choice
options (Zheng et al., 2024) and long contexts (Liu et al., 2024; Hsieh et al., 2024).""" + EXTRA + """ The gap
FLIP fills: the position preference across real pretraining checkpoints, sizes and seeds, where
inside the model it is decided, and what it costs.

""" + s[i:]

i, j = block(s, "## 6. Acknowledgement and references (50-100 words)", "## 7.")
s = s[:i] + """## 6. Acknowledgement and references (50-100 words)
Guide: [name, role]. Compute: Google Colab; Exea Labs (AMD MI300X). [Describe any AI-assistant
help truthfully, e.g. coding and drafting support.]
References: Biderman et al. 2023 (Pythia); van der Wal et al. 2025 (PolyPythias); OLMo Team 2025
(OLMo 2); Ravulapalli & Chadha 2026; Gur-Arieh et al. 2026; Wu, Geiger & Milliere 2025; Guo &
Vosoughi 2025; Feng & Steinhardt 2024; Chen et al. 2024; Zheng et al. 2024; Liu et al. 2024;
Hsieh et al. 2024. (Full list, including the two [check] works, in the full paper.)

""" + s[j:]
open(p, "w").write(s)
print("PAPER_DRAFT.md updated")
