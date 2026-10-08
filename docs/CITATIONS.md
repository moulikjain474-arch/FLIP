# Citations to add (from the 3 Oct sweep). NOT VERIFIED: check every title, author list and
# venue against arXiv / the proceedings before submission (last step; do not paste from here).

Status key: [ ] not yet checked   [x] checked against the source

## Fix / permutation self-consistency
- [x] (checked 8 Oct on arXiv: title, authors, v1 11 Oct 2023, v2 22 Apr 2024) Tang, Zhang, Ma, Lin, Ture 2023, "Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in LLMs", arXiv 2310.07712 (authors from github.com/castorini/perm-sc; check for a NAACL 2024 version)
- [x] (checked 8 Oct) Sadanandan & Behzadan 2026, "When Chain-of-Thought Backfires: Evaluating Prompt Sensitivity in Medical Language Models", arXiv 2603.25960, 26 Mar 2026. Abstract: "Shuffling answer options causes the model to change predictions 59.1% of the time" (MedGemma 4B/27B, MedMCQA and PubMedQA)
- [ ] Byerly & Khashabi, GOLD PANNING, arXiv 2510.09770
- [ ] Brown & McIlroy 2025, "Order Independence With Finetuning", arXiv 2503.23483
- [ ] Chu et al. 2026, DGAO "Towards Order Fairness", arXiv 2605.11974

## Position bias and how it arises
- [x] (checked 8 Oct: title, authors, 11 Oct 2025; trained from scratch on simple human-memory tasks) Salvatore, Wang, Zhang 2025, "Lost in the Middle: An Emergent Property...", arXiv 2510.10276
- [x] (checked 8 Oct; authors Mistry, Bajaj, Aggarwal, Maini, Tiganj; 9 Feb 2025) Mistry et al. 2025, "Emergence of Episodic Memory in Transformers: Characterizing Changes in Temporal Structure of Attention Scores During Training", arXiv 2502.06902. NOTE: the abstract describes GPT-2 attention with primacy/recency-like effects and induction-head ablation; read the paper before saying more than "studies how the attention structure changes during training"
- [ ] Layer-conductance positional-bias paper (recency grows with depth): TITLE NEEDED from Moulik
- [ ] Veseli et al., COLM 2025, "Positional Biases Shift as Inputs Approach Context Window Limits", arXiv 2508.07479
- [ ] Alur et al. 2025, "The Impossibility of Inverse Permutation Learning...", arXiv 2509.24125

## Order sensitivity in practice / stakes
- [ ] Bateni & Abdolali 2026, ClaMPAPP (pediatric appendicitis): arXiv ID needed
- [ ] Rupprecht, Ahnert, Strohmaier 2025, survey on recency bias, arXiv 2507.07188
- [ ] Zong et al., ICML 2024, "Fool Your (Vision and) Language Model...", arXiv 2310.01651
- [ ] Bao et al. 2026, NAFBench "Not What You Meant", arXiv 2609.27517
- [ ] Huynh 2026, "Consistency Without Alignment", arXiv 2609.00576

## Variable binding / mechanism / training
- [ ] Davies et al. 2023, "Discovering Variable Binding Circuitry with Desiderata", arXiv 2307.03637
- [ ] Xu 2026, "Pattern Selectivity is Not Task-Causal Structure", arXiv 2606.05378
- [ ] Dlugosz et al., EMNLP 2026, GSM-Symbolic re-evaluation, arXiv 2605.28700
- [ ] Trott 2025, "Toward a Theory of Generalizability in LLM Mech Interp", arXiv 2509.22831
- [ ] Kishino et al., ACL 2026 Findings, KL scale across Pythia, arXiv 2505.15353
- [ ] Niu et al., "Illusion or Algorithm?" (Pythia checkpoints): arXiv ID needed
- [ ] Choshen, Zhang, Andreas, ICML, "A Hitchhiker's Guide to Scaling Law Estimation", arXiv 2410.11840

## Methods precedent (style only)
- [ ] Mukherjee, Bali, Sitaram, COLM 2026, arXiv 2608.11110 (confounds table, chance floor)
- [ ] Sunkavalli 2026, pre-registered LLM-judge audit, arXiv 2608.29517

## Already used in the README / PLAN (also to verify)
- [ ] Ravulapalli & Chadha 2026; Gur-Arieh et al. 2026; Wu, Geiger & Milliere 2025; Guo & Vosoughi 2025;
      Chen et al. 2024 (premise order, ICML); Zheng et al. 2024 (option position); Hsieh et al. 2024 (RULER, COLM);
      Liu et al. 2024 (lost in the middle); Brown et al. 2020 (few-shot) if a few-shot reference is needed.

## Wording rule for the stakes line (F3)
"Reordering the sentences of a clinical note made LLMs worse at diagnosing appendicitis (Bateni & Abdolali 2026),
and shuffling answer options changed a medical model's predictions 59.1% of the time (Sadanandan & Behzadan 2026, MedGemma on MedMCQA and PubMedQA)."
Do NOT write "same note, different diagnosis": the notes were generated from records and the paper reports worse
accuracy, not specific flipped diagnoses. The 59.1% figure is checked against the abstract; Bateni & Abdolali is still unchecked (no arXiv ID yet).
