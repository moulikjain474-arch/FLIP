# FLIP: planned changes (3 Oct 2026)

Status (updated 8 Oct, see section S for what was built): **originally planned, nothing built yet.** Source: a related-work sweep on arXiv (3 Oct) plus Moulik's
ideas 4-7. None of this changes the GPU run: every new number is computed from data the run
already saves, except the frontier repeat (a few extra API calls on the laptop).

---

## A. New analyses (code only: flipstats.py, score_new.py, export_csv.py)

| # | What | Why a judge cares | Source of idea |
|---|------|-------------------|----------------|
| A1 | **Strict / lenient accuracy.** Strict = right in both orders; lenient = right in at least one. Lenient - strict = flip rate. Ensemble accuracy plotted between them (lenient is roughly the ceiling for any two-order fix). | Two words that explain the whole result instantly. | Moulik (4); same metric in Mat-Pref (2606.21830) |
| A2 | **Three-way split:** both right / flip / both wrong. | Shows "consistent" is not the same as "correct". | DGAO (2605.11974), Huynh (2609.00576) |
| A3 | **Chance baseline for flip rate:** expected flip if errors were independent of order = p1(1-p2) + p2(1-p1), using each order's own accuracy (NOT 2p(1-p), which understates the baseline by (p1-p2)^2/2 when the model prefers one position). | Turns the main number into a test: is the flipping caused by order, or just noise? | Moulik (5), corrected; Mukherjee et al. (2608.11110) "measure the chance floor" |
| A4 | **Fisher exact test on the 2x2 table** (right/wrong in original x right/wrong in swapped), with odds ratio. Odds ratio < 1 = errors anti-correlated across orders = position shortcut. | One p-value that says "this is order, not noise". | Follows from A3 |
| A5 | **Exact McNemar test for the ensemble gain:** count questions the ensemble fixed vs broke, exact binomial test. Reported next to the existing paired bootstrap. | Readable at a glance; standard paired test. | Policy-Masked Experts paper |

Labelled **exploratory / descriptive** in EXPERIMENTS.md (added after pre-registration), except
where they only re-express pre-registered numbers.

## B. Frontier test change (frontier_flip.py + EXPERIMENTS.md section 4e)

- **B1. Same-order repeat (noise floor).** Ask a subset (about 20 questions per setting) a second
  time in the identical order; report the same-order disagreement rate. API models are not
  deterministic even at temperature 0, and OpenAI reasoning models do not accept temperature at
  all. C21 is then reported with this floor next to it.
- Logged as a dated amendment to C21 **before** any frontier run. The C21 threshold itself stays.
- Open-model runs (Pythia/OLMo/Qwen) need no repeat: log-prob scoring is deterministic and the
  rename control already measures sensitivity to an irrelevant change.

## C. Claim and wording updates (README, PAPER_DRAFT, PLAN, EXPERIMENTS section 6)

- **C1. The fix is not ours.** "Ask in several orders and combine" is known as permutation
  self-consistency (Tang et al. 2023; also Sadanandan & Behzadan 2026, MUGEN 2026). New wording:
  combining both orders is a known cheap fix; FLIP measures when and why it works.
- **C2. Headline the fix result as C13** (swap-ensemble beats rename-ensemble): that shows the gain
  is not just "asking twice", and the permutation self-consistency papers do not run that control.
- **C3. Narrow the training-dynamics novelty further.** Position bias emerging during training is
  shown by Mistry et al. 2025 (GPT-2 attention) and Salvatore et al. 2025 (trained from scratch).
  Ours: in real pretraining runs the preferred direction reverses repeatedly, depends on the seed,
  and is committed mid-network.
- **C4. "Narrow but testable" phrasing**, and pre-registered nulls listed as results.
- **C5. Add to section 6 (claims we will NOT make):** that combining orders is a new method; that
  position bias appearing during training is new.

## D. Remove / downgrade / tune

- **D1. Size fit (size_fit.py): downgrade to a descriptive trend.** Five sizes plus seed-to-seed
  variation (Choshen et al., ICML) cannot support a "scaling law". Keep the plot, drop law wording.
- **D2. MMLU option test (C15): relabel as a replication** of Zong et al. (ICML 2024) and a sanity
  check, not a contribution.
- **D3. Limitations: both orders doubles inference cost** (criticism raised by DGAO and GOLD PANNING).
- **D4. Limitations: the two-order fix needs log-probabilities.** With two orders a plain vote ties
  on every flip; many frontier APIs return no log-probs, so there the fix needs several shuffles
  and a vote instead (see J4).

## E. Citations to add

Verify every title/author against arXiv before submission. IDs are from the 3 Oct search.

**Fix / permutation self-consistency**
- Tang, Zhang, Ma, Lin & Ture 2023, "Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models" (2310.07712; authors confirmed from github.com/castorini/perm-sc; check if a NAACL 2024 version exists). Method: shuffle the input list several times, aggregate the rankings (Kemeny/Borda/reciprocal rank fusion). Contrast for us: they need many shuffles and vote; with two orders a hard vote just ties on every flip, so our log-prob sum is what breaks the tie (black-box APIs without log-probs cannot do this with two orders).
- Sadanandan & Behzadan 2026, "When Chain-of-Thought Backfires..." (2603.25960)
- Byerly & Khashabi, GOLD PANNING (2510.09770)
- Brown & McIlroy 2025, "Order Independence With Finetuning" (2503.23483)
- Chu et al. 2026, DGAO "Towards Order Fairness" (2605.11974)

**Position bias and how it arises**
- Salvatore, Wang & Zhang 2025, "Lost in the Middle: An Emergent Property..." (2510.10276)
- Mistry et al. 2025, "Emergence of Episodic Memory in Transformers" (2502.06902)
- Layer-conductance positional-bias paper (recency grows with depth) [TITLE NEEDED from Moulik]
- Veseli et al., COLM 2025, "Positional Biases Shift as Inputs Approach Context Window Limits" (2508.07479)
- Alur et al. 2025, "The Impossibility of Inverse Permutation Learning..." (2509.24125)

**Order sensitivity in practice / stakes**
- Bateni & Abdolali 2026, ClaMPAPP (pediatric appendicitis) [arXiv ID needed]
- Rupprecht, Ahnert & Strohmaier 2025, survey recency bias (2507.07188)
- Zong et al., ICML 2024, "Fool Your (Vision and) Language Model..." (2310.01651)
- Bao et al. 2026, NAFBench "Not What You Meant" (2609.27517)
- Huynh 2026, "Consistency Without Alignment" (2609.00576)

**Variable binding / mechanism / training**
- Davies et al. 2023, "Discovering Variable Binding Circuitry with Desiderata" (2307.03637)
- Xu 2026, "Pattern Selectivity is Not Task-Causal Structure" (2606.05378), reason to keep mechanism claims at layer level
- Długosz et al., EMNLP 2026, GSM-Symbolic re-evaluation (2605.28700)
- Trott 2025, "Toward a Theory of Generalizability in LLM Mech Interp" (2509.22831), framing for seed results (developmental vs positional consistency)
- Kishino et al., ACL 2026 Findings, KL scale across Pythia (2505.15353), contrast: behaviour stabilises early, our preference keeps reversing
- Niu et al., "Illusion or Algorithm?" (Pythia checkpoints) [arXiv ID needed]
- Choshen, Zhang & Andreas, ICML, "A Hitchhiker's Guide to Scaling Law Estimation" (2410.11840)

**Methods precedent (style only)**
- Mukherjee, Bali & Sitaram, COLM 2026 (2608.11110): confounds table, chance floor
- Sunkavalli 2026, pre-registered LLM-judge audit (2608.29517): reporting nulls

## F. Paper / video additions

- **F1. "Confounds we ruled out" table:** token identity (rename), distance (filler), layout
  (interleaved), prompting (instruction), task format (story, code). Uses existing controls.
- **F2. Seed results on Trott's axes:** developmental (does the trajectory replicate across seeds?)
  and positional (is the commit layer the same?).
- **F3. Intro stakes (accurate wording):** "Reordering the sentences of a clinical note made LLMs
  worse at diagnosing appendicitis (Bateni & Abdolali 2026), and shuffling answer options changed a
  medical model's answer 59% of the time (Sadanandan & Behzadan 2026)."
  Do NOT say "same note, different diagnosis": the notes were generated from records, and the paper
  reports worse accuracy, not specific flipped diagnoses.
- **F4. Video hook:** "Same facts, different order, different answer." (Describes our own data.)
- **F5. Discussion:** compare our commit layer with the layer-conductance finding; contrast our
  late reversals with Kishino et al.'s early stabilisation; Alur et al. on why decoder-only
  models struggle to be order-invariant.

## G. Interview prep (docs/PLAN.md)

- Q0 "isn't this known?": add Mistry 2025, Salvatore 2025 and permutation self-consistency.
- New Q: "Isn't the flipping just noise?" -> A3/A4 chance baseline + Fisher test; rename control;
  frontier same-order repeat.
- New Q: "Your fix doubles the cost?" -> yes; it is a known cheap fix; the point is measuring when
  it helps (C13), and cheaper fixes (GOLD PANNING, fine-tuning) exist.
- New Q: "How is this different from permutation self-consistency?" -> they rank lists and vote
  over many shuffles; with only two orders a vote ties whenever the orders disagree, so FLIP sums
  the two orders' log-probabilities instead. And FLIP's main contribution is the measurement
  (training, seeds, layers, controls), not the fix.

## H. Needed from Moulik

1. Title (or arXiv ID) of the layer-conductance positional-bias paper: deferred to the final
   citation check (last step before submission).
2. "Facet-Probe": not a paper (it was Moulik's own name for the idea). Nothing to cite; the
   same-order repeat (B1) is our own design.
3. Escanor's answer on the GPU setup (1 vs 8 GPUs, 200 GB = GPU memory or RAM, shared or free):
   pending; adjusts J1 run order and the time estimate.
4. arXiv IDs for ClaMPAPP and "Illusion or Algorithm?": found at the final citation check.

## I. Considered and not doing

- New GPU experiments: none needed; nothing in the sweep justifies one.
- Mixed-effects models (Długosz et al.): paired per-question tests already cover this.
- Weight-space / SAE / circuit-level analysis: scope creep with ~9 days left; Xu 2026 supports
  staying at layer level.
- The rest of the 3 Oct search (audio, VQA, RAG, quantum, hash, graph theory, etc.): unrelated.

---

## J. Whole-project review (3 Oct, ~02:00), ranked by impact on qualifying

- **J1. Reorder run_fullday.sh: core claims first.** Today the headline claims run LAST:
  Part 6 (C8, 12B trajectory reversals), Part 7 (C9, seeds), Part 8 (C11, commit layer).
  On a shared GPU, a cut-short run would lose the thesis while keeping side tests.
  New order: 0, 1, 2, 6, 7, 8, then 3, 4, 5, 5g, 5b, 5c, 5d, 5h, 5i, 5j, 5e, and 5f (72B, most
  likely to fail on a shared 192 GB GPU) at the very end. No code change inside parts.
- **J2. Explain WHY the preference reverses (biggest science upgrade; exploratory).** Run
  attention_test.py at every 12B trajectory checkpoint (Part 6, before each `clean`, so each
  checkpoint is still downloaded once) and at the 410M seed checkpoints (Part 7). Question: when
  the behaviour flips between first and last chain, does the last token's attention shift with
  it? If yes: "the reversals line up with attention moving between chains" (mechanism of the
  reversal, not just of the answer). If no: reported as a null. Extra GPU: roughly 10-20 min per
  12B checkpoint (estimate; smoke-test first), minutes for 410M.
- **J3. Cut the story to three findings** (judges reward one clear question, not 21 claims):
  1. The position preference reverses repeatedly during pretraining and depends on the seed.
  2. It is decided mid-network (plant + patching; attention as supporting evidence).
  3. It costs real accuracy: flips beyond the chance baseline and the rename control; combining
     both orders recovers much of it; it shows up beyond the toy task (story, code, frontier).
  Every other claim and test goes to the data book / appendix, with one summary table.
- **J4. Frontier black-box fix (optional, exploratory):** `--vote 5` in frontier_flip.py:
  shuffle the facts 5 times, majority vote. Works without log-probs. Run only on levels where
  C21 shows a drop, capped question count.
- **J5. Hero figure:** 12B trajectory with the 410M seeds shaded behind it, reversals marked.
  Plus the concept figure (one question, two orders, two answers) at the top of the README,
  synopsis and video.
- **J6. Downgrade further:** base-vs-instruct (5j) to one sentence; 72B/32B (5f) treated as a
  bonus, not a dependency.

## K. Not science, but can disqualify (do before anything above matters)
- ISEF forms (1, 1A, 1B), research plan, logbook kept in real time from now, truthful AI
  disclosure, guide confirmed, project registered and fee paid. Do not backdate anything.

## L. Considered and rejected
- Human baseline (people answering both orders): judges like it, but it is human-subjects
  research needing IRB approval and extra ISEF forms; too risky this close to the deadline.
- Training-data analysis of what Pythia saw between reversals: too big for ~9 days.
- A packaged "FLIP audit" tool: nice, but presentation time is worth more.

---

## M. Venue plan (decided 4 Oct 2026)

**Order: IRIS first, UniReps second (quick non-archival fallback), full paper later.**

- **IRIS / ISEF:** submit 12 Oct (deadline 13 Oct). Nothing below may delay it.
- **UniReps 2026 workshop, extended abstract (non-archival, max 4 pages main text, anonymized, NeurIPS template, OpenReview).**
  - Deadline: use **10 Oct 17:30 IST** as the real deadline (the official page says "Oct 10 AoE", which is ~17:30 IST on 11 Oct; trust the earlier time, and check the OpenReview countdown).
  - Go / no-go on **9 Oct**, after the run results are in. Skip if results are weak or the IRIS paper is not done.
  - Content: the 4-page abstract is a cut-down copy of the IRIS paper (three findings, hero figure, one table). No new experiments. Best fit to their topics: learning dynamics, mechanistic interpretability, seed-to-seed (do different runs find the same shortcut?).
  - Before submitting, check: (1) dual-submission policy (the call page does not state it; ask unireps.organizers@gmail.com or check OpenReview/FAQ); (2) whether IRIS/ISEF has any rule on publishing or presenting work elsewhere (add to the WhatsApp question about forms 1/1A/1B); (3) UniReps is in Paris, 12 Dec 2026, and is NOT confirmed as a NeurIPS workshop, so do not call it one.
- **Full paper, after IRIS (needs the extra experiments + reviews):**
  - EACL 2027 Student Research Workshop: pre-submission mentorship due 6 Nov, paper due 15 Nov (check archival vs non-archival and eligibility).
  - NAACL 2027 main conference due 24 Dec; NAACL 2027 SRW due 12 Jan.
  - Skip: ACL ARR October (13 Oct, same day as IRIS), ALTA non-archival (10 Oct, weaker fit).
- NeurIPS 2026 interpretability workshops (InterpScience, Interp4Discovery, XAI4Science, NeurReps) are all closed.

---

## N. Sweep of 4 Oct (SPIKE, "FLIP" VLM probe, reasoning-effort blog, masking tutorial, few-shot page, well-being position paper)

Verdict: mostly nothing to take. Four small, free items; two things to verify or decide.

- **N1. Name collision, decision: keep "FLIP".** A different paper uses the name FLIP (a final-layer
  inference-time probe for vision-language models; unrelated topic, no effect on our novelty). As far
  as I know there is also the older image-text pretraining method called FLIP. Cost of renaming now
  (repo, paper, video, forms) is higher than the risk. Mitigation: define it once at first mention as
  "FLIP, short for the Order-Flip Test", and have a one-line answer ready ("unrelated, that one is a
  vision-language probe"). Do not invent a longer acronym expansion.
- **N2. Intervention-validity wording for the plant test (C11), from the VLM-FLIP abstract.** Describe
  what we already do in their terms: (a) regime structure across layers (no effect at layers 4/8/10,
  effect at some layer <= 22), (b) matched random-subspace control (`--n-rand`), (c) negative-control
  layers where planting must fail, (d) the replication gap: the plant is run on 6.9B only.
  Writing-only change. Optional extra if the GPU has time: repeat the plant on a second model
  (OLMo-2-7B); check first whether steer_stage.py supports non-Pythia models (not yet verified).
- **N3. Reasoning-effort confound for C21.** OpenAI reasoning models run at their default effort with
  a 4,000-token budget. Record the effort setting in the results; optional `--effort low|high` flag
  on one reasoning model at the hardest level, so a null (or a drop) cannot be blamed on one effort
  setting. Optional, small API cost. The blog source itself is NOT citable (practitioner post).
- **N4. Numerical-noise check for open models (new, free).** Score the same prompts at two batch
  sizes (for example 8 and 32) in smoke.sh and report how often the answer differs. Left-padded rows
  have different lengths in the original and swapped orders, so bf16/fp16 numerics can differ slightly
  between orders on near-ties. This gives the open-model counterpart of the frontier same-order repeat
  (B1) and answers "is the flipping numerical noise?".
- **N5. Verified, no change needed: padding.** model_utils.py (pkg/) sets `padding_side = "left"`,
  falls back to the EOS token as pad token, passes `attention_mask`, computes `position_ids` from the
  mask (so rotary models see correct positions on padded rows), and reads the last token at index -1.
  attention_test.py uses unbatched prompts. (From the masking tutorial's padding-side warning.)
- **Not citable / not relevant:** SPIKE (personal knowledge graphs), the well-being-mode position
  paper (sycophancy), the Skill Suites masking tutorial (background only; cite Vaswani et al. for the
  causal mask if needed), the LibraryCentral few-shot page (auto-generated filler; cite Brown et al.
  2020 directly if a few-shot reference is needed).

---

## O. IRIS rules pages read 4 Oct (FAQ + home page)

O1. Synopsis is separate from the paper, with hard word limits: Abstract 250; Introduction & Objective 100-150; Innovation 50-100; Methodology 150-250; Results & Conclusions 100-150; Acknowledgement & References 50-100. Write it to these limits. The Innovation box (50-100 words) is where the narrow, accurate novelty claim goes: trajectory across pretraining, seed dependence, committed mid-network. NOT "we invented averaging over orders" (permutation self-consistency exists).
O2. Judging criteria: originality/innovation, rigour of methodology, quality and clarity of data, depth of analysis, real-world significance. Map each to something concrete: hero trajectory figure, confounds table, strict/lenient stats, stakes line.
O3. Deadline: 13 Oct on every page, no extensions. One step on the home page says "Aug 15 or Oct 15" (stale; ignore). Target submit 12 Oct. Fee Rs 5,000 + tax is paid at submission; unpaid = not evaluated. Pay by 11 Oct.
O4. Anonymity: no school, city or state in video, paper, report, presentation or abstract. Check repo links/GitHub username, PDF and figure metadata, video overlays and voice.
O5. One project guide is required and verifies authenticity of the work. Decide who (Shivank or a teacher) and add to the WhatsApp question list.
O6. Eligibility looks fine (Indian-origin, school in India incl. Cambridge, solo OK). Only check: never registered for IRIS before as a Class 12 student.
O7. Category: ROBO (Robotics & Intelligent Machines); confirm at registration.
O8. Site suggests 12 weeks of experimentation (guidance, not a rule). Logbook dates must be true; never backdate. Be ready to say how long the project has run and what was done when.
O9. Dates conflict on the site: home page says National Fair in November, FAQ says Bengaluru in Jan/Feb. Ask on WhatsApp. Also ISEF is 8-14 May 2027 with mentoring camps Mar-Apr; if A2 exams fall in the May/June series this collides. Not a reason not to submit, but plan for it.
O10. Phone numbers differ between pages; use the WhatsApp button or contact@exstemplar.com.

---

## P. Sweep of 4 Oct night (semantic self-consistency; surprisal segmentation; sycophancy SAF/pressure direction; harmfulness subspace)

P0. Verdict: two of four are irrelevant (semantic self-consistency; surprisal-based segmentation for AutoCompressors). Two interpretability papers give one small, free idea each. Nothing here is citable padding. Do not add related-work lines for them.
P1. Semantic self-consistency: extends Wang et al. self-consistency by using reasoning paths. We read a single-token log-prob readout, so no reasoning paths exist. Skip. Our relevant self-consistency citation stays Tang et al. 2310.07712 (permutation self-consistency).
P2. Surprisal segmentation: long-context compression. Unrelated. Skip.
P3. Sycophancy (SAF + multi-layer "pressure" direction ablation): different behaviour, but both papers report that the intervention preserves baseline accuracy. Steal the idea, not the citation: add a SPECIFICITY check to the C11 plant. Report plant effect as a function of baseline log-prob margin (near-ties vs confident questions). Prediction under our thesis: the plant flips near-ties much more than high-margin questions. If Part 8 already stores per-question margins this is analysis only, no GPU. (Not yet verified in the code; check before promising.)
P4. Harmfulness subspace (55 probes, low-rank subspace, dominant-direction steering/ablation): steal one optional experiment. Sweep the RANK of the overwritten class-mean subspace (top-r components, r = 1..9 for K=10 candidates) and plot flip rate vs r. A low r that suffices supports "commitment is low-dimensional". Only if the plant script already takes a subspace and can truncate it; est. 20-30 min of GPU on 6.9B at the commit layer. Optional, below core claims in priority. Not verified in code.
P5. Nothing to remove or replace.
P6. Honest note: the last several sweeps yielded one small item each. The scarce resources now are the GPU slot, the synopsis, the logbook and the video, not more ideas. Next useful step is "go" on building the doc.

---

## Q. Sweep of 4 Oct late night (refusal-token steering; input contrasting/CAD; Gemma-2 feature universality; M-SCAR; TruthDecay; GaslightBench; methodology-prediction benchmark; XFACT translation bias (pasted twice); DiversityMedQA; idiom translation; QIANets; NusaMT)

Q0. Verdict: 9 of 13 are irrelevant to FLIP (M-SCAR, TruthDecay, methodology-prediction benchmark, XFACT x2, idiom translation, QIANets, NusaMT, CAD-style contrasting). Do not cite any of them. Three small free items below, one flag for Paper 2.
Q1. DiversityMedQA validates that each perturbation does not change the correct answer, so that any performance gap is attributable to the perturbation. Steal the habit. Add one sentence to the paper/confounds table: "The order swap preserves the ground-truth answer by construction (chains are independent), and we hand-checked N swapped items." Pick N=30 and actually do the check before writing the sentence.
Q2. Gemma-2 feature-universality abstract: middle layers carry the most similar representations across model sizes. Do not cite (abstract only, different method). Free analysis if data exists: if the commit layer has been measured for more than one model size (410M/1.4B/6.9B/12B), plot it as a FRACTION of depth. A roughly constant fraction is a clean scale-universality figure; a drifting one is an honest limitation. Check what the attention/patching outputs already cover before promising. No GPU if the data exists.
Q3. Refusal-token model diffing compares steering-vector cosine similarity across two models. Caution: cosine between residual-stream directions of different seeds or different models is not meaningful (no shared basis). Only same-run adjacent checkpoints are comparable. Optional, low priority: principal angles between class-mean subspaces at adjacent checkpoints of one run. Not doing unless the plant/subspace code is already in place and time is left.
Q4. FLAG (Paper 2, not FLIP): GaslightBench applies social/linguistic modifiers (flattery, false citations, assumptive language) to verifiable facts, 24,240 single-turn prompts + 720 four-turn dialogues, and finds multi-turn accuracy varies ~60-98%. This is in the same neighbourhood as the GaslitBench idea. Read the full paper after IRIS and decide how Paper 2 differs. Nothing to do for IRIS.
Q5. Nothing to remove or replace.

---

## R. Mentor feedback via Archana, 8 Oct (read the one-pager; comments from a mech interp mentor, relayed)

R0. Verdict: all three points are sound. Priority R3 > R1 > R2 given 4 days left and a limited MI300X window. Nothing here changes pre-registered claims C8-C11: everything below is EXPLORATORY and labelled so. Do not add or rewrite a claim after seeing this feedback.
R1. Order-patching test (her point 1). Patch the residual stream at the query token from the order-A run into the order-B run, layer by layer, and record where the chosen chain flips. This tests where the CHOICE is made, which the plant (writes a value) does not. Also run it on control questions where only one chain matches the query: if the same layer flips those, it reads out a value; if not, it chooses between chains. Start on 1.4B at the 5 checkpoints already used for the plant. Check first whether the existing attention/patching code in Part 5e/8 can be reused. Not verified in code. Est. GPU time unknown until the script exists. Best done after the mentor replies, so it runs once.
R2. Jitter (her point 2). (a) Writing change, no GPU: state that 1.4B, 6.9B and 12B have one run each, so C8 is "what happened in one training run"; a flip counts as seed-robust only if larger than the 410M spread (0.31-0.69). (b) Optional linear probe for "which chain does the question point at", per checkpoint. If the probe is steady while behaviour flips, the change is in the readout; if it moves with it, the representation changed. Cut this first if time runs short.
R3. Readout (her point 3). (a) Check once that all 10 digit candidates are single tokens in the Pythia tokenizer, including the leading-space variants matching how the prompt ends. 2 minutes. (b) Add log P(chain A value) minus log P(chain B value) per question. If results.csv stores per-candidate log-probs this is analysis only, no GPU; verify before promising. Also gives a margin per question, which is what P3 needs.
R4. Timeline: ask mentor for feedback by 9 Oct; run what he suggests once; write up 10-11 Oct; submit 12 Oct (deadline 13 Oct).
R5. Tell Archana the deadline is 13 Oct and submission 12 Oct (done). Do not mention the project-guide requirement to her. Ask the mentor himself about being the guide.

---

## S. Build log, 8 Oct 2026 (after "go")

Built and tested (unit tests pass in test_flip.py; GPU parts need smoke_extra.sh on the node first):
- A1-A5: flipstats.order_stats / fisher_exact / mcnemar_exact; extra_stats.py computes them from saved results
  (swap_test keeps only rates, so its 2x2 table is reconstructed exactly); printed by score_new.py, rows in export_csv.py.
- B1: frontier_flip.py --repeat (default 20); dated amendment to C21 in EXPERIMENTS.md.
- J4: frontier_flip.py --vote K (--vote-n). N3: --effort (own result file per effort).
- R1: patch_test.py single-chain control; Part 9 in run_fullday.sh (1.4B, 5 checkpoints, all layers).
- R3: digit-token check and per-question log-prob margin in patch_test.py (margins stored in the JSON; not in the plant script).
- J2: attention_test at 12B trajectory + 410M seed checkpoints (run_extra.sh parts X, Y), joined to position results by attn_vs_pref.py.
- Q2: commit_depth.py. N4: noise_check.py (run_extra.sh part Z).
- C1-C5, D1-D4: README, EXPERIMENTS section 6, size_fit.py docstring, PLAN interview questions 8-11.
- E: docs/CITATIONS.md (UNVERIFIED list; verify at the final citation check).
- F1: confounds table in README.

Not built / not done:
- P3 (plant effect vs baseline margin) and P4 (subspace rank sweep): need the plant script's per-question outputs; not checked, not promised.
- Q1: hand-check of 30 swapped items (must actually be done by a person before the sentence is written).
- R2b: per-checkpoint linear probe (cut first if short on time).
- F2, F3 text in the paper, F4, F5, J3 (cut the story to three findings), J5 (hero figure), O1 (synopsis to word limits): need the final results; PAPER_DRAFT.md is not in this folder.
- J1 (reorder run_fullday.sh): replaced by ONLY="..." selection; script order unchanged.
- Everything under K (ISEF forms, logbook, guide, fee) is the user's.
