# EarlyMind

**Does multi-step reasoning form inside a language model before it shows up in behavior, and does that head start grow with reasoning depth?**

Recent work shows that internal signals can come before behavior during training (Howe 2026; Xun 2026; Ravulapalli & Chadha 2026). EarlyMind asks what those studies did not:

1. **Depth (main question).** Track 1-, 2- and 3-hop variable chains across Pythia training checkpoints and seeds. Does the internal signal lead behavior, and does the lead grow with the number of hops?
2. **Controls.** Probes must beat the model's own logit decoder, an untrained (step-0) network, shuffled labels and a bag-of-words baseline. For variable tracking, every measure is also scored target-vs-distractor, so knowing which digits appear in the prompt is not enough.
3. **Use.** Erase the probed subspace at a checkpoint before behavior appears and at one after. Early erasure should cost little and late erasure a lot, compared with erasing a random subspace of the same size.
4. **Phases.** Do pretrained models pass through random → heuristic (follow the distractor, copy a worked example) → correct, as Wu et al. (2025) found in models trained from scratch?

Two bonus analyses reuse the same data at no extra cost:
- **Processing depth:** does deeper reasoning only become readable in later layers?
- **Model size:** do bigger models start reasoning earlier in training?

Copying and agreement are calibration tasks. They check the pipeline against known results.

EarlyMind is a retrospective lead-time measurement, not a deployable monitor.

## Tasks

Each task is a K-way choice of the next token.

| Task | Role | Example → answer |
|---|---|---|
| `vartrack_d1/2/3` | **Main**: 1–3 hops, with a distractor chain | `>>> e = 8 … >>> z = e … >>> b` → `8` |
| `copy` | Calibration (in-context learning) | `G G A E … \| G G A` → `E` |
| `agreement` | Surface-solvable anchor | `The songs beside the house` → ` are` |

## Measures (same items at every checkpoint)

| Measure | Meaning |
|---|---|
| `behavior` | The model's own answer (argmax over the K candidates) |
| `logit_decoder` | Classifier on the model's K candidate logits: what the output layer already knows |
| `probe_main` | Logistic probe on the middle-layer residual stream (5-fold CV, fixed C = 0.1) |
| `probe_shuffled` | Same probe on shuffled labels; should sit at chance |
| `keyword` | Bag-of-words on the prompt text: can surface cues solve it? |
| `*_td` | Variable tracking only: is the target value scored above the distractor's? (chance 0.5) |

- **Onset.** The first trained checkpoint where the lower 95% bootstrap bound clears `base + f·(1 − base)`, and the next checkpoint does too. `base` is the larger of chance and that measure's own step-0 score.
- **Lead.** `log10(behavior onset tokens) − log10(probe onset tokens)`. `+inf` means the probe crossed but behavior never did; `−inf` means the reverse. Both are kept and counted.
- **Headline figure.** `depth_leads.png` shows the lead against the threshold `f` for each depth.

## Quick start

Use Colab or Kaggle with a GPU runtime.

```bash
pip install -r requirements.txt
python smoke_test.py                                   # offline end-to-end check, tiny random model

# 1) Gate: can the models do the tasks at the end of training?
python sweep.py --model EleutherAI/pythia-1.4b --steps 143000 --n 200 --out pilot/

# 2) Full sweep (resumable; deletes each checkpoint download after use)
python sweep.py --model EleutherAI/pythia-160m --out runs/
python sweep.py --model EleutherAI/pythia-410m --out runs/
python sweep.py --model EleutherAI/pythia-1.4b --out runs/
for s in 1 2 3 4; do python sweep.py --model EleutherAI/pythia-160m-seed$s --out runs/; done

# 3) Score, analyse, error phases
python probes.py  --runs runs/ --out results.csv       # add --all-layers for the layer sweep
python analyze.py --results results.csv --figdir figs/
python errors.py  --runs runs/ --out figs/error_types.csv

# 4) Causal test: one checkpoint before behavior appears, one after
python steer.py --model EleutherAI/pythia-1.4b --step 4000   --task vartrack_d1
python steer.py --model EleutherAI/pythia-1.4b --step 143000 --task vartrack_d1
python steer_summary.py --dir steer_results --out figs/steer_summary.csv

# 5) Booth demo: what the model says vs what is readable inside, at any checkpoint
pip install gradio
python demo.py --runs runs/ --model EleutherAI/pythia-410m --task vartrack_d1   # add --share on Colab
```

## Outputs (`figs/`)

| File | Contents |
|---|---|
| `depth_leads.png` | **Headline:** lead against threshold, one line per depth, one panel per model |
| `<model>__<items>__<task>.png` | Accuracy over training for each measure, with 95% CIs, chance and the keyword baseline |
| `onsets_and_leads.csv` | Onsets, leads and threshold sensitivity (`lead@0.1` … `lead@0.5`) |
| `layer_depth.png` | **Processing depth:** target-vs-distractor readability by layer, per depth (needs `--all-layers`) |
| `scale_onsets.png` | **Model size:** onset of the inside signal and of behavior against model size, per depth |
| `seed_spread.csv` / `h1_sign_test.csv` | Lead spread across PolyPythias seeds; sign test for depth-3 lead > depth-1 lead |
| `error_types.csv`, `errors__*.png` | Answer-type phases over training |
| `steer_summary.csv` | Causal tests with Holm correction. `suppress_vs_random` is the primary test |

## Files

| File | Purpose |
|---|---|
| `tasks.py` | Item generators (fixed seed, so every checkpoint sees identical items) |
| `model_utils.py` | Loads checkpoints; runs prompts; reads and edits the last-token residual stream |
| `io_utils.py` | Shared file naming (results can never mix item sets) and atomic saves |
| `sweep.py` | Behavior + activations for every checkpoint × task |
| `probes.py` | Probes, own-logit decoder, controls, target-vs-distractor scoring, bootstrap CIs |
| `analyze.py` | Onsets, censored leads, headline figure, seed statistics |
| `errors.py` | Answer-type phases for variable tracking |
| `steer.py` | Causal erase/push tests with random and wrong-class controls |
| `steer_summary.py` | Collects causal results; Holm correction |
| `demo.py` | Interactive booth demo (Gradio): model's answer vs the probe's reading, per checkpoint |
| `smoke_test.py` | Offline end-to-end test |

## Related work

- Howe (2026). *Capability Emergence Can Be Forecast…* arXiv:2609.19000.
- Xun (2026). *Lagged Coupling: Internal Representations Become Readable Before They Become Causal.* arXiv:2609.01048.
- Ravulapalli & Chadha (2026). *Decodable In-Context State and Model Output Across Training.* arXiv:2609.31401.
- Wu, Geiger & Millière (2025). *How Do Transformers Learn Variable Binding in Symbolic Programs?* ICML.
- Billa (2026). arXiv:2602.15997.
- Biderman et al. (2023). *Pythia.*
- van der Wal et al. (2025). *PolyPythias.* ICLR.
- Hewitt & Liang (2019). *Designing and Interpreting Probes with Control Tasks.*
- Liu et al. (2023). *Cognitive Dissonance.* EMNLP.

## License

MIT
