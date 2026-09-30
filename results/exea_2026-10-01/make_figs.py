"""Draft figures for the 1 Oct 2026 Exea session. Reads results.csv (exported from the run JSONs), writes plots/*.png.

  python make_figs.py
"""
import pandas as pd, numpy as np
d=pd.read_csv(__import__("os").path.join(__import__("os").path.dirname(__file__) or ".", "results.csv"))
p=d[d.kind=="pos"].copy()
p["stepn"]=p.step.str.replace("step","").astype(int)
p=p.rename(columns={"a":"picks_later","b":"n","c":"cor_later","d":"cor_earlier","e":"dis_later","f":"dis_earlier"})
for c in ["picks_later","n","cor_later","cor_earlier","dis_later","dis_earlier"]: p[c]=p[c].astype(float)
p=p[p.group!="2.8b"]
c=d[d.kind=="causal"].copy()
c=c.rename(columns={"a":"cond","b":"correct","c":"distractor","d":"planted","e":"logp"})
for k in ["correct","distractor","planted","logp","layer"]: c[k]=c[k].astype(float)
c["stepn"]=c.step.str.replace("step","").astype(int)
import os
OUT = os.path.join(os.path.dirname(__file__) or ".", "plots"); os.makedirs(OUT, exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8})
TASKS = [("vartrack_d1", "Depth 1"), ("vartrack_d2", "Depth 2"), ("vartrack_d3", "Depth 3")]
SD = 0.023  # within-checkpoint SD from item resampling

# ---------- Fig 1: position preference across training
runs = [("410m", "pythia-410m", "410M"), ("position", "pythia-1.4b", "1.4B"),
        ("dedup", "pythia-1.4b-deduped", "1.4B-deduped"), ("position", "pythia-6.9b", "6.9B"), ("12b", "pythia-12b", "12B")]
mk = ["o", "s", "D", "^", "v"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, (t, tl) in zip(axes, TASKS):
    ax.axhspan(0.5 - 2*SD, 0.5 + 2*SD, color=GRID, alpha=0.9, lw=0)
    ax.axhline(0.5, color=MUTED, lw=1)
    for i, (g, m, lab) in enumerate(runs):
        s = p[(p.group == g) & (p.model == m) & (p.task == t)].sort_values("stepn")
        ax.plot(s.stepn, s.picks_later, marker=mk[i], ms=5, lw=2, color=CAT[i], label=lab, markeredgecolor=SURF, markeredgewidth=1)
    ax.set_xscale("log"); ax.set_title(tl, color=INK, loc="left"); ax.set_xlabel("training step (log)")
    ax.set_ylim(0.1, 0.85)
    ax.text(1100, 0.80, "prefers the LATER chain (recency)", fontsize=8, color=MUTED)
    ax.text(1100, 0.13, "prefers the FIRST chain (primacy)", fontsize=8, color=MUTED)
axes[0].set_ylabel("share of chain answers taken from the later chain")
axes[2].legend(frameon=False, fontsize=9, loc="center right")
fig.suptitle("Which chain the model grabs flips back and forth during training (grey band: ±2 SD measurement noise)", color=INK, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(__import__("os").path.join(OUT, "fig1_position_trajectory.png"), dpi=170)

# ---------- Fig 2: noise floor vs checkpoint swings
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
sets = [("position", "original questions"), ("itemseed1", "new question set 1"), ("itemseed2", "new question set 2")]
for ax, (t, tl) in zip(axes, TASKS):
    ax.axhline(0.5, color=MUTED, lw=1)
    for j, (g, lab) in enumerate(sets):
        s = p[(p.group == g) & (p.model == "pythia-1.4b") & (p.task == t) & (p.stepn.isin([5000, 6000, 7000]))].sort_values("stepn")
        ax.plot(s.stepn + (j - 1) * 90, s.picks_later, "o", ms=8, color=CAT[j], label=lab, markeredgecolor=SURF, markeredgewidth=1.5)
    ax.set_xticks([5000, 6000, 7000]); ax.set_title(tl, color=INK, loc="left"); ax.set_xlabel("training step (Pythia-1.4B)")
axes[0].set_ylabel("share taken from the later chain")
axes[2].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("Same checkpoint, new questions: nearly identical. 1000 steps apart: very different.", color=INK, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(__import__("os").path.join(OUT, "fig2_noise_floor.png"), dpi=170)

# ---------- Fig 3: end-of-training preference across sizes and seeds
fin = p[(p.stepn == 143000) & (p.group != "shots0") & (p.model != "pythia-410m-seed4")]
order = ["pythia-410m"] + [f"pythia-410m-seed{k}" for k in (1,2,3,5,6,7,8,9)] + ["pythia-1.4b", "pythia-1.4b-deduped", "pythia-6.9b", "pythia-12b"]
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.axhspan(0.5 - 2*SD, 0.5 + 2*SD, color=GRID, lw=0); ax.axhline(0.5, color=MUTED, lw=1)
for j, (t, tl) in enumerate(TASKS):
    s = fin[fin.task == t].set_index("model").reindex(order)
    ax.plot(np.arange(len(order)) + (j - 1) * 0.18, s.picks_later, "o", ms=8, color=CAT[j], label=tl, markeredgecolor=SURF, markeredgewidth=1.5)
ax.set_xticks(range(len(order))); ax.set_xticklabels([o.replace("pythia-", "") for o in order], rotation=20)
ax.set_ylabel("share from later chain\n(fully trained)"); ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper right")
ax.set_title("Fully trained models end with different habits, even with only the random seed changed", color=INK, loc="left")
fig.tight_layout(); fig.savefig(__import__("os").path.join(OUT, "fig3_final_by_run.png"), dpi=170)

# ---------- Fig 4: where the answer is committed (single-layer plant into answer-value subspace)
SEQ = ["#b7d3f2", "#86b4ea", "#4f90de", "#2a78d6", "#123e7a"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, (t, tl) in zip(axes, [TASKS[0], TASKS[2]]):
    for k, st in enumerate([3000, 5000, 10000, 32000, 143000]):
        s = c[(c.group == "steer_layers") & (c.cond == "plant_cor") & (c.task == t) & (c.stepn == st)].sort_values("layer")
        ax.plot(s.layer, s.planted, marker="o", ms=5, lw=2, color=SEQ[k], label=f"step {st}")
    ax.set_title(tl, color=INK, loc="left"); ax.set_xlabel("layer where the number is written (of 24)")
axes[0].set_ylabel("model says the written number"); axes[1].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Writing a number into one layer: ignored before layer ~12, controls the answer after ~14, at every stage of training", color=INK, x=0.01, ha="left", fontsize=10)
fig.tight_layout(); fig.savefig(__import__("os").path.join(OUT, "fig4_commit_layer.png"), dpi=170)

# ---------- Fig 5: order-swap test
sw = d[d.kind == "swap"].copy()
for k in "abcde": sw[k] = sw[k].astype(float)
sw = sw.rename(columns={"a": "acc_o", "b": "acc_s", "c": "acc_c", "d": "same", "e": "flip"})
sw["stepn"] = sw.step.str.replace("step", "").astype(int)
main = [("pythia-410m", 143000, "410M"), ("pythia-1.4b", 6000, "1.4B\nstep 6k"), ("pythia-1.4b", 32000, "1.4B\nstep 32k"),
        ("pythia-1.4b", 143000, "1.4B"), ("pythia-1.4b-deduped", 143000, "1.4B-\ndeduped"), ("pythia-6.9b", 143000, "6.9B"), ("pythia-12b", 143000, "12B")]
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
x = np.arange(len(main))
for j, (t, tl) in enumerate(TASKS):
    v = [sw[(sw.model == m) & (sw.stepn == st) & (sw.task == t)].flip.iloc[0] for m, st, _ in main]
    axes[0].plot(x + (j - 1) * 0.2, v, "o", ms=8, color=CAT[j], label=tl, markeredgecolor=SURF, markeredgewidth=1.5)
axes[0].set_xticks(x); axes[0].set_xticklabels([m[2] for m in main], fontsize=8)
axes[0].set_ylim(0, 0.65); axes[0].set_ylabel("right in one order, wrong in the other")
axes[0].set_title("Share of questions whose correctness depends only on chain order", color=INK, loc="left", fontsize=10)
axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
t = "vartrack_d1"
for i, (m, st, lab) in enumerate(main):
    r = sw[(sw.model == m) & (sw.stepn == st) & (sw.task == t)].iloc[0]
    axes[1].plot([i, i], [r.acc_o, r.acc_c], color=MUTED, lw=2)
    axes[1].plot(i, r.acc_o, "o", ms=8, color=CAT[0], markeredgecolor=SURF, markeredgewidth=1.5, label="one order" if i == 0 else None)
    axes[1].plot(i, r.acc_c, "o", ms=8, color=CAT[1], markeredgecolor=SURF, markeredgewidth=1.5, label="both orders combined" if i == 0 else None)
axes[1].set_xticks(x); axes[1].set_xticklabels([m[2] for m in main], fontsize=8)
axes[1].set_ylabel("accuracy (depth 1)"); axes[1].set_ylim(0.3, 1.0)
axes[1].set_title("Asking in both orders: free gain for trained models (12B: 79% to 91%)", color=INK, loc="left", fontsize=10)
axes[1].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(__import__("os").path.join(OUT, "fig5_order_swap.png"), dpi=170)
print("ok")
