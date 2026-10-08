"""Three figures for the poster and the video, drawn straight from the result files.

  python make_figures.py --dir MERGED_RESULTS --out figures
  1. position_over_training.png : share of answers taken from the chain written LATER, by training step
     (Pythia 1.4B / 6.9B / 12B, depth 2, 95% intervals; 0.5 = no position preference)
  2. flip_swap_vs_rename.png    : how often the answer changes when the order is reversed vs when only the
     variable names change (final checkpoints, depth 2)
  3. both_orders_gain.png       : accuracy gain from asking in both orders, with 95% intervals
Numbers come from the files only; nothing is typed in by hand.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

import flipstats as fs

BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#6b6b6b"      # colour-blind-safe pair + neutral
FINAL = ("step143000", "main")


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)


def position_points(root, task="vartrack_d2", models=("pythia-1.4b", "pythia-6.9b", "pythia-12b")):
    """{model: [(step, rate, lo, hi, n)]} sorted by step."""
    out = {m: [] for m in models}
    for f in glob.glob(os.path.join(root, "position", "**", "*.json"), recursive=True):
        s = json.load(open(f))
        m = s["model"].split("/")[-1]
        mt = re.search(r"step(\d+)", s["revision"])
        if m not in out or not mt or task not in s["tasks"]:
            continue
        pl = s["tasks"][task]["picks_later"]
        if pl["n"]:
            lo, hi = fs.wilson(pl["k"], pl["n"])
            out[m].append((int(mt.group(1)), pl["k"] / pl["n"], lo, hi, pl["n"]))
    return {m: sorted(v) for m, v in out.items()}


def final_perturb(root, task="vartrack_d2"):
    """[(model, summary, correct, ens_correct)] for final checkpoints, ordered by name."""
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "perturb", "*.json"))):
        s = json.load(open(f))
        if s["revision"] in FINAL and task in s["tasks"]:
            r = s["tasks"][task]
            rows.append((s["model"].split("/")[-1], r["summary"], r["correct"], r["ens_correct"]))
    return rows


def fig_position(root, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = position_points(root)
    pts = {m: v for m, v in pts.items() if v}
    if not pts:
        return None
    fig, axes = plt.subplots(1, len(pts), figsize=(4.2 * len(pts), 3.6), sharey=True, squeeze=False)
    for ax, (m, v) in zip(axes[0], pts.items()):
        st = np.array([p[0] for p in v]); rate = np.array([p[1] for p in v])
        lo = np.array([p[2] for p in v]); hi = np.array([p[3] for p in v])
        ax.axhline(0.5, color=GREY, lw=1, ls="--")
        ax.fill_between(st, lo, hi, color=BLUE, alpha=0.18, lw=0)
        ax.plot(st, rate, color=BLUE, lw=2, marker="o", ms=4)
        ax.set_xscale("log"); ax.set_ylim(0, 1); _style(ax)
        ax.set_title(m, loc="left", fontsize=11)
        ax.set_xlabel("training step")
    axes[0][0].set_ylabel("share of answers from the\nchain written later")
    axes[0][-1].text(0.98, 0.5, "no preference", transform=axes[0][-1].transAxes, ha="right", va="bottom", fontsize=8, color=GREY)
    fig.suptitle("Share of answers taken from the chain written later, over training (depth 2, 95% intervals)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out, "position_over_training.png")
    fig.savefig(path, dpi=200); plt.close(fig)
    return path


def fig_flip(root, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = final_perturb(root)
    if not rows:
        return None
    names = [r[0] for r in rows]
    sw = [100 * r[1]["flip_swap"] for r in rows]; rn = [100 * r[1]["flip_rename"] for r in rows]
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(5.5, 1.3 * len(rows) + 2), 3.8))
    b1 = ax.bar(x - w / 2, sw, w, color=BLUE, label="order reversed")
    b2 = ax.bar(x + w / 2, rn, w, color=ORANGE, label="variables renamed (control)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("% of questions whose correctness\nchanges between the two versions"); _style(ax)
    ax.set_ylim(0, max(sw + rn) * 1.3)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Answers that change: order reversed vs renamed (depth 2, final)", loc="left", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out, "flip_swap_vs_rename.png")
    fig.savefig(path, dpi=200); plt.close(fig)
    return path


def fig_gain(root, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = final_perturb(root)
    if not rows:
        return None
    names, g, lo, hi = [], [], [], []
    for m, s, c, e in rows:
        single = (np.array(c["orig"], float) + np.array(c["swap"], float)) / 2
        a, l, h = fs.paired_bootstrap(e["swap"], single)
        names.append(m); g.append(100 * a); lo.append(100 * l); hi.append(100 * h)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(5.5, 1.3 * len(rows) + 2), 3.8))
    ax.axhline(0, color=GREY, lw=1)
    ax.bar(x, g, 0.55, color=BLUE)
    ax.errorbar(x, g, yerr=[np.array(g) - np.array(lo), np.array(hi) - np.array(g)], fmt="none", ecolor="#222222", capsize=4, lw=1.2)
    span = max(hi) - min(lo + [0])
    for xi, v, h in zip(x, g, hi):
        ax.text(xi, max(h, v) + 0.03 * span, f"{v:+.1f}", ha="center", fontsize=8)
    ax.set_ylim(min(lo + [0]) - 0.05 * span, max(hi) + 0.14 * span)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("accuracy gain from asking in\nboth orders (points)"); _style(ax)
    ax.set_title("Accuracy gain from asking in both orders (95% interval)", loc="left", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out, "both_orders_gain.png")
    fig.savefig(path, dpi=200); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for fn in (fig_position, fig_flip, fig_gain):
        try:
            p = fn(a.dir, a.out)
            print(("wrote " + p) if p else f"{fn.__name__}: no matching results, skipped")
        except Exception as e:
            print(f"{fn.__name__}: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
