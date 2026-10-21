"""Onsets, lead times, the depth figure and seed statistics from results.csv.

Onset (pre-registered): the first TRAINED checkpoint at which the lower 95%
bootstrap bound exceeds  base + f * (1 - base)  AND the next checkpoint does too
(one noisy checkpoint cannot set an onset). base = max(chance, the same
measure's step-0 upper bound). A measure that never gets there has no onset.

Lead = log10(tokens at behavior onset) - log10(tokens at probe onset).
  > 0    the inside signal came first
  +inf   the probe crossed but behavior never did (censored: "probe first")
  -inf   behavior crossed but the probe never did (censored: "behavior first")
  blank  neither crossed
Censored outcomes are kept and counted, never dropped.

Variable-tracking leads use the target-vs-distractor (*_td) measures, where a
model that only knows which digits are present scores ~50%. The main result is
the lead as a curve over thresholds f = 0.1 ... 0.5, per depth (depth_leads.png).
Bonus figures: layer_depth.png (processing depth; needs probes.py --all-layers)
and scale_onsets.png (onsets against model size).

  python analyze.py --results results.csv --figdir figs
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

INIT_X = 1e6                       # where step 0 is drawn on the log axis
FRACS = (0.1, 0.2, 0.3, 0.4, 0.5)  # threshold sensitivity grid
DEPTH_TASKS = ["vartrack_d1", "vartrack_d2", "vartrack_d3"]
STYLE = {"behavior": ("Behavior (model's answer)", "tab:red"),
         "logit_decoder": ("Own-logit decoder", "tab:orange"),
         "probe_main": ("Inside probe (middle layer)", "tab:blue"),
         "probe_shuffled": ("Shuffled-label control", "tab:gray")}


def lead_measures(task):
    """Which measures define onsets for this task."""
    sfx = "_td" if task.startswith("vartrack") else ""
    return {k: k + sfx for k in ("behavior", "logit_decoder", "probe_main")}


def onset(df, frac):
    """Tokens at which one measure first clears its threshold twice in a row (None if never)."""
    df = df.sort_values("tokens")
    if df.empty:
        return None
    chance = 1 / df["K"].iloc[0]
    init = df[df["step"] == 0]
    base = max(chance, float(init["hi"].iloc[0])) if len(init) else chance
    trained = df[df["step"] > 0]
    ok = (trained["lo"] > base + frac * (1 - base)).to_numpy()
    stable = ok & np.append(ok[1:], True)  # the last checkpoint needs no successor
    hits = trained["tokens"].to_numpy()[stable]
    return int(hits[0]) if len(hits) else None


def lead(t_behavior, t_probe):
    if t_probe is None and t_behavior is None:
        return None
    if t_probe is None:
        return float("-inf")
    if t_behavior is None:
        return float("inf")
    log = lambda t: np.log10(max(t, INIT_X))
    return log(t_behavior) - log(t_probe)


def family(model):
    """'EleutherAI/pythia-160m-seed3' -> 'EleutherAI/pythia-160m' (for pooling seeds)."""
    return re.sub(r"-(data-|weight-)?seed\d+$", "", model)


def plot_run(g, model, task, keyword, path):
    panels = [("", "10-way accuracy")]
    if task.startswith("vartrack"):
        panels.append(("_td", "Target vs distractor accuracy"))
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4), squeeze=False)
    for ax, (sfx, ylabel) in zip(axes[0], panels):
        chance = None
        for base_name, (label, color) in STYLE.items():
            s = g[g.measure == base_name + sfx].sort_values("tokens")
            if s.empty:
                continue
            chance = 1 / s["K"].iloc[0]
            x = s["tokens"].clip(lower=INIT_X)
            ax.plot(x, s["acc"], "-o", ms=3, color=color, label=label)
            ax.fill_between(x, s["lo"], s["hi"], color=color, alpha=0.15)
        if chance is not None:
            ax.axhline(chance, ls=":", color="k", lw=1, label="Chance")
        kw = keyword.get((task, "keyword" + sfx))
        if kw is not None:
            ax.axhline(kw, ls="--", color="tab:green", lw=1, label="Keyword baseline")
        ax.set(xscale="log", ylim=(0, 1.02), ylabel=ylabel,
               xlabel="Training tokens (step 0 drawn at 1e6)",
               title=f"{model.split('/')[-1]} | {task}")
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_depth(leads, figdir):
    """Headline figure: lead vs threshold, one line per depth, one panel per model.
    Censored leads are drawn at the edge of the panel with triangle markers."""
    sub = leads[leads.task.isin(DEPTH_TASKS)]
    models = sorted(sub.model.unique())
    if not models:
        return
    finite = sub[[f"lead@{f:.1f}" for f in FRACS]].to_numpy(dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    edge = max(1.0, float(np.abs(finite).max()) + 0.5) if len(finite) else 3.0
    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 3.8), squeeze=False)
    for ax, model in zip(axes[0], models):
        has_data = False
        for task, color in zip(DEPTH_TASKS, ("tab:green", "tab:blue", "tab:purple")):
            r = sub[(sub.model == model) & (sub.task == task)]
            if r.empty:
                continue
            ys = np.array([r.iloc[0][f"lead@{f:.1f}"] for f in FRACS], dtype=float)
            has_data |= bool((~np.isnan(ys)).any())
            shown = np.clip(ys, -edge, edge)
            ax.plot(FRACS, shown, "-", color=color, label=task.replace("vartrack_d", "depth "))
            for f, y in zip(FRACS, ys):
                if np.isnan(y):
                    continue
                marker = "^" if y == np.inf else "v" if y == -np.inf else "o"
                ax.plot(f, np.clip(y, -edge, edge), marker, color=color, ms=6)
        ax.axhline(0, color="k", lw=0.8)
        ax.set(title=model.split("/")[-1], xlabel="Onset threshold f", ylim=(-edge - 0.3, edge + 0.3),
               xlim=(FRACS[0] - 0.05, FRACS[-1] + 0.05), xticks=FRACS,
               ylabel="Lead, log10 tokens (>0: inside first)")
        if not has_data:  # neither signal crossed at any threshold
            ax.text(0.5, 0.6, "no onsets yet", transform=ax.transAxes, ha="center", color="gray")
        ax.legend(fontsize=7)
    fig.text(0.01, 0.01, "▲ probe crossed, behavior never did   ▼ behavior crossed, probe never did",
             fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(figdir, "depth_leads.png"), dpi=200)
    plt.close(fig)


def model_size_m(model):
    """'EleutherAI/pythia-1.4b' -> 1400.0 (millions of parameters), or NaN if not in the name."""
    m = re.search(r"(\d+(?:\.\d+)?)([mb])(?![a-z])", model.split("/")[-1].lower())
    return float(m.group(1)) * (1000 if m.group(2) == "b" else 1) if m else np.nan


def plot_layers(runs, figdir):
    """Processing depth: at the final checkpoint, how readable is the answer at each layer?
    Hypothesis: deeper reasoning becomes readable only in later layers."""
    sub = runs[runs.task.isin(DEPTH_TASKS) & runs.measure.str.fullmatch(r"probe_L\d+_td")]
    if sub.empty or sub.layer.nunique() < 2:
        return  # needs `probes.py --all-layers`
    sub = sub[sub.step == sub.groupby("model").step.transform("max")].copy()
    sub["rel_depth"] = sub.layer / sub.n_layers
    sub.sort_values(["model", "task", "rel_depth"]).to_csv(
        os.path.join(figdir, "layer_depth.csv"), index=False)
    models = sorted(sub.model.unique(), key=model_size_m)
    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 3.8), squeeze=False)
    for ax, model in zip(axes[0], models):
        for task, color in zip(DEPTH_TASKS, ("tab:green", "tab:blue", "tab:purple")):
            s = sub[(sub.model == model) & (sub.task == task)].sort_values("rel_depth")
            if s.empty:
                continue
            ax.plot(s.rel_depth, s.acc, "-o", ms=4, color=color,
                    label=task.replace("vartrack_d", "depth "))
            ax.fill_between(s.rel_depth, s.lo, s.hi, color=color, alpha=0.15)
        ax.axhline(0.5, ls=":", color="k", lw=1)
        ax.set(title=f"{model.split('/')[-1]} (final checkpoint)", xlim=(0, 1.02), ylim=(0.3, 1.02),
               xlabel="Layer (fraction of network depth)", ylabel="Target vs distractor accuracy")
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "layer_depth.png"), dpi=200)
    plt.close(fig)


def plot_scale(leads, figdir):
    """Model size: do bigger models start reasoning earlier in training?
    Uses the main runs only (seed variants are pooled separately in seed_spread.csv)."""
    sub = leads[leads.task.isin(DEPTH_TASKS) & (leads.model == leads.family)].copy()
    sub["size_m"] = sub.model.map(model_size_m)
    sub = sub.dropna(subset=["size_m"])
    if sub.size_m.nunique() < 2:
        return
    sub.sort_values(["task", "size_m"]).to_csv(os.path.join(figdir, "scale_onsets.csv"), index=False)
    onsets = sub[["onset_behavior", "onset_probe"]].to_numpy(dtype=float)
    top = np.nanmax(onsets) * 3 if np.isfinite(onsets).any() else 1e12  # where "never" is drawn
    fig, ax = plt.subplots(figsize=(6, 4))
    for task, color in zip(DEPTH_TASKS, ("tab:green", "tab:blue", "tab:purple")):
        s = sub[sub.task == task].sort_values("size_m")
        for col, style, name in (("onset_probe", "-o", "inside"), ("onset_behavior", "--s", "behavior")):
            y = s[col].astype(float)
            ax.plot(s.size_m, y.fillna(top), style, color=color, ms=5,
                    label=f"{task.replace('vartrack_d', 'depth ')}: {name}")
    ax.axhline(top, color="k", lw=0.6, ls=":")
    ax.text(ax.get_xlim()[0], top, " never (within checkpoints tested)", va="bottom", fontsize=7)
    sizes = sorted(sub.size_m.unique())
    ax.set(xscale="log", yscale="log", xlabel="Model size (parameters)",
           ylabel="Onset (training tokens)", title="Do bigger models start reasoning earlier?")
    ax.set_xticks(sizes, [f"{v / 1000:g}B" if v >= 1000 else f"{v:g}M" for v in sizes])
    ax.minorticks_off()
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "scale_onsets.png"), dpi=200)
    plt.close(fig)


def seed_stats(leads, figdir, frac):
    """Spread of leads across seeds, and a sign test for H1 (depth-3 lead > depth-1 lead)."""
    col = f"lead@{frac:.1f}"
    df = leads.assign(lead=leads[col])
    spread = df.groupby(["family", "task"])["lead"].agg(
        n_runs="count",
        n_probe_first=lambda s: int((s == np.inf).sum()),
        n_behavior_first=lambda s: int((s == -np.inf).sum()),
        median="median", min="min", max="max").reset_index()
    spread.to_csv(os.path.join(figdir, "seed_spread.csv"), index=False)

    tests = []
    for fam, g in df[df.task.isin(["vartrack_d1", "vartrack_d3"])].groupby("family"):
        w = g.pivot_table(index="model", columns="task", values="lead", aggfunc="first")
        if not {"vartrack_d1", "vartrack_d3"} <= set(w.columns):
            continue
        w = w.dropna()
        greater = int((w["vartrack_d3"] > w["vartrack_d1"]).sum())
        less = int((w["vartrack_d3"] < w["vartrack_d1"]).sum())
        p = binomtest(greater, greater + less, 0.5, alternative="greater").pvalue if greater + less else np.nan
        tests.append(dict(family=fam, threshold=frac, runs=len(w), d3_gt_d1=greater, d3_lt_d1=less,
                          ties=len(w) - greater - less, sign_test_p=p))
    pd.DataFrame(tests).to_csv(os.path.join(figdir, "h1_sign_test.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--figdir", default="figs")
    ap.add_argument("--frac", type=float, default=0.3, help="pre-registered main threshold")
    args = ap.parse_args()
    df = pd.read_csv(args.results)
    is_kw = df.measure.str.startswith("keyword")
    keyword = {(r.task, r.measure): r.acc for r in df[is_kw].itertuples()}
    runs = df[~is_kw]
    os.makedirs(args.figdir, exist_ok=True)

    rows = []
    for (model, items, task), g in runs.groupby(["model", "items", "task"]):
        m = lead_measures(task)
        on = {k: onset(g[g.measure == v], args.frac) for k, v in m.items()}
        final = g[g.step == g.step.max()].set_index("measure")["acc"]
        rows.append(dict(
            model=model, family=family(model), items=items, task=task,
            lead_measure_suffix="_td" if task.startswith("vartrack") else "",
            onset_behavior=on["behavior"], onset_decoder=on["logit_decoder"],
            onset_probe=on["probe_main"],
            lead_probe_vs_behavior=lead(on["behavior"], on["probe_main"]),
            lead_probe_vs_decoder=lead(on["logit_decoder"], on["probe_main"]),
            final_behavior=final.get(m["behavior"]), final_decoder=final.get(m["logit_decoder"]),
            final_probe=final.get(m["probe_main"]),
            keyword=keyword.get((task, "keyword" + ("_td" if task.startswith("vartrack") else ""))),
            **{f"lead@{f:.1f}": lead(onset(g[g.measure == m["behavior"]], f),
                                     onset(g[g.measure == m["probe_main"]], f)) for f in FRACS}))
        plot_run(g, model, task, keyword,
                 os.path.join(args.figdir, f"{model.replace('/', '__')}__{items}__{task}.png"))

    leads = pd.DataFrame(rows)
    lead_cols = [c for c in leads.columns if c.startswith("lead") and c != "lead_measure_suffix"]
    leads[lead_cols] = leads[lead_cols].apply(pd.to_numeric, errors="coerce")  # None -> NaN
    leads.to_csv(os.path.join(args.figdir, "onsets_and_leads.csv"), index=False)
    plot_depth(leads, args.figdir)
    plot_layers(runs, args.figdir)
    plot_scale(leads, args.figdir)
    seed_stats(leads, args.figdir, args.frac)
    print(leads.to_string())


if __name__ == "__main__":
    main()
