"""Learning stages for variable tracking: copy -> wrong chain -> correct.

Reads one or more error-type CSVs written by errors.py (the main 13-checkpoint grid and
the dense grid can be passed together; duplicate checkpoints keep the last file's row),
labels every checkpoint by its dominant answer type, finds the pre-registered events
defined in EXPERIMENTS.md, and draws the stage figure.

  python stages.py --errors figs/error_types.csv figs/error_types_dense.csv --out figs/stages

Outputs: <out>_table.csv (one row per run x task) and <out>.png / <out>.pdf.
"""
import argparse

import numpy as np
import pandas as pd

CATS = ["correct", "distractor", "shot_copy", "other"]
SHORT = {"correct": "C", "distractor": "D", "shot_copy": "S", "other": "O"}
COLORS = {"correct": "#2a9d8f", "distractor": "#e76f51", "shot_copy": "#e9c46a", "other": "#adb5bd"}

# total parameter counts from the Pythia paper
PARAMS = {"pythia-160m": 1.62e8, "pythia-410m": 4.05e8, "pythia-1.4b": 1.41e9,
          "pythia-2.8b": 2.78e9, "pythia-6.9b": 6.86e9, "pythia-12b": 1.18e10}


def size_key(model):
    """'EleutherAI/pythia-160m-seed3' -> 'pythia-160m'."""
    name = model.split("/")[-1].replace("-deduped", "")
    return name.split("-seed")[0]


def is_seed_run(model):
    return "-seed" in model


def load(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df[df["task"].astype(str).str.startswith("vartrack")]
    df = df.drop_duplicates(["model", "items", "task", "step"], keep="last")
    return df.sort_values(["model", "items", "task", "step"]).reset_index(drop=True)


def first_true(g, mask, persist=1):
    """First checkpoint where mask holds for `persist` consecutive checkpoints.
    Returns (step, tokens) or (None, None) if it never happens (censored)."""
    m = np.asarray(mask, bool)
    for i in range(len(m) - persist + 1):
        if m[i:i + persist].all():
            return int(g["step"].iloc[i]), float(g["tokens"].iloc[i])
    return None, None


def events(g):
    """Pre-registered events for one run x task (g sorted by step). See EXPERIMENTS.md."""
    c, d, s = g["correct"].values, g["distractor"].values, g["shot_copy"].values
    top = g[CATS].values.argmax(1)
    runmax = np.maximum.accumulate(s)
    return {
        # competence: correct is the most common answer and >= 0.3, two checkpoints in a row
        "T_cor": first_true(g, (top == 0) & (c >= 0.3), persist=2),
        # baseline alarm: plain accuracy reaches 2x chance
        "B_acc": first_true(g, c >= 0.2),
        # error signal: wrong-chain answers >= 0.2 and at least as common as correct ones
        "S_err": first_true(g, (d >= 0.2) & (d >= c)),
        # copy-decline signal: copying peaked at >= 0.4 and has since fallen by >= 0.15
        "S_copy": first_true(g, (runmax >= 0.4) & (runmax - s >= 0.15)),
    }


def stage_string(g):
    labels = [SHORT[CATS[i]] for i in g[CATS].values.argmax(1)]
    out = [labels[0]]
    for lab in labels[1:]:
        if lab != out[-1]:
            out.append(lab)
    return ">".join(out)


def table(df):
    rows = []
    for (model, items, task), g in df.groupby(["model", "items", "task"]):
        ev = events(g)
        row = dict(model=model, size=size_key(model), seed_run=is_seed_run(model), items=items,
                   task=task, n_checkpoints=len(g), stages=stage_string(g))
        for k, (step, tok) in ev.items():
            row[f"{k}_step"], row[f"{k}_tokens"] = step, tok
        rows.append(row)
    return pd.DataFrame(rows)


def plot(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = sorted(df.groupby(["model", "items"]),
                  key=lambda kv: (PARAMS.get(size_key(kv[0][0]), 0), kv[0][0]))
    tasks = sorted(df["task"].unique())
    fig, axes = plt.subplots(len(runs), len(tasks), figsize=(4 * len(tasks), 2.1 * len(runs)),
                             sharex=True, sharey=True, squeeze=False)
    for i, ((model, _items), gm) in enumerate(runs):
        for j, task in enumerate(tasks):
            ax = axes[i][j]
            g = gm[gm["task"] == task]
            if g.empty:
                ax.axis("off")
                continue
            ev = events(g)
            gp = g[g["tokens"] > 0]  # log axis cannot show step 0
            ax.stackplot(gp["tokens"], *[gp[c] for c in CATS],
                         colors=[COLORS[c] for c in CATS], labels=CATS)
            ax.set_xscale("log")
            ax.set_ylim(0, 1)
            for key, ls in (("S_err", ":"), ("T_cor", "--")):
                t = ev[key][1]
                if t:
                    ax.axvline(t, color="k", ls=ls, lw=1)
            if i == 0:
                ax.set_title(task)
            if j == 0:
                ax.set_ylabel(model.split("/")[-1], fontsize=8)
    axes[0][0].legend(loc="upper left", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("training tokens (log)")
    fig.suptitle("Answer types across training (dotted: error signal, dashed: competence)")
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=160)
    fig.savefig(f"{out}.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--errors", nargs="+", required=True, help="error_types CSV(s) from errors.py")
    ap.add_argument("--out", default="figs/stages")
    args = ap.parse_args()
    df = load(args.errors)
    t = table(df)
    t.to_csv(f"{args.out}_table.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(t[["model", "task", "n_checkpoints", "stages", "S_copy_step", "S_err_step",
                 "B_acc_step", "T_cor_step"]].to_string(index=False))
    plot(df, args.out)
    print(f"wrote {args.out}_table.csv, {args.out}.png, {args.out}.pdf")


if __name__ == "__main__":
    main()
