"""Learning stages for variable tracking: copy -> wrong chain -> correct.

Reads one or more error-type CSVs written by errors.py (the main 13-checkpoint grid and
the dense grid can be passed together; duplicate checkpoints keep the last file's row),
labels every checkpoint by its dominant answer type, finds the pre-registered events
defined in EXPERIMENTS.md, and draws the stage figure.

  python stages.py --errors figs/error_types.csv figs/error_types_dense.csv --out figs/stages

Outputs: <out>_table.csv (one row per run x task) and <out>.png / <out>.pdf.

The table also scores the locked predictions P1-P4 for the large models (EXPERIMENTS.md, C3)
automatically, so nobody grades them by eye:
  P1 stage order O>S>D>C (leading O optional, one D/C flip-back allowed)
  P2 S_err and B_acc fire at the same checkpoint or one apart
  P3 T_cor at step <= 8000
  P4 wrong-chain rate at the last checkpoint >= 0.15
plus a descriptive "coin-flip window": checkpoints where correct and wrong-chain answers
are both >= 0.3 and within 0.05 of each other.
"""
import argparse
import re

import numpy as np
import pandas as pd

CATS = ["correct", "distractor", "shot_copy", "other"]
SHORT = {"correct": "C", "distractor": "D", "shot_copy": "S", "other": "O"}
COLORS = {"correct": "#2a9d8f", "distractor": "#e76f51", "shot_copy": "#e9c46a", "other": "#adb5bd"}

# total parameter counts from the Pythia paper
P1_PATTERN = re.compile(r"^(O>)?S>D>C(>D>C)?$")
COIN_MIN, COIN_TOL = 0.3, 0.05

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


def coin_flip(g):
    """Checkpoints where the model picks between the two chains about evenly."""
    c, d = g["correct"].values, g["distractor"].values
    m = (c >= COIN_MIN) & (d >= COIN_MIN) & (np.abs(c - d) < COIN_TOL)
    steps = g["step"].values[m]
    return (int(steps.min()), int(steps.max()), int(m.sum())) if m.any() else (None, None, 0)


def checkpoint_gap(g, step_a, step_b):
    """How many checkpoints apart two events are (None if either never fired)."""
    if step_a is None or step_b is None:
        return None
    steps = list(g["step"].values)
    return abs(steps.index(step_a) - steps.index(step_b))


def predictions(g, ev, stages):
    """Hit/miss for the locked predictions P1-P4 (None = cannot be scored)."""
    gap = checkpoint_gap(g, ev["S_err"][0], ev["B_acc"][0])
    t_cor = ev["T_cor"][0]
    return dict(
        P1_order=bool(P1_PATTERN.match(stages)),
        P2_err_with_acc=None if gap is None else gap <= 1,
        P3_Tcor_le_8000=None if t_cor is None else t_cor <= 8000,
        P4_final_dis_ge_015=bool(g["distractor"].values[-1] >= 0.15),
    )


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
        ev, stages = events(g), stage_string(g)
        last = g.iloc[-1]
        c0, c1, cn = coin_flip(g)
        row = dict(model=model, size=size_key(model), seed_run=is_seed_run(model), items=items,
                   task=task, n_checkpoints=len(g), last_step=int(last["step"]), stages=stages,
                   final_correct=float(last["correct"]), final_distractor=float(last["distractor"]),
                   coin_flip_first=c0, coin_flip_last=c1, coin_flip_n=cn)
        for k, (step, tok) in ev.items():
            row[f"{k}_step"], row[f"{k}_tokens"] = step, tok
        row.update(predictions(g, ev, stages))
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
                if t is not None and t > 0:
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
                 "B_acc_step", "T_cor_step", "coin_flip_first", "coin_flip_last",
                 "final_distractor"]].to_string(index=False))
        print("\nlocked predictions (scored for pythia-6.9b / pythia-12b; shown for all runs):")
        print(t[["model", "task", "P1_order", "P2_err_with_acc", "P3_Tcor_le_8000",
                 "P4_final_dis_ge_015"]].to_string(index=False))
    plot(df, args.out)
    print(f"wrote {args.out}_table.csv, {args.out}.png, {args.out}.pdf")


if __name__ == "__main__":
    main()
