"""What kind of mistakes does the model make on variable tracking, and how does
that change over training?

Wu, Geiger & Millière (ICML 2025) found that a small model trained from scratch
on this task passes through phases: random -> shallow heuristic -> real
dereferencing. This script checks whether PRETRAINED Pythia shows the same
phases. Each answer is classified as:
  correct     the target chain's value
  distractor  the other chain's value (followed the wrong chain)
  shot_copy   a digit that was an answer in one of the worked examples
  other       any other digit

  python errors.py --runs runs/ --out figs/error_types.csv
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import io_utils as iou  # noqa: E402

CATEGORIES = ["correct", "distractor", "shot_copy", "other"]


def classify(item, predicted):
    meta = item["meta"]
    if predicted == meta["answer"]:
        return "correct"
    if predicted == meta["distractor"]:
        return "distractor"
    worked_answers = [block.strip().split("\n")[-1] for block in item["prompt"].split("\n\n")[:-1]]
    return "shot_copy" if predicted in worked_answers else "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figs/error_types.csv")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.runs, "*__vartrack_d*.npz"))):
        if not iou.is_result(path):
            continue
        model, _rev, item_tag, task = iou.parse_result_name(path)
        d = np.load(path)
        items = iou.load_items(args.runs, item_tag)[task]
        assert [it["label"] for it in items] == d["labels"].tolist(), f"item/label mismatch in {path}"
        preds = d["logits"].argmax(1)
        cats = pd.Series([classify(it, it["candidates"][p]) for it, p in zip(items, preds)])
        share = cats.value_counts(normalize=True)
        rows.append(dict(model=model, items=item_tag, task=task, step=int(d["step"]),
                         tokens=int(d["tokens"]), **{c: float(share.get(c, 0.0)) for c in CATEGORIES}))
    if not rows:
        raise SystemExit(f"no variable-tracking results in {args.runs!r}")

    df = pd.DataFrame(rows).sort_values(["model", "items", "task", "tokens"])
    figdir = os.path.dirname(args.out) or "."
    os.makedirs(figdir, exist_ok=True)
    df.to_csv(args.out, index=False)

    for (model, item_tag, task), g in df.groupby(["model", "items", "task"]):
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.stackplot(g["tokens"].clip(lower=1e6), *[g[c] for c in CATEGORIES],
                     labels=CATEGORIES, alpha=0.85)
        ax.set(xscale="log", ylim=(0, 1), ylabel="Share of answers",
               xlabel="Training tokens (step 0 drawn at 1e6)",
               title=f"{model.split('/')[-1]} | {task}: answer types over training")
        ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, f"errors__{model.replace('/', '__')}__{item_tag}__{task}.png"),
                    dpi=200)
        plt.close(fig)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
