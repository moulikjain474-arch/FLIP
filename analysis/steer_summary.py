"""Collect every steer.py result into one table, with Holm correction.

Each causal test is run at several (model, checkpoint, task) settings, so the
p-values are corrected for multiple comparisons (Holm-Bonferroni) within each
test type before anything is called significant.

  python steer_summary.py --dir steer_results --out figs/steer_summary.csv
"""
import argparse
import glob
import json
import os

import pandas as pd

TESTS = ["suppress_vs_random", "anova", "enhance_vs_random", "wrong_vs_natural"]  # first = primary (H3)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input). Missing p-values count as 1."""
    pvals = [1.0 if p is None or p != p else float(p) for p in pvals]  # p != p catches NaN
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="steer_results")
    ap.add_argument("--out", default="figs/steer_summary.csv")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        with open(path) as f:
            s = json.load(f)
        row = {k: s[k] for k in ("model", "step", "task", "layer", "scope", "alpha", "n_test", "chance")}
        row.update({f"acc_{c}": v for c, v in s["acc"].items()})
        row["partial_eta2"] = s["anova"]["partial_eta2"]
        row["suppress_minus_random_logp"] = s["suppress_vs_random"]["mean_diff"]
        row.update({f"p_{t}": s[t]["p"] for t in TESTS})
        rows.append(row)
    if not rows:
        raise SystemExit(f"no steer results in {args.dir!r}; run steer.py first")

    df = pd.DataFrame(rows).sort_values(["model", "task", "step"])
    for t in TESTS:
        df[f"p_{t}_holm"] = holm(df[f"p_{t}"].tolist())
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
