"""Plant specificity and rank sweep (exploratory, P3 and P4).

  python plant_margin.py --dir $PERSIST/fullday/steer_items
Reads steer_stage.py results written with --save-items / --rank-sweep (patch_steer_stage.py adds the flags).

P3  Does planting flip near-ties more than confident questions? Confidence = the model's natural
    log p(answer) - log p(distractor); near-tie = small |confidence|. Items are split into 4 equal bins of
    |confidence|. Per bin: share answering the planted value without and with planting, and the difference.
    Prediction under the thesis: the effect is largest in the lowest bin. Also the Spearman correlation
    between |confidence| and planting success (negative = near-ties are easier to flip).
P4  Plant rate as a function of how many directions are overwritten (top-r components, r = 1..rank).
    A small r that already works supports "the commitment is low-dimensional". The matching natural rate
    and the full-rank rate are printed for reference.
Not registered claims.
"""
import argparse
import glob
import json
import os

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(mid - half), float(mid + half)


def bins_of(margin, k=4):
    """Index 0..k-1 of equal-count bins of |margin|, lowest |margin| first (ties broken by order)."""
    order = np.argsort(np.abs(margin), kind="stable")
    b = np.empty(len(margin), int)
    for i, chunk in enumerate(np.array_split(order, k)):
        b[chunk] = i
    return b


def analyse_items(it, k=4):
    m = np.asarray(it["margin_natural"], float)
    nat = np.asarray(it["planted_natural"], float)
    pl = np.asarray(it["planted_plant_dis"], float)
    b = bins_of(m, k)
    rows = []
    for i in range(k):
        s = b == i
        rows.append(dict(bin=i, n=int(s.sum()), abs_margin_lo=float(np.abs(m[s]).min()), abs_margin_hi=float(np.abs(m[s]).max()),
                         natural=float(nat[s].mean()), planted=float(pl[s].mean()),
                         effect=float(pl[s].mean() - nat[s].mean()), ci=wilson(int(pl[s].sum()), int(s.sum()))))
    rho = p = float("nan")
    if len(set(np.abs(m))) > 1 and len(set(pl)) > 1:
        from scipy.stats import spearmanr
        r = spearmanr(np.abs(m), pl)
        rho, p = float(r.statistic), float(r.pvalue)
    return rows, rho, p


def rank_rows(rates):
    out = []
    r = 1
    while f"plant_dis_orth_r{r}" in rates:
        out.append((r, rates[f"plant_dis_orth_r{r}"]["planted"]))
        r += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    root = ap.parse_args().dir
    files = sorted(glob.glob(os.path.join(root, "*.json")))
    if not files:
        print("== P3/P4: NOT RUN (no files)")
    for f in files:
        s = json.load(open(f))
        head = f"{s['model'].split('/')[-1]} {s['revision']} {s['task']} layer {s['layer']} (n_test {s['n_test']})"
        if "items" in s:
            rows, rho, p = analyse_items(s["items"])
            print(f"== P3 {head}: Spearman(|confidence|, planted) rho={rho:+.2f} p={p:.2g}  (negative = near-ties flip more)")
            for r in rows:
                print(f"   bin {r['bin']} |conf| {r['abs_margin_lo']:.2f}-{r['abs_margin_hi']:.2f} n={r['n']}: planted without {r['natural']:.3f}, "
                      f"with plant {r['planted']:.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}]  effect {r['effect']:+.3f}")
        rr = rank_rows(s["rates"])
        if rr:
            full = s["rates"]["plant_dis_orth"]["planted"]
            nat = s["rates"]["natural"]["planted"]
            print(f"== P4 {head}: natural {nat:.3f}, full primary subspace (rank {s['rank_primary']}) {full:.3f}")
            print("   top-r directions: " + "  ".join(f"r={r}: {v:.3f}" for r, v in rr))


if __name__ == "__main__":
    main()
