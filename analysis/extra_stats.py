"""Descriptive order statistics from saved results (exploratory, added 8 Oct 2026).

  python extra_stats.py --dir $PERSIST/fullday [--dir $PERSIST2/fullday ...]

For every model x checkpoint x task asked in two orders it reports:
  strict    right in BOTH orders          lenient   right in AT LEAST ONE order
  both right / flip / both wrong          (flip = right in exactly one order)
  chance    flip rate expected if errors ignored order: p1(1-p2) + p2(1-p1)
  Fisher    exact p and odds ratio on the 2x2 table (odds ratio below 1: errors anti-correlated across orders)
  McNemar   (perturb results only, which keep per-question answers) two-order ensemble vs the original
            order alone: questions fixed vs broken, exact p; and swap-ensemble vs rename-ensemble.
swap_test.py results keep only rates, but the 2x2 table follows from them exactly
(both right = (acc_original + acc_swapped - flip) / 2), so they are reported too.
Nothing here changes a registered claim.
"""
import argparse
import glob
import json
import os

import numpy as np

import flipstats as fs


def _from_rates(acc_a, acc_b, flip, n):
    """Boolean arrays with exactly the counts implied by two accuracies, the flip rate and n."""
    both = (acc_a + acc_b - flip) / 2 * n
    b, c = acc_a * n - both, acc_b * n - both
    d = n - both - b - c
    both, b, c, d = (int(round(x)) for x in (both, b, c, d))
    if min(both, b, c, d) < 0 or both + b + c + d != n:
        raise ValueError(f"inconsistent rates: acc_a={acc_a} acc_b={acc_b} flip={flip} n={n}")
    ca = np.array([1] * both + [1] * b + [0] * c + [0] * d, bool)
    cb = np.array([1] * both + [0] * b + [1] * c + [0] * d, bool)
    return ca, cb


def mcnemar_pair(x, y):
    """Exact McNemar of x vs y (bool arrays): (x only right, y only right, p)."""
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    a, b = int((x & ~y).sum()), int((~x & y).sum())
    return a, b, fs.mcnemar_exact(a, b)


def rows(root):
    """(kind, model, revision, task, metric, value, n) rows."""
    for f in sorted(glob.glob(os.path.join(root, "swap", "*.json"))):
        s = json.load(open(f))
        m = s["model"].split("/")[-1]
        for t, r in s["tasks"].items():
            ca, cb = _from_rates(r["acc_original"], r["acc_swapped"], r["correct_in_exactly_one_order"], r["n"])
            for k, v in fs.order_stats(ca, cb).items():
                yield "xstat_swaptest", m, s["revision"], t, k, v, r["n"]
    for f in sorted(glob.glob(os.path.join(root, "perturb", "*.json"))):
        s = json.load(open(f))
        m = s["model"].split("/")[-1]
        for t, r in s["tasks"].items():
            c = {k: np.array(v, bool) for k, v in r["correct"].items()}
            e = {k: np.array(v, bool) for k, v in r["ens_correct"].items()}
            n = len(c["orig"])
            for other in ("swap", "rename"):
                for k, v in fs.order_stats(c["orig"], c[other]).items():
                    yield f"xstat_perturb_{other}", m, s["revision"], t, k, v, n
            fixed, broke, p = mcnemar_pair(e["swap"], c["orig"])
            yield "xstat_ens", m, s["revision"], t, "swap_ens_fixed_vs_orig", fixed, n
            yield "xstat_ens", m, s["revision"], t, "swap_ens_broke_vs_orig", broke, n
            yield "xstat_ens", m, s["revision"], t, "swap_ens_vs_orig_mcnemar_p", p, n
            a, b, p = mcnemar_pair(e["swap"], e["rename"])
            yield "xstat_ens", m, s["revision"], t, "swap_ens_only_right_vs_rename_ens", a, n
            yield "xstat_ens", m, s["revision"], t, "rename_ens_only_right_vs_swap_ens", b, n
            yield "xstat_ens", m, s["revision"], t, "swap_vs_rename_ens_mcnemar_p", p, n


def show(root):
    table = {}
    for kind, m, rev, t, k, v, n in rows(root):
        table.setdefault((kind, m, rev, t, n), {})[k] = v
    print(f"== exploratory order statistics ({root}); descriptive, not registered claims")
    print("   strict = right in both orders; lenient = right in at least one; chance = flip rate expected by independence")
    for (kind, m, rev, t, n), d in sorted(table.items()):
        if kind in ("xstat_swaptest", "xstat_perturb_swap", "xstat_perturb_rename"):
            print(f"  {kind[6:]:15} {m} {rev} {t}: strict {d['strict']:.3f} lenient {d['lenient']:.3f} | "
                  f"both-right {d['both_right']:.3f} flip {d['flip_share']:.3f} both-wrong {d['both_wrong']:.3f} | "
                  f"chance flip {d['flip_chance']:.3f} (flip-chance {d['flip_minus_chance']:+.3f}) | "
                  f"Fisher p {d['fisher_p']:.2g} OR {d['fisher_odds']:.2f}  n={n}")
        elif kind == "xstat_ens":
            print(f"  ens (perturb)   {m} {rev} {t}: swap-ensemble vs original: fixed {int(d['swap_ens_fixed_vs_orig'])} "
                  f"broke {int(d['swap_ens_broke_vs_orig'])} p={d['swap_ens_vs_orig_mcnemar_p']:.2g} | "
                  f"swap-ens vs rename-ens: {int(d['swap_ens_only_right_vs_rename_ens'])}:"
                  f"{int(d['rename_ens_only_right_vs_swap_ens'])} p={d['swap_vs_rename_ens_mcnemar_p']:.2g}  n={n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", required=True)
    for d in ap.parse_args().dir:
        show(d)
