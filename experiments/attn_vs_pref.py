"""Does the last token's attention move with the position preference across training? (exploratory, J2)

  python attn_vs_pref.py --dir $PERSIST/fullday
Joins, per model x checkpoint x task, the position preference from position_test.py (share of
non-tied answers that take the LATER chain's value, picks_later) with the attention from
attention_test.py (attention on the second chain minus the first, averaged over all layers and heads).
Per model and task it reports the Spearman rank correlation across checkpoints (needs >= 5 points)
and how often the two agree on direction (picks_later above 0.5 vs attention above 0).
A weak or negative correlation is a result too: it means the reversals are NOT visible in where the
last token looks. Nothing here is a registered claim.
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def load(root):
    pos, att = {}, {}
    for f in glob.glob(os.path.join(root, "position", "**", "*.json"), recursive=True):
        s = json.load(open(f))
        m = s["model"].split("/")[-1]
        for t, v in s["tasks"].items():
            if v["picks_later"]["n"]:
                pos[(m, s["revision"], t)] = v["picks_later"]["rate"]
    for f in glob.glob(os.path.join(root, "attention", "*.json")):
        s = json.load(open(f))
        m = s["model"].split("/")[-1]
        for t, r in s["tasks"].items():
            att[(m, s["revision"], t)] = float(np.mean(r["layer_later_minus_earlier"]))
    return pos, att


def step_of(rev):
    g = re.search(r"(\d+)", rev)
    return int(g.group(1)) if g else 10**9


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def analyse(pos, att):
    groups = {}
    for key in sorted(set(pos) & set(att)):
        groups.setdefault((key[0], key[2]), []).append((step_of(key[1]), key[1], pos[key], att[key]))
    out = []
    for (m, t), pts in sorted(groups.items()):
        pts.sort()
        pl, at = [p[2] for p in pts], [p[3] for p in pts]
        agree = sum((a > 0.5) == (b > 0) for a, b in zip(pl, at) if a != 0.5 and b != 0)
        den = sum(1 for a, b in zip(pl, at) if a != 0.5 and b != 0)
        rho, p = spearman(pl, at) if len(pts) >= 5 and len(set(pl)) > 1 and len(set(at)) > 1 else (float("nan"), float("nan"))
        out.append(dict(model=m, task=t, n=len(pts), rho=rho, p=p, agree=agree, agree_of=den, points=pts))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    root = ap.parse_args().dir
    res = analyse(*load(root))
    print("== J2 (exploratory): does attention to the later chain track the preference for the later chain across training?")
    if not res:
        print("  NOT RUN (no matching position + attention results)")
    for r in res:
        print(f"  {r['model']} {r['task']}: n={r['n']} checkpoints  Spearman rho={r['rho']:+.2f} p={r['p']:.2g}  "
              f"direction agrees {r['agree']}/{r['agree_of']}")
        for st, rev, a, b in r["points"]:
            print(f"      {rev:>12}  picks_later {a:.3f}  attention later-first {b:+.4f}")


if __name__ == "__main__":
    main()
