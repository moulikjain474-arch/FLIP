"""Summarise chain_probe.py results across checkpoints (exploratory, R2b).

  python chain_probe_report.py --dir $PERSIST/fullday/chain_probe
Per model and task: for each checkpoint, picks_later and the best probe accuracy (and its layer), then the
Spearman correlation across checkpoints between picks_later and the best-layer probe accuracy. A flat probe with a
swinging picks_later suggests the change is in the readout; a probe that moves with it suggests the representation moved.
Chance for the probe is max(share_second, 1 - share_second) (the majority class); the shuffled-label probe is the control.
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def step_of(rev):
    g = re.search(r"(\d+)", rev)
    return int(g.group(1)) if g else 10**9


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    root = ap.parse_args().dir
    groups = {}
    for f in glob.glob(os.path.join(root, "*.json")):
        s = json.load(open(f))
        for t, r in s["tasks"].items():
            best = max(r["layers"].items(), key=lambda kv: kv[1]["acc"])
            groups.setdefault((s["model"].split("/")[-1], t), []).append(
                (step_of(s["revision"]), s["revision"], r["picks_later"], int(best[0]), best[1]["acc"],
                 best[1]["acc_shuffled"], max(r["share_second"], 1 - r["share_second"])))
    print("== R2b (exploratory): chain-identity probe vs position preference across training")
    if not groups:
        print("  NOT RUN")
    for (m, t), pts in sorted(groups.items()):
        pts.sort()
        print(f"  {m} {t}:")
        for _, rev, pl, lay, acc, sh, maj in pts:
            print(f"     {rev:>12}  picks_later {pl if pl is None else round(pl, 3)}  best probe {acc:.3f} at layer {lay} "
                  f"(shuffled {sh:.3f}, majority {maj:.3f})")
        ok = [(p[2], p[4]) for p in pts if p[2] is not None]
        if len(ok) >= 5 and len({a for a, _ in ok}) > 1 and len({b for _, b in ok}) > 1:
            from scipy.stats import spearmanr
            r = spearmanr([a for a, _ in ok], [b for _, b in ok])
            print(f"     Spearman(picks_later, probe accuracy) across {len(ok)} checkpoints: rho={r.statistic:+.2f} p={r.pvalue:.2g}")
        else:
            print("     (fewer than 5 usable checkpoints or no variation: no correlation reported)")


if __name__ == "__main__":
    main()
