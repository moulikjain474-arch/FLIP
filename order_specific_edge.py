"""Both-orders ensemble vs the equal-cost control (original + renamed), final checkpoints.
  python order_specific_edge.py --dir results_08Oct/merged
Edge = accuracy(both orders) - accuracy(original + renamed); exact McNemar on the two ensembles."""
import argparse, glob, json, os
import numpy as np
import flipstats as fs

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
a = ap.parse_args()
print("model | task | single(orig) | both-orders | orig+rename | edge (pts) | McNemar p")
for f in sorted(glob.glob(os.path.join(a.dir, "perturb", "*.json"))):
    d = json.load(open(f))
    if str(d.get("revision")) not in ("main", "step143000"):
        continue
    for t, v in d["tasks"].items():
        s, e = v["summary"], v["ens_correct"]
        x, y = np.array(e["swap"]).astype(bool), np.array(e["rename"]).astype(bool)
        oa, ob = int((x & ~y).sum()), int((~x & y).sum())
        print(f"{d['model'].split('/')[-1]:24s} {t[-2:]}  {100*s['acc_orig']:5.1f}%  {100*s['acc_ens_swap']:5.1f}%  "
              f"{100*s['acc_ens_rename']:5.1f}%  {100*(s['acc_ens_swap']-s['acc_ens_rename']):+5.1f}   "
              f"p={fs.mcnemar_exact(oa, ob):.3g}  (only-orders {oa}, only-rename {ob})")
