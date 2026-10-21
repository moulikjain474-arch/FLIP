"""Exploratory (post hoc, not a registered claim): the Order-Flip check as a free error detector.
  python3 error_catch.py --dir results_08Oct/merged
Of the questions the model gets wrong in the original order, what share is caught by also asking the swapped
order and flagging when the two answers differ in correctness (a lower bound on disagreement)? No labels needed."""
import argparse, glob, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args()
print("model | depth | wrong (orig) | caught by flip check | flagged share | P(orig wrong | flagged)")
for f in sorted(glob.glob(os.path.join(a.dir, "perturb", "*.json"))):
    d = json.load(open(f))
    if str(d.get("revision")) not in ("main", "step143000"):
        continue
    for t, v in d["tasks"].items():
        o, s = np.array(v["correct"]["orig"], bool), np.array(v["correct"]["swap"], bool)
        flagged, wrong = o != s, ~o
        caught = (wrong & flagged).sum() / max(wrong.sum(), 1)
        p = (wrong & flagged).sum() / max(flagged.sum(), 1)
        print(f"{d['model'].split('/')[-1]:24s} {t[-2:]}  {wrong.mean():6.1%}  {caught:6.1%}  {flagged.mean():6.1%}  {p:6.1%}")
