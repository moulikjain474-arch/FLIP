"""How does order dependence change with model size? (exploratory, run after the full day)

Fits  flip = c * N^beta  (a straight line in log-log) on the FINAL models in the perturb
results, where flip = share of questions correct in exactly one order and N = parameters.
Prints the fit for all models and per family, so a family difference is not hidden by the
pooled line. This is a DESCRIPTIVE TREND, not a scaling law: five sizes, one run each above 410M, and
seed-to-seed variation at 410M is large (Choshen et al., ICML), so the exponent has no meaningful interval.
Models also differ in data and recipe, not only in size. Never describe the fit as a law.

  python size_fit.py --dir $PERSIST/fullday --task vartrack_d2
"""
import argparse
import glob
import json
import os

import numpy as np

PARAMS = {  # approximate total parameters
    "pythia-410m": 4.05e8, "pythia-1.4b": 1.41e9, "pythia-6.9b": 6.86e9, "pythia-12b": 1.18e10,
    "OLMo-2-0425-1B": 1.48e9, "OLMo-2-1124-7B": 7.30e9, "OLMo-2-0325-32B": 3.22e10,
    "Qwen2.5-7B": 7.62e9, "Qwen2.5-72B": 7.27e10,
}
FINAL = {"step143000", "main"}


def fit(points):
    """(beta, c, r2) of log10(flip) = log10(c) + beta * log10(N); None if < 3 usable points."""
    pts = [(n, f) for n, f in points if f > 0]
    if len(pts) < 3:
        return None
    x, y = np.log10([p[0] for p in pts]), np.log10([p[1] for p in pts])
    beta, logc = np.polyfit(x, y, 1)
    pred = logc + beta * x
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum() if len(pts) > 2 and y.std() > 0 else float("nan")
    return float(beta), float(10 ** logc), float(r2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--task", default="vartrack_d2")
    args = ap.parse_args()
    rows = []
    for f in glob.glob(os.path.join(args.dir, "perturb", "*.json")):
        s = json.load(open(f))
        m = s["model"].split("/")[-1]
        if m in PARAMS and s["revision"] in FINAL and args.task in s["tasks"]:
            sm = s["tasks"][args.task]["summary"]
            rows.append((m, PARAMS[m], sm["flip_swap"], sm["acc_single_mean_swap"]))
    rows.sort(key=lambda r: r[1])
    print(f"{'model':18s} {'params':>8s} {'flip':>6s} {'acc':>6s}   ({args.task})")
    for m, n, flip, acc in rows:
        print(f"{m:18s} {n / 1e9:7.2f}B {flip:6.3f} {acc:6.3f}")
    for name, sel in [("all models", rows)] + [(fam, [r for r in rows if r[0].lower().startswith(fam.lower())])
                                               for fam in ("pythia", "OLMo-2", "Qwen2.5")]:
        used = [(n, f) for _, n, f, _ in sel if f > 0]
        res = fit(used)
        print(f"  {name:10s}: " + ("too few models (need 3 with flip > 0)" if res is None else
              f"flip ~ {res[1]:.3g} * N^{res[0]:+.3f}   R^2 = {res[2]:.2f}   ({len(used)} models)"))


if __name__ == "__main__":
    main()
