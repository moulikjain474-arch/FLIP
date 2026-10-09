"""Selection-matched null for the layer sweep (audit A1, defined after the data were collected, before they were scored).

  python3 selection_null.py --dir $PERSIST/fullday/plant_rand

The plant results report the best of several layers. A random subspace gets the same search: for each random draw k
we take its best layer, exactly as the learned subspace gets its best layer. The learned subspace has to beat the
best layer of EVERY random draw, not just a random draw at the learned subspace's chosen layer.

Per (model, step, task) this prints, over the layers present:
  learned best      max over layers of the planted rate with the learned subspace (and which layer)
  random best       for each draw, max over layers of its planted rate; the mean and the largest are shown
  margin            learned best - largest random best          (audit passes at >= 0.15, the C24 margin)
  same-layer margin learned - largest random draw at the same layer, per layer
With K draws the smallest possible exact permutation p-value is 1/(K+1) (5 draws: 0.167), so the margin and the
"all draws below" statement are the evidence here, not a p-value.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

MARGIN = 0.15


def collect(root):
    """{(model, step, task): {layer: {"learned": rate, "draws": [rates]}}} from steer_stage --plant-rand summaries."""
    out = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(root, "*.json"))):
        s = json.load(open(f))
        if s.get("plant_rand") is None or "rates" not in s:
            continue
        r = s["rates"]
        draws = [r[k]["planted"] for k in sorted(r) if k.startswith("plant_rand") and k != "plant_rand_mean"
                 and isinstance(r[k], dict) and "planted" in r[k]]
        key = (s["model"].split("/")[-1], s.get("step"), s["task"])
        out[key][int(s["layer"])] = dict(learned=float(r["plant_dis"]["planted"]), draws=[float(d) for d in draws],
                                         natural=float(r["natural"]["planted"]))
    return out


def audit(layers):
    """Selection-matched comparison for one (model, step, task). `layers` = {layer: {learned, draws, natural}}."""
    ls = sorted(layers)
    k = min(len(layers[l]["draws"]) for l in ls)
    if k == 0:
        return None
    best_l = max(ls, key=lambda l: layers[l]["learned"])
    learned_best = layers[best_l]["learned"]
    random_best = [max(layers[l]["draws"][i] for l in ls) for i in range(k)]
    same_layer = {l: layers[l]["learned"] - max(layers[l]["draws"][:k]) for l in ls}
    return dict(layers=ls, n_draws=k, learned_layer=best_l, learned_best=learned_best,
                random_best=random_best, random_best_mean=sum(random_best) / k, random_best_max=max(random_best),
                margin=learned_best - max(random_best), margin_vs_mean=learned_best - sum(random_best) / k,
                same_layer_margin=same_layer, natural_at_best=layers[best_l]["natural"],
                all_draws_below=all(rb < learned_best for rb in random_best))


def verdict(results, models=("pythia-6.9b",), tasks=("vartrack_d1", "vartrack_d3")):
    """PASS if, for the named models, the margin is >= MARGIN at every named depth; NOT RUN if one is missing."""
    got = {}
    for (m, step, t), a in results.items():
        if m in models and t in tasks and step == 143000 and a is not None:
            got[t] = a
    if set(tasks) - set(got):
        return "NOT RUN (needs pythia-6.9b step 143000 at depths 1 and 3, several layers)"
    return "PASS" if all(got[t]["margin"] >= MARGIN for t in tasks) else "FAIL"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    data = collect(a.dir)
    if not data:
        print("no --plant-rand results in", a.dir)
        return
    results = {k: audit(v) for k, v in data.items()}
    for (m, step, t), r in sorted(results.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
        if r is None:
            print(f"{m} {step} {t}: no random draws")
            continue
        print(f"{m} step {step} {t}: layers {r['layers']}, {r['n_draws']} draws")
        print(f"  learned best {r['learned_best']:.3f} at layer {r['learned_layer']} (natural there {r['natural_at_best']:.3f})")
        print("  random best per draw: " + ", ".join(f"{x:.3f}" for x in r["random_best"])
              + f"   mean {r['random_best_mean']:.3f}  largest {r['random_best_max']:.3f}")
        print(f"  margin vs the largest random best {r['margin']:+.3f}; vs the mean {r['margin_vs_mean']:+.3f}; "
              f"all draws below the learned best: {r['all_draws_below']}")
        print("  same-layer margin (learned - largest random draw): "
              + ", ".join(f"L{l}: {v:+.3f}" for l, v in sorted(r["same_layer_margin"].items())))
    print("A1 (selection-matched, pythia-6.9b step 143000, depths 1 and 3, margin >= 0.15):", verdict(results))


if __name__ == "__main__":
    main()
