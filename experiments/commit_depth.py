"""Where is the choice committed, as a fraction of model depth? (exploratory, Q2 / mentor R1)

  python commit_depth.py --dir $PERSIST/fullday [--dir ...]
From patch_test.py results (patch/ and patch_choice/): the first patched layer at which at least half of the
questions on which the two orders disagree take the swapped order's answer ("transfer" >= 0.5), as a share of
the model's layers. Also reports the single-chain control (a VALUE read-out) if present. Only layers that were
actually patched count, so the result is a layer GRID value (see `every`), not an exact layer.
A roughly constant fraction across sizes would support "committed at a similar relative depth"; a drifting one
is a limitation. Not a registered claim.
"""
import argparse
import glob
import json
import os


def first_layer(layers, key="transfer", thr=0.5):
    for l in sorted(layers, key=int):
        v = layers[l].get(key)
        if v is not None and v >= thr:
            return int(l)
    return None


def rows(root):
    for sub in ("patch", "patch_choice"):
        for f in sorted(glob.glob(os.path.join(root, sub, "*.json"))):
            s = json.load(open(f))
            L = s["n_layers"]
            for t, r in s["tasks"].items():
                a = first_layer(r["layers"])
                c = first_layer(r["control"]["layers"]) if r.get("control") else None
                grid = sorted(int(x) for x in r["layers"])
                step = min(b - a for a, b in zip(grid, grid[1:])) if len(grid) > 1 else 1
                yield dict(source=sub, model=s["model"].split("/")[-1], revision=s["revision"], task=t, n_layers=L,
                           choice_layer=a, choice_frac=(a / L if a else None), value_layer=c,
                           value_frac=(c / L if c else None), grid=step, n_differ=r["n_differ"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", required=True)
    print("== commit depth (exploratory): first patched layer with transfer >= 0.5, and as a fraction of depth")
    for root in ap.parse_args().dir:
        for r in rows(root):
            fmt = lambda x: "none" if x is None else f"{x:.2f}"
            print(f"  [{r['source']}] {r['model']} {r['revision']} {r['task']}: choice layer {r['choice_layer']}/{r['n_layers']} "
                  f"(= {fmt(r['choice_frac'])} of depth)  single-chain value layer {r['value_layer']} "
                  f"(= {fmt(r['value_frac'])})  grid {r['grid']}  n_differ {r['n_differ']}"
                  + ("  [few questions differ: unreliable]" if r['n_differ'] < 30 else ""))


if __name__ == "__main__":
    main()
