"""Random-subspace plant control (claim C24): does planting work because of the learned subspace?

  python plant_control_report.py --dir $PERSIST/fullday/plant_rand

Reads steer_stage.py results written with --plant-rand (patch_plant_rand.py adds the flag). For each run it prints
the share of answers equal to the planted number: with no edit (natural), planting the distractor class means in
the learned primary subspace (plant_dis), and planting the same values inside random subspaces of the same rank
(plant_rand, mean over draws), plus the paired difference plant_dis - plant_rand.

C24 (see EXPERIMENTS.md section 4g): Pythia-6.9B step 143000, layer 22, depths 1 and 3: plant_dis - plant_rand
>= 0.15 at both depths and paired p < 0.05 at both. Other layers and models are descriptive.
"""
import argparse
import glob
import json
import os


def rows(summary):
    """One printable row from a steer_stage summary written with --plant-rand; None if the run has no control."""
    pr = summary.get("plant_rand")
    if pr is None:
        return None
    r = summary["rates"]
    d = pr["plant_dis_minus_rand"]
    return dict(model=summary["model"].split("/")[-1], step=summary.get("step"), task=summary["task"],
                layer=summary["layer"], n=summary["n_test"],
                natural=r["natural"]["planted"], plant_dis=r["plant_dis"]["planted"], plant_rand=pr["rate"],
                diff=d["mean_diff"], p=d["p"])


def c24(table):
    """C24 verdict from the rows: needs Pythia-6.9B step 143000 layer 22 at depth 1 and depth 3."""
    need = {}
    for x in table:
        if x["model"] == "pythia-6.9b" and x["step"] == 143000 and x["layer"] == 22:
            need[x["task"]] = x
    if not {"vartrack_d1", "vartrack_d3"} <= set(need):
        return "NOT RUN (needs pythia-6.9b step 143000 layer 22, depths 1 and 3)"
    ok = all(need[t]["diff"] >= 0.15 and need[t]["p"] < 0.05 for t in ("vartrack_d1", "vartrack_d3"))
    return "HIT" if ok else "MISS"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    table = []
    for f in sorted(glob.glob(os.path.join(a.dir, "*.json"))):
        x = rows(json.load(open(f)))
        if x:
            table.append(x)
    if not table:
        print("no --plant-rand results in", a.dir)
        return
    print("model | step | task | layer | natural | plant (learned subspace) | plant (random subspaces) | difference | p")
    for x in sorted(table, key=lambda x: (x["model"], x["task"], x["layer"])):
        print(f"{x['model']} {x['step']} {x['task'][-2:]} L{x['layer']:>2}  natural {x['natural']:.3f}  "
              f"learned {x['plant_dis']:.3f}  random {x['plant_rand']:.3f}  diff {x['diff']:+.3f}  p={x['p']:.2g}  (n={x['n']})")
    print("C24:", c24(table))


if __name__ == "__main__":
    main()
