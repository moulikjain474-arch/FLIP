"""Checkpoint ladder (claim C30): is the chain identity readable BEFORE the learned subspace is causally used?

  python3 ladder_report.py --steer $PERSIST/fullday/ladder --probe $PERSIST/fullday/ladder_probe

Pythia-6.9B, layer 22, depths 1 and 3, checkpoints 4000, 8000, 16000, 32000, 64000, 143000. Per checkpoint:
  natural accuracy      share of held-out questions answered correctly with no edit
  readable (probe)      best layer (every 4th) of the cross-validated linear probe for "which chain does the question point
                        at" (chain_probe.py), and the majority-class chance level
  used (plant)          planted-answer rate with the learned subspace minus the mean over 5 random subspaces of the
                        same rank (steer_stage.py --plant-rand), paired p
  erase                 mean change in log p(correct) when the learned subspace is erased, minus random subspaces
A checkpoint is READABLE if probe accuracy >= majority + 0.05. It is USED if plant - random >= 0.15 and paired p < 0.05
(the C24 criterion). C30 = HIT if at BOTH depths a first readable checkpoint exists, a first used checkpoint exists,
and the readable one is strictly earlier. Anything else is MISS and the table says which of: used first, same
checkpoint, readable but never used, never readable, never used. Six checkpoints give a coarse timeline: "earlier"
means earlier among these six, not at an exact step.
"""
import argparse
import glob
import json
import os
import re

STEPS = (4000, 8000, 16000, 32000, 64000, 143000)
READ_MARGIN = 0.05
USE_MARGIN = 0.15
P = 0.05


def _step(rev):
    g = re.search(r"(\d+)", str(rev))
    return int(g.group(1)) if g else None


def load_steer(root):
    """{(step, task): row} from steer_stage summaries written with --plant-rand at the ladder layer."""
    out = {}
    for f in glob.glob(os.path.join(root, "*.json")):
        s = json.load(open(f))
        if s.get("plant_rand") is None:
            continue
        step = s.get("step") if s.get("step") is not None else _step(s.get("revision"))
        d = s["plant_rand"]["plant_dis_minus_rand"]
        e = s["primary_erase"]
        out[(step, s["task"])] = dict(
            natural=s["rates"]["natural"]["correct"], rank=s.get("rank_primary", 0), layer=s["layer"], n=s["n_test"],
            plant_diff=d["mean_diff"], plant_p=d["p"], erase_diff=e["mean_diff"], erase_p=e["p"])
    return out


def load_probe(root):
    """{(step, task): row} from chain_probe.py summaries: best-layer probe accuracy, chance, picks_later."""
    out = {}
    for f in glob.glob(os.path.join(root, "*.json")):
        s = json.load(open(f))
        step = _step(s["revision"])
        for t, r in s["tasks"].items():
            layer, v = max(r["layers"].items(), key=lambda kv: kv[1]["acc"])
            out[(step, t)] = dict(acc=v["acc"], layer=int(layer), chance=max(r["share_second"], 1 - r["share_second"]),
                                  picks_later=r["picks_later"])
    return out


def readable(p):
    return p is not None and p["acc"] >= p["chance"] + READ_MARGIN


def used(s):
    return s is not None and s["rank"] > 0 and s["plant_diff"] >= USE_MARGIN and s["plant_p"] == s["plant_p"] and s["plant_p"] < P


def first(steps, pred):
    for st in sorted(steps):
        if pred(st):
            return st
    return None


def outcome(s_read, s_use):
    if s_read is None and s_use is None:
        return "never readable, never used"
    if s_read is None:
        return "used but never readable"
    if s_use is None:
        return "readable but never used"
    if s_read < s_use:
        return "readable first"
    if s_read == s_use:
        return "same checkpoint"
    return "used first"


def evaluate(steer, probe, depths=(1, 3)):
    """({depth: (first readable step, first used step, outcome)}, verdict). Missing checkpoints count as not readable/used."""
    res = {}
    for d in depths:
        t = f"vartrack_d{d}"
        s_read = first(STEPS, lambda st: readable(probe.get((st, t))))
        s_use = first(STEPS, lambda st: used(steer.get((st, t))))
        res[d] = (s_read, s_use, outcome(s_read, s_use))
    hit = all(o == "readable first" for _, _, o in res.values())
    seen = {st for (st, _) in steer} | {st for (st, _) in probe}
    if not seen:
        return res, "NOT RUN"
    missing = [st for st in STEPS if st not in seen]
    return res, ("HIT" if hit else "MISS") + (f" (checkpoints not run: {missing})" if missing else "")


def show(steer, probe):
    print("step | depth | natural acc | probe acc (chance) @layer | picks later | plant-random (p) | erase-random dlogp (p)")
    for st in STEPS:
        for d in (1, 3):
            t = f"vartrack_d{d}"
            s, p = steer.get((st, t)), probe.get((st, t))
            ps = f"{p['acc']:.3f} ({p['chance']:.3f}) L{p['layer']}" if p else "-"
            pl = f"{p['picks_later']:.2f}" if p and p["picks_later"] is not None else "-"
            nat = f"{s['natural']:.3f}" if s else "-"
            pla = f"{s['plant_diff']:+.3f} (p={s['plant_p']:.2g})" if s else "-"
            era = f"{s['erase_diff']:+.3f} (p={s['erase_p']:.2g})" if s else "-"
            print(f"{st} | d{d} | {nat} | {ps} | {pl} | {pla} | {era}")
    res, verdict = evaluate(steer, probe)
    for d, (a, b, o) in res.items():
        print(f"  depth {d}: first readable {a}, first used {b} -> {o}")
    print("  C30:", verdict)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steer", required=True)
    ap.add_argument("--probe", required=True)
    a = ap.parse_args()
    show(load_steer(a.steer), load_probe(a.probe))


if __name__ == "__main__":
    main()
