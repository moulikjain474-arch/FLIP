"""Score the claims registered as C8-C11 in EXPERIMENTS.md from the full-day outputs.

  python score_new.py --dir $PERSIST/fullday
Prints hit / miss for each claim with the numbers behind it. Nothing here is tuned after
the fact: the thresholds are the ones written in EXPERIMENTS.md section 4b.
"""
import argparse
import glob
import json
import os

import numpy as np


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def bias(v):
    """'L' (later-biased), 'F' (first-biased) or '-' from a position_test task entry."""
    pl = v["picks_later"]
    if not pl["n"]:
        return "-"
    lo, hi = wilson(pl["k"], pl["n"])
    return "L" if lo > 0.5 else ("F" if hi < 0.5 else "-")


def load_positions(root):
    out = {}
    for f in glob.glob(os.path.join(root, "position", "**", "*.json"), recursive=True):
        s = json.load(open(f))
        step = int(s["revision"].replace("step", ""))
        for t, v in s["tasks"].items():
            out[(s["model"].split("/")[-1], step, t)] = v
    return out


def direction_changes(seq):
    b = [x for x in seq if x != "-"]
    return sum(1 for a, c in zip(b, b[1:]) if a != c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    root = ap.parse_args().dir
    pos = load_positions(root)

    print("== C8: 12B flips direction at least twice (depth 2 or 3)")
    for m in ("pythia-12b", "pythia-6.9b"):
        for t in ("vartrack_d2", "vartrack_d3"):
            steps = sorted(s for (mm, s, tt) in pos if mm == m and tt == t)
            seq = [bias(pos[(m, s, t)]) for s in steps]
            print(f"  {m} {t}: {''.join(seq)}  changes={direction_changes(seq)}  (steps {steps})")
    hit8 = any(direction_changes([bias(pos[("pythia-12b", s, t)]) for s in sorted(s for (mm, s, tt) in pos
               if mm == "pythia-12b" and tt == t)]) >= 2 for t in ("vartrack_d2", "vartrack_d3"))
    print("  C8:", "HIT" if hit8 else "MISS")

    print("== C9: seeds disagree (>=2 L and >=2 F) at depth 3, step 16000 or 64000")
    runs = ["pythia-410m"] + [f"pythia-410m-seed{k}" for k in (1, 2, 3, 5, 6, 7, 8, 9)]
    hit9 = False
    for s in (16000, 64000):
        b = [bias(pos[(r, s, "vartrack_d3")]) for r in runs if (r, s, "vartrack_d3") in pos]
        ok = b.count("L") >= 2 and b.count("F") >= 2
        hit9 |= ok
        print(f"  step {s}: {''.join(b)}  L={b.count('L')} F={b.count('F')}  {'yes' if ok else 'no'}")
    print("  C9:", "HIT" if hit9 else "MISS")

    print("== C10: correct in exactly one order >= 0.20 at depth 2 (6.9B, 12B; 16k/32k/64k)")
    sw = {}
    for f in glob.glob(os.path.join(root, "swap", "*.json")):
        s = json.load(open(f))
        sw[(s["model"].split("/")[-1], s["step"])] = s["tasks"]
    hit10 = True
    for m in ("pythia-6.9b", "pythia-12b"):
        for st in (16000, 32000, 64000):
            r = sw.get((m, st))
            if r is None:
                print(f"  {m} {st}: MISSING"); hit10 = False; continue
            v = r["vartrack_d2"]["correct_in_exactly_one_order"]
            hit10 &= v >= 0.20
            print(f"  {m} {st}: {v:.3f}")
    print("  C10:", "HIT" if hit10 else "MISS")
    for st in (32000, 64000):
        r = sw.get(("pythia-12b", st))
        if r:
            g = [r[t]["acc_combined"] - r[t]["acc_original"] for t in ("vartrack_d1", "vartrack_d2")]
            print(f"  secondary 12B {st}: combined gain d1={g[0]:+.3f} d2={g[1]:+.3f}  {'yes' if min(g) >= 0.03 else 'no'}")

    print("== C11: 6.9B plant ignored at layers <=10 (<0.10), >=0.50 at some layer <=22")
    hit11 = True
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(root, "steer_layers", "*.json"))]
    for st in (16000, 143000):
        for t in ("vartrack_d1", "vartrack_d3"):
            pts = sorted((r["layer"], r["rates"]["plant_cor"]["planted"]) for r in rows if r["step"] == st and r["task"] == t)
            if not pts:
                print(f"  step {st} {t}: MISSING"); hit11 = False; continue
            early = max(v for L, v in pts if L <= 10)
            late = max(v for L, v in pts if L <= 22)
            ok = early < 0.10 and late >= 0.50
            hit11 &= ok
            print(f"  step {st} {t}: " + " ".join(f"L{L}={v:.2f}" for L, v in pts) + f"  {'yes' if ok else 'no'}")
    print("  C11:", "HIT" if hit11 else "MISS")


if __name__ == "__main__":
    main()
