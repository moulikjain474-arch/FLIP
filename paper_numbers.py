"""Every number the synopsis, poster and video need, printed from the result files.

  python paper_numbers.py --dir MERGED_RESULTS
Sections: (1) the registered claims, hit or miss; (2) final-checkpoint accuracy with one order vs both
orders and the flip rates; (3) MMLU option-order results; (4) plain instruction vs both orders where
present. Copy numbers from here, never from memory. Rounded as in the text: 0.773 -> 77%.
"""
import argparse
import contextlib
import glob
import io
import json
import os
import re

import numpy as np

import flipstats as fs
import score_new as sn

FINAL = ("step143000", "main")


def claim_table(root):
    fns = (sn.score_c8, sn.score_c9, sn.score_c10, sn.score_c11, sn.score_controls, sn.score_layout, sn.score_benchmark,
           sn.score_olmo, sn.score_olmo_traj, sn.score_scale, sn.score_code, sn.score_c22, sn.score_c23)
    rows = []
    for fn in fns:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                fn(root)
        except Exception as e:
            rows.append((fn.__name__, f"ERROR {type(e).__name__}"))
            continue
        for line in buf.getvalue().splitlines():
            m = re.match(r"^\s+(C\d+): (HIT|MISS)(.*)$", line)
            if m:
                rows.append((m.group(1), (m.group(2) + m.group(3)).strip()))
    return rows


def headline(root):
    out = []
    for f in sorted(glob.glob(os.path.join(root, "perturb", "*.json"))):
        s = json.load(open(f))
        if s["revision"] not in FINAL:
            continue
        for t, r in sorted(s["tasks"].items()):
            sm, c, e = r["summary"], r["correct"], r["ens_correct"]
            single = (np.array(c["orig"], float) + np.array(c["swap"], float)) / 2
            g, lo, hi = fs.paired_bootstrap(e["swap"], single)
            out.append(dict(model=s["model"].split("/")[-1], revision=s["revision"], task=t, n=sm["n"],
                            single=float(single.mean()), both=float(np.mean(e["swap"])), gain=g, lo=lo, hi=hi,
                            flip_swap=sm["flip_swap"], flip_rename=sm["flip_rename"],
                            instruct=sm.get("acc_instruct_mean")))
    return out


def mmlu(root):
    out = []
    for f in sorted(glob.glob(os.path.join(root, "benchmark", "*.json"))):
        s = json.load(open(f))["summary"]
        out.append((os.path.basename(f)[:-5], s))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    root = ap.parse_args().dir
    print("== 1. Registered claims")
    for c, v in claim_table(root):
        print(f"  {c}: {v}")
    print("\n== 2. Final checkpoints: one order vs both orders (single = average of the two orders)")
    for h in headline(root):
        ins = "" if h["instruct"] is None else f" | with plain instruction {100 * h['instruct']:.1f}%"
        print(f"  {h['model']:<18} {h['task']}: single {100 * h['single']:.1f}% -> both {100 * h['both']:.1f}% "
              f"(gain {100 * h['gain']:+.1f}, 95% {100 * h['lo']:+.1f} to {100 * h['hi']:+.1f}) | answer changes: "
              f"reorder {100 * h['flip_swap']:.1f}%, rename {100 * h['flip_rename']:.1f}%{ins}  n={h['n']}")
    print("\n== 3. MMLU option order")
    for name, s in mmlu(root):
        print(f"  {name:<28} acc {100 * s['acc_orig']:.1f}%  answers change {100 * s['flip_orig_reversed']:.1f}%  "
              f"two orders {100 * s['acc_single_mean2']:.1f}% -> {100 * s['acc_ens2']:.1f}% (gain {100 * s['gain2']:+.1f}, "
              f"95% {100 * s['gain2_lo']:+.1f} to {100 * s['gain2_hi']:+.1f})")


if __name__ == "__main__":
    main()
