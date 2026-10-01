"""Score the claims registered as C8-C17 in EXPERIMENTS.md from the full-day outputs.

  python score_new.py --dir $PERSIST/fullday
Prints hit / miss for each claim with the numbers behind it. Nothing here is tuned after
the fact: the thresholds are the ones written in EXPERIMENTS.md section 4b.
"""
import argparse
import glob
import json
import os

import numpy as np

import flipstats as fs


def ge(x, thr):
    """x >= thr, tolerant of float rounding (rates are k/n, thresholds are decimals)."""
    return x >= thr - 1e-9


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


def load_positions(root, sub="position"):
    out = {}
    for f in glob.glob(os.path.join(root, sub, "**", "*.json"), recursive=True):
        s = json.load(open(f))
        step = int(s["revision"].replace("step", ""))
        for t, v in s["tasks"].items():
            out[(s["model"].split("/")[-1], step, t)] = v
    return out


def direction_changes(seq):
    b = [x for x in seq if x != "-"]
    return sum(1 for a, c in zip(b, b[1:]) if a != c)


def load_perturb(root):
    """{(model, revision): {task: entry}} from perturb_test.py outputs."""
    out = {}
    for f in glob.glob(os.path.join(root, "perturb", "*.json")):
        s = json.load(open(f))
        out[(s["model"].split("/")[-1], s["revision"])] = s["tasks"]
    return out


def score_controls(root):
    pert = load_perturb(root)
    print("== C12: reordering flips more answers than renaming (12B, depth 2, steps 32000/64000/143000)")
    wins = 0
    for m in ("pythia-12b", "pythia-6.9b"):
        for st in (32000, 64000, 143000):
            r = pert.get((m, f"step{st}"))
            if r is None:
                print(f"  {m} {st}: MISSING"); continue
            s = r["vartrack_d2"]["summary"]
            diff = s["flip_swap"] - s["flip_rename"]
            if m == "pythia-12b":
                wins += ge(diff, 0.05)
            print(f"  {m} {st}: flip_swap={s['flip_swap']:.3f} flip_rename={s['flip_rename']:.3f} diff={diff:+.3f}"
                  f"{'  (primary)' if m == 'pythia-12b' else '  (secondary)'}")
    print("  C12:", "HIT" if wins >= 2 else "MISS", f"({wins} of 3 primary cells with diff >= 0.05)")

    print("== C13: ordering ensemble beats the equal-cost rename ensemble (12B, depth 1 and 2)")
    hit13 = True
    for st in (143000, 64000, 32000):
        r = pert.get(("pythia-12b", f"step{st}"))
        if r is None:
            print(f"  12B {st}: MISSING"); hit13 &= st != 143000; continue
        g = [r[t]["summary"]["acc_ens_swap"] - r[t]["summary"]["acc_ens_rename"] for t in ("vartrack_d1", "vartrack_d2")]
        ok = ge(min(g), 0.03)
        if st == 143000:
            hit13 &= ok
        print(f"  12B {st}: orig+swap minus orig+rename d1={g[0]:+.3f} d2={g[1]:+.3f}  {'yes' if ok else 'no'}"
              f"{'  (primary)' if st == 143000 else '  (secondary)'}")
    print("  C13:", "HIT" if hit13 else "MISS")

    print("== C17: both-orders gain over the single-order average has a 95% interval above 0 (12B final, d1 and d2)")
    r = pert.get(("pythia-12b", "step143000"))
    hit17 = r is not None
    for t in ("vartrack_d1", "vartrack_d2"):
        if r is None:
            print("  MISSING"); break
        c = r[t]["correct"]
        single = (np.array(c["orig"], float) + np.array(c["swap"])) / 2
        m_, lo, hi = fs.paired_bootstrap(r[t]["ens_correct"]["swap"], single)
        hit17 &= lo > 0
        print(f"  {t}: gain {m_:+.3f}  95% interval [{lo:+.3f}, {hi:+.3f}]")
    print("  C17:", "HIT" if hit17 else "MISS")


def score_layout(root):
    print("== C14: the position shortcut also appears with interleaved chains (depth 3, final checkpoints)")
    pos = load_positions(root, "position_interleaved")
    biased = 0
    for m in ("pythia-410m", "pythia-1.4b", "pythia-6.9b"):
        v = pos.get((m, 143000, "vartrack_d3"))
        if v is None:
            print(f"  {m}: MISSING"); continue
        b = bias(v)
        biased += b != "-"
        pl = v["picks_later"]
        print(f"  {m}: picks_later={pl['rate']:.3f} (n={pl['n']})  {b}")
    print("  C14:", "HIT" if biased >= 2 else "MISS", f"({biased} of 3 biased)")


def score_benchmark(root):
    print("== C15: the Order-Flip Test finds order dependence on a real benchmark (MMLU option order)")
    flips, gains = 0, True
    want = ("OLMo-2-1124-7B", "Qwen2.5-7B", "Mistral-7B-v0.1")
    files = [os.path.join(root, "benchmark", f"{m}.json") for m in want]
    files = [f for f in files if os.path.exists(f)]
    for f in files:
        s = json.load(open(f))["summary"]
        name = os.path.basename(f)[:-5]
        flips += ge(s["flip_orig_reversed"], 0.10)
        gains &= s["acc_ens2"] > s["acc_single_mean2"]
        print(f"  {name}: acc={s['acc_orig']:.3f} flip={s['flip_orig_reversed']:.3f} "
              f"2-order {s['acc_single_mean2']:.3f}->{s['acc_ens2']:.3f} (gain {s['gain2']:+.3f} "
              f"[{s['gain2_lo']:+.3f},{s['gain2_hi']:+.3f}])  4-order {s['acc_mean4']:.3f}->{s['acc_ens4']:.3f}")
    ok = len(files) == 3 and flips >= 2 and gains
    print("  C15:", "HIT" if ok else "MISS", f"({len(files)} models, {flips} with flip >= 0.10, gain in all: {gains})")


def score_olmo(root):
    print("== C16: the order effect appears in a second model family (OLMo-2, depth 3)")
    pert = load_perturb(root)
    hit = False
    for m in ("OLMo-2-0425-1B", "OLMo-2-1124-7B"):
        r = pert.get((m, "main"))
        if r is None:
            print(f"  {m}: MISSING"); continue
        s = r["vartrack_d3"]["summary"]
        gain = s["acc_ens_swap"] - s["acc_single_mean_swap"]
        ok = ge(s["flip_swap"], 0.10) and ge(gain, 0.02)
        hit |= ok
        print(f"  {m}: acc={s['acc_orig']:.3f} flip_swap={s['flip_swap']:.3f} flip_rename={s['flip_rename']:.3f} "
              f"gain={gain:+.3f}  {'yes' if ok else 'no'}")
    print("  C16:", "HIT" if hit else "MISS")


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
            hit10 &= ge(v, 0.20)
            print(f"  {m} {st}: {v:.3f}")
    print("  C10:", "HIT" if hit10 else "MISS")
    for st in (32000, 64000):
        r = sw.get(("pythia-12b", st))
        if r:
            g = [r[t]["acc_combined"] - r[t]["acc_original"] for t in ("vartrack_d1", "vartrack_d2")]
            print(f"  secondary 12B {st}: combined gain d1={g[0]:+.3f} d2={g[1]:+.3f}  {'yes' if ge(min(g), 0.03) else 'no'}")

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
            ok = early < 0.10 - 1e-9 and ge(late, 0.50)
            hit11 &= ok
            print(f"  step {st} {t}: " + " ".join(f"L{L}={v:.2f}" for L, v in pts) + f"  {'yes' if ok else 'no'}")
    print("  C11:", "HIT" if hit11 else "MISS")
    for fn in (score_controls, score_layout, score_benchmark, score_olmo):
        fn(root)


if __name__ == "__main__":
    main()
