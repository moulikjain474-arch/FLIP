"""Score the claims registered as C8-C20, C22 and C23 in EXPERIMENTS.md from the run outputs.

  python score_new.py --dir $PERSIST/fullday
Prints hit / miss for each claim with the numbers behind it. Nothing here is tuned after
the fact: the thresholds are the ones written in EXPERIMENTS.md sections 4b, 4c and 4d.
"""
import argparse
import glob
import re
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
        step = int(re.search(r"step(\d+)", s["revision"]).group(1)) if "step" in s["revision"] else 10**9
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
                  f" tokens(swap-orig)={s.get('tokens_swap_minus_orig', float('nan')):+.2f}"
                  f" tokens(rename-orig)={s.get('tokens_rename_minus_orig', float('nan')):+.2f}"
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
    print("-- exploratory: plain instruction vs asking in both orders (12B final)")
    if r is not None:
        for t in ("vartrack_d1", "vartrack_d2", "vartrack_d3"):
            s = r[t]["summary"]
            if "acc_instruct_mean" in s:
                print(f"  {t}: single order {s['acc_single_mean_swap']:.3f} | with instruction {s['acc_instruct_mean']:.3f}"
                      f" (flip {s['flip_instruct']:.3f} vs {s['flip_swap']:.3f}) | both orders {s['acc_ens_swap']:.3f}")


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
        rate = "NA" if pl["rate"] is None else f"{pl['rate']:.3f}"
        print(f"  {m}: picks_later={rate} (n={pl['n']})  {b}")
    print("  C14:", "HIT" if biased >= 2 else "MISS", f"({biased} of 3 biased)")


def score_benchmark(root):
    print("== C15: the Order-Flip Test finds order dependence on a real benchmark (MMLU option order)")
    flips, gains = 0, True
    third = "Mistral-7B-v0.1" if os.path.exists(os.path.join(root, "benchmark", "Mistral-7B-v0.1.json")) else "falcon-7b"
    want = ("OLMo-2-1124-7B", "Qwen2.5-7B", third)
    files = [os.path.join(root, "benchmark", f"{m}.json") for m in want]
    for m, f in zip(want, files):
        if not os.path.exists(f):
            print(f"  {m}: MISSING")
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


def score_olmo_traj(root):
    print("== C18: OLMo-2-7B position habit changes direction during training (depth 2 or 3)")
    pos = load_positions(root, "position_olmo")
    hit = False
    for t in ("vartrack_d2", "vartrack_d3"):
        steps = sorted(s for (m, s, tt) in pos if m == "OLMo-2-1124-7B" and tt == t)
        seq = [bias(pos[("OLMo-2-1124-7B", s, t)]) for s in steps]
        ch = direction_changes(seq)
        hit |= ch >= 1
        print(f"  {t}: {''.join(seq) or 'MISSING'}  changes={ch}  ({len(steps)} checkpoints: {steps})")
    print("  C18:", "HIT" if hit else "MISS")


def score_scale(root):
    print("== C19: order dependence at 32B-72B scale (depth 3, flip >= 0.05 in at least one model)")
    pert = load_perturb(root)
    hit = False
    for m in ("OLMo-2-0325-32B", "Qwen2.5-72B"):
        r = pert.get((m, "main"))
        if r is None:
            print(f"  {m}: MISSING"); continue
        for t in ("vartrack_d1", "vartrack_d2", "vartrack_d3"):
            s = r[t]["summary"]
            print(f"  {m} {t}: acc={s['acc_orig']:.3f} flip_swap={s['flip_swap']:.3f} flip_rename={s['flip_rename']:.3f} "
                  f"both-orders {s['acc_single_mean_swap']:.3f}->{s['acc_ens_swap']:.3f}")
        hit |= ge(r["vartrack_d3"]["summary"]["flip_swap"], 0.05)
        b = os.path.join(root, "benchmark", f"{m}.json")
        if os.path.exists(b):
            bs = json.load(open(b))["summary"]
            print(f"  {m} MMLU: acc={bs['acc_orig']:.3f} flip={bs['flip_orig_reversed']:.3f} 2-order gain {bs['gain2']:+.3f}")
    print("  C19:", "HIT" if hit else "MISS")


def score_code(root):
    print("== C20: order dependence on short Python programs (level 2; models with accuracy >= 0.30)")
    eligible = flipped = 0
    for m, rev in C20_MODELS:
        f = os.path.join(root, "code", f"{m}__{rev}.json")
        if not os.path.exists(f):
            print(f"  {m} {rev}: MISSING"); continue
        s = json.load(open(f))
        for t in ("code_l1", "code_l2", "code_l3"):
            r = s["tasks"].get(t)
            if r:
                print(f"  {m} {t}: acc={r['acc_single_mean']:.3f} flip={r['flip']:.3f} "
                      f"both-orders {r['acc_single_mean']:.3f}->{r['acc_both']:.3f} [{r['gain_lo']:+.3f},{r['gain_hi']:+.3f}]")
        r = s["tasks"].get("code_l2")
        if r and ge(r["acc_single_mean"], 0.30):
            eligible += 1
            flipped += ge(r["flip"], 0.10)
    ok = eligible >= 2 and flipped * 2 >= eligible
    print("  C20:", "HIT" if ok else "MISS", f"({flipped} of {eligible} eligible models with flip >= 0.10)")


C12_STEPS = (32000, 64000, 143000)
C20_MODELS = [("pythia-1.4b", "step143000"), ("pythia-6.9b", "step143000"), ("pythia-12b", "step143000"),
              ("OLMo-2-1124-7B", "main"), ("Qwen2.5-7B", "main"), ("OLMo-2-0325-32B", "main"), ("Qwen2.5-72B", "main")]


def score_c8(root):
    pos = load_positions(root)
    print("== C8: 12B flips direction at least twice (depth 2 or 3)")
    want = (1000, 2000, 4000, 6000, 8000, 12000, 16000, 24000, 32000, 48000, 64000, 100000, 120000, 143000)
    hit = False
    for m in ("pythia-12b", "pythia-6.9b"):
        for t in ("vartrack_d2", "vartrack_d3"):
            steps = sorted(s for (mm, s, tt) in pos if mm == m and tt == t)
            seq = [bias(pos[(m, s, t)]) for s in steps]
            ch = direction_changes(seq)
            if m == "pythia-12b":
                hit |= ch >= 2
                missing = [s for s in want if s not in steps]
                if missing:
                    print(f"  pythia-12b {t}: MISSING steps {missing}")
            print(f"  {m} {t}: {''.join(seq)}  changes={ch}  (steps {steps})")
    print("  C8:", "HIT" if hit else "MISS")


def score_c9(root):
    pos = load_positions(root)
    print("== C9: seeds disagree (>=2 L and >=2 F) at depth 3, step 16000 or 64000")
    runs = ["pythia-410m"] + [f"pythia-410m-seed{k}" for k in (1, 2, 3, 5, 6, 7, 8, 9)]
    hit = False
    for s in (16000, 64000):
        missing = [r for r in runs if (r, s, "vartrack_d3") not in pos]
        b = [bias(pos[(r, s, "vartrack_d3")]) for r in runs if (r, s, "vartrack_d3") in pos]
        ok = b.count("L") >= 2 and b.count("F") >= 2
        hit |= ok
        print(f"  step {s}: {''.join(b)}  L={b.count('L')} F={b.count('F')}  {'yes' if ok else 'no'}"
              + (f"  MISSING {missing}" if missing else ""))
    print("  C9:", "HIT" if hit else "MISS")


def score_c10(root):
    print("== C10: correct in exactly one order >= 0.20 at depth 2 (6.9B, 12B; 16k/32k/64k)")
    sw = {}
    for f in glob.glob(os.path.join(root, "swap", "*.json")):
        s = json.load(open(f))
        sw[(s["model"].split("/")[-1], s["step"])] = s["tasks"]
    hit = True
    for m in ("pythia-6.9b", "pythia-12b"):
        for st in (16000, 32000, 64000):
            r = sw.get((m, st))
            if r is None:
                print(f"  {m} {st}: MISSING"); hit = False; continue
            v = r["vartrack_d2"]["correct_in_exactly_one_order"]
            hit &= ge(v, 0.20)
            print(f"  {m} {st}: {v:.3f}")
    print("  C10:", "HIT" if hit else "MISS")
    for st in (32000, 64000):
        r = sw.get(("pythia-12b", st))
        if r:
            g = [r[t]["acc_combined"] - r[t]["acc_original"] for t in ("vartrack_d1", "vartrack_d2")]
            print(f"  secondary 12B {st}: combined gain d1={g[0]:+.3f} d2={g[1]:+.3f}  {'yes' if ge(min(g), 0.03) else 'no'}")


def score_c11(root):
    print("== C11: 6.9B plant below 0.10 at EVERY tested layer <=10 (4, 8, 10), >=0.50 at some layer <=22")
    hit = True
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(root, "steer_layers", "*.json"))]
    for st in (16000, 143000):
        for t in ("vartrack_d1", "vartrack_d3"):
            pts = dict(sorted((r["layer"], r["rates"]["plant_cor"]["planted"]) for r in rows if r["step"] == st and r["task"] == t))
            missing_early = [L for L in (4, 8, 10) if L not in pts]
            late = [v for L, v in pts.items() if L <= 22]
            if missing_early or not late:
                print(f"  step {st} {t}: MISSING layers {missing_early or 'all <= 22'}"); hit = False; continue
            early = max(pts[L] for L in (4, 8, 10))
            ok = early < 0.10 - 1e-9 and ge(max(late), 0.50)
            hit &= ok
            print(f"  step {st} {t}: " + " ".join(f"L{L}={v:.2f}" for L, v in pts.items()) + f"  {'yes' if ok else 'no'}")
    print("  C11:", "HIT" if hit else "MISS")


def score_instruct(root):
    print("-- exploratory: base vs chat-tuned (instruct) versions, flip rate (correct in exactly one order)")
    pert = load_perturb(root)
    for base in ("Qwen2.5-7B", "OLMo-2-1124-7B"):
        a, b = pert.get((base, "main")), pert.get((base + "-Instruct", "main"))
        if a is None or b is None:
            print(f"  {base}: MISSING" + (" base" if a is None else "") + (" instruct" if b is None else "")); continue
        for t in ("vartrack_d1", "vartrack_d2", "vartrack_d3"):
            sa, sb = a[t]["summary"], b[t]["summary"]
            print(f"  {base} {t}: base flip {sa['flip_swap']:.3f} (acc {sa['acc_single_mean_swap']:.3f})  "
                  f"instruct flip {sb['flip_swap']:.3f} (acc {sb['acc_single_mean_swap']:.3f})")
        for name in (base, base + "-Instruct"):
            f = os.path.join(root, "benchmark", f"{name}.json")
            if os.path.exists(f):
                s = json.load(open(f))["summary"]
                print(f"  {name} MMLU: acc {s['acc_orig']:.3f}  flip {s['flip_orig_reversed']:.3f}")


C22_MODELS = ("pythia-6.9b", "pythia-12b")


def score_c22(root):
    print("== C22: low-confidence third flips >= 2x the high-confidence third, exact Fisher p < 0.05 "
          "(6.9B and 12B, step 143000, depth 2, confidence = original-order top-2 margin)")
    passed = 0
    for m in C22_MODELS:
        f = os.path.join(root, "uncertainty", f"{m}__step143000.json")
        if not os.path.exists(f):
            print(f"  {m}: MISSING"); continue
        r = json.load(open(f))["tasks"].get("vartrack_d2")
        if r is None:
            print(f"  {m}: MISSING task vartrack_d2"); continue
        g = r["summary"]["gate_orig_margin"]
        ok = ge(g["low_flip"], 2 * g["high_flip"]) and g["low_flip"] > 0 and g["fisher_p"] < 0.05
        passed += ok
        ratio = "n/a (high third never flips)" if g["ratio"] is None else f"{g['ratio']:.2f}"
        print(f"  {m}: flip by confidence tertile (low->high) " + " ".join(f"{x:.3f}" for x in g["group_flip"])
              + f" | ratio {ratio} | low-high {g['diff']:+.3f} [{g['diff_lo']:+.3f},{g['diff_hi']:+.3f}]"
              + f" | Fisher p={g['fisher_p']:.3g} | Spearman(margin, flip)={g['spearman_margin_flip']:+.2f}"
              + f" | acc by tertile " + " ".join(f"{x:.2f}" for x in r["summary"]["acc_orig_by_group"]) + f"  {'yes' if ok else 'no'}")
    print("  C22:", "HIT" if passed == len(C22_MODELS) else "MISS", f"({passed} of {len(C22_MODELS)} models)")


def score_c23(root):
    print("== C23: fresh question set (seed >= 1): swap flip >= 0.20, flip(swap) - flip(rename) >= 0.05, "
          "both-orders gain interval above 0 (6.9B and 12B, step 143000, depth 2)")
    passed = tested = 0
    for m in C22_MODELS:
        files = sorted(glob.glob(os.path.join(root, "perturb_seed*", f"{m}__step143000__seed*.json")))
        if not files:
            print(f"  {m}: MISSING"); continue
        for f in files:
            s = json.load(open(f))
            r = s["tasks"].get("vartrack_d2")
            if r is None:
                print(f"  {m} seed {s.get('seed')}: MISSING task vartrack_d2"); continue
            sm, c = r["summary"], r["correct"]
            single = (np.array(c["orig"], float) + np.array(c["swap"], float)) / 2
            gain, lo, hi = fs.paired_bootstrap(r["ens_correct"]["swap"], single)
            ok = ge(sm["flip_swap"], 0.20) and ge(sm["flip_swap"] - sm["flip_rename"], 0.05) and lo > 0
            tested += 1
            passed += ok
            print(f"  {m} seed {s.get('seed')}: flip swap {sm['flip_swap']:.3f} rename {sm['flip_rename']:.3f} | "
                  f"both-orders gain {gain:+.3f} [{lo:+.3f},{hi:+.3f}] | n={sm['n']}  {'yes' if ok else 'no'}")
    # a model counts once: HIT needs every tested model/seed to pass and both models to be present
    have = {m for m in C22_MODELS if glob.glob(os.path.join(root, "perturb_seed*", f"{m}__step143000__seed*.json"))}
    print("  C23:", "HIT" if tested and passed == tested and have == set(C22_MODELS) else "MISS",
          f"({passed} of {tested} model-seed runs; models present: {sorted(have)})")


C25_MODELS = ["OLMo-2-1124-7B", "Qwen2.5-7B", "Mistral-7B-v0.1", "OLMo-2-1124-7B-Instruct", "Qwen2.5-7B-Instruct",
              "Qwen2.5-14B", "OLMo-2-1124-13B", "Qwen2.5-32B", "OLMo-2-0325-32B", "Qwen2.5-72B"]


def score_c24(root):
    import plant_control_report as pcr
    print("== C24: plant in the learned subspace beats plant in random subspaces of the same rank by >= 0.15, "
          "paired p < 0.05 (6.9B step 143000, layer 22, depths 1 and 3)")
    table = []
    for f in sorted(glob.glob(os.path.join(root, "plant_rand", "*.json"))):
        x = pcr.rows(json.load(open(f)))
        if x:
            table.append(x)
    for x in table:
        if x["model"] == "pythia-6.9b" and x["step"] == 143000 and x["layer"] == 22:
            print(f"  {x['task']}: learned {x['plant_dis']:.3f} random {x['plant_rand']:.3f} "
                  f"diff {x['diff']:+.3f} p={x['p']:.2g} n={x['n']}")
    print("  C24:", pcr.c24(table))


def score_c25(root):
    print("== C25: ARC-Challenge option order: flip(original, reversed) >= 0.03 AND both-orders gain interval above 0, "
          "in at least 7 of 10 models (a model not run counts as not supported)")
    supported, seen = 0, set()
    for f in sorted(glob.glob(os.path.join(root, "benchmark_arc", "*.json"))):
        s = json.load(open(f))
        m = s["model"].split("/")[-1]
        sm = s["summary"]
        ok = ge(sm["flip_orig_reversed"], 0.03) and sm["gain2_lo"] > 0
        if m in C25_MODELS:
            seen.add(m)
            supported += ok
        print(f"  {m}: acc {sm['acc_orig']:.3f} | flip {sm['flip_orig_reversed']:.3f} | two orders gain "
              f"{sm['gain2']:+.3f} [{sm['gain2_lo']:+.3f},{sm['gain2_hi']:+.3f}] n={sm['n']}  {'yes' if ok else 'no'}")
    missing = [m for m in C25_MODELS if m not in seen]
    print("  C25:", "HIT" if supported >= 7 else "MISS", f"({supported} of {len(C25_MODELS)} supported"
          + (f"; NOT RUN: {missing}" if missing else "") + ")")


C26_MODELS = ["Qwen2.5-7B-Instruct", "OLMo-2-1124-7B-Instruct", "Qwen2.5-14B", "OLMo-2-1124-13B", "Qwen2.5-32B",
              "OLMo-2-0325-32B", "Qwen2.5-72B"]


def score_c26(root):
    print("== C26: medication-record wording, depth 3: flip >= 0.05 AND both-orders gain interval above 0, "
          "in at least 4 of 7 models (a model not run counts as not supported)")
    supported, seen = 0, set()
    for f in sorted(glob.glob(os.path.join(root, "story_med", "*.json"))):
        s = json.load(open(f))
        m = s["model"].split("/")[-1]
        r = s["tasks"].get("med_d3")
        if r is None:
            continue
        ok = ge(r["flip"], 0.05) and r["gain_lo"] > 0
        if m in C26_MODELS:
            seen.add(m)
            supported += ok
        print(f"  {m} {s.get('revision')}: acc {r['acc_orig']:.3f}/{r['acc_swapped']:.3f} | flip {r['flip']:.3f} | "
              f"both orders gain {r['gain']:+.3f} [{r['gain_lo']:+.3f},{r['gain_hi']:+.3f}] n={r['n']}  {'yes' if ok else 'no'}")
    missing = [m for m in C26_MODELS if m not in seen]
    print("  C26:", "HIT" if supported >= 4 else "MISS", f"({supported} of {len(C26_MODELS)} supported"
          + (f"; NOT RUN: {missing}" if missing else "") + ")")


def score_c27(root):
    print("== C27: harder medication wording (depth 6 or 9): Qwen2.5-72B or Qwen2.5-72B-Instruct has flip >= 0.05 AND "
          "both-orders gain interval above 0 at depth 6 or 9")
    hit, seen = False, False
    for f in sorted(glob.glob(os.path.join(root, "story_med_deep", "*.json"))):
        s = json.load(open(f))
        m = s["model"].split("/")[-1]
        for t, r in sorted(s["tasks"].items()):
            ok = ge(r["flip"], 0.05) and r["gain_lo"] > 0
            if m in ("Qwen2.5-72B", "Qwen2.5-72B-Instruct"):
                seen = True
                hit = hit or ok
            print(f"  {m} {t}: acc {r['acc_orig']:.3f}/{r['acc_swapped']:.3f} | flip {r['flip']:.3f} | "
                  f"gain {r['gain']:+.3f} [{r['gain_lo']:+.3f},{r['gain_hi']:+.3f}]  {'yes' if ok else 'no'}")
    print("  C27:", "HIT" if hit else ("MISS" if seen else "NOT RUN (no 72B results in story_med_deep)"))


def score_extra(root):
    """Exploratory descriptive statistics (not claims): strict/lenient, chance flip rate, Fisher, McNemar."""
    import extra_stats
    extra_stats.show(root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    root = ap.parse_args().dir
    for fn in (score_c8, score_c9, score_c10, score_c11, score_controls, score_layout, score_benchmark,
               score_olmo, score_olmo_traj, score_scale, score_code, score_c22, score_c23, score_c24, score_c25, score_c26, score_c27, score_instruct, score_extra):
        try:
            fn(root)
        except Exception as e:          # one broken result must not hide the other claims
            print(f"  ERROR while scoring ({fn.__name__}): {type(e).__name__}: {e}  -> treat as NOT SCORED")


if __name__ == "__main__":
    main()
