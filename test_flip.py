"""Checks that need no GPU:  python test_flip.py"""
import random

import numpy as np

import benchmark_flip as bf
import flipstats as fs
import perturb as pt
import story_task as st
import code_task as ct

HEAD = ">>> x = 4\n>>> y = x\n>>> y\n4\n\n"
NAMES = list("abcdefghijklmnopqrstuvwxyz")


def make_prompt(rng, depth):
    names = rng.sample(NAMES, 2 * depth)
    chains = []
    for c in range(2):
        ns = names[c * depth:(c + 1) * depth]
        lines = [f">>> {ns[0]} = {rng.randint(0, 9)}"] + [f">>> {ns[i]} = {ns[i - 1]}" for i in range(1, depth)]
        chains.append(lines)
    q = rng.choice([chains[0][-1], chains[1][-1]]).split(" = ")[0][4:]
    return HEAD + "\n".join(chains[0] + chains[1]) + f"\n>>> {q}\n"


def test_perturbations():
    rng = random.Random(0)
    for i in range(500):
        p = make_prompt(rng, rng.randint(1, 3))
        ans = pt.final_answer(p)
        assert ans is not None
        v = pt.variants(p, seed=i)
        assert all(pt.final_answer(x) == ans for x in v.values()), v
        assert pt.swap_chains(v["swap"]) == p                       # swapping twice is the identity
        assert v["swap"] != p and v["rename"] != p
        assert v["orig"][: p.rfind("\n\n") + 2] == v["rename"][: p.rfind("\n\n") + 2]   # shots untouched
        _, a, b, _ = pt.split_problem(p)
        _, ra, rb, rtail = pt.split_problem(v["rename"])
        old = {pt.ASSIGN.match(l).group(1) for l in a + b}
        new = {pt.ASSIGN.match(l).group(1) for l in ra + rb}
        assert not old & new and len(new) == len(old)               # names fully replaced
        assert [len(pt.ASSIGN.match(l).group(1)) for l in ra + rb] == [len(pt.ASSIGN.match(l).group(1)) for l in a + b]
        assert [l.split(" = ")[1] if l.split(" = ")[1].isdigit() else "ref" for l in ra + rb] == \
               [l.split(" = ")[1] if l.split(" = ")[1].isdigit() else "ref" for l in a + b]   # same structure
    assert pt.rename_chains(p, 7) == pt.rename_chains(p, 7)          # deterministic
    v = pt.variants(p, 0, instruct=True)
    assert v["instruct"] == pt.INSTRUCTION + p and v["instruct_swap"] == pt.INSTRUCTION + v["swap"]
    assert pt.final_answer(v["instruct"]) == pt.final_answer(v["instruct_swap"]) == pt.final_answer(p)
    print("perturbations ok")


def test_filler():
    rng = random.Random(1)
    for i in range(300):
        p = make_prompt(rng, rng.randint(1, 3))
        for where in ("between", "before_query"):
            for k in (2, 4):
                f = pt.add_filler(p, where, k)
                assert f.count(pt.FILLER) == k and pt.strip_filler(f) == p
                assert pt.final_answer(pt.strip_filler(f)) == pt.final_answer(p)
                body = f[f.rfind("\n\n") + 2:].split("\n")
                first_pass = body.index(pt.FILLER)
                n_assign = sum(1 for l in body if pt.ASSIGN.match(l))
                assert first_pass == (n_assign // 2 if where == "between" else n_assign)
    print("filler ok")


def test_story():
    for d in (1, 2, 3):
        items = st.make_story(200, d)
        assert items == st.make_story(200, d)                       # deterministic
        for it in items:
            a, b = it["prompt"], it["prompt_swapped"]
            assert a != b and a.rsplit("\n\n", 1)[0] == b.rsplit("\n\n", 1)[0]   # same shots
            qa, qb = a.rsplit("\n\n", 1)[1].split("\n"), b.rsplit("\n\n", 1)[1].split("\n")
            assert qa[0] == qb[1] and qa[1] == qb[0] and qa[2:] == qb[2:]         # chains swapped
            assert a.endswith("A:\n") and it["candidates"] == [str(x) for x in range(10)]
            name = qa[2].split("does ")[1].split(" have")[0]
            facts, env = " ".join(qa[:2]).split(". "), {}
            assert " 1 coins" not in a
            for f in facts:          # resolve the story and check the label
                w = f.rstrip(".").split()
                env[w[0]] = w[2] if w[2].isdigit() else env[w[-1]]
            assert env[name] == it["meta"]["answer"] == st.DIGITS[it["label"]]
            tgt_line = qa[0] if it["meta"]["target_first"] else qa[1]
            assert name in tgt_line
    print("story ok")


def test_code():
    for level in (1, 2, 3):
        for it in ct.make_code(150, level):
            a = it["prompt"].rsplit("```python\n", 1)[1].split("\n```")[0]
            b = it["prompt_swapped"].rsplit("```python\n", 1)[1].split("\n```")[0]
            assert a != b and ct.run(a) == ct.run(b) == str(it["label"])
            assert it["prompt"].endswith("Output:\n") and it["candidates"] == [str(d) for d in range(10)]
            assert "\nOutput:\n" in it["prompt"].split("```python")[1]                  # shots use the same format
    lp = np.log(np.full((4, 10), 0.1)); lp[:, 3] = 0
    m = fs.pair_metrics(lp, lp, [3, 3, 3, 3])
    assert m["acc_both"] == 1 and m["flip"] == 0 and m["gain"] == 0
    print("code ok")


def test_chain_spans():
    rng = random.Random(3)
    for i in range(300):
        p = make_prompt(rng, rng.randint(1, 3))
        (s1, e1), (s2, e2) = pt.chain_char_spans(p)
        _, a, b, _ = pt.split_problem(p)
        assert p[s1:e1] == "\n".join(a) and p[s2:e2] == "\n".join(b) and e1 < s2
        vals, target = pt.chain_values(p)
        q = p.rstrip("\n").split("\n")[-1][4:]
        assert q in "\n".join([a, b][target]) and pt.final_answer(p) == vals[target]
    print("chain spans ok")


def test_bad_format_is_caught():
    for bad in (HEAD + ">>> a = 1\n>>> b = c\n>>> a\n",              # not self-contained / odd
                HEAD + ">>> a = 1\n>>> b = 2\n>>> c = 3\n>>> a\n"):
        try:
            pt.swap_chains(bad)
        except AssertionError:
            continue
        raise SystemExit("bad prompt not caught")
    print("format checks ok")


def test_single_chain_control():
    rng = random.Random(1)
    for i in range(300):
        p = make_prompt(rng, rng.randint(1, 3))
        src, tgt, vs, vt = pt.single_chain_control(p, i)
        assert vs == pt.final_answer(p) and vt != vs
        assert src.startswith(p[: p.rfind("\n\n") + 2]) and tgt.startswith(p[: p.rfind("\n\n") + 2])
        bs, bt = src[src.rfind("\n\n") + 2:].strip("\n").split("\n"), tgt[tgt.rfind("\n\n") + 2:].strip("\n").split("\n")
        assert len(bs) == len(bt) and sum(x != y for x, y in zip(bs, bt)) == 1   # only the start digit differs
        env = {}
        for l in bt[:-1]:
            lhs, rhs = l[4:].split(" = ")
            env[lhs] = rhs if rhs.isdigit() else env[rhs]
        assert env[bt[-1][4:].strip()] == vt                                   # target answer is the new digit
    print("single-chain control ok")


def test_extra_stats():
    import extra_stats as ex
    ca, cb = ex._from_rates(0.6, 0.5, 0.3, 100)          # both right 40, a-only 20, b-only 10, both wrong 30
    assert ca.sum() == 60 and cb.sum() == 50 and (ca != cb).sum() == 30 and (ca & cb).sum() == 40
    st = fs.order_stats(ca, cb)
    assert abs(st["strict"] - 0.4) < 1e-9 and abs(st["lenient"] - 0.7) < 1e-9 and abs(st["both_wrong"] - 0.3) < 1e-9
    try:
        ex._from_rates(0.9, 0.9, 1.0, 10)
        raise SystemExit("inconsistent rates not rejected")
    except ValueError:
        pass
    # chance flip rate with unequal accuracies: p1(1-p2)+p2(1-p1), not 2p(1-p)
    ca2, cb2 = ex._from_rates(0.9, 0.5, 0.4, 100)
    assert abs(fs.order_stats(ca2, cb2)["flip_chance"] - (0.9 * 0.5 + 0.5 * 0.1)) < 1e-9
    # ensemble fixed / broke counts against the original-order answer
    import numpy as _np
    lpa = _np.log(_np.array([[.9, .1], [.2, .8], [.6, .4], [.3, .7]]))
    lpb = _np.log(_np.array([[.4, .6], [.1, .9], [.1, .9], [.2, .8]]))
    pm = fs.pair_metrics(lpa, lpb, [0, 1, 1, 0])
    # ensemble picks: q0 .9*.4 vs .1*.6 -> 0 (right); q1 -> 1 (right); q2 .6*.1=.06 vs .4*.9=.36 -> 1 (right); q3 .06 vs .56 -> 1 (wrong)
    # original picks right on q0,q1 ; wrong on q2 (picks 0), q3 (picks 1 -> label 0 wrong)
    assert pm["ens_fixed"] == 1 and pm["ens_broke"] == 0, pm
    # rows() contract on fake result files
    import tempfile, json as _json, os as _os
    d = tempfile.mkdtemp()
    _os.makedirs(d + "/swap"); _os.makedirs(d + "/perturb")
    _json.dump(dict(model="x/m", revision="step1", tasks={"t": dict(acc_original=.6, acc_swapped=.5,
               correct_in_exactly_one_order=.3, n=100)}), open(d + "/swap/a.json", "w"))
    cc = {k: [1, 0, 1, 1] for k in ("orig", "swap", "rename", "swap_rename")}
    _json.dump(dict(model="x/m", revision="step1", tasks={"t": dict(correct=cc, ens_correct={k: [1, 1, 1, 1] for k in ("swap", "rename", "all4")})}),
               open(d + "/perturb/a.json", "w"))
    kinds = {r[0] for r in ex.rows(d)}
    assert kinds == {"xstat_swaptest", "xstat_perturb_swap", "xstat_perturb_rename", "xstat_ens"}, kinds
    a, b, p = ex.mcnemar_pair([1, 1, 1, 0, 0], [0, 0, 0, 0, 1])
    assert (a, b) == (3, 1) and 0 < p <= 1
    print("extra stats ok")


def test_confidence_gate_and_claims():
    import numpy as np, tempfile, json as _json, os as _os, io, contextlib
    import uncertainty_gate as ug
    rng = np.random.default_rng(5)
    # top-2 margin and tertile cutting
    lp = np.log(np.array([[.5, .3, .2], [.9, .05, .05], [.4, .35, .25]]))
    assert np.allclose(fs.top2_margin(lp), [np.log(.5 / .3), np.log(.9 / .05), np.log(.4 / .35)])
    m = np.arange(30, dtype=float)
    f = np.array([True] * 10 + [False] * 20)            # all flips in the lowest-confidence third
    g = fs.confidence_gate(m, f, n_boot=200)
    assert g["group_flip"] == [1.0, 0.0, 0.0] and g["group_n"] == [10, 10, 10] and g["ratio"] is None
    assert g["diff"] == 1.0 and g["fisher_p"] < 1e-4 and g["spearman_margin_flip"] < -0.5
    # no relationship -> no gating
    m2, f2 = rng.uniform(size=900), rng.uniform(size=900) < 0.4
    g2 = fs.confidence_gate(m2, f2, n_boot=200)
    assert abs(g2["diff"]) < 0.1 and g2["diff_lo"] < 0 < g2["diff_hi"]
    # ties keep question order (stable), 7 questions in 3 groups -> 3/2/2
    assert fs.confidence_gate(np.zeros(7), np.array([1, 0, 0, 0, 0, 0, 0], bool), n_boot=20)["group_n"] == [3, 2, 2]
    assert abs(fs.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12 and np.isnan(fs.spearman([1, 1, 1], [1, 2, 3]))
    # uncertainty_gate.analyse: a model that is unsure exactly when the order changes its answer
    n = 600
    y = rng.integers(0, 10, n)
    unsure = rng.uniform(size=n) < 0.5
    def lps(wrong):
        lg = rng.normal(0, .3, (n, 10))
        lg[np.arange(n), y] += 4.0                       # confident and right ...
        lg[np.arange(n), (y + 1) % 10] += np.where(unsure, 4.2, -9.0) * wrong   # ... unless unsure and wrong
        return lg - np.log(np.exp(lg).sum(1, keepdims=True))
    lo = lps(np.ones(n))
    ls = lps(np.where(unsure & (rng.uniform(size=n) < .5), 0.0, 1.0))
    sm, raw = ug.analyse(lo, ls, y)
    gg = sm["gate_orig_margin"]
    assert gg["low_flip"] > 2 * max(gg["high_flip"], 0.01) and sm["acc_orig_by_group"][-1] > sm["acc_orig_by_group"][0], sm
    assert len(raw["margin_orig"]) == n
    # score_c22 / score_c23 on fake result files
    import score_new as sn
    d = tempfile.mkdtemp()
    _os.makedirs(d + "/uncertainty"); _os.makedirs(d + "/perturb_seed1")
    for mname in ("pythia-6.9b", "pythia-12b"):
        _json.dump(dict(model="EleutherAI/" + mname, revision="step143000", seed=0, tasks={"vartrack_d2": dict(summary=sm)}),
                   open(f"{d}/uncertainty/{mname}__step143000.json", "w"))
        yy = rng.integers(0, 2, 600)
        co = (rng.uniform(size=600) < .6).astype(int)
        cs = np.where(rng.uniform(size=600) < .5, co, 1 - co)            # about half of the answers change
        cr = np.where(rng.uniform(size=600) < .85, co, 1 - co)
        ens = np.where(rng.uniform(size=600) < .8, np.maximum(co, cs), np.minimum(co, cs))   # ensemble leans to the right one
        summ = dict(flip_swap=float((co != cs).mean()), flip_rename=float((co != cr).mean()), n=600)
        _json.dump(dict(model="EleutherAI/" + mname, revision="step143000", seed=1, tasks={"vartrack_d2": dict(
            summary=summ, correct=dict(orig=co.tolist(), swap=cs.tolist(), rename=cr.tolist()),
            ens_correct=dict(swap=ens.tolist()))}), open(f"{d}/perturb_seed1/{mname}__step143000__seed1.json", "w"))
    def grab(fn):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(d)
        return buf.getvalue()
    out22, out23 = grab(sn.score_c22), grab(sn.score_c23)
    assert "C22: HIT (2 of 2" in out22, out22
    assert "C23: HIT (2 of 2" in out23, out23
    _os.remove(d + "/uncertainty/pythia-12b__step143000.json")
    assert "MISSING" in grab(sn.score_c22) and "C22: MISS" in grab(sn.score_c22)
    _os.remove(d + "/perturb_seed1/pythia-12b__step143000__seed1.json")
    assert "C23: MISS" in grab(sn.score_c23)
    import export_csv as ec
    assert any(r[0] == "gate" for r in ec.gate_rows(d)) and any(r[0] == "perturb_seed1" for r in ec.gate_rows(d))
    print("confidence gate and claims ok")


def test_chain_probe_labels():
    import chain_probe as cp
    rng = random.Random(3)
    prompts = [make_prompt(rng, rng.randint(1, 3)) for _ in range(200)]
    cands = [str(d) for d in range(10)]
    # an answerer that always picks the SECOND chain's value -> picks_later == 1; the FIRST -> 0
    second = [cands.index(pt.chain_values(p)[0][1]) for p in prompts]
    first = [cands.index(pt.chain_values(p)[0][0]) for p in prompts]
    y, pl = cp.labels_and_picks(prompts, second, cands)
    assert pl == 1.0 and set(y.tolist()) <= {0, 1} and 0.2 < y.mean() < 0.8
    assert cp.labels_and_picks(prompts, first, cands)[1] == 0.0
    # answers equal to neither chain value are ignored
    other = [next(i for i, c in enumerate(cands) if c not in pt.chain_values(p)[0]) for p in prompts]
    assert cp.labels_and_picks(prompts, other, cands)[1] is None
    print("chain probe labels ok")


def test_plant_margin():
    import plant_margin as pm
    m = np.linspace(-3, 3, 40)
    b = pm.bins_of(m, 4)
    assert sorted(set(b.tolist())) == [0, 1, 2, 3] and (b == 0).sum() == 10
    assert np.abs(m[b == 0]).max() <= np.abs(m[b == 1]).min() + 1e-12      # bin 0 holds the smallest |margin|
    print("plant margin ok")


def test_stats():
    assert fs.flip_rate([1, 1, 0, 0], [1, 0, 1, 0]) == 0.5
    lp = [np.log(np.array([[0.6, 0.4], [0.2, 0.8]])), np.log(np.array([[0.3, 0.7], [0.1, 0.9]]))]
    assert fs.ensemble_correct(lp, [1, 1]).tolist() == [True, True]
    m, lo, hi = fs.paired_bootstrap(np.ones(200), np.zeros(200))
    assert m == 1 and lo == 1 and hi == 1
    m, lo, hi = fs.paired_bootstrap(np.r_[np.ones(100), np.zeros(100)], np.r_[np.zeros(100), np.ones(100)])
    assert lo < 0 < hi
    lo, hi = fs.wilson(50, 100)
    assert lo < 0.5 < hi
    o, p = fs.fisher_exact(10, 0, 0, 10)
    assert p < 1e-4 and o == float("inf")
    assert abs(fs.fisher_exact(5, 5, 5, 5)[1] - 1.0) < 1e-9
    assert abs(fs.fisher_exact(3, 1, 1, 3)[1] - 0.4857142857) < 1e-6
    assert fs.mcnemar_exact(0, 0) == 1.0 and abs(fs.mcnemar_exact(10, 0) - 2 / 1024) < 1e-12
    ca, cb = np.array([1, 1, 0, 0, 1, 0], bool), np.array([1, 0, 1, 0, 1, 0], bool)
    st = fs.order_stats(ca, cb)
    assert abs(st["strict"] - 2 / 6) < 1e-9 and abs(st["lenient"] - 4 / 6) < 1e-9
    assert abs(st["both_right"] + st["both_wrong"] + st["flip_share"] - 1) < 1e-9
    assert abs(st["flip_chance"] - (0.5 * 0.5 + 0.5 * 0.5)) < 1e-9
    print("stats ok")


if __name__ == "__main__":
    test_perturbations()
    test_filler()
    test_story()
    test_code()
    test_chain_spans()
    test_bad_format_is_caught()
    test_single_chain_control()
    test_extra_stats()
    test_confidence_gate_and_claims()
    test_chain_probe_labels()
    test_plant_margin()
    test_stats()
    bf.selftest()
