"""Checks that need no GPU:  python test_flip.py"""
import random

import numpy as np

import benchmark_flip as bf
import flipstats as fs
import perturb as pt

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
    print("perturbations ok")


def test_bad_format_is_caught():
    for bad in (HEAD + ">>> a = 1\n>>> b = c\n>>> a\n",              # not self-contained / odd
                HEAD + ">>> a = 1\n>>> b = 2\n>>> c = 3\n>>> a\n"):
        try:
            pt.swap_chains(bad)
        except AssertionError:
            continue
        raise SystemExit("bad prompt not caught")
    print("format checks ok")


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
    print("stats ok")


if __name__ == "__main__":
    test_perturbations()
    test_bad_format_is_caught()
    test_stats()
    bf.selftest()
