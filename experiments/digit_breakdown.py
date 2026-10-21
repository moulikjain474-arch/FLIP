"""Is the flip rate driven by WHICH digits are the answer or the distractor? (audit A2, exploratory)

  python3 digit_breakdown.py --model Qwen/Qwen2.5-7B-Instruct --revision main --theme med --depths 3
  python3 digit_breakdown.py --selftest

The Order-Flip Test pairs every question with its own order-swapped copy, so a digit that is easy or hard cancels
within a pair. This script checks that directly: it builds the same items as story_test.py (same seed, same
prompts), and for each depth reports the flip rate (correct in exactly one of the two orders) by
  - the correct answer digit
  - the distractor digit (the other chain's start value)
with the number of items per digit, and a permutation test: if the digits were irrelevant, how often would the
spread between the highest and lowest digit rate be as large as observed? (2000 shuffles, fixed seed.)
A small p means a digit effect is present; the pooled flip rate then needs the per-digit table next to it.
"""
import argparse
import json
import os

import numpy as np

from story_task import DIGITS, make_story


def by_digit(flips, digits, min_n=10):
    """{digit: (rate, n)} for digits with at least min_n items."""
    out = {}
    for d in sorted(set(digits.tolist())):
        m = digits == d
        if m.sum() >= min_n:
            out[int(d)] = (float(flips[m].mean()), int(m.sum()))
    return out


def spread_p(flips, digits, reps=2000, seed=0, min_n=10):
    """(observed max-min rate over digits, permutation p) for the null that digits do not matter."""
    rng = np.random.default_rng(seed)

    def spread(f):
        r = [f[digits == d].mean() for d in set(digits.tolist()) if (digits == d).sum() >= min_n]
        return max(r) - min(r) if len(r) > 1 else 0.0

    obs = spread(flips)
    hits = sum(spread(rng.permutation(flips)) >= obs for _ in range(reps))
    return obs, (hits + 1) / (reps + 1)


def analyse(flips, answer, distractor):
    flips = np.asarray(flips, float)
    res = {}
    for name, dg in (("answer", np.asarray(answer)), ("distractor", np.asarray(distractor))):
        obs, p = spread_p(flips, dg)
        res[name] = dict(table=by_digit(flips, dg), spread=obs, p=p)
    return res


def show(depth, flip_rate, res):
    print(f"depth {depth}: pooled flip {flip_rate:.3f}")
    for name in ("answer", "distractor"):
        r = res[name]
        print(f"  by {name} digit: " + "  ".join(f"{d}:{v:.2f}(n={n})" for d, (v, n) in r["table"].items()))
        print(f"    spread (highest - lowest digit) {r['spread']:.3f}, permutation p = {r['p']:.3f}")


def selftest():
    rng = np.random.default_rng(0)
    n = 4000
    ans = rng.integers(0, 10, n)
    dis = (ans + rng.integers(1, 10, n)) % 10
    null = (rng.random(n) < 0.2).astype(float)
    res = analyse(null, ans, dis)
    assert res["answer"]["p"] > 0.01 and res["distractor"]["p"] > 0.01, res
    biased = (rng.random(n) < np.where(ans == 1, 0.5, 0.2)).astype(float)       # digit 1 flips far more
    res = analyse(biased, ans, dis)
    assert res["answer"]["p"] < 0.01 and res["answer"]["table"][1][0] > 0.4, res
    for theme in ("coins", "med"):                                              # the items the real run uses
        items = make_story(50, 3, 0, theme=theme)
        assert all(it["meta"]["distractor"] != it["meta"]["answer"] for it in items)
    print("digit_breakdown selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--depths", type=int, nargs="+", default=[3])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--theme", choices=("coins", "med"), default="med")
    ap.add_argument("--out", default="digit_breakdown_results")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.model or (args.step is None) == (args.revision is None):
        raise SystemExit("give --model and exactly one of --step or --revision")
    revision = args.revision or f"step{args.step}"

    import torch
    import model_utils as mu

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.float16 if "pythia" in args.model else torch.bfloat16) if dev == "cuda" else torch.float32
    tok, model = mu.load(args.model, revision, dev, dtype)
    out = dict(model=args.model, revision=revision, theme=args.theme, depths={})
    for d in args.depths:
        items = make_story(args.n, d, args.seed, theme=args.theme)
        cids = mu.candidate_ids(tok, items[0]["candidates"])
        y = np.array([it["label"] for it in items])
        ans = np.array([DIGITS.index(it["meta"]["answer"]) for it in items])
        dis = np.array([DIGITS.index(it["meta"]["distractor"]) for it in items])
        lo, _ = mu.run(tok, model, [it["prompt"] for it in items], cids, batch_size=args.batch_size, device=dev)
        ls, _ = mu.run(tok, model, [it["prompt_swapped"] for it in items], cids, batch_size=args.batch_size, device=dev)
        po, ps = lo.argmax(-1).numpy(), ls.argmax(-1).numpy()
        flips = ((po == y) ^ (ps == y)).astype(float)
        res = analyse(flips, ans, dis)
        show(d, float(flips.mean()), res)
        out["depths"][f"d{d}"] = dict(n=len(items), flip=float(flips.mean()), answer=res["answer"], distractor=res["distractor"])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.model.split('/')[-1]}__{revision}__{args.theme}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
