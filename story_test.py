"""Order-Flip Test on a second binding task told as a story (exploratory).

  python story_test.py --model EleutherAI/pythia-12b --step 143000
Reports per depth: accuracy in each order, flip rate (correct in exactly one order), the
both-orders ensemble, and how often the model takes the LATER chain's value when it answers
with one of the two chain values (0.5 = no position habit).
"""
import argparse
import json
import os

import numpy as np

import flipstats as fs
from story_task import DIGITS, make_story


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="story_results")
    args = ap.parse_args()
    if (args.step is None) == (args.revision is None):
        raise SystemExit("give exactly one of --step or --revision")
    revision = args.revision or f"step{args.step}"

    import torch
    import model_utils as mu

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.float16 if "pythia" in args.model else torch.bfloat16) if dev == "cuda" else torch.float32
    tok, model = mu.load(args.model, revision, dev, dtype)
    out = dict(model=args.model, revision=revision, tasks={})
    for d in args.depths:
        items = make_story(args.n, d, args.seed)
        if d == args.depths[0]:
            print(items[0]["prompt"][-200:] + " ->" + items[0]["candidates"][items[0]["label"]])
        cids = mu.candidate_ids(tok, items[0]["candidates"])
        y = np.array([it["label"] for it in items])
        dis = np.array([DIGITS.index(it["meta"]["distractor"]) for it in items])
        tf = np.array([it["meta"]["target_first"] for it in items])
        lo, _ = mu.run(tok, model, [it["prompt"] for it in items], cids, batch_size=args.batch_size, device=dev)
        ls, _ = mu.run(tok, model, [it["prompt_swapped"] for it in items], cids, batch_size=args.batch_size, device=dev)
        lpo, lps = torch.log_softmax(lo, -1).numpy(), torch.log_softmax(ls, -1).numpy()
        po, ps = lpo.argmax(-1), lps.argmax(-1)
        co, cs = po == y, ps == y
        ens = fs.ensemble_correct([lpo, lps], y)
        later = later_n = 0      # in the original prompt the target is later iff not target_first
        for p, target_later in ((po, ~tf), (ps, tf)):
            took_t, took_d = p == y, p == dis
            later += int((took_t & target_later).sum() + (took_d & ~target_later).sum())
            later_n += int((took_t | took_d).sum())
        single = (co.astype(float) + cs) / 2
        g = fs.paired_bootstrap(ens, single)
        r = dict(acc_orig=float(co.mean()), acc_swapped=float(cs.mean()), acc_single_mean=float(single.mean()),
                 acc_both=float(ens.mean()), gain=g[0], gain_lo=g[1], gain_hi=g[2], flip=fs.flip_rate(co, cs),
                 distractor_rate=float(((po == dis).mean() + (ps == dis).mean()) / 2),
                 picks_later=later / later_n if later_n else None, picks_later_n=later_n, n=int(len(y)))
        lo95, hi95 = fs.wilson(later, later_n) if later_n else (None, None)
        r.update(picks_later_lo=lo95, picks_later_hi=hi95)
        out["tasks"][f"story_d{d}"] = r
        print(f"story_d{d}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.model.split('/')[-1]}__{revision}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
