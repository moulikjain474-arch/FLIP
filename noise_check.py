"""Is the flipping numerical noise? Score the same prompts at two batch sizes (exploratory, N4).

Left-padded rows have different lengths in the original and swapped orders, so fp16/bf16 numerics can
differ slightly between orders, which matters only on near-ties. This asks the question directly: how often
does the SAME prompt get a different answer when only the batch composition changes?

  python noise_check.py --model EleutherAI/pythia-1.4b --step 143000 --n 300
Reports, per task, the share of prompts whose answer differs between batch size A and B, for the original
and the swapped order, next to the flip rate between the two orders (the quantity this is meant to explain).
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tasks", nargs="+", default=["vartrack_d1", "vartrack_d2", "vartrack_d3"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs-a", type=int, default=8)
    ap.add_argument("--bs-b", type=int, default=32)
    ap.add_argument("--out", default="noise_results")
    args = ap.parse_args()
    if (args.step is None) == (args.revision is None):
        raise SystemExit("give exactly one of --step or --revision")
    revision = args.revision or f"step{args.step}"

    import torch
    import model_utils as mu
    from perturb import swap_chains
    from tasks import build_all

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.float16 if "pythia" in args.model else torch.bfloat16) if dev == "cuda" else torch.float32
    tok, model = mu.load(args.model, revision, dev, dtype)
    items_all = build_all(args.n, args.seed, shots=2, layout="blocked")
    out = dict(model=args.model, revision=revision, bs=[args.bs_a, args.bs_b], tasks={})
    for t in args.tasks:
        items = items_all[t]
        orig = [it["prompt"] for it in items]
        swp = [swap_chains(p) for p in orig]
        cids = mu.candidate_ids(tok, items[0]["candidates"])
        pred = {}
        for name, prompts in (("orig", orig), ("swap", swp)):
            for bs in (args.bs_a, args.bs_b):
                lo, _ = mu.run(tok, model, prompts, cids, batch_size=bs, device=dev)
                pred[(name, bs)] = lo.argmax(-1).numpy()
        y = np.array([it["label"] for it in items])
        a, b = args.bs_a, args.bs_b
        r = dict(n=int(len(y)),
                 batch_disagree_orig=float((pred[("orig", a)] != pred[("orig", b)]).mean()),
                 batch_disagree_swap=float((pred[("swap", a)] != pred[("swap", b)]).mean()),
                 order_disagree_bs_b=float((pred[("orig", b)] != pred[("swap", b)]).mean()),
                 flip_bs_b=float(((pred[("orig", b)] == y) != (pred[("swap", b)] == y)).mean()))
        out["tasks"][t] = r
        print(t, {k: round(v, 3) for k, v in r.items()}, flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.model.split('/')[-1]}__{revision}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
