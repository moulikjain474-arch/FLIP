"""Registered claim C22: does the order effect concentrate where the model is unsure?

Every question is asked in two orders (original, chains swapped). The model's confidence in the
ORIGINAL order is the gap between its best and second-best answer (log-probability, over the ten
candidate digits). A question "flips" if it is answered correctly in exactly one of the two orders.
Questions are cut into three equal groups by confidence; if the order effect really is what an
unsure model does, the least-confident third should flip far more often than the most-confident third.

  python uncertainty_gate.py --model EleutherAI/pythia-12b --step 143000
  python uncertainty_gate.py --model EleutherAI/pythia-6.9b --step 143000 --out $PERSIST/fullday/uncertainty

Saves per-question margins and correctness so the analysis can be redone without the model.
Margin is measured only in the original order (what a user would have); the margin in the swapped
order and the smaller of the two are reported as extra descriptive rows.
"""
import argparse
import json
import os

import numpy as np

import flipstats as fs


def analyse(logp_orig, logp_swap, y):
    """Pure function (no model): per-task summary from the two [N, K] log-prob arrays and labels."""
    y = np.asarray(y)
    co, cs = logp_orig.argmax(-1) == y, logp_swap.argmax(-1) == y
    flip = co != cs
    mo, ms = fs.top2_margin(logp_orig), fs.top2_margin(logp_swap)
    out = dict(n=int(len(y)), acc_orig=float(co.mean()), acc_swap=float(cs.mean()), flip=float(flip.mean()))
    out["gate_orig_margin"] = fs.confidence_gate(mo, flip)             # the registered measure
    out["gate_swap_margin"] = fs.confidence_gate(ms, flip)             # descriptive
    out["gate_min_margin"] = fs.confidence_gate(np.minimum(mo, ms), flip)   # descriptive
    parts = np.array_split(np.argsort(mo, kind="stable"), 3)
    out["acc_orig_by_group"] = [float(co[p].mean()) for p in parts]    # sanity: confident third should be more accurate
    return out, dict(margin_orig=mo.tolist(), margin_swap=ms.tolist(),
                     correct_orig=co.astype(int).tolist(), correct_swap=cs.astype(int).tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tasks", nargs="+", default=["vartrack_d1", "vartrack_d2", "vartrack_d3"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dtype", choices=("fp16", "bf16"), default=None)
    ap.add_argument("--out", default="uncertainty_results")
    args = ap.parse_args()
    if (args.step is None) == (args.revision is None):
        raise SystemExit("give exactly one of --step or --revision")
    revision = args.revision or f"step{args.step}"

    import torch
    import model_utils as mu
    from perturb import swap_chains
    from tasks import build_all

    items_all = build_all(args.n, args.seed, shots=2, layout="blocked")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = args.dtype or ("fp16" if "pythia" in args.model else "bf16")
    tok, model = mu.load(args.model, revision, dev,
                         (torch.float16 if dtype == "fp16" else torch.bfloat16) if dev == "cuda" else torch.float32)
    out = dict(model=args.model, revision=revision, step=args.step, seed=args.seed, tasks={})
    for t in args.tasks:
        items = items_all[t]
        orig = [it["prompt"] for it in items]
        swp = [swap_chains(p) for p in orig]
        y = np.array([it["label"] for it in items])
        cids = mu.candidate_ids(tok, items[0]["candidates"])
        lo, _ = mu.run(tok, model, orig, cids, batch_size=args.batch_size, device=dev)
        ls, _ = mu.run(tok, model, swp, cids, batch_size=args.batch_size, device=dev)
        summary, raw = analyse(torch.log_softmax(lo, -1).numpy(), torch.log_softmax(ls, -1).numpy(), y)
        out["tasks"][t] = dict(summary=summary, **raw)
        g = summary["gate_orig_margin"]
        print(t, f"flip {summary['flip']:.3f} | by confidence (low->high) "
              + " ".join(f"{r:.3f}" for r in g["group_flip"])
              + f" | low-high {g['diff']:+.3f} [{g['diff_lo']:+.3f},{g['diff_hi']:+.3f}] ratio "
              + ("n/a" if g["ratio"] is None else f"{g['ratio']:.2f}") + f" fisher p={g['fisher_p']:.2g}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    name = f"{args.model.split('/')[-1]}__{revision}" + (f"__seed{args.seed}" if args.seed else "") + ".json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
