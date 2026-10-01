"""Order-swap test: ask every question twice, with the two chains in each order.

For each item the final problem block (after the last blank line) is split into its two
chains, which are swapped; the answer is unchanged. Reports accuracy on the original order,
the swapped order, and a combined answer (sum of log-probs over both orders), plus how often
the model gives the SAME answer to both orders (consistency).

  python swap_test.py --model EleutherAI/pythia-1.4b --step 143000
"""
import argparse, json, os
import numpy as np
import torch
import model_utils as mu
from perturb import swap_chains
from tasks import build_all

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--revision", default=None, help="e.g. main, for models without step checkpoints")
    ap.add_argument("--tasks", nargs="+", default=["vartrack_d1", "vartrack_d2", "vartrack_d3"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="swap_results")
    args = ap.parse_args()
    if (args.step is None) == (args.revision is None):
        raise SystemExit("give exactly one of --step or --revision")
    revision = args.revision or f"step{args.step}"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = mu.load(args.model, revision, dev, torch.float16 if dev == "cuda" else torch.float32)
    items_all = build_all(args.n, args.seed, shots=2, layout="blocked")
    out = dict(model=args.model, step=args.step, revision=revision, tasks={})
    for i, t in enumerate(args.tasks):
        items = items_all[t]
        orig = [it["prompt"] for it in items]
        swp = [swap_chains(p) for p in orig]
        if i == 0:
            print("--- original tail ---\n" + orig[0][-160:] + "\n--- swapped tail ---\n" + swp[0][-160:] + "\n")
        cids = mu.candidate_ids(tok, items[0]["candidates"])
        lo, _ = mu.run(tok, model, orig, cids, batch_size=args.batch_size, device=dev)
        ls, _ = mu.run(tok, model, swp, cids, batch_size=args.batch_size, device=dev)
        y = np.array([it["label"] for it in items])
        po, ps = lo.argmax(-1).numpy(), ls.argmax(-1).numpy()
        pc = (torch.log_softmax(lo, -1) + torch.log_softmax(ls, -1)).argmax(-1).numpy()
        r = dict(acc_original=float((po == y).mean()), acc_swapped=float((ps == y).mean()),
                 acc_combined=float((pc == y).mean()), same_answer_both_orders=float((po == ps).mean()),
                 correct_in_exactly_one_order=float(((po == y) != (ps == y)).mean()), n=int(len(y)))
        out["tasks"][t] = r
        print(t, {k: round(v, 3) for k, v in r.items()}, flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.model.split('/')[-1]}__{revision}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
