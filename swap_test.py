"""Order-swap test: ask every question twice, with the two chains in each order.

For each item the final problem block (after the last blank line) is split into its two
chains, which are swapped; the answer is unchanged. Reports accuracy on the original order,
the swapped order, and a combined answer (sum of log-probs over both orders), plus how often
the model gives the SAME answer to both orders (consistency).

  python swap_test.py --model EleutherAI/pythia-1.4b --step 143000
"""
import argparse, json, os, re
import numpy as np
import torch
import model_utils as mu
from tasks import build_all

ASSIGN = re.compile(r"^>>> (\w+) = (\w+)\s*$")


def swap_chains(prompt):
    """Return the prompt with the two chains of the final problem swapped (answer unchanged)."""
    cut = prompt.rstrip("\n").rfind("\n\n") + 2
    head, block = prompt[:cut], prompt[cut:]
    lines = block.split("\n")
    q = max(i for i, l in enumerate(lines) if l.startswith(">>> ") and not ASSIGN.match(l))
    assigns, tail = lines[:q], lines[q:]
    assert all(ASSIGN.match(l) for l in assigns), f"unexpected line in problem block: {assigns}"
    assert len(assigns) % 2 == 0, f"odd number of assignment lines: {assigns}"
    h = len(assigns) // 2
    a, b = assigns[:h], assigns[h:]
    for chain in (a, b):  # each chain starts from a constant and then refers only to itself
        assert ASSIGN.match(chain[0]).group(2).isdigit(), f"chain does not start with a number: {chain}"
        names = {ASSIGN.match(chain[0]).group(1)}
        for l in chain[1:]:
            lhs, rhs = ASSIGN.match(l).groups()
            assert rhs in names, f"chain is not self-contained: {chain}"
            names.add(lhs)
    return head + "\n".join(b + a + tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--tasks", nargs="+", default=["vartrack_d1", "vartrack_d2", "vartrack_d3"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="swap_results")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = mu.load(args.model, f"step{args.step}", dev, torch.float16 if dev == "cuda" else torch.float32)
    items_all = build_all(args.n, args.seed, shots=2, layout="blocked")
    out = dict(model=args.model, step=args.step, tasks={})
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
    with open(os.path.join(args.out, f"{args.model.split('/')[-1]}__step{args.step}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
