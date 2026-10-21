"""Exploratory (8 Oct 2026, written AFTER seeing C22 at 12B): which chain does the model pick in each order?

Each question is asked in the original order and with the two chains swapped. Whatever the model
answers is matched to the start value of the chain written first or second, so every answer is
"the first chain", "the second chain" or "neither". For each question the pair of answers is then
sorted into one of five groups (A = chain written first originally, B = chain written second):
  right_both        the queried chain in both orders (content-driven and correct)
  wrong_chain_both  the other chain in both orders (content-driven but wrong)
  first_both        the chain written FIRST in both orders (A, then B after the swap): position-driven
  second_both       the chain written SECOND in both orders (B, then A after the swap): position-driven
  neither           at least one answer matches neither chain
Position-driven answers are correct in exactly one order, so they are a subset of the flips.
Also reported: the same shares inside confidence thirds (original-order margin), to connect to C22,
and the usual gate numbers. Nothing here is a registered claim.

  python position_decomposition.py --model EleutherAI/pythia-12b --step 143000
  python position_decomposition.py --model allenai/OLMo-2-1124-7B --revision main
"""
import argparse
import json
import os

import numpy as np

import flipstats as fs

GROUPS = ("right_both", "wrong_chain_both", "first_both", "second_both", "neither")


def classify(prompt_orig, prompt_swap, pred_orig, pred_swap, cands):
    """Group of one question. pred_* are candidate indices; prompts are the two versions of the question."""
    from perturb import chain_values
    (vo, t), (vs, _) = chain_values(prompt_orig), chain_values(prompt_swap)
    a, b = vo                                    # start values of chain A (first) and chain B (second) originally
    assert vs == [b, a], "swapped prompt does not hold the same chains in reverse order"
    if a == b:                                   # chains indistinguishable by value (tasks.py never draws this): left out
        return "ambiguous"
    who = lambda ans: "A" if ans == a else "B" if ans == b else None     # chain identity, not position
    co, cs = who(cands[int(pred_orig)]), who(cands[int(pred_swap)])
    if co is None or cs is None:
        return "neither"
    queried = "A" if t == 0 else "B"
    if co == cs:
        return "right_both" if co == queried else "wrong_chain_both"
    return "first_both" if co == "A" else "second_both"        # A first then B after swap = first slot twice


def decompose(prompts_orig, prompts_swap, preds_orig, preds_swap, cands, margin_orig=None):
    g = [classify(po, ps, a, b, cands) for po, ps, a, b in zip(prompts_orig, prompts_swap, preds_orig, preds_swap)]
    keep = [i for i, x in enumerate(g) if x != "ambiguous"]
    n = len(keep)
    d = max(n, 1)                                  # all-ambiguous input: shares are 0, not a crash
    out = dict(n=n, n_ambiguous=len(g) - n, share={k: sum(g[i] == k for i in keep) / d for k in GROUPS},
               position_driven=sum(g[i] in ("first_both", "second_both") for i in keep) / d)
    if margin_orig is not None and n >= 3:
        m = np.asarray(margin_orig)
        parts = np.array_split(np.array(keep)[np.argsort(m[keep], kind="stable")], 3)
        out["by_confidence"] = [{k: sum(g[i] == k for i in p) / len(p) for k in GROUPS}
                                | {"position_driven": sum(g[i] in ("first_both", "second_both") for i in p) / len(p)}
                                for p in parts]
    return out, g


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
    ap.add_argument("--out", default="decomp_results")
    args = ap.parse_args()
    if (args.step is None) == (args.revision is None):
        raise SystemExit("give exactly one of --step or --revision")
    revision = args.revision or f"step{args.step}"

    import torch
    import model_utils as mu
    import uncertainty_gate as ug
    from perturb import swap_chains
    from tasks import build_all

    items_all = build_all(args.n, args.seed, shots=2, layout="blocked")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = args.dtype or ("fp16" if "pythia" in args.model else "bf16")
    tok, model = mu.load(args.model, revision, dev,
                         (torch.float16 if dtype == "fp16" else torch.bfloat16) if dev == "cuda" else torch.float32)
    out = dict(model=args.model, revision=revision, seed=args.seed, tasks={})
    for t in args.tasks:
        items = items_all[t]
        orig = [it["prompt"] for it in items]
        swp = [swap_chains(p) for p in orig]
        y = np.array([it["label"] for it in items])
        cands = [str(c).strip() for c in items[0]["candidates"]]
        cids = mu.candidate_ids(tok, items[0]["candidates"])
        lo, _ = mu.run(tok, model, orig, cids, batch_size=args.batch_size, device=dev)
        ls, _ = mu.run(tok, model, swp, cids, batch_size=args.batch_size, device=dev)
        lpo, lps = torch.log_softmax(lo, -1).numpy(), torch.log_softmax(ls, -1).numpy()
        gate, raw = ug.analyse(lpo, lps, y)
        dec, groups = decompose(orig, swp, lpo.argmax(-1), lps.argmax(-1), cands, margin_orig=raw["margin_orig"])
        out["tasks"][t] = dict(gate=gate, decomposition=dec, groups=groups)
        sh = dec["share"]
        print(t, " ".join(f"{k} {sh[k]:.3f}" for k in GROUPS), f"| position-driven {dec['position_driven']:.3f}"
              + " | by confidence (low->high): " + " ".join(f"{c['position_driven']:.2f}" for c in dec["by_confidence"]), flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.model.split('/')[-1]}__{revision}.json"), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
