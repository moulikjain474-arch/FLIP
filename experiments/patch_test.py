"""Where inside the model does reordering change the answer? (exploratory activation patching)

For each question, run the SWAPPED prompt and record the residual stream at the last token
after every block. Then run the ORIGINAL prompt and, at one layer at a time, replace its
last-token state with the swapped run's. If the answer then becomes the swapped run's answer,
the choice that reordering changes has already reached the last token by that layer.

Reported per layer, over questions where the two orders give different answers:
  transfer = share whose patched answer equals the swapped-order answer.
Expect ~0 at early layers and ~1 at late layers; the layer where it rises is the result.

Also (exploratory, added 8 Oct after mentor feedback):
  * single-chain control: the same patch on prompts that contain ONLY the queried chain, source and
    target differing in the chain's start digit. transfer there = where a VALUE is read out; compare
    with the two-chain curve = where the CHOICE of chain is made.
  * per-question margin log P(first chain's value) - log P(second chain's value), original and swapped.
  * digit-token check: every candidate digit must be exactly one token at the end of the prompt.

  python patch_test.py --model EleutherAI/pythia-1.4b --step 143000
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
    ap.add_argument("--tasks", nargs="+", default=["vartrack_d1", "vartrack_d3"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=2, help="patch every k-th layer (plus the last)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-control", type=int, default=200, help="questions for the single-chain control (0 = skip)")
    ap.add_argument("--out", default="patch_results")
    args = ap.parse_args()
    if (args.step is None) == (args.revision is None):
        raise SystemExit("give exactly one of --step or --revision")
    revision = args.revision or f"step{args.step}"

    import torch
    import model_utils as mu
    from perturb import swap_chains, chain_values, single_chain_control
    from tasks import build_all

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.float16 if "pythia" in args.model else torch.bfloat16) if dev == "cuda" else torch.float32
    tok, model = mu.load(args.model, revision, dev, dtype)
    L = mu.n_layers(model)
    layers = sorted(set(range(args.every, L + 1, args.every)) | {1, L})
    items_all = build_all(args.n, args.seed, shots=2, layout="blocked")
    out = dict(model=args.model, revision=revision, n_layers=L, layers=layers, tasks={})

    for t in args.tasks:
        items = items_all[t]
        orig = [it["prompt"] for it in items]
        swp = [swap_chains(p) for p in orig]
        y = np.array([it["label"] for it in items])
        cids = mu.candidate_ids(tok, items[0]["candidates"])
        lo, _ = mu.run(tok, model, orig, cids, batch_size=args.batch_size, device=dev)
        ls, acts = mu.run(tok, model, swp, cids, read_layers=layers, batch_size=args.batch_size, device=dev)
        po, ps = lo.argmax(-1).numpy(), ls.argmax(-1).numpy()
        cands = [str(c).strip() for c in items[0]["candidates"]]
        bad = [c for c, i in zip(cands, cids) if len(tok(orig[0] + c, add_special_tokens=False).input_ids)
               - len(tok(orig[0], add_special_tokens=False).input_ids) != 1
               or tok(orig[0] + c, add_special_tokens=False).input_ids[-1] != i]
        print(t, "digit-token check:", "OK, all candidates are single tokens" if not bad else f"PROBLEM for {bad}", flush=True)

        def margins(prompts, logits):
            lp = torch.log_softmax(logits.float(), -1).numpy()
            out_m = []
            for i, pr in enumerate(prompts):
                vals, _ = chain_values(pr)
                try:
                    out_m.append(float(lp[i, cands.index(vals[0])] - lp[i, cands.index(vals[1])]))
                except ValueError:                      # a chain start value is not a candidate: keep the slot, mark it
                    out_m.append(None)
            return out_m
        differ = po != ps
        res = dict(n=int(len(y)), n_differ=int(differ.sum()), acc_orig=float((po == y).mean()),
                   acc_swap=float((ps == y).mean()), layers={}, digit_token_problems=bad,
                   margin_first_minus_second_orig=margins(orig, lo), margin_first_minus_second_swap=margins(swp, ls))
        for l in layers:
            pp = []
            for i in range(0, len(orig), args.batch_size):
                src = acts[l][i:i + args.batch_size]

                def put(h, _layer, src=src):
                    return src.to(h.device, h.dtype)

                lg, _ = mu.run(tok, model, orig[i:i + args.batch_size], cids, edit=(l, put),
                               batch_size=args.batch_size, device=dev)
                pp.append(lg.argmax(-1).numpy())
            pp = np.concatenate(pp)
            res["layers"][l] = dict(
                transfer=float((pp[differ] == ps[differ]).mean()) if differ.any() else None,
                keep_orig=float((pp[differ] == po[differ]).mean()) if differ.any() else None,
                acc=float((pp == y).mean()))
            print(t, "layer", l, res["layers"][l], flush=True)
        if args.n_control:
            k = min(args.n_control, len(orig))
            ctrl = []
            for i, p in enumerate(orig[:k]):
                c = single_chain_control(p, args.seed * 100003 + i)
                if c[2] in cands and c[3] in cands:      # both digits must be answer candidates
                    ctrl.append(c)
            k = len(ctrl)
            if k < 10:
                raise SystemExit(f"single-chain control: only {k} usable pairs (candidates {cands}); refusing to report")
            src_p, tgt_p = [c[0] for c in ctrl], [c[1] for c in ctrl]
            i_src, i_tgt = [cands.index(c[2]) for c in ctrl], [cands.index(c[3]) for c in ctrl]
            lsrc, cacts = mu.run(tok, model, src_p, cids, read_layers=layers, batch_size=args.batch_size, device=dev)
            ltgt, _ = mu.run(tok, model, tgt_p, cids, batch_size=args.batch_size, device=dev)
            res["control"] = dict(n=k, acc_src=float((lsrc.argmax(-1).numpy() == i_src).mean()),
                                  acc_tgt=float((ltgt.argmax(-1).numpy() == i_tgt).mean()), layers={})
            for l in layers:
                pp = []
                for i in range(0, k, args.batch_size):
                    src = cacts[l][i:i + args.batch_size]

                    def put(h, _layer, src=src):
                        return src.to(h.device, h.dtype)

                    lg, _ = mu.run(tok, model, tgt_p[i:i + args.batch_size], cids, edit=(l, put),
                                   batch_size=args.batch_size, device=dev)
                    pp.append(lg.argmax(-1).numpy())
                pp = np.concatenate(pp)
                res["control"]["layers"][l] = dict(transfer=float((pp == np.array(i_src)).mean()),
                                                   keep_target=float((pp == np.array(i_tgt)).mean()))
                print(t, "control layer", l, res["control"]["layers"][l], flush=True)
        out["tasks"][t] = res
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.model.split('/')[-1]}__{revision}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
