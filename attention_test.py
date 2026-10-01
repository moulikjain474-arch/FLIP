"""Where does the last token look: the first chain or the second? (exploratory attention analysis)

For each question (blocked layout), read the attention from the LAST prompt token to the
tokens of each chain, in every layer and head. Reports, per layer and head:
  later_minus_earlier   attention on the second chain minus attention on the first
and how that differs between questions where the model answered with the LATER chain's
value and those where it answered with the EARLIER chain's value. Heads where the gap is
large are candidates for the heads that choose the chain.

Hypotheses from the long-context literature (to check, not assumed): a preference for the
FIRST chain could come from attention drawn toward the start of the input (the "attention
sink" at the first tokens); a preference for the LAST chain from causal masking and position
decay favouring nearby tokens.

  python attention_test.py --model EleutherAI/pythia-1.4b --step 143000
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
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default="attention_results")
    args = ap.parse_args()
    if (args.step is None) == (args.revision is None):
        raise SystemExit("give exactly one of --step or --revision")
    revision = args.revision or f"step{args.step}"

    import torch
    import model_utils as mu
    from perturb import chain_char_spans, chain_values
    from tasks import build_all

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # fp32: eager attention in fp16 can overflow; one question at a time, so memory is fine (12B ~ 48 GB)
    tok, model = mu.load(args.model, revision, dev, torch.float32)
    # attention weights are only returned by the plain ("eager") attention implementation
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    else:
        model.config._attn_implementation = "eager"
    items_all = build_all(args.n, args.seed, shots=2, layout="blocked")
    out = dict(model=args.model, revision=revision, tasks={})

    for t in args.tasks:
        items = items_all[t]
        cids = torch.tensor(mu.candidate_ids(tok, items[0]["candidates"]), device=dev)
        diffs, picked_later, target_later, correct = [], [], [], []
        for it in items:
            p = it["prompt"]
            spans = chain_char_spans(p)
            vals, target = chain_values(p)
            enc = tok(p, return_offsets_mapping=True, add_special_tokens=False)
            offs = enc["offset_mapping"]
            idx = [[i for i, (s, e) in enumerate(offs) if e > s and s >= lo and e <= hi] for lo, hi in spans]
            assert idx[0] and idx[1], "a chain has no tokens"
            ids = torch.tensor([enc["input_ids"]], device=dev)
            with torch.no_grad():
                o = model(input_ids=ids, output_attentions=True)
            if o.attentions is None or o.attentions[0] is None:
                raise SystemExit("this model/transformers version did not return attention weights")
            att = torch.stack([a[0, :, -1, :].float() for a in o.attentions])        # [L, H, T]
            if not (torch.isfinite(att).all() and torch.isfinite(o.logits[0, -1]).all()):
                raise SystemExit("non-finite attention or logits: refusing to report garbage")
            first, second = att[:, :, idx[0]].sum(-1), att[:, :, idx[1]].sum(-1)     # [L, H]
            diffs.append((second - first).cpu().numpy())
            pred = int(o.logits[0, -1, cids].argmax())
            ans = str(it["candidates"][pred]).strip()
            picked_later.append(1 if ans == vals[1] else 0 if ans == vals[0] else -1)
            target_later.append(target == 1)
            correct.append(pred == it["label"])
        d = np.stack(diffs)                                       # [N, L, H]
        pl = np.array(picked_later)
        L, H = d.shape[1:]
        later, earlier = d[pl == 1], d[pl == 0]
        gap = (later.mean(0) - earlier.mean(0)) if len(later) and len(earlier) else np.full((L, H), np.nan)
        flat = [(abs(float(gap[l, h])), l + 1, h, float(gap[l, h])) for l in range(L) for h in range(H) if not np.isnan(gap[l, h])]
        top = sorted(flat, reverse=True)[:args.top]
        r = dict(n=int(len(items)), accuracy=float(np.mean(correct)), n_picked_later=int((pl == 1).sum()),
                 n_picked_earlier=int((pl == 0).sum()),
                 layer_later_minus_earlier=[float(x) for x in d.mean(axis=(0, 2))],
                 layer_gap_by_choice=[float(x) for x in np.nanmean(gap, axis=1)],
                 top_heads=[dict(layer=l, head=h, gap=g) for _, l, h, g in top],
                 head_gap=[[None if np.isnan(x) else float(x) for x in row] for row in gap])
        out["tasks"][t] = r
        print(t, "acc", round(r["accuracy"], 3), "picked later/earlier", r["n_picked_later"], r["n_picked_earlier"],
              "top heads", [(x["layer"], x["head"], round(x["gap"], 3)) for x in r["top_heads"][:5]], flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.model.split('/')[-1]}__{revision}.json"), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
