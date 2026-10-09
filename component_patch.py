"""Which sublayer carries the order-dependent choice? Component-level activation patching (audit A4, exploratory).

patch_test.py patches the whole last-token residual stream after block l. This patches ONE sublayer's output at the last
token: the attention output or the MLP output of block l. For each question we run the order-swapped prompt, record every
sublayer's last-token output, then run the original prompt and replace one sublayer's last-token output with the swapped
run's. Over questions where the two orders give different answers:
  transfer = share whose patched answer equals the swapped-order answer
  keep     = share whose patched answer still equals the original answer
Where attention-output transfer rises is where the choice of chain is moved to the last token; MLP-output transfer is
the other route. Patching one sublayer can only show that it carries part of the difference, not that it is the only place.

  python3 component_patch.py --model EleutherAI/pythia-6.9b --step 143000 --every 4
  python3 component_patch.py --selftest        # tiny random models on CPU: checks the patching machinery
"""
import argparse
import contextlib
import json
import os

import numpy as np


def parts(model):
    """[(attention module, MLP module)] per block, for GPT-NeoX/Pythia and Llama-style (incl. OLMo, Qwen) blocks."""
    import model_utils as mu
    out = []
    for layer in mu.get_layers(model):
        attn = next((getattr(layer, n) for n in ("attention", "self_attn", "attn") if hasattr(layer, n)), None)
        if attn is None or not hasattr(layer, "mlp"):
            raise ValueError("unknown block layout: need an attention module and .mlp")
        out.append((attn, layer.mlp))
    return out


@contextlib.contextmanager
def comp_hooks(model, read=(), edit=None):
    """read: iterable of (layer, 'attn'|'mlp') whose last-token output is recorded (before any edit, float32 cpu).
    edit: {(layer, 'attn'|'mlp'): tensor [B, d]} written into the last-token output of that sublayer.
    Layers are numbered 1..n like model_utils."""
    import model_utils as mu
    mods = parts(model)
    edit = edit or {}
    keys = set(read) | set(edit)
    store, handles = {}, []

    def make(key):
        def hook(_module, _inputs, out):
            h = mu._hidden(out)
            if key in read:
                store[key] = h[:, -1, :].float().cpu()
            if key in edit:
                h = h.clone()
                h[:, -1, :] = edit[key].to(h.device, h.dtype)
                out = mu._replace(out, h)
            return out
        return hook

    for (l, c) in sorted(keys):
        handles.append(mods[l - 1][0 if c == "attn" else 1].register_forward_hook(make((l, c))))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def run_comp(tok, model, prompts, cids, read=(), edit_fn=None, batch_size=32, device="cuda"):
    """Candidate logits [N, K] and {(layer, comp): [N, d]} last-token outputs. edit_fn(i0, i1) -> {(l, c): [B, d]}."""
    import torch
    import model_utils as mu
    cand = torch.tensor(cids, device=device)
    logits_all, acts = [], {k: [] for k in read}
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            enc, pos = mu.encode(tok, prompts[i:i + batch_size], device)
            edit = edit_fn(i, min(i + batch_size, len(prompts))) if edit_fn else None
            with comp_hooks(model, read, edit) as store:
                lg = mu.last_logits(model, enc, pos)
            logits_all.append(lg.index_select(-1, cand).float().cpu())
            for k in read:
                acts[k].append(store[k])
    return torch.cat(logits_all), {k: torch.cat(v) for k, v in acts.items()}


def selftest():
    import torch
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    common = dict(vocab_size=64, hidden_size=32, num_hidden_layers=3, num_attention_heads=2, intermediate_size=64,
                  max_position_embeddings=64)
    models = (GPTNeoXForCausalLM(GPTNeoXConfig(**common)),
              LlamaForCausalLM(LlamaConfig(num_key_value_heads=2, **common)))
    for model in models:
        model.eval()
        n = len(parts(model))
        a = torch.randint(0, 64, (4, 12))
        b = torch.randint(0, 64, (4, 12))
        b[:, -1] = a[:, -1]                               # same last token: same embedding at the last position
        keys = [(l, c) for l in range(1, n + 1) for c in ("attn", "mlp")]
        mask = torch.ones_like(a)

        def logits(ids, read=(), edit=None):
            with torch.no_grad(), comp_hooks(model, read, edit) as st:
                lg = model(input_ids=ids, attention_mask=mask).logits[:, -1, :]
            return lg, st
        la, sa = logits(a, read=keys)
        lb, sb = logits(b, read=keys)
        assert not torch.allclose(la, lb, atol=1e-3), "the two prompts must give different outputs"
        l0, _ = logits(a, edit={})                        # no edit: unchanged
        assert torch.allclose(l0, la, atol=1e-5)
        l_self, _ = logits(a, edit={k: sa[k] for k in keys})   # patch a's own outputs back: unchanged
        assert torch.allclose(l_self, la, atol=1e-4)
        l_all, _ = logits(a, edit={k: sb[k] for k in keys})    # every sublayer from b: equals b's output
        assert torch.allclose(l_all, lb, atol=1e-4), float((l_all - lb).abs().max())
        l_one, _ = logits(a, edit={(1, "attn"): sb[(1, "attn")]})   # one sublayer: changes, but not all the way
        assert not torch.allclose(l_one, la, atol=1e-6) and not torch.allclose(l_one, lb, atol=1e-4)
    print("component_patch selftest ok (GPT-NeoX and Llama: no edit, self edit, full edit, partial edit)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tasks", nargs="+", default=["vartrack_d1", "vartrack_d3"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=2, help="patch every k-th layer (plus the first and last)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="component_patch_results")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.model or (args.step is None) == (args.revision is None):
        raise SystemExit("give --model and exactly one of --step or --revision")
    revision = args.revision or f"step{args.step}"

    import torch
    import model_utils as mu
    from perturb import swap_chains
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
        keys = [(l, c) for l in layers for c in ("attn", "mlp")]
        lo, _ = run_comp(tok, model, orig, cids, batch_size=args.batch_size, device=dev)
        ls, src = run_comp(tok, model, swp, cids, read=keys, batch_size=args.batch_size, device=dev)
        po, ps = lo.argmax(-1).numpy(), ls.argmax(-1).numpy()
        differ = po != ps
        res = dict(n=int(len(y)), n_differ=int(differ.sum()), acc_orig=float((po == y).mean()),
                   acc_swap=float((ps == y).mean()), layers={})
        print(t, "questions where the two orders differ:", int(differ.sum()), "of", len(y), flush=True)
        for l in layers:
            res["layers"][l] = {}
            for c in ("attn", "mlp"):
                lg, _ = run_comp(tok, model, orig, cids, edit_fn=lambda i0, i1, l=l, c=c: {(l, c): src[(l, c)][i0:i1]},
                                 batch_size=args.batch_size, device=dev)
                pp = lg.argmax(-1).numpy()
                res["layers"][l][c] = dict(
                    transfer=float((pp[differ] == ps[differ]).mean()) if differ.any() else None,
                    keep=float((pp[differ] == po[differ]).mean()) if differ.any() else None,
                    acc=float((pp == y).mean()))
            a, m = res["layers"][l]["attn"], res["layers"][l]["mlp"]
            print(t, f"layer {l:>2}: attn transfer {a['transfer']:.3f} keep {a['keep']:.3f} | "
                  f"mlp transfer {m['transfer']:.3f} keep {m['keep']:.3f}" if differ.any() else f"layer {l}: no differing questions",
                  flush=True)
        out["tasks"][t] = res
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.model.split('/')[-1]}__{revision}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
